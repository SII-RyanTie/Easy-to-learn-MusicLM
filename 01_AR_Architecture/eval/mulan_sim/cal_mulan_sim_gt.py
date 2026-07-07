import os
import json
import re
import csv
import torch
import librosa
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 💡 核心：必须在 import 之前设置离线模式
# ==========================================
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from muq import MuQMuLan

# ==========================================
# 1. 单卡工作进程 (每张卡一首一首处理)
# ==========================================
def worker(rank, world_size, dataset, output_csv, custom_cache_path):
    device = torch.device(f'cuda:{rank}')
    
    # 1. 数据切片：当前卡只领走属于自己的那部分数据
    my_data = dataset[rank::world_size]
    if len(my_data) == 0:
        return

    # 2. 加载模型到当前专属 GPU
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir=custom_cache_path)
    mulan = mulan.to(device).eval()

    results = []
    temp_csv = output_csv.replace('.csv', f'_rank{rank}.csv')

    # 仅让主卡 (rank 0) 打印进度条，保持终端整洁
    iterator = tqdm(my_data, desc=f"GPU 0 进度", unit="首") if rank == 0 else my_data

    # 3. 核心：一首一首处理 (保持你原有的极简逻辑)
    for item in iterator:
        track_id = item["track_id"]
        audio_path = item["audio_path"]
        text_prompt = item["caption"]
        
        try:
            wav, sr = librosa.load(audio_path, sr=24000)
            wavs = torch.tensor(wav).unsqueeze(0).to(device)
            
            with torch.no_grad():
                audio_embeds = mulan(wavs=wavs)
                text_embeds = mulan(texts=[text_prompt])
                sim = mulan.calc_similarity(audio_embeds, text_embeds)[0][0].item()
                
                results.append({
                    "track_id": track_id, 
                    "similarity": round(sim, 4),
                    "audio_path": audio_path,
                    "caption": text_prompt
                })
                
        except Exception as e:
            if rank == 0:
                tqdm.write(f"❌ GPU {rank} 处理音频 {track_id} 异常: {e}")
            else:
                print(f"❌ GPU {rank} 处理音频 {track_id} 异常: {e}")

    # 4. 保存当前卡的临时结果
    with open(temp_csv, mode='w', newline='', encoding='utf-8') as f:
        fieldnames = ["track_id", "similarity", "audio_path", "caption"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

# ==========================================
# 2. 主程序 (数据准备与多进程启动)
# ==========================================
def main():
    custom_cache_path = "/inspire/qb-ilm/project/powersystem/public/tieruian_public/hf_ckpt"
    pt_dir = "/inspire/qb-ilm/project/powersystem/public/tieruian_public/MTG_dataset/mtg_tokens_mucodec_val_w_MSMlabel"
    audio_base_dir = "/inspire/qb-ilm/project/powersystem/public/tieruian_public/MTG_dataset/audio"
    jsonl_path = "/inspire/qb-ilm/project/powersystem/public/tieruian_public/MTG_dataset/MTGTop50.val.jsonl"
    
    output_csv = "/inspire/hdd/project/powersystem/tieruian-253108120115/chenxie/music_project/eval/mulan_sim/MTG_ground_truth_MSMlabel.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # --- 1. 数据准备 ---
    id_to_audiopath = {}
    print("🔍 [1/3] 正在解析 JSONL 文件建立路径映射...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            raw_path = data.get("audio_path", "")
            if raw_path:
                filename = raw_path.split('/')[-1] 
                track_id = filename.split('.')[0]  
                sub_folder = raw_path.split('/')[-2] 
                id_to_audiopath[track_id] = os.path.join(audio_base_dir, sub_folder, filename)

    input_path = Path(pt_dir)
    all_pt_files = sorted([p.resolve() for p in input_path.iterdir() if p.is_file() and p.suffix == '.pt' and 'track' in p.name])

    final_dataset = []
    print(f"📂 [2/3] 找到 {len(all_pt_files)} 个 PT 文件，正在提取文本标签...")
    for pt_file in tqdm(all_pt_files, desc="提取文本进度"):
        match = re.search(r'\d+', pt_file.name)
        if match:
            track_id = match.group()
            if track_id in id_to_audiopath:
                try:
                    pt_data = torch.load(pt_file, map_location='cpu')
                    caption = pt_data.get('text_prompt')
                    if caption:
                        final_dataset.append({
                            "track_id": track_id,
                            "audio_path": id_to_audiopath[track_id],
                            "caption": caption
                        })
                except:
                    pass

    print(f"✅ 成功获取了 {len(final_dataset)} 条数据！\n")

    # --- 2. 启动多进程 ---
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise ValueError("未检测到 GPU！")
        
    print(f"🚀 [3/3] 检测到 {world_size} 张 GPU，开始分发任务 (每张卡一首一首处理)...")
    
    # 启动多进程，把数据分给各个 GPU
    mp.spawn(
        worker,
        args=(world_size, final_dataset, output_csv, custom_cache_path),
        nprocs=world_size,
        join=True
    )

    # --- 3. 合并多卡结果 ---
    print("\n💾 正在合并所有 GPU 的计算结果...")
    all_results = []
    for rank in range(world_size):
        temp_csv = output_csv.replace('.csv', f'_rank{rank}.csv')
        if os.path.exists(temp_csv):
            with open(temp_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                all_results.extend(list(reader))
            os.remove(temp_csv) # 合并完直接删掉临时文件

    # 写入最终的 CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "similarity", "audio_path", "caption"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"🎉 完美收工！共保存 {len(all_results)} 条结果至: {output_csv}")

if __name__ == "__main__":
    main()
