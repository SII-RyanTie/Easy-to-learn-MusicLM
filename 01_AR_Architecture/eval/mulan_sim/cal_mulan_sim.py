# import os
# import csv
# import argparse
# import torch
# import librosa
# from tqdm import tqdm
# from muq import MuQMuLan

# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"

# # ==========================================
# # 0. 命令行参数解析
# # ==========================================
# parser = argparse.ArgumentParser(description="批量计算音频与文本的 MuLan 相似度")
# parser.add_argument(
#     "--model_name", 
#     type=str, 
#     required=True, 
#     help="要评估的模型名称，例如: stable_audio_open 或 musicgen"
# )
# args = parser.parse_args()

# # ==========================================
# # 1. 初始化与配置
# # ==========================================
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# custom_cache_path = "/inspire/qb-ilm/project/powersystem/public/tieruian_public/hf_ckpt"

# # 动态拼接目录路径
# wav_dir = f"/inspire/hdd/project/powersystem/tieruian-253108120115/chenxie/music_project/output_MTG/{args.model_name}"
# pt_base_dir = "/inspire/qb-ilm/project/powersystem/public/tieruian_public/MTG_dataset/mtg_tokens_mucodec_val_w_MSMlabel"
# output_csv = f"/inspire/hdd/project/powersystem/tieruian-253108120115/chenxie/music_project/eval/mulan_sim/{args.model_name}.csv"

# # 安全检查：确保输入的音频文件夹存在
# if not os.path.exists(wav_dir):
#     raise FileNotFoundError(f"错误：找不到音频文件夹 {wav_dir}，请检查模型名称是否拼写正确！")

# # 安全检查：确保输出 CSV 的父文件夹存在，不存在则自动创建
# os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# print(f"当前评估模型: {args.model_name}")
# print("正在加载 MuLan 模型...")
# mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir=custom_cache_path)
# mulan = mulan.to(device).eval()

# # 获取目录下所有的 .wav 文件
# wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
# results = []

# print(f"共找到 {len(wav_files)} 个音频文件，开始批量处理...")

# # ==========================================
# # 2. 核心遍历逻辑
# # ==========================================
# for wav_name in tqdm(wav_files):
#     track_id = os.path.splitext(wav_name)[0]
#     wav_path = os.path.join(wav_dir, wav_name)
#     pt_path = os.path.join(pt_base_dir, f"{track_id}.pt")
    
#     # --- 步骤 A: 读取对应的文本提示词 ---
#     if not os.path.exists(pt_path):
#         print(f"\n[跳过] 找不到对应的标签文件: {pt_path}")
#         continue
        
#     try:
#         pt_data = torch.load(pt_path, map_location='cpu')
#         text_prompt = pt_data.get('text_prompt')
#         if not text_prompt:
#             print(f"\n[跳过] PT文件中缺失 text_prompt 键: {pt_path}")
#             continue
#     except Exception as e:
#         print(f"\n[错误] 读取 {pt_path} 失败: {e}")
#         continue

#     # --- 步骤 B: 提取音频与文本特征，计算相似度 ---
#     try:
#         wav, sr = librosa.load(wav_path, sr=24000)
#         wavs = torch.tensor(wav).unsqueeze(0).to(device)
        
#         with torch.no_grad():
#             audio_embeds = mulan(wavs=wavs)
#             text_embeds = mulan(texts=[text_prompt])
#             sim = mulan.calc_similarity(audio_embeds, text_embeds)[0][0].item()
            
#             results.append({
#                 "filename": wav_name, 
#                 "similarity": round(sim, 4)
#             })
            
#     except Exception as e:
#         print(f"\n[错误] 处理音频 {wav_name} 时发生异常: {e}")

# # ==========================================
# # 3. 结果保存
# # ==========================================
# print(f"\n处理完成！正在将 {len(results)} 条结果保存到 {output_csv} ...")

# with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
#     writer = csv.DictWriter(f, fieldnames=["filename", "similarity"])
#     writer.writeheader()
#     writer.writerows(results)

# print("🎉 所有任务已圆满结束！")

import os
import csv
import argparse
import torch
import librosa
from tqdm import tqdm
from muq import MuQMuLan

# ==========================================
# 💡 核心：必须在 import datasets 之前设置离线模式
# ==========================================
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from datasets import load_dataset

# ==========================================
# 0. 命令行参数解析
# ==========================================
parser = argparse.ArgumentParser(description="批量计算音频与文本的 MuLan 相似度")
parser.add_argument(
    "--model_name", 
    type=str, 
    required=True, 
    help="要评估的模型名称，例如: stable_audio_open 或 musicgen_large"
)
args = parser.parse_args()

# ==========================================
# 1. 初始化与配置
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
custom_cache_path = "/inspire/qb-ilm/project/powersystem/public/tieruian_public/hf_ckpt"

# 动态拼接目录路径
wav_dir = f"/inspire/hdd/project/powersystem/tieruian-253108120115/chenxie/music_project/outputs_MusicCaps/{args.model_name}"
output_csv = f"/inspire/hdd/project/powersystem/tieruian-253108120115/chenxie/music_project/eval/mulan_sim/{args.model_name}.csv"

# 安全检查
if not os.path.exists(wav_dir):
    raise FileNotFoundError(f"错误：找不到音频文件夹 {wav_dir}，请检查模型名称是否拼写正确！")
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# ==========================================
# 2. 加载 MusicCaps 文本数据 (构建查找字典)
# ==========================================
print("🔍 正在离线加载 MusicCaps 数据集以获取文本提示词...")
ds = load_dataset("google/MusicCaps", cache_dir=custom_cache_path)

# 💡 必须保持与生成音频时完全一致的 seed 和抽取数量！
SEED = 42
sampled_data = ds['train'].shuffle(seed=SEED).select(range(100))

# 构建 { ytid : caption } 的字典，方便后续 O(1) 极速查找
ytid_to_caption = {row['ytid']: row['caption'] for row in sampled_data}
print(f"✅ 成功提取了 {len(ytid_to_caption)} 条文本提示词映射！")

# ==========================================
# 3. 加载 MuLan 模型
# ==========================================
print(f"\n当前评估模型: {args.model_name}")
print("🧠 正在加载 MuLan 模型...")
mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir=custom_cache_path)
mulan = mulan.to(device).eval()

wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
results = []

print(f"🎵 共找到 {len(wav_files)} 个音频文件，开始批量计算相似度...")

# ==========================================
# 4. 核心遍历逻辑
# ==========================================
for wav_name in tqdm(wav_files):
    track_id = os.path.splitext(wav_name)[0] # 提取 ytid
    wav_path = os.path.join(wav_dir, wav_name)
    
    # --- 步骤 A: 从字典中极速获取文本提示词 ---
    if track_id not in ytid_to_caption:
        print(f"\n[跳过] 找不到 {wav_name} 对应的文本描述 (可能不在抽取的100条内)")
        continue
        
    text_prompt = ytid_to_caption[track_id]

    # --- 步骤 B: 提取音频与文本特征，计算相似度 ---
    try:
        wav, sr = librosa.load(wav_path, sr=24000)
        wavs = torch.tensor(wav).unsqueeze(0).to(device)
        
        with torch.no_grad():
            audio_embeds = mulan(wavs=wavs)
            text_embeds = mulan(texts=[text_prompt])
            sim = mulan.calc_similarity(audio_embeds, text_embeds)[0][0].item()
            
            results.append({
                "filename": wav_name, 
                "similarity": round(sim, 4)
            })
            
    except Exception as e:
        print(f"\n[错误] 处理音频 {wav_name} 时发生异常: {e}")

# ==========================================
# 5. 结果保存
# ==========================================
print(f"\n处理完成！正在将 {len(results)} 条结果保存到 {output_csv} ...")

with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "similarity"])
    writer.writeheader()
    writer.writerows(results)

print("🎉 所有任务已圆满结束！")
