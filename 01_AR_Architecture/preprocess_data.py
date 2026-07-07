import os
import sys
import json
import math
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
import torch.multiprocessing as mp
from transformers import T5Tokenizer, T5EncoderModel

# ==========================================
# Workaround: Resolve import conflicts between huggingface_hub and legacy diffusers
# ==========================================
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    def dummy_cached_download(*args, **kwargs):
        return huggingface_hub.hf_hub_download(*args, **kwargs)
    huggingface_hub.cached_download = dummy_cached_download

# ==========================================
# TODO: User Configuration Area
# Update these paths according to your local environment
# ==========================================
os.environ["HF_HOME"] = "./hf_cache"

MTG_BASE_DIR = "./data/MTG_dataset"
JSONL_PATH = os.path.join(MTG_BASE_DIR, "MTGTop50.val.jsonl")
OUTPUT_DIR = os.path.join(MTG_BASE_DIR, "mtg_tokens_mucodec_val_w_MSMlabel")

MUCODEC_CKPT_PATH = "./MuCodec/ckpt/mucodec.pt" 
T5_MODEL_NAME = "google/flan-t5-base"
MOSS_MODEL_PATH = "./MOSS_Music/weights/MOSS-Music-8B-Instruct"

# Audio processing constants
TARGET_SR = 48000
CHUNK_SEC = 10
TOKENS_PER_SEC = 25  # MuCodec feature frame rate (48000 / 1920 ≈ 25)
TOKENS_PER_CHUNK = CHUNK_SEC * TOKENS_PER_SEC

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Append local module paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "MuCodec"))
sys.path.append(os.path.join(current_dir, "MOSS_Music"))

from generate import MuCodec
from MOSS_Music.src.audio_io import load_audio
from MOSS_Music.src.modeling_moss_music import MossMusicModel
from MOSS_Music.src.processing_moss_music import MossMusicProcessor

# ==========================================
# Core Processing Function (Runs independently on each GPU)
# ==========================================
def process_chunk_on_gpu(gpu_id, df_chunk, ckpt_path):
    device = f"cuda:{gpu_id}"
    
    # 1. Initialize MuCodec
    model = MuCodec(model_path=ckpt_path, layer_num=7, load_main_model=True, device=device)

    # 2. Initialize FLAN-T5
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
    t5_model = T5EncoderModel.from_pretrained(T5_MODEL_NAME).to(device)
    t5_model.eval()

    # 3. Initialize MOSS-Music (Using bfloat16 to optimize VRAM usage)
    moss_processor = MossMusicProcessor.from_pretrained(
        MOSS_MODEL_PATH, trust_remote_code=True, enable_time_marker=True
    )
    moss_model = MossMusicModel.from_pretrained(
        MOSS_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, 
        device_map=device,          
    )
    moss_model.eval()

    local_index = []
    local_chunks_processed = 0

    for idx, row in tqdm(df_chunk.iterrows(), total=len(df_chunk), position=gpu_id, desc=f"GPU {gpu_id}"):
        
        audio_path = os.path.join(MTG_BASE_DIR, row['audio_path'].replace("data/MTG/", "")) 
        track_id = os.path.basename(audio_path).split('.')[0]
        
        if not os.path.exists(audio_path):
            continue

        # --- A. Generate high-quality text descriptions via MOSS-Music ---
        try:
            raw_audio_moss = load_audio(audio_path, sample_rate=moss_processor.config.mel_sr)
            prompt = "Describe this music track in one concise sentence, highlighting its genre, main instruments, and mood. Keep it under 15 words."
            
            inputs = moss_processor(text=prompt, audios=[raw_audio_moss], return_tensors="pt").to(device)
            
            if inputs.get("audio_data") is not None:
                inputs["audio_data"] = inputs["audio_data"].to(moss_model.dtype)
                
            inputs["audio_input_mask"] = inputs["input_ids"] == moss_processor.audio_token_id

            with torch.no_grad():
                generated_ids = moss_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    num_beams=1,
                    temperature=1.0,
                    top_p=0.8,
                    top_k=50,
                    use_cache=True,
                )

            input_len = inputs["input_ids"].shape[1]
            text_prompt = moss_processor.decode(generated_ids[0, input_len:], skip_special_tokens=True).strip()
            
            if not text_prompt:
                text_prompt = "A generic music track."
                
        except Exception as e:
            print(f"MOSS Inference failed for {track_id}: {e}")
            continue 

        # --- B. Extract T5 text embeddings ---
        with torch.no_grad():
            text_inputs = tokenizer(text_prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            text_embeds = t5_model(**text_inputs).last_hidden_state.cpu() 

        # --- C. Load, resample, and convert audio to stereo for MuCodec ---
        try:
            wav, sr = torchaudio.load(audio_path)
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                wav = resampler(wav)
            
            if wav.shape[0] == 1:
                wav = torch.cat([wav, wav], dim=0)
            elif wav.shape[0] > 2:
                wav = wav[:2, :]
                
            wav = wav.to(device)
        except Exception:
            continue # Skip corrupted audio files

        # --- D. Extract MuCodec discrete tokens and split into chunks ---
        try:
            with torch.no_grad():
                codes = model.sound2code(wav) 
                codes = codes.squeeze(0).cpu() 
        except Exception:
            continue

        total_tokens = codes.shape[1]
        song_tokens = []
        
        for start in range(0, total_tokens, TOKENS_PER_CHUNK):
            chunk_codes = codes[:, start:start + TOKENS_PER_CHUNK]
            if chunk_codes.shape[1] < TOKENS_PER_CHUNK:
                continue
            song_tokens.append(chunk_codes)

        if len(song_tokens) == 0:
            continue
            
        local_chunks_processed += len(song_tokens)

        # --- E. Save processed data ---
        save_filename = f"track_{track_id}.pt"
        save_path = os.path.join(OUTPUT_DIR, save_filename)
        
        save_data = {
            "track_id": track_id,
            "tokens": song_tokens,
            "text_prompt": text_prompt,        
            "text_embeds": text_embeds,        
            "num_chunks": len(song_tokens),
            "sample_rate": TARGET_SR,
            "chunk_size_sec": CHUNK_SEC,
            "codec": "mucodec_layer7"
        }
        torch.save(save_data, save_path)

        local_index.append({"track_id": track_id, "file": save_filename})

    return local_index, local_chunks_processed

# ==========================================
# Main Execution (Task distribution and merging)
# ==========================================
if __name__ == '__main__':
    # Spawn method is required for CUDA multiprocessing
    mp.set_start_method('spawn', force=True)
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No GPUs found! Please check your CUDA environment.")
    
    print(f"🔥 Detected {num_gpus} GPUs. Preparing for parallel processing...")

    df_meta = pd.read_json(JSONL_PATH, lines=True)
    total_tracks = len(df_meta)
    print(f"Total tracks to process: {total_tracks}")

    # Split dataset evenly across available GPUs
    chunk_size = math.ceil(total_tracks / num_gpus)
    df_chunks = [df_meta.iloc[i:i + chunk_size] for i in range(0, total_tracks, chunk_size)]

    print("Starting multi-processing pool...")
    
    with mp.Pool(processes=num_gpus) as pool:
        results = []
        for gpu_id in range(num_gpus):
            if gpu_id < len(df_chunks):
                res = pool.apply_async(process_chunk_on_gpu, args=(gpu_id, df_chunks[gpu_id], MUCODEC_CKPT_PATH))
                results.append(res)
        
        pool.close()
        pool.join()

    # Merge results from all GPUs
    print("\nAll GPUs finished! Merging index data...")
    final_music_list = []
    total_chunks = 0

    for res in results:
        local_idx, local_chunks = res.get()
        final_music_list.extend(local_idx)
        total_chunks += local_chunks

    final_index_data = {
        "num_music": len(final_music_list),
        "codec": "mucodec_layer7",
        "sample_rate": TARGET_SR,
        "chunk_size_sec": CHUNK_SEC,
        "tokens_per_chunk": TOKENS_PER_CHUNK,
        "music": final_music_list
    }

    index_path = os.path.join(OUTPUT_DIR, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(final_index_data, f, indent=2, ensure_ascii=False)

    print(f"✅ All done! Successfully processed {len(final_music_list)} tracks.")
    print(f"Total {CHUNK_SEC}s chunks generated: {total_chunks}")
    print(f"Index saved to: {index_path}")
