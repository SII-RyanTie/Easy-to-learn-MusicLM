import os
import yaml
import random
import argparse
import torch
import soundfile as sf
import torch.nn.functional as F
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 💡 Offline Mode Configuration
# Note: If running on a server without internet access, ensure the 
# datasets and models are pre-downloaded to your cache directory.
# ==========================================
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from datasets import load_dataset
from transformers import T5Tokenizer, T5EncoderModel, set_seed
import huggingface_hub

# Workaround: Resolve import conflicts for legacy huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    def dummy_cached_download(*args, **kwargs):
        return huggingface_hub.hf_hub_download(*args, **kwargs)
    huggingface_hub.cached_download = dummy_cached_download

from MuCodec.generate import MuCodec
from model.music_transformer_mucodec_w_label import MusicTransformerText_delay

# ==========================================
# Core Generation & Delay Pattern Logic
# ==========================================
@torch.no_grad()
def generate_delay(args, model, start_tokens, prompt_text_token, attention_mask=None, max_new_tokens=200, temperature=1.0, rank=None, cfg_scale=3.0):
    """
    Autoregressive generation loop with Classifier-Free Guidance (CFG) 
    and multi-codebook delay pattern support.
    """
    model.eval()
    x = start_tokens.to(rank)
    K = args.num_codebooks
    B = x.size(0)

    # Prepare unconditional embeddings for CFG (all zeros)
    uncond_text_embeds = torch.zeros((B, 1, 768), dtype=torch.float32, device=rank)
    uncond_mask = torch.ones((B, 1), dtype=torch.bool, device=rank)

    for step in range(max_new_tokens):
        x_cond = x[:, -args.max_seq_len:]
        
        # Conditional forward pass
        cond_logits = model(x_cond, prompt_text_token, attention_mask)
        
        # Classifier-Free Guidance (CFG) logic
        if cfg_scale != 1.0:
            uncond_logits = model(x_cond, uncond_text_embeds, uncond_mask)
            logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
        else:
            logits = cond_logits
        
        # Sample next tokens for each codebook
        next_tokens = []
        for k in range(K):
            last_logits = logits[:, -1, k, :] / temperature
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_tokens.append(next_token)
            
        next_tokens = torch.stack(next_tokens, dim=-1)
        x = torch.cat([x, next_tokens], dim=1)

    # Realign tokens (undo the delay pattern)
    T_total = x.shape[1]
    aligned_full = torch.zeros_like(x)
    
    for k in range(K):
        valid_len = T_total - k
        if valid_len > 0:
            aligned_full[:, :valid_len, k] = x[:, k:, k]
            
    aligned_full = aligned_full[:, :-K, :] if K > 1 else aligned_full[:, :-1, :]
    return aligned_full

def apply_delay_pattern(raw_tokens, bos_token=16384):
    """Applies the delay pattern required for multi-codebook autoregressive modeling."""
    T, K = raw_tokens.shape
    delayed_seq = torch.full((T + K - 1, K), bos_token, dtype=raw_tokens.dtype)
    for k in range(K):
        delayed_seq[k : T + k, k] = raw_tokens[:, k]
    return delayed_seq.unsqueeze(0)

# ==========================================
# Multi-GPU Worker Process
# ==========================================
def worker(rank, world_size, args, config, all_prompts, all_filenames):
    """
    Independent worker process for parallel inference.
    Args:
        rank: Current GPU ID (0, 1, 2...)
        world_size: Total number of available GPUs
    """
    device = torch.device(f'cuda:{rank}')
    
    # 1. Dataset Chunking: Assign a specific slice of data to this GPU
    chunk_size = len(all_prompts) // world_size
    start_idx = rank * chunk_size
    # The last GPU processes any remaining samples
    end_idx = start_idx + chunk_size if rank != world_size - 1 else len(all_prompts)
    
    my_prompts = all_prompts[start_idx:end_idx]
    my_filenames = all_filenames[start_idx:end_idx]
    
    if len(my_prompts) == 0:
        return # Exit if no data is assigned to this GPU

    # 2. Initialize Models on the specific GPU
    bos_token = 16384
    codec = MuCodec(model_path=args.mucodec_path, layer_num=7, load_main_model=True, device=device)

    tokenizer = T5Tokenizer.from_pretrained(args.t5_model_name)
    t5_model = T5EncoderModel.from_pretrained(args.t5_model_name).to(device)
    t5_model.eval()

    model = MusicTransformerText_delay(
        vocab_size=config.get('vocab_size', 1024),
        num_codebooks=config.get('num_codebooks', 4),
        embed_dim=config.get('embed_dim', 256),
        max_seq_len=config.get('max_seq_len', 1024),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        dropout=0.0
    ).to(device)
    
    state_dict = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict['weights'], strict=True)

    # 3. Setup Generation Parameters
    fps = 25
    gen_len = int(args.gen_sec * fps)
    num_codebooks = config.get('num_codebooks', 4)

    dummy_prompt = torch.zeros((1, num_codebooks), dtype=torch.long)
    delayed_start_pattern = apply_delay_pattern(dummy_prompt, bos_token=bos_token)

    # 4. Inference Loop
    # Only display the progress bar on GPU 0 to prevent terminal clutter
    iterator = range(0, len(my_prompts), args.batch_size)
    if rank == 0:
        iterator = tqdm(iterator, desc=f"GPU 0 Progress")

    for i in iterator:
        batch_texts = my_prompts[i : i + args.batch_size]
        batch_filenames = my_filenames[i : i + args.batch_size]
        current_b_size = len(batch_texts)
        
        try:
            with torch.no_grad():
                text_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                text_embeds = t5_model(**text_inputs).last_hidden_state
                attention_mask = text_inputs.attention_mask.bool()

            start_tokens_tensor = delayed_start_pattern.repeat(current_b_size, 1, 1).to(device)

            aligned_outputs = generate_delay(
                args, model, start_tokens_tensor, text_embeds, attention_mask,
                max_new_tokens=gen_len, temperature=args.temperature, rank=device, cfg_scale=args.cfg_scale
            )

            # Decode tokens to audio and save
            for b_idx in range(current_b_size):
                codes = aligned_outputs[b_idx, 1:].permute(1, 0).unsqueeze(0) 
                codes = torch.clamp(codes, min=0, max=bos_token-1)
                
                with torch.no_grad():
                    wav = codec.code2sound(codes)
                    audio_data = wav[0].cpu().numpy()
                
                out_filename = f"{batch_filenames[b_idx]}.wav"
                out_path = os.path.join(args.output_dir, out_filename)
                sf.write(out_path, audio_data, samplerate=48000)
                
        except Exception as e:
            print(f"\n❌ Error processing batch on GPU {rank}: {e}")
            continue

# ==========================================
# Main Entry Point
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Inference Script for Music Generation")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the model checkpoint (.pt)")
    parser.add_argument('--output_dir', type=str, default="./outputs/my_model", help="Directory to save generated audio")
    parser.add_argument('--cache_dir', type=str, default="./hf_cache", help="Hugging Face cache directory")
    parser.add_argument('--t5_model_name', type=str, default="google/flan-t5-base", help="T5 model name or local path")
    parser.add_argument('--mucodec_path', type=str, default="./MuCodec/ckpt/mucodec.pt", help="Path to MuCodec checkpoint")
    parser.add_argument('--num_samples', type=int, default=100, help="Number of random samples to generate")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size per GPU") 
    parser.add_argument('--gen_sec', type=float, default=10.0, help="Length of generated audio in seconds")
    parser.add_argument('--temperature', type=float, default=1.0, help="Sampling temperature")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--cfg_scale', type=float, default=3.0, help="Classifier-Free Guidance (CFG) scale")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("🔍 Loading MusicCaps dataset (Offline Mode)...")
    ds = load_dataset("google/MusicCaps", cache_dir=args.cache_dir)
    sampled_data = ds['train'].shuffle(seed=args.seed).select(range(args.num_samples))
    
    all_prompts = [row['caption'] for row in sampled_data]
    all_filenames = [row['ytid'] for row in sampled_data]
    print(f"✅ Successfully extracted {len(all_prompts)} prompts.")

    print(f"📄 Loading configuration from: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        vars(args).update(config)

    # Detect GPUs and spawn processes
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise ValueError("No GPUs detected! Please check your CUDA environment.")
        
    print(f"🚀 Detected {world_size} GPUs. Starting parallel inference...")
    
    mp.spawn(
        worker,
        args=(world_size, args, config, all_prompts, all_filenames),
        nprocs=world_size,
        join=True
    )
    
    print(f"\n🎉 Generation complete! Audio files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
