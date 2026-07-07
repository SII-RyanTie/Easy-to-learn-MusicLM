import torch
from tqdm import tqdm

import torch.distributed as dist
import soundfile as sf
import torch.nn.functional as F
import random

# test.py 中的 generate 函数修改思路

@torch.no_grad()
def generate_delay(args, model, start_tokens, text_embeds,mask, max_new_tokens=200, temperature=1.0, rank=None):
    model.eval()
    
    x = start_tokens.to(rank)
    K = args.num_codebooks

    for step in range(max_new_tokens):
        x_cond = x[:, -args.max_seq_len:]
        logits = model(x_cond, text_embeds,mask)  # [B, T, K, vocab]
        
        next_tokens = []
        for k in range(K):
            last_logits = logits[:, -1, k, :] / temperature
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_tokens.append(next_token)
            
        next_tokens = torch.stack(next_tokens, dim=-1)  # [B, 1, K]
        x = torch.cat([x, next_tokens], dim=1)

    # ==========================================
    # 全局反向对齐 (Delay 还原)
    # ==========================================
    T_total = x.shape[1]
    aligned_full = torch.zeros_like(x)
    
    for k in range(K):
        valid_len = T_total - k
        if valid_len > 0:
            aligned_full[:, :valid_len, k] = x[:, k:, k]
            
    # 如果是多码本，截断最后 K 步；如果是单码本 (K=1)，截断 1 步
    aligned_full = aligned_full[:, :-K, :] if K > 1 else aligned_full[:, :-1, :]
    
    return aligned_full



def test(args, model, rank, world_size, test_loader,decoder,val_loss1,current_time,model_name,iteration,writer=None):
    # raw_model = model.module if hasattr(model, "module") else model
    model.eval()  # 注意这里用 raw_model.eval()
    counter = torch.zeros(1).to(rank)

    MAX_K = 8 
    loss_boxes = [torch.zeros(1).to(rank) for _ in range(MAX_K)]


    test_iter = tqdm(test_loader, desc=f"Val Epoch {iteration}") if rank == 0 else test_loader
    # iteration_base = len(test_loader) * epoch
    
    with torch.no_grad():
        target_bidx = random.randint(0, len(test_loader) - 1) if rank == 0 else -1
        for bidx, data in enumerate(test_iter):
            # iteration = iteration_base + bidx
            x,y, text_embeds, attention_mask, prompt = data
            x,y, text_embeds, attention_mask = x.to(rank), y.to(rank), text_embeds.to(rank), attention_mask.to(rank)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x, text_embeds, attention_mask)
                
                B, T, K, V = logits.shape
                
                # 💡 动态计算每个码本的 Loss
                for k in range(K):
                    cb_loss = val_loss1(logits[:, :, k, :].reshape(-1, V), y[:, :, k].reshape(-1))
                    loss_boxes[k] += cb_loss * x.size(0)


            counter += x.size(0) 
            
            if rank == 0 and bidx == target_bidx:
                x_sample = x[0]
                text_sample = text_embeds[0].unsqueeze(0)
                mask_sample = attention_mask[0].unsqueeze(0)
                prompt_sample = prompt[0]
                start = x_sample[:512].unsqueeze(0)
                
                generated_tokens = generate_delay(
                    args,
                    model,
                    start_tokens=start,
                    text_embeds = text_sample,
                    mask = mask_sample,
                    max_new_tokens=512,
                    temperature=1.0,
                    rank = rank
                )
                
                for k in range(K):
                    print(f"Codebook {k} unique tokens:", generated_tokens[0, :, k].unique().shape[0])
                    
                codes = generated_tokens[0].permute(1, 0).unsqueeze(0).to(rank)
                
                with torch.no_grad():

                    wav = decoder.code2sound(codes)
                    audio = wav[0].cpu().numpy()
                    sample_rate = 48000
                    sf.write(f"./log/{current_time}_{model_name}/pictures/{iteration}.wav", audio, sample_rate)
                
            


    dist.all_reduce(counter, op=dist.ReduceOp.SUM)

    counter = counter / world_size

    for k in range(K): # 这里的 K 是最后一个 batch 的 K，依然有效
        dist.all_reduce(loss_boxes[k], op=dist.ReduceOp.SUM)
        loss_boxes[k] = loss_boxes[k] / world_size
        
        if rank == 0:
            loss_mean = (loss_boxes[k] / counter).cpu().numpy()
            writer.add_scalar(f'val/cb{k+1}', loss_mean, global_step=iteration) 

