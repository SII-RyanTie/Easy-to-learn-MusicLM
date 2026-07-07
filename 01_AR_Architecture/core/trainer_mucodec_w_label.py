from tqdm import tqdm
import torch.distributed as dist
import torch
import torch.nn.functional as F
from core.test_mucodec_w_label import test
import os

def train(args, model, rank, world_size, train_loader, test_loader, codec, optimizer, epoch,train_loss,grad_scaler,current_time,model_name,scheduler=None,sampler=None,writer=None):
    model.train()
    if sampler:
        sampler.set_epoch(epoch)

    train_iter = tqdm(train_loader, desc=f"Train Epoch {epoch}") if rank == 0 else train_loader
    iteration_base = len(train_loader) * epoch
    for batch_idx, data in enumerate(train_iter):
        iteration = iteration_base + batch_idx
        
        x,y, text_embeds, attention_mask, _ = data
        x,y, text_embeds, attention_mask = x.to(rank), y.to(rank), text_embeds.to(rank), attention_mask.to(rank)

        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x, text_embeds, attention_mask)
            
            B, T, K, V = logits.shape  # V 就是词表大小 (1024 或 16384)
            
            weights = [1.0] * K 
            
            
            loss = 0
            
            for k in range(K):
                l_k = train_loss(
                    logits[:, :, k, :].reshape(-1, V),  # 👈 动态词表大小 V
                    y[:, :, k].reshape(-1)
                )
                loss += weights[k] * l_k
        
        loss_value = loss
        loss_value_for_reduce = loss_value.detach().clone()

        ####使用FP16才用
        # grad_scaler.scale(loss_value).backward()
        # grad_scaler.unscale_(optimizer)
        # model.clip_grad_norm_(max_norm=args.clip_max_norm)
        # grad_scaler.step(optimizer)
        # grad_scaler.update()
        loss_value.backward()
        model.clip_grad_norm_(max_norm=args.clip_max_norm)
        optimizer.step()
        
        optimizer.zero_grad(set_to_none=True)
        
        # ==========================================
        # 💡 修改 2：在每次参数更新后，推进 scheduler
        # ==========================================
        if scheduler is not None:
            scheduler.step()
        # ==========================================

        dist.all_reduce(loss_value_for_reduce, op=dist.ReduceOp.SUM)
        loss_value_for_reduce = loss_value_for_reduce / world_size
        
        # dist.barrier(device_ids=[rank])

        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            # print(current_lr)
            writer.add_scalar('learning_rate', current_lr, global_step=iteration)
            writer.add_scalar(f'train/loss', loss_value_for_reduce.item(), global_step=iteration)
            
        
        #####这里是验证逻辑
        if iteration > 0 and iteration % args.val_iters == 0:
            if rank == 0:
                print(f"\n[Iteration {iteration}] 🚀 触发验证与权重保存...")
            
            # 1. 跑验证集并生成音频 (test 内部会调用 model.eval())
            test(args, model, rank, world_size, test_loader, codec, train_loss, current_time, model_name,iteration, writer=writer)
                        # 2. 保存 Checkpoint (以 iteration 命名)
            dist.barrier(device_ids=[rank])
            save_dir = f'./log/{current_time}_{model_name}/checkpoints'
            os.makedirs(save_dir, exist_ok=True)
            checkpoint = {
                    'weights': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch
                    }
            
            if rank == 0:
                torch.save(checkpoint, f"{save_dir}/step_{iteration}.pth")
            
            # 3. ⚠️ 极其重要：验证完毕后，必须把模型切回训练模式！
            model.train()
