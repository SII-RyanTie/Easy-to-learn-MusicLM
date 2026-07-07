import os
import argparse
import logging
import shutil
import functools
import math
from datetime import datetime, timedelta
import pytz
import yaml

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

# ==========================================
# Suppress DeepSpeed/FSDP verbose INFO logs
# ==========================================
logging.getLogger("deepspeed").setLevel(logging.WARNING)
logging.getLogger("deepspeed.accelerator").setLevel(logging.WARNING)

# ==========================================
# Workaround: Resolve import conflicts for huggingface_hub
# ==========================================
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    def dummy_cached_download(*args, **kwargs):
        return huggingface_hub.hf_hub_download(*args, **kwargs)
    huggingface_hub.cached_download = dummy_cached_download

# FSDP Imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import _module_wrap_policy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

# Local module imports
from MuCodec.generate import MuCodec
from model.music_transformer_mucodec_w_label import MusicTransformerText_delay
from core.dataset_mucodec_w_label import MusicTextDataset_delay, collate_fn_with_text
from core.trainer_mucodec_w_label import train

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# ==========================================
# Distributed Training Setup
# ==========================================
def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    # Initialize the process group with NCCL backend
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=1800))

def cleanup():
    dist.destroy_process_group()

# ==========================================
# Main FSDP Training Loop
# ==========================================
def fsdp_main(rank, world_size, args):
    setup(rank, world_size, args.port)
    torch.cuda.set_device(rank)

    # ------------------------------------------
    # 1. Dataset & DataLoader Initialization
    # TODO: Update these paths to your local dataset directories
    # ------------------------------------------
    train_dir = "./data/MTG_dataset/mtg_tokens_mucodec_train_w_MSMlabel"
    val_dir = "./data/MTG_dataset/mtg_tokens_mucodec_val_w_MSMlabel"

    train_dataset = MusicTextDataset_delay(train_dir, block_size=1024, stride=256)
    val_dataset = MusicTextDataset_delay(val_dir, block_size=1024, stride=256, rate=0.01)

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    loader_kwargs = {
        'num_workers': 20,
        'pin_memory': True,
        'shuffle': False,
        'drop_last': True,
        'collate_fn': collate_fn_with_text
    }

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler1, **loader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.test_batch_size, sampler=sampler2, **loader_kwargs
    )

    # ------------------------------------------
    # 2. Model Initialization & FSDP Wrapping
    # ------------------------------------------
    if args.backbone == "MusicTransformer":
        model = MusicTransformerText_delay(
            vocab_size=args.vocab_size,
            num_codebooks=args.num_codebooks,
            embed_dim=args.embed_dim,
            max_seq_len=args.max_seq_len,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout
        ).to(rank)
        module_classes = [MusicTransformerText_delay]
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")

    # Configure FSDP wrapping policy
    my_auto_wrap_policy = functools.partial(_module_wrap_policy, module_classes=module_classes)

    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=dtype_dict['bf16'],
            reduce_dtype=dtype_dict['bf16'],
            buffer_dtype=dtype_dict['bf16'],
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        forward_prefetch=True, 
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE, 
        device_id=torch.cuda.current_device(),
        use_orig_params=True
    )

    # ------------------------------------------
    # 3. Optimizer & Scheduler (Iteration-based)
    # ------------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-6, weight_decay=1e-1)
    grad_scaler = ShardedGradScaler()
    
    total_iters = len(train_loader) * args.epochs
    warmup_iters = getattr(args, 'warmup_iters', 5000) # Default to 5000 if not in yaml

    def lr_lambda(current_step: int):
        # Linear Warmup
        if current_step < warmup_iters:
            return float(current_step) / float(max(1, warmup_iters))
        # Cosine Decay
        progress = float(current_step - warmup_iters) / float(max(1, total_iters - warmup_iters))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # ------------------------------------------
    # 4. Load Audio Tokenizer (MuCodec)
    # ------------------------------------------
    bos_token = 16384
    codec = MuCodec(model_path="./MuCodec/ckpt/mucodec.pt", layer_num=7, load_main_model=True, device=f"cuda:{rank}")
    
    # ------------------------------------------
    # 5. Loss Function & TensorBoard
    # ------------------------------------------
    train_loss = nn.CrossEntropyLoss(ignore_index=bos_token)
    
    current_time = args.current_time
    model_name = args.model_name
    writer = SummaryWriter(f"./log/{current_time}_{model_name}") if rank == 0 else None
    
    # ------------------------------------------
    # 6. Training Loop
    # ------------------------------------------
    for epoch in range(0, args.epochs):
        train(
            args=args, 
            model=model, 
            rank=rank, 
            world_size=world_size, 
            train_loader=train_loader, 
            test_loader=test_loader, 
            codec=codec, 
            optimizer=optimizer, 
            epoch=epoch,
            train_loss=train_loss,
            grad_scaler=grad_scaler,
            current_time=current_time,
            model_name=model_name,
            scheduler=scheduler, 
            sampler=sampler1,
            writer=writer
        )
        
        # Note: Checkpoint saving logic should be handled inside the `train` function 
        # or added here depending on your project structure.

    cleanup()

# ==========================================
# Entry Point
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MusicLM Training Script')
    parser.add_argument('--yamlPath', type=str, default='./config/train_config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load configuration from YAML
    config = yaml.load(open(args.yamlPath, 'rb'), Loader=yaml.FullLoader)
    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)
            
    torch.manual_seed(args.seed)
    
    # Setup logging directories
    beijing_tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.now(beijing_tz).strftime("%Y-%m-%dT%H-%M-%S")
    log_path = f"./log/{current_time}_{args.model_name}"
    
    os.makedirs(f"{log_path}/pictures", exist_ok=True)
    os.makedirs(f"{log_path}/checkpoints", exist_ok=True)
    
    # Backup scripts for reproducibility
    shutil.copy(f"./main.py", f"{log_path}")
    shutil.copy(args.yamlPath, f"{log_path}")

    args.current_time = current_time
    WORLD_SIZE = torch.cuda.device_count()
    
    print(f"🚀 Starting FSDP Training on {WORLD_SIZE} GPUs...")
    
    mp.spawn(
        fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True
    )
