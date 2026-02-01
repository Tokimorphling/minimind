import os
import sys
import argparse
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

# --- Megatron-Core Compatibility Patch ---
# Some versions of megatron-core have a hard dependency on transformer-engine metadata.
# If transformer-engine is uninstalled due to binary compatibility issues, we mock it.
import importlib.metadata
from unittest.mock import MagicMock

_orig_version = importlib.metadata.version
def _mock_version(package_name):
    try:
        return _orig_version(package_name)
    except importlib.metadata.PackageNotFoundError:
        if package_name == "transformer-engine":
            return "0.0.0"  # Return a version that doesn't trigger TE-specific features
        raise

importlib.metadata.version = _mock_version

try:
    import transformer_engine
except ImportError:
    # Create a dummy module to satisfy static imports in megatron-core
    sys.modules["transformer_engine"] = MagicMock()
# -----------------------------------------

# Add project root to sys.path
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig
from model.model_minimind_megatron import MiniMindMegatronForCausalLM
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, setup_seed, SkipBatchSampler
from optim.optimizer import get_optimizer

# Megatron imports
from megatron.core import parallel_state, tensor_parallel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
def initialize_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
    """Initializes Megatron-Core distributed state."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    # Initialize parallel state
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size
    )

    model_parallel_cuda_manual_seed(1234)
def train_epoch(epoch, loader, iters, optimizer, model, args, start_step=0, wandb=None, autocast_ctx=None, scaler=None):
    start_time = time.time()
    model.train()
    
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, lr: {current_lr:.8f}, time: {eta_min:.1f}min')
            if wandb and is_main_process():
                wandb.log({"loss": current_loss, "learning_rate": current_lr})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            ckp = f'{args.save_dir}/{args.save_weight}_megatron.pth'
            torch.save(model.state_dict(), ckp)

def main():
    parser = argparse.ArgumentParser(description="MiniMind Megatron Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='pretrain_megatron', type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallelism size")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain-Megatron")
    
    args = parser.parse_args()

    # 1. Initialize Megatron
    initialize_megatron(tensor_model_parallel_size=args.tp_size)
    
    setup_seed(42 + parallel_state.get_tensor_model_parallel_rank())
    
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
    # 2. Config & Model
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len,
        params_dtype=dtype
    )
    
    model = MiniMindMegatronForCausalLM(lm_config).to(args.device)
    model.to(dtype)
    
    # 3. Optimizer & Scaler
    optimizer = get_optimizer("adamw", model, lr=args.learning_rate)
    scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))
    


    # 4. wandb (swanlab)
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_run_name = f"MiniMind-Megatron-Pretrain-TP{args.tp_size}-BS{args.batch_size}-LR{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    # 5. Data
    from trainer.trainer_utils import init_model
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len
    )
    # 使用绝对路径，避免 transformers 将相对路径识别为远程 repo
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.abspath(os.path.join(current_dir, '../model'))

    _, tokenizer = init_model(lm_config, 'none', tokenizer_path=tokenizer_path, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    
    # For Megatron, data parallel rank is used for sampling
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_world_size = parallel_state.get_data_parallel_world_size()
    
    train_sampler = DistributedSampler(
        train_ds, 
        num_replicas=dp_world_size, 
        rank=dp_rank, 
        shuffle=True
    )
    
    # 6. Training Loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
        train_epoch(epoch, loader, len(loader), optimizer, model, args, wandb=wandb, scaler=scaler, autocast_ctx=autocast_ctx)

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
