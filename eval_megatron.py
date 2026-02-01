import os
import sys
import argparse
import torch
import warnings
from transformers import AutoTokenizer, TextStreamer
from unittest.mock import MagicMock
import importlib.metadata

# --- Megatron-Core Compatibility Patch ---
_orig_version = importlib.metadata.version
def _mock_version(package_name):
    try:
        return _orig_version(package_name)
    except importlib.metadata.PackageNotFoundError:
        if package_name == "transformer-engine":
            return "0.0.0"
        raise

importlib.metadata.version = _mock_version

try:
    import transformer_engine
except ImportError:
    sys.modules["transformer_engine"] = MagicMock()
# -----------------------------------------

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from model.model_minimind import MiniMindConfig
from model.model_minimind_megatron import MiniMindMegatronForCausalLM
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

def initialize_megatron(tensor_model_parallel_size=1):
    # Set default distributed env vars if not present (for single process run)
    if not torch.distributed.is_initialized():
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"
            
        torch.distributed.init_process_group(backend="nccl")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=1
        )
        
    model_parallel_cuda_manual_seed(1234)

# Patch prepare_inputs_for_generation to support caching
def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
    if past_key_values:
        input_ids = input_ids[:, -1:]
    return {"input_ids": input_ids, "past_key_values": past_key_values, **kwargs}

MiniMindMegatronForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation

def main():
    parser = argparse.ArgumentParser(description="MiniMind Megatron Eval")
    parser.add_argument("--load_path", type=str, default="out/pretrain_megatron_megatron.pth", help="Path to megatron checkpoint")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallelism size (must match training)")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    
    args = parser.parse_args()

    # Initialize Megatron
    initialize_megatron(tensor_model_parallel_size=args.tp_size)

    # Config
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len,
        params_dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    )

    # Build Model
    model = MiniMindMegatronForCausalLM(config).to(args.device)
    if args.dtype == 'bfloat16':
        model.to(torch.bfloat16)
    else:
        model.to(torch.float16)
    
    # Load Checkpoint
    if os.path.exists(args.load_path):
        print(f"Loading checkpoint from {args.load_path}...")
        state_dict = torch.load(args.load_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    else:
        print(f"Warning: Checkpoint {args.load_path} not found. Using random weights.")

    model.eval()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    
    # Test Prompts
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        'Recommend some food in China.',
    ]

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print(f"\n{'='*20} Start Evaluation {'='*20}")
    for prompt in prompts:
        print(f"\n💬: {prompt}")
        print("🤖: ", end="")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                inputs=inputs.input_ids,
                max_new_tokens=args.max_seq_len,
                do_sample=True,
                streamer=streamer,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True
            )
    print(f"\n{'='*20} End Evaluation {'='*20}")

if __name__ == "__main__":
    main()
