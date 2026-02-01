#!/bin/bash

# --- 硬件环境配置 ---
# 测试环境 (2080Ti): GPUS_PER_NODE=1, TP_SIZE=1
# 生产环境 (4*L20):   GPUS_PER_NODE=4, TP_SIZE=2
GPUS_PER_NODE=1
TP_SIZE=1

# 分布式网络配置
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

# --- 训练超参数 ---
# L20 显存大，生产环境可以将 BATCH_SIZE 调至 32 或更高
BATCH_SIZE=32
ACC_STEPS=4
LR=5e-4
MAX_SEQ_LEN=512

# --- 生产环境性能优化环境变量 ---
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果没有 InfiniBand 网络请禁用
export NCCL_P2P_LEVEL=NVL # L20 支持 NVLink，强制 P2P 走 NVLink
export CUDA_DEVICE_MAX_CONNECTIONS=1 # Megatron 推荐设置

# 运行训练脚本
/home/asuka/miniconda3/envs/llm/bin/torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    trainer/train_pretrain_megatron.py \
    --tp_size=$TP_SIZE \
    --batch_size=$BATCH_SIZE \
    --accumulation_steps=$ACC_STEPS \
    --learning_rate=$LR \
    --max_seq_len=$MAX_SEQ_LEN \
    --data_path="./dataset/pretrain_hq.jsonl" \
    --save_dir="./out" \
    --dtype="bfloat16" \
    --use_wandb
