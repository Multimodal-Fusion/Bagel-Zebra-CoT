#!/bin/bash
# Training script for chess ThinkTrace dataset
# Interleaved text and image generation with chain-of-thought reasoning

# Environment setup
export HF_HOME=/home/colligo/ssd/.cache/huggingface
# export CUDA_VISIBLE_DEVICES=0  # Use single GPU for toy dataset

# Distributed training settings (single node, single GPU for toy dataset)
NUM_NODES=$WORLD_SIZE
NODE_RANK=$RANK
MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=$MASTER_PORT
NPROC_PER_NODE=8   # Single GPU for toy dataset


# Model path - adjust this to your model location
# MODEL_PATH=/home/colligo/project/vlm/Bagel/hf/Qwen2.5-0.5B-Instruct  # Small model for testing
# For production, use: MODEL_PATH=/dev/shm/models/BAGEL-7B-MoT
MODEL_PATH=/home/colligo/project/vlm/Bagel/models/BAGEL-7B-MoT

# Create results directories
mkdir -p results/chess_thinktrace
mkdir -p results/checkpoints/chess_thinktrace

echo "=========================================="
echo "Training Chess ThinkTrace Dataset"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset config: ./data/configs/chess_thinktrace.yaml"
echo "GPUs: $NPROC_PER_NODE"
echo "Output: results/chess_thinktrace"
echo "NUM_NODES: $NUM_NODES"
echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "=========================================="

# Launch training
torchrun \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/chess_thinktrace.yaml \
  --model_path $MODEL_PATH \
  --layer_module Qwen2DecoderLayer \
  --max_latent_size 32 \
  --resume-from $MODEL_PATH \
  --finetune_from_hf True \
  --auto_resume False \
  --resume-model-only True \
  --finetune-from-ema False \
  --log_every 1 \
  --lr 1e-5 \
  --lr_scheduler cosine \
  --min_lr 1e-6 \
  --num_worker 1 \
  --expected_num_tokens 20000 \
  --max_num_tokens 30000 \
  --max_num_tokens_per_sample 20000 \
  --prefer_buffer_before 20000 \
  --num_shard=$NPROC_PER_NODE \
  --sharding_strategy="HYBRID_SHARD" \
  --wandb_project "chess-thinktrace" \
  --wandb_name "chess-thinktrace-$(date +%Y%m%d_%H%M%S)" \
  --save_every 50 \
  --warmup_steps 10 \
  --total_steps 100 \
  --results_dir results/chess_thinktrace/ \
  --checkpoint_dir results/checkpoints/chess_thinktrace/ \
  --text_cond_dropout_prob 0.1 \
  --vae_cond_dropout_prob 0.1 \
  --vit_cond_dropout_prob 0.4

echo "=========================================="
echo "Training completed!"
echo "Check results in: results/chess_thinktrace/"
echo "Checkpoints saved in: results/checkpoints/chess_thinktrace/"
echo "=========================================="