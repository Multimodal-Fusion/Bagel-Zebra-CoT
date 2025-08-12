#!/bin/bash
# Training script for chess ThinkTrace dataset
# Interleaved text and image generation with chain-of-thought reasoning

########################################################
# experiment config
########################################################
export MAIN_DIR="$HOME/project/vlm/Bagel-Zebra-CoT"
DATASET_CONFIG="$MAIN_DIR/data/configs/chess_thinktrace.yaml"
MODEL_PATH="$MAIN_DIR/models/BAGEL-7B-MoT"
EXPERIMENT_NAME="bagel-chess-thinktrace-visualcot-v1"

# modality params
VISUAL_GEN=true
VISUAL_UND=true

# training hyperparams
LEARNING_RATE=2e-5
MIN_LEARNING_RATE=1e-6
LR_SCHEDULER="cosine"

# training hyperparams
WARMUP_STEPS=10
TOTAL_STEPS=20
SAVE_EVERY=10
EXPECTED_NUM_TOKENS=20000
MAX_NUM_TOKENS=30000
MAX_NUM_TOKENS_PER_SAMPLE=20000
PREFER_BUFFER_BEFORE=20000
NUM_WORKER=1 # use small num_workers since the num_used_data (1000) are not enough to split

# logging hyperparams
LOG_EVERY=1

# resume training hyperparams
AUTO_RESUME=false
RESUME_MODEL_ONLY=true
FINETUNE_FROM_EMA=false
########################################################
# set the variables
########################################################
export NPROC_PER_NODE=${NUM_OF_GPUS:-8}
export NUM_NODES=${WORLD_SIZE:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-"29500"}
export WANDB_ENTITY="genai-x"

# add main directory to python path
export PYTHONPATH="$MAIN_DIR:$PYTHONPATH"

########################################################
# print the variables
########################################################
echo "================================================"
echo "num_nodes: $NUM_NODES"
echo "node_rank: $NODE_RANK" 
echo "master_addr: $MASTER_ADDR"
echo "master_port: $MASTER_PORT"
echo "model_path: $MODEL_PATH"
echo "WANDB_ENTITY: $WANDB_ENTITY"
echo "DATASET_CONFIG: $DATASET_CONFIG"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "================================================"


########################################################
# run the experiment
########################################################
torchrun \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train/pretrain_unified_navit.py \
  --dataset_config_file $DATASET_CONFIG \
  --model_path $MODEL_PATH \
  --layer_module Qwen2DecoderLayer \
  --max_latent_size 32 \
  --resume-from $MODEL_PATH \
  --visual_gen $VISUAL_GEN \
  --visual_und $VISUAL_UND \
  --finetune_from_hf True \
  --auto_resume $AUTO_RESUME \
  --resume-model-only $RESUME_MODEL_ONLY \
  --finetune-from-ema $FINETUNE_FROM_EMA \
  --log_every $LOG_EVERY \
  --lr $LEARNING_RATE \
  --lr_scheduler $LR_SCHEDULER \
  --min_lr $MIN_LEARNING_RATE \
  --num_worker $NUM_WORKER \
  --expected_num_tokens $EXPECTED_NUM_TOKENS \
  --max_num_tokens $MAX_NUM_TOKENS \
  --max_num_tokens_per_sample $MAX_NUM_TOKENS_PER_SAMPLE \
  --prefer_buffer_before $PREFER_BUFFER_BEFORE \
  --num_shard=$NPROC_PER_NODE \
  --sharding_strategy="HYBRID_SHARD" \
  --save_every $SAVE_EVERY \
  --warmup_steps $WARMUP_STEPS \
  --total_steps $TOTAL_STEPS \
  --wandb_name $EXPERIMENT_NAME \
  --checkpoint_dir "results/$EXPERIMENT_NAME/checkpoints" \
  --results_dir "results/$EXPERIMENT_NAME" \
  --text_cond_dropout_prob 0.1 \
  --vae_cond_dropout_prob 0.1 \
  --save_every $SAVE_EVERY

echo "=========================================="
echo "🎉 Training completed!"
echo "Check results in: results/$EXPERIMENT_NAME/"
echo "Checkpoints saved in: results/$EXPERIMENT_NAME/checkpoints/"
echo "=========================================="