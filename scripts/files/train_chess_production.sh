#!/bin/bash
# Production training script for chess ThinkTrace dataset
# For use with larger models and multi-GPU setups

# Parse command line arguments
MODEL_SIZE=${1:-"7B"}  # Default to 7B, can pass "0.5B" for testing
GPUS=${2:-8}           # Number of GPUs to use
BATCH_TOKENS=${3:-60000}  # Tokens per batch

# Environment setup
export HF_HOME=/dev/shm/
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=INFO

# Set model path based on size
if [ "$MODEL_SIZE" = "0.5B" ]; then
    MODEL_PATH=/home/colligo/project/vlm/Bagel/hf/Qwen2.5-0.5B-Instruct
    LAYER_MODULE=Qwen2DecoderLayer
    LR=2e-5
elif [ "$MODEL_SIZE" = "7B" ]; then
    MODEL_PATH=/dev/shm/models/BAGEL-7B-MoT
    LAYER_MODULE=Qwen2MoTDecoderLayer
    LR=1e-5
else
    echo "Invalid model size. Use '0.5B' or '7B'"
    exit 1
fi

# Distributed training settings
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29502
NPROC_PER_NODE=$GPUS

# Training parameters
WARMUP_STEPS=100
TOTAL_STEPS=5000
SAVE_EVERY=100
LOG_EVERY=10

# Create output directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="chess_thinktrace_${MODEL_SIZE}_${TIMESTAMP}"
RESULTS_DIR="results/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints"

mkdir -p $RESULTS_DIR
mkdir -p $CHECKPOINT_DIR

# Save configuration
cat > ${RESULTS_DIR}/config.txt << EOL
========================================
Training Configuration
========================================
Experiment: $EXPERIMENT_NAME
Model: $MODEL_PATH ($MODEL_SIZE)
Dataset: Chess ThinkTrace (10 samples)
GPUs: $NPROC_PER_NODE
Batch tokens: $BATCH_TOKENS
Learning rate: $LR
Warmup steps: $WARMUP_STEPS
Total steps: $TOTAL_STEPS
========================================
Features:
- Chain-of-thought reasoning with <think> tokens
- Visual reasoning with generated images
- Interleaved text (CE loss) and images (MSE loss)
- Triple image representation (VAE, VIT)
- Special token loss for image generation prediction
========================================
EOL

cat ${RESULTS_DIR}/config.txt

echo "Starting training..."
echo "Logs: ${RESULTS_DIR}/train.log"
echo "Errors: ${RESULTS_DIR}/train.err"

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
  --layer_module $LAYER_MODULE \
  --max_latent_size 32 \
  --resume-from $MODEL_PATH \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema $([ "$MODEL_SIZE" = "7B" ] && echo "True" || echo "False") \
  --log_every $LOG_EVERY \
  --lr $LR \
  --lr_scheduler cosine \
  --min_lr 1e-6 \
  --num_worker 4 \
  --expected_num_tokens $BATCH_TOKENS \
  --max_num_tokens $(($BATCH_TOKENS + 10000)) \
  --max_num_tokens_per_sample $BATCH_TOKENS \
  --prefer_buffer_before $(($BATCH_TOKENS / 2)) \
  --num_shard=$NPROC_PER_NODE \
  --sharding_strategy=$([ "$GPUS" -gt 1 ] && echo "HYBRID_SHARD" || echo "NO_SHARD") \
  --wandb_project "chess-thinktrace" \
  --wandb_name "$EXPERIMENT_NAME" \
  --save_every $SAVE_EVERY \
  --warmup_steps $WARMUP_STEPS \
  --total_steps $TOTAL_STEPS \
  --results_dir $RESULTS_DIR \
  --checkpoint_dir $CHECKPOINT_DIR \
  --text_cond_dropout_prob 0.1 \
  --vae_cond_dropout_prob 0.1 \
  --vit_cond_dropout_prob 0.4 \
  --vae_image_downsample 16 \
  --vit_patch_size 14 \
  --max_num_patch_per_side 70 \
  --gradient_checkpointing True \
  --mixed_precision "bf16" \
  > ${RESULTS_DIR}/train.log 2> ${RESULTS_DIR}/train.err &

# Get the PID of the training process
TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo "PID: $TRAIN_PID" > ${RESULTS_DIR}/train.pid

# Monitor training
echo ""
echo "=========================================="
echo "Training in progress..."
echo "Monitor with: tail -f ${RESULTS_DIR}/train.log"
echo "Check errors: tail -f ${RESULTS_DIR}/train.err"
echo "Stop training: kill $TRAIN_PID"
echo "=========================================="

# Wait for training to complete
wait $TRAIN_PID

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved in: $RESULTS_DIR"
echo "Checkpoints in: $CHECKPOINT_DIR"
echo "==========================================" 