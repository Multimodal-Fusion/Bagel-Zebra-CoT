#!/bin/bash

# Single checkpoint conversion script
# 
# Usage:
#   ./convert_checkpoint.sh CHECKPOINT_PATH [REPLACE] [NO_MERGE_WEIGHTS]
#
# Arguments:
#   CHECKPOINT_PATH (required)       - Path to single checkpoint (e.g., ./results/EXPNAME/checkpoints/0000500)
#   REPLACE (default: false)         - Replace existing _hf directory if it exists
#   NO_MERGE_WEIGHTS (default: false) - Disable merging missing generation weights
#
# Examples:
#   ./convert_checkpoint.sh ./results/EXPNAME/checkpoints/0000500
#   ./convert_checkpoint.sh ./results/EXPNAME/checkpoints/0000500 true
#   ./convert_checkpoint.sh ./results/EXPNAME/checkpoints/0000500 false true
#
# Configuration
BASE_MODEL="/home/colligo/project/vlm/Bagel-Zebra-CoT/models/BAGEL-7B-MoT"  # Base model for weight merging

# Check if checkpoint path argument is provided
if [ $# -lt 1 ]; then
    echo "Error: CHECKPOINT_PATH is required as the first argument"
    echo ""
    echo "Usage: $0 CHECKPOINT_PATH [REPLACE] [NO_MERGE_WEIGHTS]"
    echo ""
    echo "Example:"
    echo "  $0 ./results/bagel-chess-thinktrace-visualcot-v1/checkpoints/0000500"
    exit 1
fi

# Parse arguments
CHECKPOINT_PATH="$1"          # Required: path to single checkpoint
REPLACE=${2:-false}           # Default to false
NO_MERGE_WEIGHTS=${3:-false}  # Default to false (merge weights)

# Validate checkpoint path exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: The specified checkpoint path does not exist: $CHECKPOINT_PATH"
    exit 1
fi

# Check if EMA checkpoint exists
if [ ! -f "$CHECKPOINT_PATH/ema.safetensors" ]; then
    echo "Error: No ema.safetensors found in: $CHECKPOINT_PATH"
    echo ""
    echo "Contents of $CHECKPOINT_PATH:"
    ls -la "$CHECKPOINT_PATH" 2>/dev/null | head -20
    exit 1
fi

# Determine output path
HF_PATH="${CHECKPOINT_PATH}_hf"

# Check if already exists
if [ -d "$HF_PATH" ] && [ "$REPLACE" = "false" ]; then
    echo "Error: Output directory already exists: $HF_PATH"
    echo "Use 'true' as second argument to replace existing directory"
    echo ""
    echo "Example:"
    echo "  $0 $CHECKPOINT_PATH true"
    exit 1
elif [ -d "$HF_PATH" ] && [ "$REPLACE" = "true" ]; then
    echo "Replacing existing directory: $HF_PATH"
    rm -rf "$HF_PATH"
fi

echo "Converting checkpoint: $CHECKPOINT_PATH"
echo "Base model for merging: $BASE_MODEL"
echo "Output directory: $HF_PATH"
echo "Merge missing weights: $([ "$NO_MERGE_WEIGHTS" = "true" ] && echo "false" || echo "true")"
echo ""

# Build prepare_ckpt.py command with appropriate flags
PREPARE_CMD="python helper_scripts/prepare_ckpt.py \"$CHECKPOINT_PATH\" --base_model_path \"$BASE_MODEL\""

# Add no_merge_weights flag if specified
if [ "$NO_MERGE_WEIGHTS" = "true" ]; then
    PREPARE_CMD="$PREPARE_CMD --no_merge_weights"
fi

# Execute the conversion
echo "Running: $PREPARE_CMD"
eval $PREPARE_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Checkpoint conversion completed successfully!"
    echo "📁 Output directory: $HF_PATH"
    if [ "$NO_MERGE_WEIGHTS" = "true" ]; then
        echo "⚠️  Note: Weight merging was disabled - checkpoint may not support generation"
    else
        echo "🎯 Checkpoint includes generation weights from $BASE_MODEL"
    fi
else
    echo ""
    echo "❌ Checkpoint conversion failed!"
    exit 1
fi