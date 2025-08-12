#!/bin/bash

# Enhanced checkpoint conversion script with parallel processing and weight merging
# 
# Usage:
#   ./convert_all_checkpoints.sh [REPLACE] [MAX_PARALLEL] [NO_MERGE_WEIGHTS]
#
# Arguments:
#   REPLACE (default: false)        - Replace existing _hf directories if they exist
#   MAX_PARALLEL (default: 4)       - Maximum number of parallel conversions
#   NO_MERGE_WEIGHTS (default: false) - Disable merging missing generation weights
#
# Examples:
#   ./convert_all_checkpoints.sh                    # Default: merge weights, don't replace, 4 parallel
#   ./convert_all_checkpoints.sh true               # Replace existing, merge weights
#   ./convert_all_checkpoints.sh true 8             # Replace, 8 parallel processes
#   ./convert_all_checkpoints.sh false 4 true       # Don't replace, old behavior (no weight merging)
#
# Configuration
BASE_MODEL="/home/colligo/project/vlm/Bagel-Zebra-CoT/models/BAGEL-7B-MoT"  # Base model for weight merging

# Example checkpoint paths - update BASE_PATH to your specific checkpoint directory
# BASE_PATH="./results/bagel-chess-thinktrace-visualcot-v1/checkpoints"
BASE_PATH="./results/bagel-chess-thinktrace-visualcot-v1/checkpoints"

# Parse arguments
REPLACE=${1:-false}           # Default to false, can override with: ./script.sh true
MAX_PARALLEL=${2:-8}          # Maximum parallel conversions, default 4
NO_MERGE_WEIGHTS=${3:-false}  # Default to false (merge weights), set true to disable merging

# Dynamically find all checkpoint directories (only pure numeric directories)
CHECKPOINTS=()
for dir in "$BASE_PATH"/*; do
    if [ -d "$dir" ]; then
        checkpoint=$(basename "$dir")
        # Only include directories that are purely numeric (e.g., 0000500, 0001000)
        if [[ "$checkpoint" =~ ^[0-9]+$ ]]; then
            CHECKPOINTS+=("$checkpoint")
        fi
    fi
done

# Sort checkpoints numerically
IFS=$'\n' CHECKPOINTS=($(sort -n <<<"${CHECKPOINTS[*]}"))
unset IFS

echo "Converting checkpoints: ${CHECKPOINTS[*]}"
echo "Base model for merging: $BASE_MODEL"
echo "Replace existing: $REPLACE"
echo "Max parallel processes: $MAX_PARALLEL"
echo "Merge missing weights: $([ "$NO_MERGE_WEIGHTS" = "true" ] && echo "false" || echo "true")"
echo "Starting enhanced conversion at $(date)"

# Function to convert a single checkpoint
convert_checkpoint() {
    local ckpt=$1
    local HF_PATH="$BASE_PATH/${ckpt}_hf"
    
    if [ -d "$HF_PATH" ] && [ "$REPLACE" = "false" ]; then
        echo "Skipping checkpoint $ckpt (already exists: $HF_PATH)"
        return 0
    elif [ -d "$HF_PATH" ] && [ "$REPLACE" = "true" ]; then
        echo "Replacing existing checkpoint $ckpt..."
        rm -rf "$HF_PATH"
    fi
    
    echo "Converting checkpoint $ckpt..."
    
    # Build prepare_ckpt.py command with appropriate flags
    PREPARE_CMD="python helper_scripts/prepare_ckpt.py \"$BASE_PATH/$ckpt\" --base_model_path \"$BASE_MODEL\""
    
    # Add no_merge_weights flag if specified
    if [ "$NO_MERGE_WEIGHTS" = "true" ]; then
        PREPARE_CMD="$PREPARE_CMD --no_merge_weights"
    fi
    
    # Execute the conversion
    eval $PREPARE_CMD
    echo "Completed checkpoint $ckpt"
}

# Export function for parallel execution
export -f convert_checkpoint
export BASE_PATH
export BASE_MODEL
export REPLACE
export NO_MERGE_WEIGHTS

# Use parallel processing with limited concurrent jobs
printf '%s\n' "${CHECKPOINTS[@]}" | xargs -n 1 -P "$MAX_PARALLEL" -I {} bash -c 'convert_checkpoint "$@"' _ {}

echo "All checkpoint conversions completed at $(date)"
echo ""
echo "✅ Enhanced checkpoints ready for both VLM understanding AND image generation!"
if [ "$NO_MERGE_WEIGHTS" = "true" ]; then
    echo "⚠️  Note: Weight merging was disabled - checkpoints may not support generation"
else
    echo "🎯 All VLM checkpoints now include generation weights from $BASE_MODEL"
fi
