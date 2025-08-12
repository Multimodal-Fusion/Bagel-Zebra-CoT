#!/bin/bash

# Enhanced checkpoint conversion script with parallel processing and weight merging
# 
# Usage:
#   ./convert_all_checkpoints.sh BASE_PATH [REPLACE] [MAX_PARALLEL] [NO_MERGE_WEIGHTS]
#
# Arguments:
#   BASE_PATH (required)             - Path to checkpoint directory (e.g., ./results/EXPNAME/checkpoints)
#   REPLACE (default: false)         - Replace existing _hf directories if they exist
#   MAX_PARALLEL (default: 8)        - Maximum number of parallel conversions
#   NO_MERGE_WEIGHTS (default: false) - Disable merging missing generation weights
#
# Examples:
#   ./convert_all_checkpoints.sh ./results/EXPNAME/checkpoints                    # Default: merge weights, don't replace, 8 parallel
#   ./convert_all_checkpoints.sh ./results/EXPNAME/checkpoints true               # Replace existing, merge weights
#   ./convert_all_checkpoints.sh ./results/EXPNAME/checkpoints true 8             # Replace, 8 parallel processes
#   ./convert_all_checkpoints.sh ./results/EXPNAME/checkpoints false 4 true       # Don't replace, old behavior (no weight merging)
#
# Configuration
BASE_MODEL="/home/colligo/project/vlm/Bagel-Zebra-CoT/models/BAGEL-7B-MoT"  # Base model for weight merging

# Check if base path argument is provided
if [ $# -lt 1 ]; then
    echo "Error: BASE_PATH is required as the first argument"
    echo ""
    echo "Usage: $0 BASE_PATH [REPLACE] [MAX_PARALLEL] [NO_MERGE_WEIGHTS]"
    echo ""
    echo "Example:"
    echo "  $0 ./results/bagel-chess-thinktrace-visualcot-v1/checkpoints"
    exit 1
fi

# Parse arguments
BASE_PATH="$1"                # Required: path to checkpoint directory
REPLACE=${2:-false}           # Default to false, can override with: ./script.sh path true
MAX_PARALLEL=${3:-8}          # Maximum parallel conversions, default 8
NO_MERGE_WEIGHTS=${4:-false}  # Default to false (merge weights), set true to disable merging

# Validate base path exists
if [ ! -d "$BASE_PATH" ]; then
    echo "Error: The specified path does not exist: $BASE_PATH"
    exit 1
fi

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

# Check if any checkpoints were found
if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "Error: No checkpoint directories found in: $BASE_PATH"
    echo ""
    echo "Expected to find directories with numeric names (e.g., 0000500, 0001000)"
    echo ""
    echo "Contents of $BASE_PATH:"
    ls -la "$BASE_PATH" 2>/dev/null | head -20
    exit 1
fi

# Sort checkpoints numerically
IFS=$'\n' CHECKPOINTS=($(sort -n <<<"${CHECKPOINTS[*]}"))
unset IFS

echo "Found ${#CHECKPOINTS[@]} checkpoints in: $BASE_PATH"
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
