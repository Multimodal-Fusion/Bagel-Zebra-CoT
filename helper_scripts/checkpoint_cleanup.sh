#!/bin/bash

# Checkpoint Cleanup Script for Bagel-Zebra-CoT Models
# Purpose: Keep only the latest original checkpoint per model, preserve all HF checkpoints
# Usage: ./checkpoint_cleanup.sh [--dry-run] [--auto-confirm]

set -e

RESULTS_DIR="/home/colligo/project/vlm/Bagel-Zebra-CoT/results"
DRY_RUN=false
AUTO_CONFIRM=false
ALSO_DELETE_LATEST_CHECKPOINT=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --auto-confirm)
            AUTO_CONFIRM=true
            shift
            ;;
        --delete-all-original)
            ALSO_DELETE_LATEST_CHECKPOINT=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--auto-confirm] [--delete-all-original]"
            echo "  --dry-run            Show what would be deleted without actually deleting"
            echo "  --auto-confirm       Skip confirmation prompts (use with caution!)"
            echo "  --delete-all-original Delete ALL original checkpoints including latest (keeps only HF versions)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to format bytes to human readable
format_bytes() {
    local bytes=$1
    if [[ $bytes -gt 1073741824 ]]; then
        echo "$(( bytes / 1073741824 ))GB"
    elif [[ $bytes -gt 1048576 ]]; then
        echo "$(( bytes / 1048576 ))MB"
    elif [[ $bytes -gt 1024 ]]; then
        echo "$(( bytes / 1024 ))KB"
    else
        echo "${bytes}B"
    fi
}

# Check if results directory exists
if [[ ! -d "$RESULTS_DIR" ]]; then
    echo -e "${RED}Error: Results directory not found: $RESULTS_DIR${NC}"
    exit 1
fi

echo -e "${BLUE}=== Bagel-Zebra-CoT Checkpoint Cleanup Script ===${NC}"
echo -e "Target directory: ${YELLOW}$RESULTS_DIR${NC}"
if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}DRY RUN MODE - No files will be deleted${NC}"
fi
if [[ "$ALSO_DELETE_LATEST_CHECKPOINT" == "true" ]]; then
    echo -e "${RED}WARNING: Will delete ALL original checkpoints (keeping only HF versions)${NC}"
fi
echo

# Arrays to store cleanup information
declare -a MODELS_TO_CLEAN
declare -a FOLDERS_TO_DELETE
declare -a FOLDER_SIZES
TOTAL_SIZE_TO_FREE=0

# Find all model directories (supporting both bagel-* and other patterns)
echo -e "${BLUE}Scanning for model directories...${NC}"
for model_dir in "$RESULTS_DIR"/*; do
    if [[ -d "$model_dir" ]]; then
        model_name=$(basename "$model_dir")
        checkpoints_dir="$model_dir/checkpoints"
        
        if [[ ! -d "$checkpoints_dir" ]]; then
            echo -e "${YELLOW}Warning: No checkpoints directory found in $model_name${NC}"
            continue
        fi
        
        # Find all original checkpoint directories (numeric names without _hf suffix)
        original_checkpoints=()
        for checkpoint in "$checkpoints_dir"/*; do
            if [[ -d "$checkpoint" ]]; then
                checkpoint_name=$(basename "$checkpoint")
                # Check if it's a numeric checkpoint (not ending with _hf)
                if [[ "$checkpoint_name" =~ ^[0-9]+$ ]]; then
                    original_checkpoints+=("$checkpoint_name")
                fi
            fi
        done
        
        # Sort checkpoints numerically and find the latest
        if [[ ${#original_checkpoints[@]} -gt 0 ]]; then
            IFS=$'\n' sorted_checkpoints=($(sort -n <<<"${original_checkpoints[*]}"))
            unset IFS
            
            latest_checkpoint="${sorted_checkpoints[-1]}"
            
            echo -e "${GREEN}Model: $model_name${NC}"
            echo -e "  Original checkpoints found: ${#original_checkpoints[@]}"
            
            if [[ "$ALSO_DELETE_LATEST_CHECKPOINT" == "true" ]]; then
                echo -e "  ${RED}Will delete ALL original checkpoints${NC}"
                
                # Mark all checkpoints for deletion
                for checkpoint in "${sorted_checkpoints[@]}"; do
                    folder_path="$checkpoints_dir/$checkpoint"
                    folder_size=$(du -sb "$folder_path" 2>/dev/null | cut -f1 || echo "0")
                    
                    MODELS_TO_CLEAN+=("$model_name")
                    FOLDERS_TO_DELETE+=("$folder_path")
                    FOLDER_SIZES+=("$folder_size")
                    TOTAL_SIZE_TO_FREE=$((TOTAL_SIZE_TO_FREE + folder_size))
                    
                    echo -e "    ${RED}Will delete: $checkpoint ($(format_bytes $folder_size))${NC}"
                done
            elif [[ ${#original_checkpoints[@]} -gt 1 ]]; then
                echo -e "  Latest checkpoint (to keep): ${BLUE}$latest_checkpoint${NC}"
                
                # Mark older checkpoints for deletion
                for checkpoint in "${sorted_checkpoints[@]}"; do
                    if [[ "$checkpoint" != "$latest_checkpoint" ]]; then
                        folder_path="$checkpoints_dir/$checkpoint"
                        folder_size=$(du -sb "$folder_path" 2>/dev/null | cut -f1 || echo "0")
                        
                        MODELS_TO_CLEAN+=("$model_name")
                        FOLDERS_TO_DELETE+=("$folder_path")
                        FOLDER_SIZES+=("$folder_size")
                        TOTAL_SIZE_TO_FREE=$((TOTAL_SIZE_TO_FREE + folder_size))
                        
                        echo -e "    ${RED}Will delete: $checkpoint ($(format_bytes $folder_size))${NC}"
                    fi
                done
            else
                echo -e "  Only 1 original checkpoint - nothing to clean (use --delete-all-original to remove)"
            fi
            echo
        else
            echo -e "${YELLOW}Model: $model_name${NC} - No original checkpoints found"
        fi
    fi
done

# Summary
echo -e "${BLUE}=== Cleanup Summary ===${NC}"
if [[ ${#FOLDERS_TO_DELETE[@]} -eq 0 ]]; then
    echo -e "${GREEN}No cleanup needed! All models have only their latest checkpoint.${NC}"
    exit 0
fi

echo -e "Models to clean: ${#MODELS_TO_CLEAN[@]}"
echo -e "Folders to delete: ${#FOLDERS_TO_DELETE[@]}"
echo -e "Space to free: ${GREEN}$(format_bytes $TOTAL_SIZE_TO_FREE)${NC}"
echo

# Show detailed list of what will be deleted
echo -e "${YELLOW}Detailed deletion list:${NC}"
for i in "${!FOLDERS_TO_DELETE[@]}"; do
    folder="${FOLDERS_TO_DELETE[$i]}"
    size="${FOLDER_SIZES[$i]}"
    echo -e "  ${RED}$folder${NC} ($(format_bytes $size))"
done
echo

# Confirmation
if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}DRY RUN COMPLETE - No files were deleted${NC}"
    exit 0
fi

if [[ "$AUTO_CONFIRM" != "true" ]]; then
    echo -e "${YELLOW}This will permanently delete the folders listed above.${NC}"
    if [[ "$ALSO_DELETE_LATEST_CHECKPOINT" == "true" ]]; then
        echo -e "${RED}WARNING: This includes ALL original checkpoints! Only HF versions will remain.${NC}"
    fi
    echo -e "${RED}This action cannot be undone!${NC}"
    echo
    read -p "Do you want to proceed? (type 'yes' to continue): " confirmation
    
    if [[ "$confirmation" != "yes" ]]; then
        echo -e "${YELLOW}Cleanup cancelled by user.${NC}"
        exit 0
    fi
fi

# Perform cleanup
echo -e "${BLUE}Starting cleanup...${NC}"
deleted_count=0
freed_space=0

for i in "${!FOLDERS_TO_DELETE[@]}"; do
    folder="${FOLDERS_TO_DELETE[$i]}"
    size="${FOLDER_SIZES[$i]}"
    
    echo -n "Deleting $(basename "$folder")... "
    
    if rm -rf "$folder"; then
        echo -e "${GREEN}✓${NC}"
        deleted_count=$((deleted_count + 1))
        freed_space=$((freed_space + size))
    else
        echo -e "${RED}✗ Failed${NC}"
    fi
done

echo
echo -e "${GREEN}=== Cleanup Complete ===${NC}"
echo -e "Deleted folders: $deleted_count/${#FOLDERS_TO_DELETE[@]}"
echo -e "Space freed: ${GREEN}$(format_bytes $freed_space)${NC}"

if [[ $deleted_count -eq ${#FOLDERS_TO_DELETE[@]} ]]; then
    echo -e "${GREEN}All cleanup operations completed successfully!${NC}"
else
    echo -e "${YELLOW}Some deletions failed. Check permissions and disk space.${NC}"
    exit 1
fi