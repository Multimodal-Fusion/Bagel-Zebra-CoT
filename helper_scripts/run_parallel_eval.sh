#!/bin/bash

# Script for parallel evaluation across multiple GPUs
# Usage: ./run_parallel_eval.sh --num-workers 4 [other eval args...]

# add current path to pythonpath
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "added $(pwd) to pythonpath"

# Default values
NUM_WORKERS=1
ARGS=""
CHECKPOINT_DIR=""
DATASET_PATH=""
MAX_SAMPLES=""
EVAL_NAME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            ARGS="$ARGS --checkpoint-dir $2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            ARGS="$ARGS --dataset-path $2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            ARGS="$ARGS --max-samples $2"
            shift 2
            ;;
        --eval-name)
            EVAL_NAME="$2"
            ARGS="$ARGS --eval-name $2"
            shift 2
            ;;
        *)
            # Pass through other arguments
            ARGS="$ARGS $1"
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$CHECKPOINT_DIR" ] || [ -z "$DATASET_PATH" ]; then
    echo "Error: --checkpoint-dir and --dataset-path are required"
    exit 1
fi

# Generate eval name if not provided
if [ -z "$EVAL_NAME" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    DATASET_NAME=$(basename "$DATASET_PATH" .jsonl)
    EVAL_NAME="eval_${DATASET_NAME}_${TIMESTAMP}"
    ARGS="$ARGS --eval-name $EVAL_NAME"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
OUTPUT_PATH="$OUTPUT_DIR/$EVAL_NAME"

echo "=============================================="
echo "Parallel Evaluation Configuration"
echo "=============================================="
echo "Number of workers: $NUM_WORKERS"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Dataset: $DATASET_PATH"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "Output: $OUTPUT_PATH"
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Get total number of samples
echo "Counting samples in dataset..."
if [ -n "$MAX_SAMPLES" ]; then
    TOTAL_SAMPLES=$MAX_SAMPLES
else
    TOTAL_SAMPLES=$(wc -l < "$DATASET_PATH")
fi
echo "Total samples to process: $TOTAL_SAMPLES"

# Calculate samples per worker
SAMPLES_PER_WORKER=$((TOTAL_SAMPLES / NUM_WORKERS))
REMAINDER=$((TOTAL_SAMPLES % NUM_WORKERS))

echo "Samples per worker: ~$SAMPLES_PER_WORKER"
echo ""

# Launch workers in parallel
PIDS=()
for (( i=0; i<$NUM_WORKERS; i++ )); do
    # Calculate start and end indices for this worker
    if [ $i -lt $REMAINDER ]; then
        START_IDX=$((i * (SAMPLES_PER_WORKER + 1)))
        END_IDX=$((START_IDX + SAMPLES_PER_WORKER + 1))
    else
        START_IDX=$((i * SAMPLES_PER_WORKER + REMAINDER))
        END_IDX=$((START_IDX + SAMPLES_PER_WORKER))
    fi
    
    echo "Launching Worker $i (GPU $i): samples $START_IDX to $((END_IDX-1))"
    
    # Launch worker with specific GPU
    CUDA_VISIBLE_DEVICES=$i python helper_scripts/eval_thinktrace.py \
        $ARGS \
        --start-idx $START_IDX \
        --end-idx $END_IDX \
        --worker-id $i \
        > "$OUTPUT_PATH/worker_${i}.log" 2>&1 &
    
    PIDS+=($!)
done

echo ""
echo "All workers launched. Monitoring progress..."
echo ""

# Monitor workers
while true; do
    RUNNING=0
    for PID in "${PIDS[@]}"; do
        if kill -0 $PID 2>/dev/null; then
            RUNNING=$((RUNNING + 1))
        fi
    done
    
    if [ $RUNNING -eq 0 ]; then
        break
    fi
    
    echo -ne "\rWorkers still running: $RUNNING/$NUM_WORKERS"
    sleep 5
done

echo ""
echo ""
echo "All workers completed. Aggregating results..."
echo ""

# Check if all worker results exist
MISSING_WORKERS=()
for (( i=0; i<$NUM_WORKERS; i++ )); do
    if [ ! -f "$OUTPUT_PATH/results_worker_${i}.json" ]; then
        MISSING_WORKERS+=($i)
    fi
done

if [ ${#MISSING_WORKERS[@]} -gt 0 ]; then
    echo "Warning: Missing results from workers: ${MISSING_WORKERS[@]}"
    echo "Check worker logs for errors:"
    for i in "${MISSING_WORKERS[@]}"; do
        echo "  - $OUTPUT_PATH/worker_${i}.log"
    done
fi

# Aggregate results using Python
python - <<EOF
import json
import os
from datetime import datetime

output_path = "$OUTPUT_PATH"
num_workers = $NUM_WORKERS

# Collect all worker results
all_results = []
total_correct = 0
total_samples = 0

for i in range(num_workers):
    worker_file = os.path.join(output_path, f"results_worker_{i}.json")
    if os.path.exists(worker_file):
        with open(worker_file, 'r') as f:
            worker_data = json.load(f)
            all_results.extend(worker_data['results'])
            total_correct += worker_data['metrics']['correct']
            total_samples += worker_data['metrics']['total']
        print(f"Worker {i}: {worker_data['metrics']['correct']}/{worker_data['metrics']['total']} correct ({worker_data['metrics']['accuracy']:.2f}%)")

# Sort results by sample index
all_results.sort(key=lambda x: x['sample_idx'])

# Calculate final accuracy
final_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0

# Create aggregated output
output_data = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_dir": "$CHECKPOINT_DIR",
        "dataset_path": "$DATASET_PATH",
        "total_samples": total_samples,
        "evaluated_samples": total_samples,
        "num_workers": num_workers,
        "eval_name": "$EVAL_NAME"
    },
    "metrics": {
        "accuracy": final_accuracy,
        "correct": total_correct,
        "total": total_samples
    },
    "results": all_results
}

# Save aggregated results
results_path = os.path.join(output_path, "results.json")
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print("Aggregation Complete!")
print("="*60)
print(f"Total samples: {total_samples}")
print(f"Correct: {total_correct}")
print(f"Accuracy: {final_accuracy:.2f}%")
print(f"Results saved to: {results_path}")
print("="*60)
EOF

echo ""
echo "Evaluation complete! Check $OUTPUT_PATH/results.json for final results."
echo "Worker logs available in: $OUTPUT_PATH/worker_*.log"