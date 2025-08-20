#!/bin/bash

# Script to generate all job scripts for training with different configurations
# Creates 36 job scripts: 3 tasks × 3 trace types × 4 sample sizes

JOBS_DIR="../jobs-19-08-2025"
TEMPLATE_FILE="job_template.sh"

# Configuration arrays
TASKS=("frozenlake" "maze_small" "tetris_small")
TRACE_TYPES=("sft" "textual-cot" "visual-cot")
SAMPLE_SIZES=(1000 2000 5000 10000)

# Fixed settings
SNAPSHOT_ID="59d4aa01-f739-42de-be4a-5cd488338f6f"
PROJECT="GPU-213 - Discretionary + Interns for Research - ILO"
ACCELERATOR="NVIDIA_A100_80GB"

# Create jobs directory if it doesn't exist
mkdir -p "$JOBS_DIR"

# Create base template
cat > "$TEMPLATE_FILE" << 'EOF'
#!/bin/bash -il 
# '-i' = interactive, '-l' = login (loads ~/.bash_profile)

########################################################
# ABATCH directives for sbatch job submission
# Usage: sbatch jobs-19-08-2025/this_script.sh
########################################################
#ABATCH --prefix unifiedlearning
#ABATCH --name bagel_TASK_NAME_TRACE_TYPE_NUM_SAMPLESsamples_5000steps
#ABATCH --num-pods 1
#ABATCH --xpus-per-pod 8
#ABATCH --accelerator-type ACCELERATOR_TYPE
#ABATCH --project "PROJECT_NAME"
#ABATCH --image docker-matrix-experiments-snapshot.ff.adobe.io/colligo/colligo-dev:v34
#ABATCH --snapshot-id SNAPSHOT_ID_VALUE
#ABATCH --init-script /home/colligo/init.sh
#ABATCH --start
#ABATCH --job-type training

set -exv

########################################################
# initialize env variables
########################################################
export WANDB_ENTITY="grepa"
export WANDB_PROJECT="grepa"
export REPO_DIR="/home/colligo/project/vlm/Bagel-Zebra-CoT"
export CONDA_ENV="bagel"
export GIT_BRANCH="main"
export GIT_COMMIT_HASH="ec07fb9ac80cb50e9c993fa0c540e06b73360714"

# export relevant keys/tokens
export HF_TOKEN=$HF_TOKEN
export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_ENTITY="transfusion"
export WANDB_PROJECT="transfusion"
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
export SPATH=s3://dit-scale-up/sensei-fs/users/jaskirats

# Task-specific variables
TASK_NAME="TASK_NAME_VALUE"
TRACE_TYPE="TRACE_TYPE_VALUE"
NUM_SAMPLES=NUM_SAMPLES_VALUE

# Define sd as a function instead of alias
sd() {
    bash /home/colligo/setup/ssd-symlinks.sh "$@"
}

# Initialize conda environment
. /opt/conda/etc/profile.d/conda.sh
# print all env variables as sanity check
printenv

########################################################
# init
########################################################
# go to the project root
cd $REPO_DIR
# activate the conda environment
conda activate $CONDA_ENV

# init sd paths
echo "Checking if sd is working..."
sd --help || echo "sd command not found"

# init sd paths
sd init
sd apply
echo "Done: sd paths initialized"

# get data and pretrained_models using rsync
sd rsync models
echo "Done: models synced"

# also go to /home/colligo/project/vlm/implicit-explicit-reasoning
cd /home/colligo/project/vlm/implicit-explicit-reasoning

# download the data
sd init
sd apply
sd rsync data/$TASK_NAME
echo "Done: data synced"

# go back to the repo root
cd $REPO_DIR
# add repo path to python path
export PYTHONPATH=$REPO_DIR:$PYTHONPATH

########################################################
# Force git to desired state (ignore local changes/conflicts)
########################################################
if [ "$GIT_COMMIT_HASH" == "skip" ]; then
    echo "Skipping git operations (GIT_COMMIT_HASH=skip)..."
    echo "Using current git state: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
else
    echo "Forcing git to desired state..."
    # Clean up any local changes
    git reset --hard HEAD
    git clean -fd
    # Fetch latest from remote
    git fetch origin
    if [ "$GIT_COMMIT_HASH" == "latest" ]; then
        # Force to latest remote branch
        git checkout -B $GIT_BRANCH origin/$GIT_BRANCH
        git reset --hard origin/$GIT_BRANCH
    else
        # Force to specific commit
        git checkout -f $GIT_COMMIT_HASH
    fi
    echo "Git state: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
fi

########################################################
# print env variables, JOB vars
########################################################
echo "WANDB_ENTITY: $WANDB_ENTITY"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "JOB_UUID: $JOB_UUID"
echo "JOB_NAME: $JOB_NAME"
echo "SPATH: $SPATH"
echo "PWD: $PWD"
echo "USER: $USER"
echo "DESIRED_GIT_BRANCH: $GIT_BRANCH"
echo "DESIRED_GIT_COMMIT_HASH: $GIT_COMMIT_HASH"
echo "CURRENT_GIT_BRANCH: $(git rev-parse --abbrev-ref HEAD)"
echo "CURRENT_GIT_COMMIT_HASH: $(git rev-parse HEAD)"

########################################################
# exp specific configs
########################################################
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1

echo "Launching training job with TASK: $TASK_NAME, TRACE: $TRACE_TYPE, SAMPLES: $NUM_SAMPLES"
# launch the training job
# run the training script in interactive mode
bash -i scripts/train_thinktrace.sh $TASK_NAME $TRACE_TYPE $NUM_SAMPLES

# sync the results folder to s3 using sd sync
sd sync results

# finish training
echo "Training finished!"
EOF

# Generate all job scripts
echo "Generating job scripts..."
for task in "${TASKS[@]}"; do
    for trace in "${TRACE_TYPES[@]}"; do
        for samples in "${SAMPLE_SIZES[@]}"; do
            # Create job script filename
            job_file="${JOBS_DIR}/bagel_${task}_${trace}_${samples}samples.sh"
            
            # Copy template and replace placeholders
            cp "$TEMPLATE_FILE" "$job_file"
            
            # Replace all placeholders - ORDER MATTERS!
            # First replace the _VALUE placeholders
            sed -i "s/TASK_NAME_VALUE/${task}/g" "$job_file"
            sed -i "s/TRACE_TYPE_VALUE/${trace}/g" "$job_file"
            sed -i "s/NUM_SAMPLES_VALUE/${samples}/g" "$job_file"
            # Then replace in the job name (be more specific to avoid replacing the values we just set)
            sed -i "s/#ABATCH --name bagel_TASK_NAME_TRACE_TYPE_NUM_SAMPLESsamples/#ABATCH --name bagel_${task}_${trace}_${samples}samples/g" "$job_file"
            sed -i "s/ACCELERATOR_TYPE/${ACCELERATOR}/g" "$job_file"
            sed -i "s/PROJECT_NAME/${PROJECT}/g" "$job_file"
            sed -i "s/SNAPSHOT_ID_VALUE/${SNAPSHOT_ID}/g" "$job_file"
            
            # Make executable
            chmod +x "$job_file"
            
            echo "Created: $job_file"
        done
    done
done

# Clean up template
rm "$TEMPLATE_FILE"

echo "Done! Generated $(ls -1 ${JOBS_DIR}/*.sh | wc -l) job scripts in ${JOBS_DIR}/"
echo ""
echo "To submit a job, use:"
echo "  sbatch ${JOBS_DIR}/bagel_<task>_<trace>_<samples>samples.sh"
echo ""
echo "Example:"
echo "  sbatch ${JOBS_DIR}/bagel_frozenlake_visual-cot_1000samples.sh"