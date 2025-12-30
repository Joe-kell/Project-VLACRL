#!/bin/bash
#SBATCH --job-name=eval_lora_scale      # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=80              # Number of CPUs per task
#SBATCH --gres=gpu:4                    # Number of GPUs (RECOMMENDED: 4+ GPUs)
#SBATCH --time=4:00:00                  # Maximum execution time (HH:MM:SS)
#SBATCH --mem=490G                       # Memory per node (adjust as needed)
# #SBATCH --partition=mll            # Partition name (adjust to your cluster's partition)
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jiahengh@utexas.edu
#SBATCH --exclude=slurm-node-008,slurm-node-011

# =============================================================================
# MODULE LOADING
# =============================================================================
# #SBATCH --output=%x-%j.out              # Output file name (%x = job name, %j = job ID)
# Common modules to load (uncomment and adjust for your cluster):
# module purge                          # Clear existing modules
# module load cuda/11.8                 # CUDA toolkit (adjust version)
# module load python/3.11               # Python (if not using conda/venv)
# module load gcc/9.3.0                 # Compiler (if needed)
# module list                           # Verify loaded modules

### Usage: sbatch eval_embodiment_lora_scale.slurm CHECKPOINT_LOCATION LORA_SCALE
### Example: sbatch eval_embodiment_lora_scale.slurm logs/bcrl_logit/0.3/task_0 0.5
### Note: CHECKPOINT_LOCATION should be relative to workspace root (e.g., logs/bcrl_logit/0.3/task_0)
###       The script will construct the full path: ${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_10/actor/
###       LORA_SCALE is the coefficient for weighted merging (0.0 = base only, 1.0 = full LoRA)

CHECKPOINT_LOCATION=$1
LORA_SCALE=$2

if [ -z "$CHECKPOINT_LOCATION" ] || [ -z "$LORA_SCALE" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: sbatch examples/mll_cluster/eval_embodiment_lora_scale.slurm CHECKPOINT_LOCATION LORA_SCALE"
    echo "Example: sbatch examples/mll_cluster/eval_embodiment_lora_scale.slurm logs/bcrl_logit/0.3/task_0 0.5"
    exit 1
fi

# Validate LORA_SCALE is a valid number between 0.0 and 1.0
if ! [[ "$LORA_SCALE" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "ERROR: LORA_SCALE must be a number, got: $LORA_SCALE"
    exit 1
fi

# Use awk for floating point comparison (more portable than bc)
if ! awk "BEGIN {exit !($LORA_SCALE >= 0 && $LORA_SCALE <= 1)}"; then
    echo "ERROR: LORA_SCALE must be between 0.0 and 1.0, got: $LORA_SCALE"
    exit 1
fi

# Get workspace root (SLURM_SUBMIT_DIR is the directory where sbatch was run from)
cd $SLURM_SUBMIT_DIR
WORKSPACE_ROOT=$(pwd)

# Construct full checkpoint path (remove any trailing slashes from CHECKPOINT_LOCATION first)
CHECKPOINT_LOCATION="${CHECKPOINT_LOCATION%/}"
CHECKPOINT_PATH="${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_10/actor/"

# Verify checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
echo ""
echo "Checkpoint Location: $CHECKPOINT_LOCATION"
echo "Full Checkpoint Path: $CHECKPOINT_PATH"
echo "LoRA Scale: $LORA_SCALE"
echo ""
echo "Note: Log path will be automatically constructed by eval_embodiment.sh"
echo "      and will include lora_scale in the directory name."
echo ""

# Pass lora_scale to the eval script via Hydra override
# The eval script will automatically handle log path construction including lora_scale
bash examples/embodiment/eval_embodiment.sh libero_spatial_grpo_openvlaoft_eval \
    +actor.model.lora_path="${CHECKPOINT_PATH}" \
    actor.model.lora_scale="${LORA_SCALE}"

# Print completion time
echo "End Time: $(date)"
