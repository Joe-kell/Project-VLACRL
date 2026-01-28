#!/bin/bash
#SBATCH --job-name=train_embodiment_ewc      # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=90              # Number of CPUs per task
#SBATCH --gres=gpu:8                    # Number of GPUs (RECOMMENDED: 4+ GPUs)
#SBATCH --time=70:00:00                  # Maximum execution time (HH:MM:SS)
#SBATCH --mem=900G                      # Memory per node (adjust as needed)
#SBATCH --partition=mll                 # Partition name (adjust to your cluster's partition)
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jiahengh@utexas.edu
#SBATCH --exclude=slurm-node-008,slurm-node-011
#SBATCH --output=logs/slurm/%x-%j.out              # Output file name (%x = job name, %j = job ID)
#SBATCH --error=logs/slurm/%x-%j.err               # Error file name


### Usage: sbatch examples/mll_cluster/run_embodiment_naive_lora_ewc.slurm [TASK_ID_OR_RANGE] [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
### Example (single task): sbatch examples/mll_cluster/run_embodiment_naive_lora_ewc.slurm 0
### Example (task range): sbatch examples/mll_cluster/run_embodiment_naive_lora_ewc.slurm "0,3"
### Example (with max_epoch): sbatch examples/mll_cluster/run_embodiment_naive_lora_ewc.slurm 0 "" 15
### Example (continue from checkpoint): sbatch examples/mll_cluster/run_embodiment_naive_lora_ewc.slurm 0 ./logs/naive_lora_ewc/task_0_seed1234/checkpoints/global_step_10/actor 20
### Example (with seed): sbatch examples/mll_cluster/run_embodiment_naive_lora_ewc.slurm 0 "" "" "" 42
### Note: TASK_ID_OR_RANGE can be:
###       - A single task ID (e.g., "0") - trains that task only
###       - A tuple "a,b" where a < b (e.g., "0,3") - trains tasks from a to b sequentially
###       CHECKPOINT_PATH is optional and will be auto-generated from previous task if not provided
###       If CHECKPOINT_PATH is provided for a range, it will only be used for the first task
###       MAX_EPOCH is optional and can always be specified to override the default max_epochs
###       SEED is optional and defaults to 1234 if not provided

TASK_INPUT=${1:-0}
MANUAL_CHECKPOINT_PATH=$2
MAX_EPOCH=$3
CONFIG_NAME=${4:-mll_cluster/libero_spatial_grpo_openvlaoft}
SEED=${5:-1234}

cd $SLURM_SUBMIT_DIR

# Call the shell script
bash examples/mll_cluster/run_embodiment_naive_lora_ewc.sh "${TASK_INPUT}" "${MANUAL_CHECKPOINT_PATH}" "${MAX_EPOCH}" "${CONFIG_NAME}" "${SEED}"

exit $?
