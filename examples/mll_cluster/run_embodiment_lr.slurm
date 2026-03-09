#!/bin/bash
#SBATCH --job-name=train_embodiment_lr      # Job name
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=1                 # Number of tasks per node
#SBATCH --cpus-per-task=90                  # Number of CPUs per task
#SBATCH --gres=gpu:8                        # Number of GPUs (RECOMMENDED: 4+ GPUs)
#SBATCH --time=70:00:00                     # Maximum execution time (HH:MM:SS)
#SBATCH --mem=900G                          # Memory per node (adjust as needed)
#SBATCH --partition=mll                     # Partition name (adjust to your cluster's partition)
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jiahengh@utexas.edu
#SBATCH --exclude=slurm-node-008,slurm-node-011
#SBATCH --output=logs/slurm/%x-%j.out              # Output file name (%x = job name, %j = job ID)
#SBATCH --error=logs/slurm/%x-%j.err               # Error file name

### Usage: sbatch examples/mll_cluster/run_embodiment_lr.slurm [TASK_ID_OR_RANGE] [LR_STRING] [CONFIG_NAME] [SEED]
### Example (single task): sbatch examples/mll_cluster/run_embodiment_lr.slurm 0
### Example (task range): sbatch examples/mll_cluster/run_embodiment_lr.slurm "0,3"
### Example (with LR): sbatch examples/mll_cluster/run_embodiment_lr.slurm "1,4" "2e-6,2e-6,1e-5"
### Example (with config): sbatch examples/mll_cluster/run_embodiment_lr.slurm "0,2" "2e-6,2e-6,1e-5" mll_cluster/libero_spatial_grpo_openvlaoft_lr
### Example (with seed): sbatch examples/mll_cluster/run_embodiment_lr.slurm "0,2" "" "" 42
### Note: TASK_ID_OR_RANGE can be:
###       - A single task ID (e.g., "0") - trains that task only
###       - A tuple "a,b" where a < b (e.g., "0,3") - trains tasks from a to b sequentially
###       LR_STRING is comma-separated: "vision_lora_lr,llm_lora_lr,llm_head_lora_lr"
###       If not provided, uses default values from config
###       SEED is optional and defaults to 1234 if not provided

bash examples/mll_cluster/run_embodiment_lr.sh "$@"
exit $?
