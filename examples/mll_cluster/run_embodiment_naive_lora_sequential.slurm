#!/bin/bash
#SBATCH --job-name=seq_lora      # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=90              # Number of CPUs per task
#SBATCH --gres=gpu:8                    # Number of GPUs (RECOMMENDED: 4+ GPUs)
#SBATCH --time=70:00:00                  # Maximum execution time (HH:MM:SS)
#SBATCH --mem=900G                      # Memory per node (adjust as needed)
#SBATCH --partition=mll                 # Partition name (adjust to your cluster's partition)
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jiahengh@utexas.edu
#SBATCH --exclude=slurm-node-008,slurm-node-011,slurm-node-004
#SBATCH --output=logs/slurm/%x-%j.out              # Output file name (%x = job name, %j = job ID)
#SBATCH --error=logs/slurm/%x-%j.err               # Error file name

### NOTE: this script is meant for sequential launching of tasks in hand-crafted order.
### Usage: sbatch examples/mll_cluster/run_embodiment_naive_lora_sequential.slurm TASK_IDS [RUN_ID] [MAX_EPOCH] [CONFIG_NAME] [SEED] [INIT_CHECKPOINT]
### Example: sbatch examples/mll_cluster/run_embodiment_naive_lora_sequential.slurm "0,1,2"
### Example (with run_id): sbatch examples/mll_cluster/run_embodiment_naive_lora_sequential.slurm "0,1,2" "run1"
### Example (with initial checkpoint): sbatch examples/mll_cluster/run_embodiment_naive_lora_sequential.slurm "0,1,2" "" "" "" 42 logs/naive_lora_sequential/seq_0_1_2_seed42/task_0/checkpoints/global_step_10/actor

bash examples/mll_cluster/run_embodiment_naive_lora_sequential.sh "$@"
exit $?
