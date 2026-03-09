#!/bin/bash
#SBATCH --job-name=train_embodiment_multilora   # Job name
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Number of tasks per node
#SBATCH --cpus-per-task=90                      # Number of CPUs per task
#SBATCH --gres=gpu:8                            # Number of GPUs
#SBATCH --time=70:00:00                         # Maximum execution time (HH:MM:SS)
#SBATCH --mem=900G                              # Memory per node (adjust as needed)
#SBATCH --partition=mll                         # Partition name
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jiahengh@utexas.edu
#SBATCH --exclude=slurm-node-008,slurm-node-011
#SBATCH --output=logs/slurm/%x-%j.out              # Output file name (%x = job name, %j = job ID)
#SBATCH --error=logs/slurm/%x-%j.err               # Error file name

### Usage: sbatch examples/mll_cluster/run_embodiment_multilora.slurm TASK_ID_OR_RANGE [MERGE_COEFFICIENT] [CONFIG_NAME] [SEED]
### Example (single task): sbatch examples/mll_cluster/run_embodiment_multilora.slurm 0
### Example (single task): sbatch examples/mll_cluster/run_embodiment_multilora.slurm 1 0.8
### Example (with seed):   sbatch examples/mll_cluster/run_embodiment_multilora.slurm 0 "" "" 42
### Example (range):       sbatch examples/mll_cluster/run_embodiment_multilora.slurm "0,3" 0.8

bash examples/mll_cluster/run_embodiment_multilora.sh "$@"
exit $?
