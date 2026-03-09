#!/bin/bash
#SBATCH --job-name=eval_lora_scale      # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=96              # Number of CPUs per task
#SBATCH --gres=gpu:8                    # Number of GPUs (RECOMMENDED: 8 GPUs for evaluation)
#SBATCH --time=24:00:00                  # Maximum execution time (HH:MM:SS)
#SBATCH --mem=980G                       # Memory per node (adjusted as needed)
# #SBATCH --partition=mll            # Partition name (adjust to your cluster's partition)
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jiahengh@utexas.edu
#SBATCH --exclude=slurm-node-008,slurm-node-011
#SBATCH --output=logs/slurm/%x-%j.out              # Output file name (%x = job name, %j = job ID)
#SBATCH --error=logs/slurm/%x-%j.err               # Error file name

### Usage: sbatch examples/mll_cluster/eval_embodiment_lora_scale.slurm CHECKPOINT_LOCATION [CURRENT_LORA_SCALE] [PREVIOUS_LORA_COEFF] [STEP_NUMBER] [CONFIG_NAME]
### Example (single LoRA): sbatch examples/mll_cluster/eval_embodiment_lora_scale.slurm logs/bcrl_logit/0.3/task_0 0.5
### Example (multi-LoRA): sbatch examples/mll_cluster/eval_embodiment_lora_scale.slurm logs/naive_lora_multilora/task_2_coeff_0_9
### Example (multi-LoRA with current scale): sbatch examples/mll_cluster/eval_embodiment_lora_scale.slurm logs/naive_lora_multilora/task_2_coeff_0_9 0.7

bash examples/mll_cluster/eval_embodiment_lora_scale.sh "$@"
exit $?
