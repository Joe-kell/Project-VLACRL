#!/bin/bash
#SBATCH --job-name=train_embodiment      # Job name
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


### Usage: sbatch examples/mll_cluster/run_embodiment_bcrl_logit.slurm TASK_ID BC_COEFF [CHECKPOINT_PATH] [CONFIG_NAME] [SEED]
### Example: sbatch examples/mll_cluster/run_embodiment_bcrl_logit.slurm 0 0.1 libero_spatial_grpo_openvlaoft_bcrl_logit
### Example (with seed): sbatch examples/mll_cluster/run_embodiment_bcrl_logit.slurm 0 0.1 "" "" 42
### Note: CHECKPOINT_PATH is now optional and will be auto-generated from previous task if not provided
###       SEED is optional and defaults to 1234 if not provided

TASK_INPUT=${1:-0}
BC_COEFF=$2
MANUAL_CHECKPOINT_PATH=$3
CONFIG_NAME=${4:-libero_spatial_grpo_openvlaoft_bcrl_logit}
SEED=${5:-1234}

# Source common functions
source "examples/mll_cluster/common_functions.sh"

if [ -z "$TASK_INPUT" ] || [ -z "$BC_COEFF" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: sbatch examples/mll_cluster/run_embodiment_bcrl_logit.slurm TASK_ID_OR_RANGE BC_COEFF [CHECKPOINT_PATH] [CONFIG_NAME] [SEED]"
    exit 1
fi

# Extract config tag and derive eval config name
CONFIG_TAG=$(extract_config_tag "$CONFIG_NAME")
EVAL_CONFIG_NAME=$(derive_eval_config_name "$CONFIG_NAME")
GLOBAL_STEP=$(get_default_global_step "$CONFIG_NAME")
FIRST_TASK_ID=$(get_first_task_id "$CONFIG_NAME")

# Parse TASK_INPUT to determine if it's a single task or a range
if [[ "$TASK_INPUT" == *,* ]]; then
    # It's a tuple (a,b)
    IFS=',' read -r TASK_START TASK_END <<< "$TASK_INPUT"
    TASK_START=$(echo "$TASK_START" | tr -d '()[] ')
    TASK_END=$(echo "$TASK_END" | tr -d '()[] ')

    # Validate that both values are numeric and start < end
    if ! [[ "$TASK_START" =~ ^[0-9]+$ ]] || ! [[ "$TASK_END" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Task range must contain two numeric values: \"a,b\" where a and b are integers"
        echo "       Example: \"0,3\" or \"1,5\""
        exit 1
    fi

    if [ "$TASK_START" -ge "$TASK_END" ]; then
        echo "ERROR: First task ID ($TASK_START) must be smaller than second task ID ($TASK_END)"
        echo "       Example: \"0,3\" (trains tasks 0, 1, 2, 3)"
        exit 1
    fi

    IS_RANGE=true
    NUM_TASKS=$((TASK_END - TASK_START + 1))
else
    # It's a single task ID
    if ! [[ "$TASK_INPUT" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Task ID must be a numeric value"
        echo "       Example: 0 or \"0,3\" for a range"
        exit 1
    fi
    IS_RANGE=false
    TASK_START=$TASK_INPUT
    TASK_END=$TASK_INPUT
    NUM_TASKS=1
fi

# Validate seed is a number
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEED must be a non-negative integer, got: $SEED"
    exit 1
fi

mkdir -p logs/slurm
mkdir -p logs/bcrl_logit
mkdir -p logs/bcrl_logit/${BC_COEFF}

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
echo ""

# Main training loop
OVERALL_EXIT_CODE=0

for TASK_ID in $(seq $TASK_START $TASK_END); do
    echo ""
    echo "========================================="
    if [ "$IS_RANGE" = true ]; then
        echo "Sequential Training - Task ${TASK_ID} (${TASK_START} to ${TASK_END})"
    else
        echo "Lifelong Learning - Single Task Training (BCRL Logit)"
    fi
    echo "BC Coefficient: $BC_COEFF"
    echo "========================================="

    # Determine checkpoint path
    if [ "$TASK_ID" -eq "$TASK_START" ] && [ -n "$MANUAL_CHECKPOINT_PATH" ]; then
        # Use manual checkpoint path for first task if provided
        CHECKPOINT_PATH="$MANUAL_CHECKPOINT_PATH"
    elif [ "$TASK_ID" -eq $FIRST_TASK_ID ]; then
        # First task in sequence, no checkpoint
        CHECKPOINT_PATH=""
    else
        # Use checkpoint from previous task
        PREV_TASK_ID=$((TASK_ID - 1))
        PREV_LOG_DIR="./logs/bcrl_logit/${BC_COEFF}/task_${PREV_TASK_ID}_seed${SEED}"
        # Inject config tag into PREV_LOG_DIR to match where previous task saved checkpoint
        if [ -n "$CONFIG_TAG" ]; then
            PREV_LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$PREV_LOG_DIR" "$CONFIG_TAG")
            if [ -z "$PREV_LOG_DIR_TRANSFORMED" ]; then
                echo "  ERROR: Failed to transform previous task log directory for task $PREV_TASK_ID"
                echo "         Original PREV_LOG_DIR: [$PREV_LOG_DIR]"
                echo "         CONFIG_TAG: [$CONFIG_TAG]"
                OVERALL_EXIT_CODE=1
                break
            fi
        else
            PREV_LOG_DIR_TRANSFORMED="$PREV_LOG_DIR"
        fi

        CHECKPOINT_PATH="${PREV_LOG_DIR_TRANSFORMED}/checkpoints/global_step_${GLOBAL_STEP}/actor"

        # Additional validation: ensure CHECKPOINT_PATH is a valid relative or absolute path
        if [[ "$CHECKPOINT_PATH" =~ ^/checkpoints/ ]]; then
            echo "  ERROR: Invalid checkpoint path construction detected"
            echo "         PREV_LOG_DIR was likely empty or malformed"
            echo "         PREV_LOG_DIR: [$PREV_LOG_DIR]"
            echo "         CHECKPOINT_PATH: [$CHECKPOINT_PATH]"
            OVERALL_EXIT_CODE=1
            break
        fi
    fi

    # Determine LOG_DIR
    if [ "$TASK_ID" -eq "$TASK_START" ] && [ -n "$MANUAL_CHECKPOINT_PATH" ]; then
        # Extract task ID and global_step from checkpoint path for a descriptive log dir
        if [[ "$CHECKPOINT_PATH" =~ task_([0-9]+) ]]; then
            SOURCE_TASK="${BASH_REMATCH[1]}"
            if [[ "$CHECKPOINT_PATH" =~ global_step_([0-9]+) ]]; then
                SOURCE_STEP="${BASH_REMATCH[1]}"
                LOG_DIR="./logs/bcrl_logit/${BC_COEFF}/task_${TASK_ID}_from_task_${SOURCE_TASK}_step_${SOURCE_STEP}_seed${SEED}"
            else
                echo "ERROR: Could not extract global_step from checkpoint path: $CHECKPOINT_PATH"
                echo "       Expected format: .../checkpoints/global_step_<M>/actor"
                OVERALL_EXIT_CODE=1
                break
            fi
        else
            echo "ERROR: Could not extract task ID from checkpoint path: $CHECKPOINT_PATH"
            echo "       Expected format: .../task_<N>/checkpoints/global_step_<M>/actor"
            OVERALL_EXIT_CODE=1
            break
        fi
    else
        LOG_DIR="./logs/bcrl_logit/${BC_COEFF}/task_${TASK_ID}_seed${SEED}"
    fi

    # Inject config tag into LOG_DIR
    if [ -n "$CONFIG_TAG" ]; then
        LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$LOG_DIR" "$CONFIG_TAG")
        if [ -z "$LOG_DIR_TRANSFORMED" ]; then
            echo "  ERROR: Failed to transform LOG_DIR with config tag"
            echo "         Original LOG_DIR: [$LOG_DIR]"
            echo "         CONFIG_TAG: [$CONFIG_TAG]"
            OVERALL_EXIT_CODE=1
            break
        fi
        LOG_DIR="$LOG_DIR_TRANSFORMED"
    fi

    # Validate LOG_DIR is not empty
    if [ -z "$LOG_DIR" ]; then
        echo "  ERROR: LOG_DIR is empty after path construction"
        OVERALL_EXIT_CODE=1
        break
    fi

    export LOG_DIR
    mkdir -p "${LOG_DIR}"

    # Set experiment name
    EXPERIMENT_NAME="bcrl_logit_task_${TASK_ID}_${BC_COEFF}"
    if [ -n "$CONFIG_TAG" ]; then
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_${CONFIG_TAG}"
    fi

    # Update SLURM job name (only for first task)
    if [ "$TASK_ID" -eq "$TASK_START" ] && command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_ID" ]; then
        if [ "$IS_RANGE" = true ]; then
            JOB_NAME="bcrl_logit_${BC_COEFF}_tasks_${TASK_START}_to_${TASK_END}"
            if [ -n "$CONFIG_TAG" ]; then
                JOB_NAME="${JOB_NAME}_${CONFIG_TAG}"
            fi
        else
            LOG_DIR_PARENT=$(basename "$(dirname "$LOG_DIR")")
            LOG_DIR_BASENAME=$(basename "$LOG_DIR")
            JOB_NAME="bcrl_logit_${LOG_DIR_PARENT}_${LOG_DIR_BASENAME}"
            if [ -n "$CONFIG_TAG" ]; then
                JOB_NAME="${JOB_NAME}_${CONFIG_TAG}"
            fi
        fi
        scontrol update job=$SLURM_JOB_ID name="${JOB_NAME}" 2>/dev/null || true
    fi

    echo "Configuration:"
    echo "  Task ID: $TASK_ID"
    if [ "$IS_RANGE" = true ]; then
        echo "  Task Range: ${TASK_START} to ${TASK_END}"
    fi
    echo "  BC Coefficient: $BC_COEFF"
    echo "  Experiment Name: $EXPERIMENT_NAME"
    echo "  Checkpoint Save Path: $LOG_DIR"
    echo "  Config Name: $CONFIG_NAME"
    echo "  Random Seed: $SEED"

    if [ -n "$CHECKPOINT_PATH" ]; then
        if [ ! -d "$CHECKPOINT_PATH" ]; then
            echo "  ERROR: Checkpoint not found at $CHECKPOINT_PATH"
            if [ "$TASK_ID" -gt $FIRST_TASK_ID ]; then
                PREV_TASK_ID=$((TASK_ID - 1))
                echo "         Task $TASK_ID requires checkpoint from task $PREV_TASK_ID"
                echo "         Make sure task $PREV_TASK_ID has been trained first, or provide a manual checkpoint path"
            fi
            OVERALL_EXIT_CODE=1
            break
        fi
        echo "  Loading from checkpoint: $CHECKPOINT_PATH"
    else
        echo "  Training from base model (SFT checkpoint) - First task (task $FIRST_TASK_ID)"
    fi

    echo "========================================="
    echo ""

    # Build Hydra overrides
    OVERRIDES="algorithm.use_experience_replay=True \
        +algorithm.use_reference_logits_bc=True \
        +algorithm.use_cached_bc_logits=True \
        algorithm.bc_coeff=${BC_COEFF} \
        env.fixed_task_ids=[${TASK_ID}] \
        runner.logger.experiment_name=${EXPERIMENT_NAME} \
        actor.seed=${SEED}"

    if [ -n "$CHECKPOINT_PATH" ]; then
        OVERRIDES="$OVERRIDES +actor.model.lora_path=${CHECKPOINT_PATH}"
    fi

    echo "Running with Hydra overrides:"
    echo "$OVERRIDES"
    echo ""

    bash examples/embodiment/run_embodiment.sh ${CONFIG_NAME} $OVERRIDES

    EXIT_CODE=$?
    echo ""
    echo "========================================="
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Task $TASK_ID with BC_COEFF=$BC_COEFF completed successfully"
        echo ""
        echo "Checkpoint saved to: ${LOG_DIR}"

        # Run evaluation
        CHECKPOINT_LOCATION=$(echo "$LOG_DIR" | sed 's|^\./||')
        echo ""
        echo "Running evaluation for: ${CHECKPOINT_LOCATION}"
        bash examples/mll_cluster/eval_embodiment.sh "${CHECKPOINT_LOCATION}" "" "${EVAL_CONFIG_NAME}"
    else
        echo "✗ Task $TASK_ID failed with exit code $EXIT_CODE"
        echo "  Check logs at: logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
        OVERALL_EXIT_CODE=$EXIT_CODE
        if [ "$IS_RANGE" = true ]; then
            echo "  Stopping sequential training due to failure"
            break
        fi
    fi
    echo "========================================="
done

echo ""
echo "========================================="
if [ "$IS_RANGE" = true ]; then
    if [ $OVERALL_EXIT_CODE -eq 0 ]; then
        echo "All tasks (${TASK_START} to ${TASK_END}) with BC_COEFF=$BC_COEFF completed successfully!"
    else
        echo "Sequential training failed. Completed up to task $((TASK_ID - 1))"
    fi
else
    if [ $OVERALL_EXIT_CODE -eq 0 ]; then
        echo "Task $TASK_START with BC_COEFF=$BC_COEFF completed successfully"
    else
        echo "Task $TASK_START failed"
    fi
fi
echo "Finished at: $(date)"
echo "========================================="

exit $OVERALL_EXIT_CODE
