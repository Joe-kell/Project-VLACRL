#!/bin/bash
### Usage: bash examples/crl_experiment/run_embodiment_slca.sh TASK_ID_OR_RANGE [LR_STRING] [CONFIG_NAME] [SEED]
### Example (single task): bash examples/crl_experiment/run_embodiment_slca.sh 0
### Example (task range): bash examples/crl_experiment/run_embodiment_slca.sh "0,3"
### Example (with LR): bash examples/crl_experiment/run_embodiment_slca.sh "1,4" "2e-6,2e-6,1e-5"
### Example (with config): bash examples/crl_experiment/run_embodiment_slca.sh "0,2" "2e-6,2e-6,1e-5" crl_experiment/libero_spatial_grpo_openvlaoft_lr
### Example (with seed): bash examples/crl_experiment/run_embodiment_slca.sh "0,2" "" "" 42
### Note: TASK_ID_OR_RANGE can be:
###       - A single task ID (e.g., "0") - trains that task only
###       - A tuple "a,b" where a < b (e.g., "0,3") - trains tasks from a to b sequentially
###       LR_STRING is comma-separated: "vision_lora_lr,llm_lora_lr,llm_head_lora_lr"
###       If not provided, uses default values from config
###       SEED is optional and defaults to 1234 if not provided

TASK_INPUT=${1:-0}
LR_STRING=$2
CONFIG_NAME=${3:-crl_experiment/libero_spatial_grpo_openvlaoft_lr}
SEED=${4:-1234}

# Parse TASK_INPUT to determine if it's a single task or a range
if [[ "$TASK_INPUT" == *,* ]]; then
    IFS=',' read -r TASK_START TASK_END <<< "$TASK_INPUT"
    TASK_START=$(echo "$TASK_START" | tr -d '()[] ')
    TASK_END=$(echo "$TASK_END" | tr -d '()[] ')
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

# Parse learning rates if provided (parse once, use for all tasks in range)
if [ -n "$LR_STRING" ]; then
    IFS=',' read -r VISION_LR LLM_LR HEAD_LR <<< "$LR_STRING"
    if [ -z "$VISION_LR" ] || [ -z "$LLM_LR" ] || [ -z "$HEAD_LR" ]; then
        echo "ERROR: LR_STRING must contain exactly 3 comma-separated values: vision_lora_lr,llm_lora_lr,llm_head_lora_lr"
        echo "       Example: \"2e-6,2e-6,1e-5\""
        exit 1
    fi
    V_LR_STR=$(echo "$VISION_LR" | sed 's/\.0*e/e/g' | sed 's/\.//g' | sed 's/e-/e/g')
    L_LR_STR=$(echo "$LLM_LR" | sed 's/\.0*e/e/g' | sed 's/\.//g' | sed 's/e-/e/g')
    H_LR_STR=$(echo "$HEAD_LR" | sed 's/\.0*e/e/g' | sed 's/\.//g' | sed 's/e-/e/g')
fi

mkdir -p logs/slurm
mkdir -p logs/slca_experiment

# Print job information (only if running under SLURM)
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Job ID: $SLURM_JOB_ID"
    echo "Job Name: $SLURM_JOB_NAME"
    echo "Node: $SLURM_NODELIST"
    echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
    echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
fi
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Change to repo root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
    REPO_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))
    cd "$REPO_ROOT"
fi

source "examples/crl_experiment/common_functions.sh"

CONFIG_TAG=$(extract_config_tag "$CONFIG_NAME")
EVAL_CONFIG_NAME=$(derive_eval_config_name "$CONFIG_NAME")
GLOBAL_STEP=$(get_default_global_step "$CONFIG_NAME")
FIRST_TASK_ID=$(get_first_task_id "$CONFIG_NAME")

# For _lr configs, use the existing eval config (without _lr)
if [[ "$CONFIG_NAME" =~ _lr ]]; then
    if [[ "$CONFIG_NAME" =~ _lr_([^/]+)$ ]]; then
        LR_TAG="${BASH_REMATCH[1]}"
        EVAL_CONFIG_NAME=$(echo "$CONFIG_NAME" | sed "s|_lr_${LR_TAG}$|_eval_${LR_TAG}|")
    elif [[ "$CONFIG_NAME" =~ _lr$ ]]; then
        EVAL_CONFIG_NAME=$(echo "$CONFIG_NAME" | sed 's|_lr$|_eval|')
    fi
fi

# Main training loop
OVERALL_EXIT_CODE=0

for TASK_ID in $(seq $TASK_START $TASK_END); do
    echo ""
    echo "========================================="
    if [ "$IS_RANGE" = true ]; then
        echo "Sequential Training - Task ${TASK_ID} (${TASK_START} to ${TASK_END})"
    else
        echo "Lifelong Learning - SLCA (learning rate) Experiment"
    fi
    echo "========================================="

    if [ -n "$LR_STRING" ]; then
        TASK_LOG_DIR="./logs/slca_experiment/task_${TASK_ID}_lr_v${V_LR_STR}_l${L_LR_STR}_h${H_LR_STR}_seed${SEED}"
    else
        TASK_LOG_DIR="./logs/slca_experiment/task_${TASK_ID}_lr_default_seed${SEED}"
    fi

    if [ -n "$CONFIG_TAG" ]; then
        TASK_LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$TASK_LOG_DIR" "$CONFIG_TAG")
        if [ -z "$TASK_LOG_DIR_TRANSFORMED" ]; then
            echo "  ERROR: Failed to transform TASK_LOG_DIR with config tag"
            OVERALL_EXIT_CODE=1
            break
        fi
        TASK_LOG_DIR="$TASK_LOG_DIR_TRANSFORMED"
    fi

    if [ -z "$TASK_LOG_DIR" ]; then
        echo "  ERROR: TASK_LOG_DIR is empty after path construction"
        OVERALL_EXIT_CODE=1
        break
    fi

    export LOG_DIR="${TASK_LOG_DIR}"
    mkdir -p "${TASK_LOG_DIR}"

    EXPERIMENT_NAME=$(basename "$TASK_LOG_DIR")
    if [ -n "$CONFIG_TAG" ]; then
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_${CONFIG_TAG}"
    fi

    if [ "$TASK_ID" -eq "$TASK_START" ] && [ -n "$SLURM_JOB_ID" ] && command -v scontrol &> /dev/null; then
        if [ "$IS_RANGE" = true ]; then
            if [ -n "$CONFIG_TAG" ]; then
                scontrol update job=$SLURM_JOB_ID name="slca_tasks_${TASK_START}_to_${TASK_END}_${CONFIG_TAG}" 2>/dev/null || true
            else
                scontrol update job=$SLURM_JOB_ID name="slca_tasks_${TASK_START}_to_${TASK_END}" 2>/dev/null || true
            fi
        else
            scontrol update job=$SLURM_JOB_ID name="${EXPERIMENT_NAME}" 2>/dev/null || true
        fi
    fi

    echo "Configuration:"
    echo "  Task ID: $TASK_ID"
    if [ "$IS_RANGE" = true ]; then
        echo "  Task Range: ${TASK_START} to ${TASK_END}"
    fi
    echo "  Experiment Name: $EXPERIMENT_NAME"
    echo "  Checkpoint Save Path: $TASK_LOG_DIR"
    echo "  Config Name: $CONFIG_NAME"
    echo "  Random Seed: $SEED"
    if [ -n "$LR_STRING" ]; then
        echo "  Vision LoRA LR: $VISION_LR"
        echo "  LLM LoRA LR: $LLM_LR"
        echo "  LLM Head LoRA LR: $HEAD_LR"
    else
        echo "  Using default learning rates from config"
    fi

    if [ "$TASK_ID" -eq "$FIRST_TASK_ID" ]; then
        CHECKPOINT_PATH=""
        echo "  Training from base model (SFT checkpoint) - no LoRA path"
    else
        PREV_TASK_ID=$((TASK_ID - 1))
        if [ -n "$LR_STRING" ]; then
            PREV_TASK_LOG_DIR="./logs/slca_experiment/task_${PREV_TASK_ID}_lr_v${V_LR_STR}_l${L_LR_STR}_h${H_LR_STR}_seed${SEED}"
        else
            PREV_TASK_LOG_DIR="./logs/slca_experiment/task_${PREV_TASK_ID}_lr_default_seed${SEED}"
        fi
        [ -n "$CONFIG_TAG" ] && PREV_TASK_LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$PREV_TASK_LOG_DIR" "$CONFIG_TAG") || PREV_TASK_LOG_DIR_TRANSFORMED="$PREV_TASK_LOG_DIR"
        if [ -z "$PREV_TASK_LOG_DIR_TRANSFORMED" ]; then
            echo "  ERROR: Failed to construct previous task log directory for task $PREV_TASK_ID"
            OVERALL_EXIT_CODE=1
            break
        fi
        PREV_TASK_LOG_DIR="$PREV_TASK_LOG_DIR_TRANSFORMED"
        CHECKPOINT_PATH="${PREV_TASK_LOG_DIR}/checkpoints/global_step_${GLOBAL_STEP}/actor"
        if [[ "$CHECKPOINT_PATH" =~ ^/checkpoints/ ]]; then
            echo "  ERROR: Invalid checkpoint path construction detected"
            OVERALL_EXIT_CODE=1
            break
        fi
        if [ ! -d "$CHECKPOINT_PATH" ]; then
            echo "ERROR: Previous checkpoint does not exist: $CHECKPOINT_PATH"
            OVERALL_EXIT_CODE=1
            break
        fi
        echo "  Loading checkpoint from previous task: $CHECKPOINT_PATH"
    fi
    echo "========================================="
    echo ""

    OVERRIDES="env.fixed_task_ids=[${TASK_ID}] \
    	runner.logger.experiment_name=${EXPERIMENT_NAME} \
    	actor.seed=${SEED}"
    if [ -n "$LR_STRING" ]; then
        OVERRIDES="$OVERRIDES actor.optim.vision_lora_lr=${VISION_LR} \
    	actor.optim.llm_lora_lr=${LLM_LR} \
    	actor.optim.llm_head_lora_lr=${HEAD_LR}"
    fi
    [ -n "$CHECKPOINT_PATH" ] && OVERRIDES="${OVERRIDES} +actor.model.lora_path=${CHECKPOINT_PATH}"

    echo "Running with Hydra overrides:"
    echo "$OVERRIDES"
    echo ""

    bash examples/embodiment/run_embodiment.sh ${CONFIG_NAME} $OVERRIDES

    EXIT_CODE=$?
    echo ""
    echo "========================================="
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Task $TASK_ID completed successfully"
        echo ""
        echo "Checkpoint saved to: ${TASK_LOG_DIR}"
        CHECKPOINT_LOCATION=$(echo "$TASK_LOG_DIR" | sed 's|^\./||')
        echo ""
        echo "Running evaluation for: ${CHECKPOINT_LOCATION}"
        bash examples/crl_experiment/eval_embodiment.sh "${CHECKPOINT_LOCATION}" "${GLOBAL_STEP}" "${EVAL_CONFIG_NAME}"
    else
        echo "✗ Task $TASK_ID failed with exit code $EXIT_CODE"
        [ -n "$SLURM_JOB_ID" ] && echo "  Check logs at: logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
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
    [ $OVERALL_EXIT_CODE -eq 0 ] && echo "All tasks (${TASK_START} to ${TASK_END}) completed successfully!" || echo "Sequential training failed. Completed up to task $((TASK_ID - 1))"
else
    [ $OVERALL_EXIT_CODE -eq 0 ] && echo "Task $TASK_START completed successfully" || echo "Task $TASK_START failed"
fi
echo "Finished at: $(date)"
echo "========================================="

exit $OVERALL_EXIT_CODE
