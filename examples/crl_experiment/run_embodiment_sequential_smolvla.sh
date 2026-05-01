#!/bin/bash
### Usage: bash examples/crl_experiment/run_embodiment_sequential_smolvla.sh TASK_ID_OR_RANGE [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
### Example (single task): bash examples/crl_experiment/run_embodiment_sequential_smolvla.sh 5
### Example (task range): bash examples/crl_experiment/run_embodiment_sequential_smolvla.sh "5,7"
### Example (resume): bash examples/crl_experiment/run_embodiment_sequential_smolvla.sh 6 ./logs_smolvla/sequential_smolvla/task_5_seed1234/checkpoints/global_step_10/actor/model.pt 10
### Notes:
###   - This script chains full SmolVLA checkpoints via actor/model.pt (not LoRA adapter dirs).
###   - If you enable LoRA (`actor.model.is_lora=true`), use the existing LoRA sequential
###     driver script pattern (`run_embodiment_sequential.sh`) with +actor.model.lora_path.
###   - For task ranges, only the first task can take a manual checkpoint path.
###   - Subsequent tasks auto-load from the previous task's actor/model.pt checkpoint.

set -euo pipefail

TASK_INPUT=${1:-5}
MANUAL_CHECKPOINT_PATH=${2:-}
MAX_EPOCH=${3:-}
CONFIG_NAME=${4:-crl_experiment/libero_object_grpo_smolvla_object}
SEED=${5:-1234}
EXPERIMENT_TYPE="sequential_smolvla"

# Optional: base LeRobot SmolVLA policy path used when checkpoint_load_path points to model.pt.
SMOLVLA_BASE_POLICY_PATH="${SMOLVLA_BASE_POLICY_PATH:-/home/s2758621/Octo_RL/checkpoints/smolvla_libero_object_20demo/job_3441070/checkpoints/last/pretrained_model}"

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEED must be a non-negative integer, got: $SEED"
    exit 1
fi

if [[ "$TASK_INPUT" == *,* ]]; then
    IFS=',' read -r TASK_START TASK_END <<< "$TASK_INPUT"
    TASK_START=$(echo "$TASK_START" | tr -d '()[] ')
    TASK_END=$(echo "$TASK_END" | tr -d '()[] ')
    if ! [[ "$TASK_START" =~ ^[0-9]+$ ]] || ! [[ "$TASK_END" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Task range must be \"a,b\" with integer values"
        exit 1
    fi
    if [ "$TASK_START" -ge "$TASK_END" ]; then
        echo "ERROR: range start must be < range end"
        exit 1
    fi
    IS_RANGE=true
else
    if ! [[ "$TASK_INPUT" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Task ID must be an integer"
        exit 1
    fi
    TASK_START=$TASK_INPUT
    TASK_END=$TASK_INPUT
    IS_RANGE=false
fi

mkdir -p "logs/${EXPERIMENT_TYPE}"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "SMOLVLA_BASE_POLICY_PATH: $SMOLVLA_BASE_POLICY_PATH"
echo ""

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))
cd "$REPO_ROOT"

source "examples/crl_experiment/common_functions.sh"
CONFIG_TAG=$(extract_config_tag "$CONFIG_NAME")
GLOBAL_STEP=$(get_default_global_step "$CONFIG_NAME")
FIRST_TASK_ID=$(get_first_task_id "$CONFIG_NAME")
EVAL_CONFIG_NAME="${EVAL_CONFIG_NAME:-crl_experiment/libero_object_grpo_smolvla_eval_object}"

OVERALL_EXIT_CODE=0

for TASK_ID in $(seq "$TASK_START" "$TASK_END"); do
    echo ""
    echo "========================================="
    if [ "$IS_RANGE" = true ]; then
        echo "SmolVLA Sequential Training - Task ${TASK_ID} (${TASK_START} to ${TASK_END})"
    else
        echo "SmolVLA Sequential Training - Single Task"
    fi
    echo "========================================="

    if [ "$TASK_ID" -eq "$TASK_START" ] && [ -n "$MANUAL_CHECKPOINT_PATH" ]; then
        CHECKPOINT_PATH="$MANUAL_CHECKPOINT_PATH"
    elif [ "$TASK_ID" -eq "$FIRST_TASK_ID" ]; then
        CHECKPOINT_PATH=""
    else
        PREV_TASK_ID=$((TASK_ID - 1))
        PREV_LOG_DIR="./logs/${EXPERIMENT_TYPE}/task_${PREV_TASK_ID}_seed${SEED}"
        if [ -n "$CONFIG_TAG" ]; then
            PREV_LOG_DIR=$(inject_config_tag_into_log_path "$PREV_LOG_DIR" "$CONFIG_TAG")
        fi
        CHECKPOINT_PATH="${PREV_LOG_DIR}/checkpoints/global_step_${GLOBAL_STEP}/actor/model.pt"
    fi

    if [ "$TASK_ID" -eq "$TASK_START" ] && [ -n "$MANUAL_CHECKPOINT_PATH" ]; then
        if [[ "$CHECKPOINT_PATH" =~ task_([0-9]+) ]] && [[ "$CHECKPOINT_PATH" =~ global_step_([0-9]+) ]]; then
            LOG_DIR="./logs/${EXPERIMENT_TYPE}/task_${TASK_ID}_from_task_${BASH_REMATCH[1]}_step_${BASH_REMATCH[2]}_seed${SEED}"
        else
            LOG_DIR="./logs/${EXPERIMENT_TYPE}/task_${TASK_ID}_seed${SEED}"
        fi
    else
        LOG_DIR="./logs/${EXPERIMENT_TYPE}/task_${TASK_ID}_seed${SEED}"
    fi

    if [ -n "$CONFIG_TAG" ]; then
        LOG_DIR=$(inject_config_tag_into_log_path "$LOG_DIR" "$CONFIG_TAG")
    fi

    export LOG_DIR
    mkdir -p "$LOG_DIR"
    EXPERIMENT_NAME=$(basename "$LOG_DIR")

    echo "Configuration:"
    echo "  Task ID: $TASK_ID"
    echo "  Config Name: $CONFIG_NAME"
    echo "  Seed: $SEED"
    echo "  Log Dir: $LOG_DIR"
    echo "  Eval Config: $EVAL_CONFIG_NAME"

    if [ -n "$CHECKPOINT_PATH" ]; then
        if [ ! -f "$CHECKPOINT_PATH" ]; then
            echo "ERROR: checkpoint not found: $CHECKPOINT_PATH"
            OVERALL_EXIT_CODE=1
            break
        fi
        echo "  Loading checkpoint: $CHECKPOINT_PATH"
    else
        echo "  First task: training from base SmolVLA policy path in config"
    fi

    OVERRIDES="env.fixed_task_ids=[${TASK_ID}] runner.logger.experiment_name=${EXPERIMENT_NAME} actor.seed=${SEED} actor.model.base_policy_path=${SMOLVLA_BASE_POLICY_PATH}"

    if [ -n "$CHECKPOINT_PATH" ]; then
        OVERRIDES="$OVERRIDES actor.checkpoint_load_path=${CHECKPOINT_PATH} rollout.model_dir=${CHECKPOINT_PATH}"
    else
        OVERRIDES="$OVERRIDES actor.checkpoint_load_path=${SMOLVLA_BASE_POLICY_PATH} rollout.model_dir=${SMOLVLA_BASE_POLICY_PATH}"
    fi

    if [ -n "$MAX_EPOCH" ]; then
        if ! [[ "$MAX_EPOCH" =~ ^[0-9]+$ ]] || [ "$MAX_EPOCH" -le 0 ]; then
            echo "ERROR: MAX_EPOCH must be a positive integer"
            OVERALL_EXIT_CODE=1
            break
        fi
        OVERRIDES="$OVERRIDES runner.max_epochs=${MAX_EPOCH}"
    fi

    echo "Running with overrides:"
    echo "$OVERRIDES"
    echo ""

    bash examples/embodiment/run_embodiment.sh ${CONFIG_NAME} $OVERRIDES
    EXIT_CODE=$?

    echo ""
    echo "========================================="
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Task $TASK_ID completed successfully"
        CHECKPOINT_LOCATION=$(echo "$LOG_DIR" | sed 's|^\./||')
        bash examples/crl_experiment/eval_smolvla_rl_embodiment.sh "${CHECKPOINT_LOCATION}" "${GLOBAL_STEP}" "${EVAL_CONFIG_NAME}" "${SMOLVLA_BASE_POLICY_PATH}"
    else
        echo "Task $TASK_ID failed with exit code $EXIT_CODE"
        OVERALL_EXIT_CODE=$EXIT_CODE
        break
    fi
    echo "========================================="
done

echo ""
echo "Finished at: $(date)"
exit $OVERALL_EXIT_CODE
