#!/bin/bash

# Sequential Task Training Script
# Trains on tasks sequentially, loading checkpoints from previous task
# Evaluates after each training run
#
# Usage:
#   ./run_lifelong.sh [config_name] [bc_coeff] [num_tasks] [--start-with-eval]
#
# Flags:
#   --start-with-eval   Skip training for the first TASK_ID and run eval first.
#                       Requires PREV_CHECKPOINT_PATH to be set manually in
#                       this script (see below), since there is no prior
#                       training run to source it from.
#
# Example (resuming after eval crashed on task 5):
#   Edit PREV_CHECKPOINT_PATH manually to point to task_5's checkpoint, then:
#   ./run_lifelong.sh myconfig 0.0 5 --start-with-eval

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Get script directory
SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
RUN_EMBODIMENT_SCRIPT="${SCRIPT_DIR}/run_embodiment.sh"
EVAL_EMBODIMENT_SCRIPT="${SCRIPT_DIR}/../../examples/embodiment/eval_embodiment.sh"

# Check if run_embodiment.sh exists
if [ ! -f "$RUN_EMBODIMENT_SCRIPT" ]; then
    echo "ERROR: run_embodiment.sh not found at: $RUN_EMBODIMENT_SCRIPT"
    exit 1
fi

# Check if eval_embodiment.sh exists
if [ ! -f "$EVAL_EMBODIMENT_SCRIPT" ]; then
    echo "ERROR: eval_embodiment.sh not found at: $EVAL_EMBODIMENT_SCRIPT"
    exit 1
fi

# Default values
DEFAULT_CONFIG="libero_spatial_grpo_openvlaoft"
DEFAULT_BC_COEFF=0.00
DEFAULT_NUM_TASKS=5

# Parse flags and positional arguments
START_WITH_EVAL=false
POSITIONAL_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --start-with-eval)
            START_WITH_EVAL=true
            ;;
        *)
            POSITIONAL_ARGS+=("$arg")
            ;;
    esac
done

CONFIG_NAME="${POSITIONAL_ARGS[0]:-$DEFAULT_CONFIG}"
BC_COEFF="${POSITIONAL_ARGS[1]:-$DEFAULT_BC_COEFF}"
NUM_TASKS="${POSITIONAL_ARGS[2]:-$DEFAULT_NUM_TASKS}"
EVAL_CONFIG="$CONFIG_NAME"

# Get REPO_PATH (run_embodiment.sh will set this, but we need it for log dir)
export EMBODIED_PATH="$SCRIPT_DIR"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))

# Format BC coefficient for directory name (e.g., 0.03 -> 03, 0.3 -> 3, 0.005 -> 005)
BC_COEFF_FORMATTED=$(echo "$BC_COEFF" | sed 's/^0\.//' | sed 's/\.//g')

# Base log directory
BASE_LOG_DIR="${REPO_PATH}/logs/full_ft/libero_10"
mkdir -p "${BASE_LOG_DIR}"

# ============================================================================
# PREV_CHECKPOINT_PATH
# When using --start-with-eval, manually set this to the checkpoint of the
# task you want to evaluate first (i.e. the task that just finished training
# before the crash). The loop will eval it, then continue training from there.
# ============================================================================
PREV_CHECKPOINT_PATH="./logs/bcrl_logit/

if [ "$START_WITH_EVAL" = true ]; then
    if [ -z "$PREV_CHECKPOINT_PATH" ]; then
        echo "ERROR: --start-with-eval requires PREV_CHECKPOINT_PATH to be set manually in the script."
        exit 1
    fi
    echo "INFO: --start-with-eval is set. Will skip training for the first task and run eval first."
    echo "INFO: Using manually set PREV_CHECKPOINT_PATH: $PREV_CHECKPOINT_PATH"
fi

# ============================================================================
# Print Configuration
# ============================================================================

echo "========================================================================"
echo "Sequential Task Training"
echo "========================================================================"
echo "Config:           $CONFIG_NAME"
echo "Eval Config:      $EVAL_CONFIG"
echo "BC Coefficient:   $BC_COEFF"
echo "Num Tasks:        $NUM_TASKS (3 to $((NUM_TASKS-1+3)))"
echo "Base Log Dir:     $BASE_LOG_DIR"
echo "Start With Eval:  $START_WITH_EVAL"
echo "========================================================================"
echo ""

# ============================================================================
# Training Loop
# ============================================================================

FIRST_TASK=true

for TASK_ID in $(seq 3 $((NUM_TASKS-1+3))); do
    echo ""
    echo "========================================================================"
    echo "Task ${TASK_ID}"
    echo "========================================================================"

    # Create task-specific log directory
    TASK_LOG_DIR="${BASE_LOG_DIR}/task_${TASK_ID}"
    mkdir -p "${TASK_LOG_DIR}"

    # Derive this task's checkpoint path (used whether we train or skip)
    CHECKPOINT_DIR="${TASK_LOG_DIR}/checkpoints/global_step_5/actor/model.pt"

    # ------------------------------------------------------------------
    # Optionally skip training for the very first task in this run
    # ------------------------------------------------------------------
    if [ "$FIRST_TASK" = true ] && [ "$START_WITH_EVAL" = true ]; then
        echo "Skipping training for task ${TASK_ID} (--start-with-eval is set)"
        echo "Using PREV_CHECKPOINT_PATH: $PREV_CHECKPOINT_PATH"
        # Point CHECKPOINT_DIR at the manually set path so eval uses it
        CHECKPOINT_DIR="$PREV_CHECKPOINT_PATH"
    else
        # ------------------------------------------------------------------
        # Normal training path
        # ------------------------------------------------------------------
        OVERRIDES="env.fixed_task_ids=[${TASK_ID}] actor.seed=${BC_COEFF}"

        if [ $TASK_ID -gt 3 ]; then
            if [ -z "$PREV_CHECKPOINT_PATH" ]; then
                echo "ERROR: Previous checkpoint path is empty for task ${TASK_ID}"
                exit 1
            fi

            # if [ ! -d "$PREV_CHECKPOINT_PATH" ]; then
            #     echo "ERROR: Previous checkpoint does not exist: $PREV_CHECKPOINT_PATH"
            #     exit 1
            # fi

            echo "Loading checkpoint from: $PREV_CHECKPOINT_PATH"
            OVERRIDES="${OVERRIDES} +actor.model.ckpt_path=${PREV_CHECKPOINT_PATH}"
        fi

        echo "Task ${TASK_ID} overrides: ${OVERRIDES}"
        echo "Logging to: ${TASK_LOG_DIR}"
        echo ""

        export LOG_DIR="${TASK_LOG_DIR}"

        bash ${RUN_EMBODIMENT_SCRIPT} ${CONFIG_NAME} ${OVERRIDES}

        if [ $? -ne 0 ]; then
            echo ""
            echo "ERROR: Training failed for task ${TASK_ID}"
            echo "Check log file: ${TASK_LOG_DIR}/run_embodiment.log"
            exit 1
        fi

        if [ ! -d "$CHECKPOINT_DIR" ]; then
            echo ""
            echo "ERROR: Checkpoint not found at expected location: $CHECKPOINT_DIR"
            echo "Training may have completed but checkpoint was not saved correctly"
            exit 1
        fi

        echo ""
        echo "Task ${TASK_ID} training completed successfully"
        echo "Checkpoint saved at: $CHECKPOINT_DIR"
    fi

    # --------------------------------------------------------------------------
    # Evaluation after training on Task ${TASK_ID}
    # --------------------------------------------------------------------------
    echo ""
    echo "========================================================================"
    echo "Evaluating after Task ${TASK_ID} training"
    echo "========================================================================"

    TASK_EVAL_LOG_DIR="${BASE_LOG_DIR}/task_${TASK_ID}/eval"
    mkdir -p "${TASK_EVAL_LOG_DIR}"

    echo "Eval checkpoint: $CHECKPOINT_DIR"
    echo "Eval log dir:    $TASK_EVAL_LOG_DIR"
    echo ""

    export LOG_DIR="${TASK_EVAL_LOG_DIR}"

    bash ${EVAL_EMBODIMENT_SCRIPT} ${EVAL_CONFIG} "+actor.model.ckpt_path=${CHECKPOINT_DIR}"

    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Evaluation failed for task ${TASK_ID}"
        echo "Check log file: ${TASK_EVAL_LOG_DIR}/eval_embodiment.log"
        exit 1
    fi

    echo ""
    echo "Task ${TASK_ID} evaluation completed successfully"
    echo "Eval results saved at: $TASK_EVAL_LOG_DIR"
    # --------------------------------------------------------------------------

    # Set checkpoint path for next iteration
    PREV_CHECKPOINT_PATH="$CHECKPOINT_DIR"
    FIRST_TASK=false

    echo "========================================================================"
done

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================================================"
echo "All tasks completed successfully!"
echo "========================================================================"
echo "Results saved in: $BASE_LOG_DIR"
echo ""
echo "Task directories:"
for TASK_ID in $(seq 0 $((NUM_TASKS-1))); do
    echo "  Task ${TASK_ID}: ${BASE_LOG_DIR}/task_${TASK_ID}"
    echo "    Train: ${BASE_LOG_DIR}/task_${TASK_ID}/"
    echo "    Eval:  ${BASE_LOG_DIR}/task_${TASK_ID}/eval/"
done
echo "========================================================================"
