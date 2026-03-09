#!/bin/bash
#
# Sequential Task Training Script
# Trains on tasks sequentially, loading checkpoints from previous task
#
# Usage:
#   ./run_lifelong.sh [config_name] [seed] [num_tasks] [lora_rank]

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Get script directory
SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
RUN_EMBODIMENT_SCRIPT="${SCRIPT_DIR}/run_embodiment.sh"

# Check if run_embodiment.sh exists
if [ ! -f "$RUN_EMBODIMENT_SCRIPT" ]; then
    echo "ERROR: run_embodiment.sh not found at: $RUN_EMBODIMENT_SCRIPT"
    exit 1
fi

# Default values
DEFAULT_CONFIG="libero_spatial_grpo_openvlaoft"
DEFAULT_SEED=0
DEFAULT_NUM_TASKS=5
DEFAULT_LORA_RANK=8

# Get arguments
CONFIG_NAME="${1:-$DEFAULT_CONFIG}"
SEED="${2:-$DEFAULT_SEED}"
NUM_TASKS="${3:-$DEFAULT_NUM_TASKS}"
LORA_RANK="${4:-$DEFAULT_LORA_RANK}"

# Get REPO_PATH
export EMBODIED_PATH="$SCRIPT_DIR"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))

# Base log directory
BASE_LOG_DIR="${REPO_PATH}/logs/naive_LoRA/"
mkdir -p "${BASE_LOG_DIR}"

# ============================================================================
# Print Configuration
# ============================================================================

echo "========================================================================"
echo "Sequential Task Training"
echo "========================================================================"
echo "Config:      $CONFIG_NAME"
echo "Seed:        $SEED"
echo "Num Tasks:   $NUM_TASKS (0 to $((NUM_TASKS-1)))"
echo "LoRA Rank:   $LORA_RANK"
echo "Base Log Dir:$BASE_LOG_DIR"
echo "========================================================================"
echo ""

# ============================================================================
# Training Loop
# ============================================================================

PREV_CHECKPOINT_PATH=""

for TASK_ID in $(seq 0 $((NUM_TASKS-1))); do
    echo ""
    echo "========================================================================"
    echo "Training on Task ${TASK_ID}"
    echo "========================================================================"

    TASK_LOG_DIR="${BASE_LOG_DIR}/task_${TASK_ID}"
    mkdir -p "${TASK_LOG_DIR}"

    # Updated overrides
    OVERRIDES="env.fixed_task_ids=[${TASK_ID}] actor.seed=${SEED} actor.model.lora_rank=${LORA_RANK}"

    if [ $TASK_ID -gt 0 ]; then
        if [ -z "$PREV_CHECKPOINT_PATH" ]; then
            echo "ERROR: Previous checkpoint path is empty for task ${TASK_ID}"
            exit 1
        fi

        if [ ! -d "$PREV_CHECKPOINT_PATH" ]; then
            echo "ERROR: Previous checkpoint does not exist: $PREV_CHECKPOINT_PATH"
            exit 1
        fi

        echo "Loading checkpoint from: $PREV_CHECKPOINT_PATH"
        OVERRIDES="${OVERRIDES} +actor.model.lora_path=${PREV_CHECKPOINT_PATH}"
    fi

    echo "Task ${TASK_ID} overrides: ${OVERRIDES}"
    echo "Logging to: ${TASK_LOG_DIR}"
    echo ""

    export LOG_DIR="${TASK_LOG_DIR}"

    bash ${RUN_EMBODIMENT_SCRIPT} ${CONFIG_NAME} ${OVERRIDES}

    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Training failed for task ${TASK_ID}"
        exit 1
    fi

    CHECKPOINT_DIR="${TASK_LOG_DIR}/checkpoints/global_step_10/actor"

    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo ""
        echo "ERROR: Checkpoint not found at: $CHECKPOINT_DIR"
        exit 1
    fi

    echo ""
    echo "Task ${TASK_ID} completed successfully"
    echo "Checkpoint saved at: $CHECKPOINT_DIR"

    PREV_CHECKPOINT_PATH="$CHECKPOINT_DIR"

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
for TASK_ID in $(seq 0 $((NUM_TASKS-1))); do
    echo "  Task ${TASK_ID}: ${BASE_LOG_DIR}/task_${TASK_ID}"
done
echo "========================================================================"
