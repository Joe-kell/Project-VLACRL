#!/bin/bash
#
# Evaluate SmolVLA RL checkpoints saved as actor/model.pt in CRL.
#
# Usage:
#   bash examples/crl_experiment/eval_smolvla_rl_embodiment.sh CHECKPOINT_LOCATION [STEP_NUMBER] [CONFIG_NAME] [SMOLVLA_BASE_POLICY_PATH]
#
# Examples:
#   bash examples/crl_experiment/eval_smolvla_rl_embodiment.sh logs/sequential_smolvla/task_5_seed1234
#   bash examples/crl_experiment/eval_smolvla_rl_embodiment.sh logs/sequential_smolvla/task_5_seed1234 10
#   bash examples/crl_experiment/eval_smolvla_rl_embodiment.sh logs/sequential_smolvla/task_5_seed1234 10 crl_experiment/libero_object_grpo_smolvla_eval_object /path/to/pretrained_model

set -euo pipefail

CHECKPOINT_LOCATION=${1:-}
STEP_NUMBER=${2:-10}
CONFIG_NAME=${3:-crl_experiment/libero_object_grpo_smolvla_eval_object}
SMOLVLA_BASE_POLICY_PATH_ARG=${4:-}

if [ -z "$CHECKPOINT_LOCATION" ]; then
    echo "ERROR: Missing CHECKPOINT_LOCATION"
    echo "Usage: bash examples/crl_experiment/eval_smolvla_rl_embodiment.sh CHECKPOINT_LOCATION [STEP_NUMBER] [CONFIG_NAME] [SMOLVLA_BASE_POLICY_PATH]"
    exit 1
fi

if ! [[ "$STEP_NUMBER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: STEP_NUMBER must be a non-negative integer, got: $STEP_NUMBER"
    exit 1
fi

WORKSPACE_ROOT=$(pwd)
CHECKPOINT_LOCATION="${CHECKPOINT_LOCATION%/}"
CHECKPOINT_PATH="${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_${STEP_NUMBER}/actor/model.pt"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: SmolVLA RL checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

SMOLVLA_BASE_POLICY_PATH="${SMOLVLA_BASE_POLICY_PATH_ARG:-${SMOLVLA_BASE_POLICY_PATH:-/home/s2758621/Octo_RL/checkpoints/smolvla_libero_object_20demo/job_3441070/checkpoints/last/pretrained_model}}"

if [ ! -f "${SMOLVLA_BASE_POLICY_PATH}/config.json" ] || [ ! -f "${SMOLVLA_BASE_POLICY_PATH}/model.safetensors" ]; then
    echo "ERROR: SMOLVLA_BASE_POLICY_PATH is not a valid LeRobot pretrained_model directory:"
    echo "  ${SMOLVLA_BASE_POLICY_PATH}"
    exit 1
fi

echo "Working Directory: $(pwd)"
echo "Checkpoint Location: $CHECKPOINT_LOCATION"
echo "Checkpoint File: $CHECKPOINT_PATH"
echo "Base SmolVLA Policy: $SMOLVLA_BASE_POLICY_PATH"
echo "Config Name: $CONFIG_NAME"
echo "Start Time: $(date)"
echo ""

export SMOLVLA_POLICY_PATH="$SMOLVLA_BASE_POLICY_PATH"
export EVAL_STEP_NUMBER="$STEP_NUMBER"

bash examples/embodiment/eval_embodiment.sh \
    "${CONFIG_NAME}" \
    rollout.model_dir="${CHECKPOINT_PATH}" \
    actor.checkpoint_load_path="${CHECKPOINT_PATH}" \
    actor.model.base_policy_path="${SMOLVLA_BASE_POLICY_PATH}" \
    actor.tokenizer.tokenizer_model="${SMOLVLA_BASE_POLICY_PATH}" \
    actor.model.model_name=smolvla \
    actor.model.is_lora=False

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully."
else
    echo "Evaluation failed with exit code $EXIT_CODE"
fi
echo "End Time: $(date)"
exit $EXIT_CODE
