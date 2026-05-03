#!/usr/bin/env bash
#SBATCH --job-name=vlacrl_controller
#SBATCH --output=/home/s2758621/Continual_VLA_RL/logs/test-controller-%j.out
#SBATCH --error=/home/s2758621/Continual_VLA_RL/logs/test-controller-%j.err
#SBATCH --partition=ICF-Free
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=400G
#SBATCH --gres=gpu:nvidia_l40s:4

### Usage: bash examples/crl_experiment/run_embodiment_sequential_smolvla.sh TASK_ID_OR_RANGE [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
### Example (single task): bash examples/crl_experiment/run_embodiment_sequential_smolvla.sh 5
### Example (task range): bash examples/crl_experiment/run_embodiment_sequential_smolvla.sh "5,7"
### Example (resume LoRA): bash examples/crl_experiment/run_embodiment_sequential_smolvla.sh 6 ./logs_smolvla/sequential_smolvla/task_5_seed1234/checkpoints/global_step_10/actor 10
### Notes:
###   - This file can now be submitted directly with `sbatch`.
###   - Default placement uses 4 GPUs as two actor ranks plus two
###     rollout/env ranks. This preserves the OpenVLA-OFT paper budget
###     (10 * 11 * 12 * 8 = 10560 episodes/task) while splitting rollout
###     buffer memory across two workers.
###   - The launcher selects a hardware profile and applies matching Hydra overrides.
###   - Override detection with `SMOLVLA_HW_PROFILE={rtx2080ti|a40|l40s}` if needed.
###   - Precision is no longer overridden here; it comes from the Hydra config / model path.
###   - LoRA is enabled by default. Sequential tasks chain PEFT adapter dirs from
###     checkpoints/global_step_<N>/actor. Set SMOLVLA_IS_LORA=false for full-model
###     state_dict chaining via actor/model.pt.
###   - For task ranges, only the first task can take a manual checkpoint path.
###   - Subsequent tasks auto-load from the previous task's actor checkpoint.

set -euo pipefail

TASK_INPUT=${1:-5}
MANUAL_CHECKPOINT_PATH=${2:-}
MAX_EPOCH=${3:-10}
CONFIG_NAME=${4:-crl_experiment/libero_object_grpo_smolvla_object}
SEED=${5:-1234}
EXPERIMENT_TYPE="sequential_smolvla"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlacrl_libplus_smolvla}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"

# Optional: base LeRobot SmolVLA policy path used when checkpoint_load_path points to model.pt.
SMOLVLA_BASE_POLICY_PATH="${SMOLVLA_BASE_POLICY_PATH:-/home/s2758621/Continual_VLA_RL/model/smolvla/libero_object_20demo_base/pretrained_model}"

detect_smolvla_hw_profile() {
    local node_name="${SLURMD_NODENAME:-${HOSTNAME:-unknown}}"
    case "$node_name" in
        damnii*)
            echo "rtx2080ti"
            ;;
        scotia*)
            echo "l40s"
            ;;
        *)
            echo "a40"
            ;;
    esac
}

SMOLVLA_HW_PROFILE="${SMOLVLA_HW_PROFILE:-$(detect_smolvla_hw_profile)}"

case "$SMOLVLA_HW_PROFILE" in
    rtx2080ti)
        DEFAULT_SMOLVLA_ATTN_IMPL="sdpa"
        DEFAULT_SMOLVLA_MICRO_BATCH_SIZE="1"
        DEFAULT_SMOLVLA_GLOBAL_BATCH_SIZE="8192"
        DEFAULT_SMOLVLA_USE_AMP="true"
        DEFAULT_SMOLVLA_GRADIENT_CHECKPOINTING="true"
        ;;
    a40)
        DEFAULT_SMOLVLA_ATTN_IMPL="sdpa"
        DEFAULT_SMOLVLA_MICRO_BATCH_SIZE="1"
        DEFAULT_SMOLVLA_GLOBAL_BATCH_SIZE="8192"
        DEFAULT_SMOLVLA_USE_AMP="true"
        DEFAULT_SMOLVLA_GRADIENT_CHECKPOINTING="true"
        ;;
    l40s)
        DEFAULT_SMOLVLA_ATTN_IMPL="flash_attention_2"
        DEFAULT_SMOLVLA_MICRO_BATCH_SIZE="16"
        DEFAULT_SMOLVLA_GLOBAL_BATCH_SIZE="8192"
        DEFAULT_SMOLVLA_USE_AMP="true"
        DEFAULT_SMOLVLA_GRADIENT_CHECKPOINTING="true"
        ;;
    *)
        echo "ERROR: unsupported SMOLVLA_HW_PROFILE: $SMOLVLA_HW_PROFILE"
        echo "Valid values: rtx2080ti, a40, l40s"
        exit 1
        ;;
esac

# Profile defaults can still be overridden explicitly via environment.
SMOLVLA_ATTN_IMPL="${SMOLVLA_ATTN_IMPL:-$DEFAULT_SMOLVLA_ATTN_IMPL}"
SMOLVLA_MICRO_BATCH_SIZE="${SMOLVLA_MICRO_BATCH_SIZE:-$DEFAULT_SMOLVLA_MICRO_BATCH_SIZE}"
SMOLVLA_GLOBAL_BATCH_SIZE="${SMOLVLA_GLOBAL_BATCH_SIZE:-$DEFAULT_SMOLVLA_GLOBAL_BATCH_SIZE}"
SMOLVLA_USE_AMP="${SMOLVLA_USE_AMP:-$DEFAULT_SMOLVLA_USE_AMP}"
SMOLVLA_GRADIENT_CHECKPOINTING="${SMOLVLA_GRADIENT_CHECKPOINTING:-$DEFAULT_SMOLVLA_GRADIENT_CHECKPOINTING}"
SMOLVLA_ACTOR_GPUS="${SMOLVLA_ACTOR_GPUS:-0-1}"
SMOLVLA_ROLLOUT_GPUS="${SMOLVLA_ROLLOUT_GPUS:-2-3}"
SMOLVLA_ENV_GPUS="${SMOLVLA_ENV_GPUS:-2-3}"
SMOLVLA_NUM_GROUP_ENVS="${SMOLVLA_NUM_GROUP_ENVS:-12}"
SMOLVLA_ROLLOUT_EPOCH="${SMOLVLA_ROLLOUT_EPOCH:-11}"
SMOLVLA_IS_LORA="${SMOLVLA_IS_LORA:-true}"
SMOLVLA_LORA_RANK="${SMOLVLA_LORA_RANK:-32}"
SMOLVLA_LORA_ALPHA="${SMOLVLA_LORA_ALPHA:-32}"
SMOLVLA_LORA_DROPOUT="${SMOLVLA_LORA_DROPOUT:-0.0}"

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
echo "SMOLVLA_HW_PROFILE: $SMOLVLA_HW_PROFILE"
echo "SLURMD_NODENAME: ${SLURMD_NODENAME:-<unset>}"
echo "SMOLVLA_BASE_POLICY_PATH: $SMOLVLA_BASE_POLICY_PATH"
echo "SMOLVLA_ATTN_IMPL: $SMOLVLA_ATTN_IMPL"
echo "SMOLVLA_MICRO_BATCH_SIZE: $SMOLVLA_MICRO_BATCH_SIZE"
echo "SMOLVLA_GLOBAL_BATCH_SIZE: $SMOLVLA_GLOBAL_BATCH_SIZE"
echo "SMOLVLA_USE_AMP: $SMOLVLA_USE_AMP"
echo "SMOLVLA_GRADIENT_CHECKPOINTING: $SMOLVLA_GRADIENT_CHECKPOINTING"
echo "SMOLVLA_ACTOR_GPUS: $SMOLVLA_ACTOR_GPUS"
echo "SMOLVLA_ROLLOUT_GPUS: $SMOLVLA_ROLLOUT_GPUS"
echo "SMOLVLA_ENV_GPUS: $SMOLVLA_ENV_GPUS"
echo "SMOLVLA_NUM_GROUP_ENVS: $SMOLVLA_NUM_GROUP_ENVS"
echo "SMOLVLA_ROLLOUT_EPOCH: $SMOLVLA_ROLLOUT_EPOCH"
echo "MAX_EPOCH: ${MAX_EPOCH:-<config default>}"
echo "SMOLVLA_IS_LORA: $SMOLVLA_IS_LORA"
echo "SMOLVLA_LORA_RANK: $SMOLVLA_LORA_RANK"
echo "SMOLVLA_LORA_ALPHA: $SMOLVLA_LORA_ALPHA"
echo "SMOLVLA_LORA_DROPOUT: $SMOLVLA_LORA_DROPOUT"

# Keep Ray's Unix socket paths short enough for AF_UNIX limits.
RAY_TMP_BASE_DEFAULT="/tmp/r_${USER:-u}_${SLURM_JOB_ID:-smolvla}"
export RAY_TMPDIR="${RAY_TMPDIR:-$RAY_TMP_BASE_DEFAULT}"
export TMPDIR="$RAY_TMPDIR"
export TMP="$RAY_TMPDIR"
export TEMP="$RAY_TMPDIR"
mkdir -p "$RAY_TMPDIR"
chmod 700 "$RAY_TMPDIR" 2>/dev/null || true
echo "RAY_TMPDIR: $RAY_TMPDIR"
echo ""

REPO_ROOT="${CONTINUAL_VLA_RL_ROOT:-/home/s2758621/Continual_VLA_RL}"
if [ ! -d "$REPO_ROOT" ]; then
    echo "ERROR: REPO_ROOT does not exist: $REPO_ROOT"
    exit 1
fi
cd "$REPO_ROOT"

# GPU monitoring. Set GPU_MONITOR_ENABLED=0 to disable.
GPU_MONITOR_ENABLED="${GPU_MONITOR_ENABLED:-1}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-5}"
GPU_MONITOR_DIR="${GPU_MONITOR_DIR:-${REPO_ROOT}/logs/gpu_monitor}"
GPU_MONITOR_PREFIX="${GPU_MONITOR_PREFIX:-smolvla_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"

start_gpu_monitor() {
    if [ "$GPU_MONITOR_ENABLED" != "1" ]; then
        echo "GPU monitor disabled."
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "GPU monitor disabled: nvidia-smi not found."
        return 0
    fi

    mkdir -p "$GPU_MONITOR_DIR"
    GPU_MONITOR_GPU_LOG="${GPU_MONITOR_DIR}/${GPU_MONITOR_PREFIX}_gpu.csv"
    GPU_MONITOR_PMON_LOG="${GPU_MONITOR_DIR}/${GPU_MONITOR_PREFIX}_pmon.log"

    echo "GPU monitor logs:"
    echo "  GPU summary: $GPU_MONITOR_GPU_LOG"
    echo "  GPU process: $GPU_MONITOR_PMON_LOG"

    (
        echo "wall_time,smi_timestamp,index,name,util_gpu_pct,util_mem_pct,memory_used_mib,memory_total_mib,power_w,temp_c"
        while true; do
            wall_time="$(date --iso-8601=seconds)"
            nvidia-smi \
                --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
                --format=csv,noheader,nounits \
                | awk -v wall_time="$wall_time" 'BEGIN { FS=", "; OFS="," } { print wall_time,$0 }' \
                || true
            sleep "$GPU_MONITOR_INTERVAL"
        done
    ) >> "$GPU_MONITOR_GPU_LOG" &
    GPU_MONITOR_GPU_PID=$!

    (
        while true; do
            echo "===== $(date --iso-8601=seconds) ====="
            nvidia-smi pmon -c 1 -s um || true
            sleep "$GPU_MONITOR_INTERVAL"
        done
    ) >> "$GPU_MONITOR_PMON_LOG" &
    GPU_MONITOR_PMON_PID=$!
}

stop_gpu_monitor() {
    if [ -n "${GPU_MONITOR_GPU_PID:-}" ]; then
        kill "$GPU_MONITOR_GPU_PID" 2>/dev/null || true
    fi
    if [ -n "${GPU_MONITOR_PMON_PID:-}" ]; then
        kill "$GPU_MONITOR_PMON_PID" 2>/dev/null || true
    fi
}

start_gpu_monitor
trap stop_gpu_monitor EXIT

if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
elif [[ -x "$CONDA_BASE/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/bin/activate" "$CONDA_ENV_NAME"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
else
    echo "ERROR: Could not find conda activation scripts."
    exit 1
fi

echo "CONDA_ENV_NAME: $CONDA_ENV_NAME"
python - <<'PY'
import sys
import yaml

print(f"PYTHON_EXECUTABLE: {sys.executable}")
print(f"PYyaml_VERSION: {yaml.__version__}")
PY
echo ""

COMMON_FUNCTIONS_SH="${REPO_ROOT}/examples/crl_experiment/common_functions.sh"
if [ ! -f "$COMMON_FUNCTIONS_SH" ]; then
    echo "ERROR: common_functions.sh not found: $COMMON_FUNCTIONS_SH"
    exit 1
fi
source "$COMMON_FUNCTIONS_SH"
CONFIG_TAG=$(extract_config_tag "$CONFIG_NAME")
GLOBAL_STEP="${SMOLVLA_GLOBAL_STEP:-$(get_default_global_step "$CONFIG_NAME")}"
if [ -n "$MAX_EPOCH" ] && [ -z "${SMOLVLA_GLOBAL_STEP:-}" ]; then
    GLOBAL_STEP="$MAX_EPOCH"
fi
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
        if [ "$SMOLVLA_IS_LORA" = "true" ]; then
            CHECKPOINT_PATH="${PREV_LOG_DIR}/checkpoints/global_step_${GLOBAL_STEP}/actor"
        else
            CHECKPOINT_PATH="${PREV_LOG_DIR}/checkpoints/global_step_${GLOBAL_STEP}/actor/model.pt"
        fi
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
        if [ "$SMOLVLA_IS_LORA" = "true" ]; then
            if [ ! -f "${CHECKPOINT_PATH}/adapter_config.json" ] || [ ! -f "${CHECKPOINT_PATH}/adapter_model.bin" ]; then
                echo "ERROR: LoRA adapter checkpoint not found at: $CHECKPOINT_PATH"
                echo "       Expected adapter_config.json and adapter_model.bin"
                OVERALL_EXIT_CODE=1
                break
            fi
        else
            if [ ! -f "$CHECKPOINT_PATH" ]; then
                echo "ERROR: checkpoint not found: $CHECKPOINT_PATH"
                OVERALL_EXIT_CODE=1
                break
            fi
        fi
        echo "  Loading checkpoint: $CHECKPOINT_PATH"
    else
        echo "  First task: training from base SmolVLA policy path in config"
    fi

    if [ ! -f "${SMOLVLA_BASE_POLICY_PATH}/config.json" ] || [ ! -f "${SMOLVLA_BASE_POLICY_PATH}/model.safetensors" ]; then
        echo "ERROR: invalid SmolVLA base policy path: $SMOLVLA_BASE_POLICY_PATH"
        OVERALL_EXIT_CODE=1
        break
    fi

    OVERRIDES="env.fixed_task_ids=[${TASK_ID}] runner.logger.experiment_name=${EXPERIMENT_NAME} actor.seed=${SEED} smolvla.base_policy_path=${SMOLVLA_BASE_POLICY_PATH} ++cluster.component_placement.actor=${SMOLVLA_ACTOR_GPUS} ++cluster.component_placement.rollout=${SMOLVLA_ROLLOUT_GPUS} ++cluster.component_placement.env=${SMOLVLA_ENV_GPUS} algorithm.num_group_envs=${SMOLVLA_NUM_GROUP_ENVS} algorithm.rollout_epoch=${SMOLVLA_ROLLOUT_EPOCH} actor.micro_batch_size=${SMOLVLA_MICRO_BATCH_SIZE} actor.global_batch_size=${SMOLVLA_GLOBAL_BATCH_SIZE} actor.model.attn_implementation=${SMOLVLA_ATTN_IMPL} actor.model.use_amp=${SMOLVLA_USE_AMP} actor.model.gradient_checkpointing=${SMOLVLA_GRADIENT_CHECKPOINTING} actor.model.is_lora=${SMOLVLA_IS_LORA} actor.model.lora_rank=${SMOLVLA_LORA_RANK} actor.model.lora_alpha=${SMOLVLA_LORA_ALPHA} actor.model.lora_dropout=${SMOLVLA_LORA_DROPOUT} env.train.num_images_in_input=2 env.eval.num_images_in_input=2"

    if [ -n "$CHECKPOINT_PATH" ]; then
        if [ "$SMOLVLA_IS_LORA" = "true" ]; then
            OVERRIDES="$OVERRIDES actor.checkpoint_load_path=${SMOLVLA_BASE_POLICY_PATH} rollout.model_dir=${SMOLVLA_BASE_POLICY_PATH} +actor.model.lora_path=${CHECKPOINT_PATH}"
        else
            OVERRIDES="$OVERRIDES actor.checkpoint_load_path=${CHECKPOINT_PATH} rollout.model_dir=${CHECKPOINT_PATH}"
        fi
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
