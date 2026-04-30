#!/usr/bin/env bash
#SBATCH --job-name=vlacrl_controller
#SBATCH --output=/home/s2758621/Continual_VLA_RL/logs/test-controller-%j.out
#SBATCH --error=/home/s2758621/Continual_VLA_RL/logs/test-controller-%j.err
#SBATCH --partition=ICF-Free
#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=500G
#SBATCH --gres=gpu:a40:4

set -uo pipefail

DEFAULT_REPO_ROOT="/home/s2758621/Continual_VLA_RL"

if [[ -n "${REPO_ROOT:-}" ]] && [[ -d "${REPO_ROOT}/examples" ]]; then
  REPO_ROOT="$REPO_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -d "${SLURM_SUBMIT_DIR}/examples" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
elif [[ -d "${DEFAULT_REPO_ROOT}/examples" ]]; then
  REPO_ROOT="$DEFAULT_REPO_ROOT"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$SCRIPT_DIR"
fi

cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlacrl}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"

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

export HF_HOME="${HF_HOME:-$REPO_ROOT/model}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-$HF_HUB_OFFLINE}"
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_DATASETS_CACHE"

RAY_TMP_BASE_DEFAULT="/tmp/ray_${USER:-unknown}"
export RAY_TMPDIR="${RAY_TMPDIR:-$RAY_TMP_BASE_DEFAULT}"
export TMPDIR="${TMPDIR:-$RAY_TMPDIR}"
export TMP="${TMP:-$RAY_TMPDIR}"
export TEMP="${TEMP:-$RAY_TMPDIR}"
mkdir -p "$RAY_TMPDIR"
chmod 700 "$RAY_TMPDIR" 2>/dev/null || true

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,ENV}"

CONFIG_NAME="${CONFIG_NAME:-crl_experiment/libero_10_grpo_openvlaoft_long}"
SEED="${SEED:-1234}"
LIBERO_TYPE="${LIBERO_TYPE:-standard}"
LIBERO_SUFFIX="${LIBERO_SUFFIX:-}"
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
MAX_RETRIES="${MAX_RETRIES:-2}"
MANUAL_CHECKPOINT_PATH="${MANUAL_CHECKPOINT_PATH:-}"

ACTOR_GPU_MAP="${ACTOR_GPU_MAP:-0-2}"
ROLLOUT_GPU_MAP="${ROLLOUT_GPU_MAP:-3}"
ENV_GPU_MAP="${ENV_GPU_MAP:-3}"
TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-4}"
TRAIN_NUM_GROUP_ENVS="${TRAIN_NUM_GROUP_ENVS:-6}"
TRAIN_ROLLOUT_EPOCH="${TRAIN_ROLLOUT_EPOCH:-15}"
ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-40}"
ACTOR_GLOBAL_BATCH_SIZE="${ACTOR_GLOBAL_BATCH_SIZE:-1440}"
ACTOR_ENABLE_OFFLOAD="${ACTOR_ENABLE_OFFLOAD:-0}"
PRESERVE_EPISODE_BUDGET="${PRESERVE_EPISODE_BUDGET:-0}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-20}"
PRESERVE_EVAL_BUDGET="${PRESERVE_EVAL_BUDGET:-1}"
STRICT_EPISODE_BUDGET="${STRICT_EPISODE_BUDGET:-1}"
LOGGER_BACKENDS="${LOGGER_BACKENDS:-tensorboard}"
TRUE_RESUME_ENABLED="${TRUE_RESUME_ENABLED:-1}"
ROLLING_PARTIAL_CHECKPOINT_ENABLED="${ROLLING_PARTIAL_CHECKPOINT_ENABLED:-1}"
PARTIAL_RESUME_EXIT_CODE="${PARTIAL_RESUME_EXIT_CODE:-85}"
RUNNER_RESUME_SAVE_GRACE_SECONDS="${RUNNER_RESUME_SAVE_GRACE_SECONDS:-1800}"
ROLLING_LATEST_CHECKPOINT_NAME="${ROLLING_LATEST_CHECKPOINT_NAME:-latest_partial}"
ROLLING_PREVIOUS_CHECKPOINT_NAME="${ROLLING_PREVIOUS_CHECKPOINT_NAME:-previous_partial}"
ROLLING_TMP_CHECKPOINT_NAME="${ROLLING_TMP_CHECKPOINT_NAME:-latest_partial_tmp}"

source "examples/crl_experiment/common_functions.sh"

CONFIG_TAG="$(extract_config_tag "$CONFIG_NAME")"
BASE_EVAL_CONFIG_NAME="$(derive_eval_config_name "$CONFIG_NAME")"
WRAPPER_EVAL_DEFAULT_STEP=10

if [[ "$CONFIG_TAG" == "object" ]]; then
  SUITE_NAME="libero_object"
  TASK_IDS=(5 6 7 8 9)
  BASE_MAX_EPOCHS=10
  BASE_ROLLOUT_EPOCH=16
  BASE_GROUP_SIZE=8
  BASE_NUM_GROUP_ENVS=8
  BASE_EVAL_NUM_ENVS=80
  BASE_EVAL_ROLLOUT_EPOCH=8
  BASE_MAX_EPISODE_STEPS=512
  MODEL_NUM_ACTION_CHUNKS=8
elif [[ "$CONFIG_TAG" == "long" ]]; then
  SUITE_NAME="libero_long"
  TASK_IDS=(3 4 5 6 7)
  BASE_MAX_EPOCHS=5
  BASE_ROLLOUT_EPOCH=16
  BASE_GROUP_SIZE=8
  BASE_NUM_GROUP_ENVS=8
  BASE_EVAL_NUM_ENVS=80
  BASE_EVAL_ROLLOUT_EPOCH=8
  BASE_MAX_EPISODE_STEPS=512
  MODEL_NUM_ACTION_CHUNKS=8
else
  SUITE_NAME="libero_spatial"
  TASK_IDS=(0 1 2 3 4)
  BASE_MAX_EPOCHS=10
  BASE_ROLLOUT_EPOCH=16
  BASE_GROUP_SIZE=8
  BASE_NUM_GROUP_ENVS=8
  BASE_EVAL_NUM_ENVS=80
  BASE_EVAL_ROLLOUT_EPOCH=20
  BASE_MAX_EPISODE_STEPS=512
  MODEL_NUM_ACTION_CHUNKS=8
fi

INPUT_MAX_EPOCHS="${MAX_EPOCH:-14}"
if [[ -z "$LOGGER_BACKENDS" ]] && [[ "$PRESERVE_EPISODE_BUDGET" == "0" ]] && [[ "$INPUT_MAX_EPOCHS" == "1" ]]; then
  LOGGER_BACKENDS="tensorboard"
fi
BASE_BUDGET_EPOCHS="${BASE_BUDGET_EPOCHS:-$BASE_MAX_EPOCHS}"
TRAIN_N_CHUNK_STEPS=$((BASE_MAX_EPISODE_STEPS / MODEL_NUM_ACTION_CHUNKS))

count_gpu_slots() {
  local map="$1"
  local count=0
  local token
  local start
  local end
  local tmp

  IFS=',' read -r -a tokens <<< "$map"
  for token in "${tokens[@]}"; do
    token="${token//[[:space:]]/}"
    [[ -z "$token" ]] && continue
    if [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start="${BASH_REMATCH[1]}"
      end="${BASH_REMATCH[2]}"
      if (( end < start )); then
        tmp="$start"
        start="$end"
        end="$tmp"
      fi
      count=$((count + end - start + 1))
    elif [[ "$token" =~ ^[0-9]+$ ]]; then
      count=$((count + 1))
    else
      echo "ERROR: invalid GPU map token '$token' in map '$map'"
      exit 1
    fi
  done
  echo "$count"
}

ACTOR_WORLD_SIZE="$(count_gpu_slots "$ACTOR_GPU_MAP")"
if (( ACTOR_WORLD_SIZE < 2 )); then
  echo "ERROR: Controller requires multi-actor placement. Got ACTOR_GPU_MAP='$ACTOR_GPU_MAP'."
  exit 1
fi
if [[ -z "$TRAIN_NUM_GROUP_ENVS" ]]; then
  if [[ "$PRESERVE_EPISODE_BUDGET" == "1" ]]; then
    case "$ACTOR_WORLD_SIZE" in
      2)
        TRAIN_NUM_GROUP_ENVS=16
        ;;
      4)
        TRAIN_NUM_GROUP_ENVS=4
        ;;
      3)
        TRAIN_NUM_GROUP_ENVS=6
        ;;
      *)
        TRAIN_NUM_GROUP_ENVS=4
        ;;
    esac
  else
    case "$ACTOR_WORLD_SIZE" in
      3)
        if (( TRAIN_ROLLOUT_EPOCH == 1 )); then
          TRAIN_NUM_GROUP_ENVS=3
        else
          TRAIN_NUM_GROUP_ENVS=6
        fi
        ;;
      *)
        TRAIN_NUM_GROUP_ENVS=4
        ;;
    esac
  fi
fi
if [[ -z "$ACTOR_GLOBAL_BATCH_SIZE" ]]; then
  if (( (TRAIN_ROLLOUT_EPOCH * TRAIN_NUM_GROUP_ENVS) % ACTOR_WORLD_SIZE != 0 )); then
    echo "ERROR: Invalid actor/local rollout geometry for world size $ACTOR_WORLD_SIZE"
    echo "       TRAIN_ROLLOUT_EPOCH * TRAIN_NUM_GROUP_ENVS must be divisible by actor_world_size"
    echo "       rollout_epoch=$TRAIN_ROLLOUT_EPOCH num_group_envs=$TRAIN_NUM_GROUP_ENVS world=$ACTOR_WORLD_SIZE"
    exit 1
  fi

  ROLLOUT_BATCH_PER_RANK=$((TRAIN_ROLLOUT_EPOCH * TRAIN_GROUP_SIZE * TRAIN_NUM_GROUP_ENVS / ACTOR_WORLD_SIZE))
  ROLLOUT_SIZE_PER_RANK=$((TRAIN_N_CHUNK_STEPS * ROLLOUT_BATCH_PER_RANK))
  TARGET_LOCAL_BATCH_SIZE=512
  BEST_LOCAL_BATCH_SIZE=0
  BEST_LOCAL_BATCH_DISTANCE=0

  for ((CANDIDATE_LOCAL_BATCH=ACTOR_MICRO_BATCH_SIZE; CANDIDATE_LOCAL_BATCH<=ROLLOUT_SIZE_PER_RANK; CANDIDATE_LOCAL_BATCH+=ACTOR_MICRO_BATCH_SIZE)); do
    if (( ROLLOUT_SIZE_PER_RANK % CANDIDATE_LOCAL_BATCH != 0 )); then
      continue
    fi
    CANDIDATE_DISTANCE=$((TARGET_LOCAL_BATCH_SIZE - CANDIDATE_LOCAL_BATCH))
    if (( CANDIDATE_DISTANCE < 0 )); then
      CANDIDATE_DISTANCE=$(( -CANDIDATE_DISTANCE ))
    fi
    if (( BEST_LOCAL_BATCH_SIZE == 0 || CANDIDATE_DISTANCE < BEST_LOCAL_BATCH_DISTANCE || (CANDIDATE_DISTANCE == BEST_LOCAL_BATCH_DISTANCE && CANDIDATE_LOCAL_BATCH < BEST_LOCAL_BATCH_SIZE) )); then
      BEST_LOCAL_BATCH_SIZE=$CANDIDATE_LOCAL_BATCH
      BEST_LOCAL_BATCH_DISTANCE=$CANDIDATE_DISTANCE
    fi
  done

  if (( BEST_LOCAL_BATCH_SIZE == 0 )); then
    echo "ERROR: No valid local batch size exists for the current rollout geometry."
    echo "       local_rollout_size=$ROLLOUT_SIZE_PER_RANK micro_batch_size=$ACTOR_MICRO_BATCH_SIZE"
    echo "       This means the actor dataloader would fail even if global_batch_size satisfied the basic divisibility check."
    exit 1
  fi

  ACTOR_GLOBAL_BATCH_SIZE=$((BEST_LOCAL_BATCH_SIZE * ACTOR_WORLD_SIZE))
fi
if (( ACTOR_GLOBAL_BATCH_SIZE % (ACTOR_MICRO_BATCH_SIZE * ACTOR_WORLD_SIZE) != 0 )); then
  echo "ERROR: Invalid batch arithmetic: global_batch_size must be divisible by micro_batch_size * actor_world_size"
  echo "       global=$ACTOR_GLOBAL_BATCH_SIZE micro=$ACTOR_MICRO_BATCH_SIZE world=$ACTOR_WORLD_SIZE"
  exit 1
fi
if (( TRAIN_NUM_GROUP_ENVS % ACTOR_WORLD_SIZE != 0 )); then
  echo "ERROR: Invalid GRPO grouping for actor world size $ACTOR_WORLD_SIZE"
  echo "       TRAIN_NUM_GROUP_ENVS=$TRAIN_NUM_GROUP_ENVS must be divisible by actor_world_size"
  echo "       Otherwise actor-local rollout batches stop being divisible by TRAIN_GROUP_SIZE=$TRAIN_GROUP_SIZE."
  exit 1
fi
if (( (TRAIN_ROLLOUT_EPOCH * TRAIN_NUM_GROUP_ENVS) % ACTOR_WORLD_SIZE != 0 )); then
  echo "ERROR: Invalid rollout reshaping for actor world size $ACTOR_WORLD_SIZE"
  echo "       TRAIN_ROLLOUT_EPOCH * TRAIN_NUM_GROUP_ENVS must be divisible by actor_world_size"
  echo "       Otherwise post-rollout batch_size stops being divisible by TRAIN_GROUP_SIZE=$TRAIN_GROUP_SIZE."
  exit 1
fi

ROLLOUT_BATCH_PER_RANK=$((TRAIN_ROLLOUT_EPOCH * TRAIN_GROUP_SIZE * TRAIN_NUM_GROUP_ENVS / ACTOR_WORLD_SIZE))
ROLLOUT_SIZE_PER_RANK=$((TRAIN_N_CHUNK_STEPS * ROLLOUT_BATCH_PER_RANK))
BATCH_SIZE_PER_RANK=$((ACTOR_GLOBAL_BATCH_SIZE / ACTOR_WORLD_SIZE))
if (( ROLLOUT_SIZE_PER_RANK % BATCH_SIZE_PER_RANK != 0 )); then
  echo "ERROR: Invalid actor dataloader arithmetic for the current rollout geometry"
  echo "       local_rollout_size=$ROLLOUT_SIZE_PER_RANK batch_size_per_rank=$BATCH_SIZE_PER_RANK"
  echo "       This would fail fsdp_actor_worker.py assertion rollout_size % batch_size_per_rank == 0"
  exit 1
fi

if [[ "$PRESERVE_EPISODE_BUDGET" == "1" ]]; then
  BASE_SAMPLES_PER_TASK=$((BASE_BUDGET_EPOCHS * BASE_ROLLOUT_EPOCH * BASE_GROUP_SIZE * BASE_NUM_GROUP_ENVS))
  SCALED_SAMPLES_PER_EPOCH=$((TRAIN_ROLLOUT_EPOCH * TRAIN_GROUP_SIZE * TRAIN_NUM_GROUP_ENVS))
  if (( SCALED_SAMPLES_PER_EPOCH <= 0 )); then
    echo "ERROR: Invalid scaled training budget; got $SCALED_SAMPLES_PER_EPOCH samples/epoch"
    exit 1
  fi
  if [[ "$STRICT_EPISODE_BUDGET" == "1" ]] && (( BASE_SAMPLES_PER_TASK % SCALED_SAMPLES_PER_EPOCH != 0 )); then
    echo "ERROR: Exact episode-budget parity is impossible with the current scaled settings."
    echo "       base_samples_per_task=$BASE_SAMPLES_PER_TASK scaled_samples_per_epoch=$SCALED_SAMPLES_PER_EPOCH"
    echo "       The current layout is suitable for smoke/profiling, but not exact paper-budget matching."
    exit 1
  fi
  EFFECTIVE_TRAIN_EPOCHS=$(((BASE_SAMPLES_PER_TASK + SCALED_SAMPLES_PER_EPOCH - 1) / SCALED_SAMPLES_PER_EPOCH))
else
  EFFECTIVE_TRAIN_EPOCHS="$INPUT_MAX_EPOCHS"
fi
TRAIN_CHECKPOINT_STEP="$EFFECTIVE_TRAIN_EPOCHS"

if [[ "$PRESERVE_EVAL_BUDGET" == "1" ]]; then
  BASE_EVAL_EPISODES=$((BASE_EVAL_NUM_ENVS * BASE_EVAL_ROLLOUT_EPOCH))
  if (( EVAL_NUM_ENVS <= 0 )); then
    echo "ERROR: EVAL_NUM_ENVS must be > 0"
    exit 1
  fi
  EFFECTIVE_EVAL_ROLLOUT_EPOCH=$(((BASE_EVAL_EPISODES + EVAL_NUM_ENVS - 1) / EVAL_NUM_ENVS))
else
  EFFECTIVE_EVAL_ROLLOUT_EPOCH="$BASE_EVAL_ROLLOUT_EPOCH"
fi

VARIANT_LOG_SUFFIX=""
if [[ "$LIBERO_TYPE" != "standard" ]]; then
  VARIANT_LOG_SUFFIX="_${LIBERO_TYPE}"
  if [[ -n "$LIBERO_SUFFIX" ]]; then
    SAFE_LIBERO_SUFFIX="$(echo "$LIBERO_SUFFIX" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/_/g')"
    VARIANT_LOG_SUFFIX="${VARIANT_LOG_SUFFIX}_${SAFE_LIBERO_SUFFIX}"
  fi
fi

STATE_ROOT="${STATE_ROOT:-$REPO_ROOT/online_RL_logs}"
STATE_DIR_DEFAULT="$STATE_ROOT/${SUITE_NAME}/seed${SEED}${VARIANT_LOG_SUFFIX}"
STATE_DIR="${STATE_DIR:-$STATE_DIR_DEFAULT}"
STATE_FILE="${CONTROLLER_STATE_FILE:-$STATE_DIR/state.env}"
HISTORY_FILE="$STATE_DIR/history.tsv"
PATHS_FILE="$STATE_DIR/paths.env"
SUMMARY_FILE="$STATE_DIR/task_summary.tsv"
GENERATED_CONFIG_DIR="$REPO_ROOT/examples/embodiment/config/crl_controller/${SUITE_NAME}_seed${SEED}${VARIANT_LOG_SUFFIX}"
mkdir -p "$STATE_DIR" "$GENERATED_CONFIG_DIR"

TRAIN_CONFIG_BASENAME="$(basename "$CONFIG_NAME")"
EVAL_CONFIG_BASENAME="$(basename "$BASE_EVAL_CONFIG_NAME")"
CONTROLLER_TRAIN_CONFIG_NAME="crl_controller/${SUITE_NAME}_seed${SEED}${VARIANT_LOG_SUFFIX}/${TRAIN_CONFIG_BASENAME}"
CONTROLLER_EVAL_CONFIG_NAME="crl_controller/${SUITE_NAME}_seed${SEED}${VARIANT_LOG_SUFFIX}/${EVAL_CONFIG_BASENAME}"

save_state() {
  cat > "$STATE_FILE" <<STATEEOF
TASK_CURSOR=${TASK_CURSOR@Q}
RETRY_COUNT=${RETRY_COUNT@Q}
LAST_COMPLETED_TASK=${LAST_COMPLETED_TASK@Q}
LAST_CHECKPOINT=${LAST_CHECKPOINT@Q}
LAST_LOG_DIR=${LAST_LOG_DIR@Q}
LAST_TASK_RUNTIME_SECONDS=${LAST_TASK_RUNTIME_SECONDS@Q}
ACTIVE_TASK_ID=${ACTIVE_TASK_ID@Q}
ACTIVE_TASK_CHECKPOINT=${ACTIVE_TASK_CHECKPOINT@Q}
ACTIVE_TASK_LOG_DIR=${ACTIVE_TASK_LOG_DIR@Q}
ACTIVE_TASK_STEP=${ACTIVE_TASK_STEP@Q}
ACTIVE_TASK_MAX_EPOCHS=${ACTIVE_TASK_MAX_EPOCHS@Q}
RUN_NAMESPACE=${RUN_NAMESPACE@Q}
WRAPPER_WORKDIR=${WRAPPER_WORKDIR@Q}
STATEEOF
}

save_paths() {
  cat > "$PATHS_FILE" <<PATHSEOF
SUITE_NAME=${SUITE_NAME@Q}
SEED=${SEED@Q}
STATE_DIR=${STATE_DIR@Q}
STATE_FILE=${STATE_FILE@Q}
HISTORY_FILE=${HISTORY_FILE@Q}
SUMMARY_FILE=${SUMMARY_FILE@Q}
GENERATED_CONFIG_DIR=${GENERATED_CONFIG_DIR@Q}
LAST_COMPLETED_TASK=${LAST_COMPLETED_TASK@Q}
LAST_CHECKPOINT=${LAST_CHECKPOINT@Q}
LAST_LOG_DIR=${LAST_LOG_DIR@Q}
ACTIVE_TASK_ID=${ACTIVE_TASK_ID@Q}
ACTIVE_TASK_CHECKPOINT=${ACTIVE_TASK_CHECKPOINT@Q}
ACTIVE_TASK_LOG_DIR=${ACTIVE_TASK_LOG_DIR@Q}
ACTIVE_TASK_STEP=${ACTIVE_TASK_STEP@Q}
ACTIVE_TASK_MAX_EPOCHS=${ACTIVE_TASK_MAX_EPOCHS@Q}
RUN_NAMESPACE=${RUN_NAMESPACE@Q}
WRAPPER_WORKDIR=${WRAPPER_WORKDIR@Q}
PATHSEOF
}

append_summary() {
  local completed_task="$1"
  local runtime_seconds="$2"
  local next_action="$3"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$SUITE_NAME" \
    "$SEED" \
    "$completed_task" \
    "${SLURM_JOB_ID:-manual}" \
    "$runtime_seconds" \
    "$LAST_CHECKPOINT" \
    "$LAST_LOG_DIR" \
    "$next_action" >> "$SUMMARY_FILE"
}

append_history() {
  local status="$1"
  local stage="$2"
  local task_id="$3"
  local detail="$4"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$status" \
    "$stage" \
    "$task_id" \
    "${SLURM_JOB_ID:-manual}" \
    "$detail" >> "$HISTORY_FILE"
}

if [[ ! -f "$HISTORY_FILE" ]]; then
  printf 'timestamp_utc\tstatus\tstage\ttask_id\tjob_id\tdetail\n' > "$HISTORY_FILE"
fi

if [[ ! -f "$SUMMARY_FILE" ]]; then
  printf 'timestamp_utc\tsuite\tseed\tcompleted_task\tjob_id\truntime_seconds\tcheckpoint_path\tlog_dir\tnext_action\n' > "$SUMMARY_FILE"
fi

if [[ -f "$STATE_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$STATE_FILE"
else
  TASK_CURSOR=0
  RETRY_COUNT=0
  LAST_COMPLETED_TASK=""
  LAST_CHECKPOINT=""
  LAST_LOG_DIR=""
  LAST_TASK_RUNTIME_SECONDS=0
  ACTIVE_TASK_ID=""
  ACTIVE_TASK_CHECKPOINT=""
  ACTIVE_TASK_LOG_DIR=""
  ACTIVE_TASK_STEP=0
  ACTIVE_TASK_MAX_EPOCHS=0
  RUN_NAMESPACE="${RUN_NAMESPACE:-run_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-manual}}"
  WRAPPER_WORKDIR="${WRAPPER_WORKDIR:-$STATE_DIR/.wrapper_runs/$RUN_NAMESPACE}"
  mkdir -p "$WRAPPER_WORKDIR"
  save_state
  save_paths
fi

if [[ -z "${RUN_NAMESPACE:-}" ]]; then
  RUN_NAMESPACE="run_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-manual}"
fi
if [[ -z "${WRAPPER_WORKDIR:-}" ]]; then
  WRAPPER_WORKDIR="$STATE_DIR/.wrapper_runs/$RUN_NAMESPACE"
fi
ACTIVE_TASK_ID="${ACTIVE_TASK_ID:-}"
ACTIVE_TASK_CHECKPOINT="${ACTIVE_TASK_CHECKPOINT:-}"
ACTIVE_TASK_LOG_DIR="${ACTIVE_TASK_LOG_DIR:-}"
ACTIVE_TASK_STEP="${ACTIVE_TASK_STEP:-0}"
ACTIVE_TASK_MAX_EPOCHS="${ACTIVE_TASK_MAX_EPOCHS:-0}"
mkdir -p "$WRAPPER_WORKDIR"
ensure_wrapper_symlink() {
  local name="$1"
  local target="$2"
  local link_path="$WRAPPER_WORKDIR/$name"

  if [[ -L "$link_path" ]]; then
    local current_target
    current_target="$(readlink "$link_path" || true)"
    if [[ "$current_target" != "$target" ]]; then
      rm -f "$link_path"
    fi
  elif [[ -e "$link_path" ]]; then
    echo "ERROR: Wrapper workspace path already exists and is not a symlink: $link_path"
    exit 1
  fi

  if [[ ! -e "$link_path" ]]; then
    ln -s "$target" "$link_path"
  fi
}

for wrapper_item in examples rlinf LIBERO openvla-oft transformers-openvla-oft third_party model src ray_utils; do
  ensure_wrapper_symlink "$wrapper_item" "$REPO_ROOT/$wrapper_item"
done

parse_slurm_time_to_seconds() {
  local time_str="$1"
  local days=0
  local hours=0
  local minutes=0
  local seconds=0

  if [[ "$time_str" =~ ^([0-9]+)-([0-9]{1,2}):([0-9]{2}):([0-9]{2})$ ]]; then
    days="${BASH_REMATCH[1]}"
    hours="${BASH_REMATCH[2]}"
    minutes="${BASH_REMATCH[3]}"
    seconds="${BASH_REMATCH[4]}"
  elif [[ "$time_str" =~ ^([0-9]{1,2}):([0-9]{2}):([0-9]{2})$ ]]; then
    hours="${BASH_REMATCH[1]}"
    minutes="${BASH_REMATCH[2]}"
    seconds="${BASH_REMATCH[3]}"
  elif [[ "$time_str" =~ ^([0-9]{1,2}):([0-9]{2})$ ]]; then
    minutes="${BASH_REMATCH[1]}"
    seconds="${BASH_REMATCH[2]}"
  else
    echo "ERROR: Could not parse Slurm time string: $time_str" >&2
    return 1
  fi

  echo $((days * 86400 + hours * 3600 + minutes * 60 + seconds))
}

resolve_time_limit_seconds() {
  local time_str="${SLURM_TIMELIMIT:-}"
  if [[ -z "$time_str" || "$time_str" == "UNLIMITED" ]]; then
    time_str="$(awk -F= '/^#SBATCH --time=/{print $2; exit}' "$0")"
  fi
  parse_slurm_time_to_seconds "$time_str"
}

write_controller_configs() {
  local train_src="$REPO_ROOT/examples/embodiment/config/${CONFIG_NAME}.yaml"
  local eval_src="$REPO_ROOT/examples/embodiment/config/${BASE_EVAL_CONFIG_NAME}.yaml"
  local train_dst="$GENERATED_CONFIG_DIR/${TRAIN_CONFIG_BASENAME}.yaml"
  local eval_dst="$GENERATED_CONFIG_DIR/${EVAL_CONFIG_BASENAME}.yaml"

  python - "$train_src" "$eval_src" "$train_dst" "$eval_dst" \
    "$EFFECTIVE_TRAIN_EPOCHS" "$TRAIN_CHECKPOINT_STEP" "$TRAIN_ROLLOUT_EPOCH" \
    "$TRAIN_GROUP_SIZE" "$TRAIN_NUM_GROUP_ENVS" "$EVAL_NUM_ENVS" \
    "$EFFECTIVE_EVAL_ROLLOUT_EPOCH" "$ACTOR_MICRO_BATCH_SIZE" "$ACTOR_GLOBAL_BATCH_SIZE" \
    "$ACTOR_ENABLE_OFFLOAD" "$LOGGER_BACKENDS" \
    "$ACTOR_GPU_MAP" "$ROLLOUT_GPU_MAP" "$ENV_GPU_MAP" \
    "$RUNNER_RESUME_GLOBAL_STEP" "$RUNNER_STOP_UNIX_TIME" "$PARTIAL_RESUME_EXIT_CODE" \
    "$RUNNER_RESUME_SAVE_GRACE_SECONDS" "$ROLLING_PARTIAL_CHECKPOINT_ENABLED" \
    "$ROLLING_LATEST_CHECKPOINT_NAME" "$ROLLING_PREVIOUS_CHECKPOINT_NAME" \
    "$ROLLING_TMP_CHECKPOINT_NAME" "$RUNNER_RESUME_CHECKPOINT_PATH" <<'PY'
import sys
from pathlib import Path
import yaml

(train_src, eval_src, train_dst, eval_dst,
 train_epochs, train_ckpt_step, train_rollout_epoch,
 train_group_size, train_num_group_envs, eval_num_envs,
 eval_rollout_epoch, actor_micro_batch_size, actor_global_batch_size,
 actor_enable_offload, logger_backends,
 actor_gpu_map, rollout_gpu_map, env_gpu_map,
 runner_resume_global_step, runner_stop_unix_time, partial_resume_exit_code,
 runner_resume_save_grace_seconds, rolling_partial_checkpoint_enabled,
 rolling_latest_checkpoint_name, rolling_previous_checkpoint_name,
 rolling_tmp_checkpoint_name, runner_resume_checkpoint_path) = sys.argv[1:]

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)

def patch_common(cfg):
    cfg.setdefault('cluster', {})
    cfg['cluster'].setdefault('component_placement', {})
    cfg['cluster']['component_placement'] = {
        'actor': actor_gpu_map,
        'rollout': rollout_gpu_map,
        'env': env_gpu_map,
    }
    cfg.setdefault('env', {})
    cfg['env'].setdefault('eval', {})
    cfg['env']['eval']['num_envs'] = int(eval_num_envs)
    cfg['env']['eval']['eval_per_task'] = '${int_div:${env.eval.num_envs}, 10}'
    cfg.setdefault('algorithm', {})
    cfg['algorithm']['eval_rollout_epoch'] = int(eval_rollout_epoch)
    if logger_backends:
        cfg.setdefault('runner', {})
        cfg['runner'].setdefault('logger', {})
        cfg['runner']['logger']['logger_backends'] = [
            backend.strip() for backend in logger_backends.split(',') if backend.strip()
        ]
    return cfg

train_cfg = patch_common(load_yaml(train_src))
train_cfg.setdefault('runner', {})
train_cfg['runner']['max_epochs'] = int(train_epochs)
train_cfg['runner']['save_interval'] = int(train_ckpt_step)
train_cfg['runner']['resume_global_step'] = int(runner_resume_global_step)
train_cfg['runner']['stop_unix_time'] = int(runner_stop_unix_time)
train_cfg['runner']['partial_resume_exit_code'] = int(partial_resume_exit_code)
train_cfg['runner']['resume_save_grace_seconds'] = int(runner_resume_save_grace_seconds)
train_cfg['runner']['rolling_checkpoint'] = {
    'enabled': bool(int(rolling_partial_checkpoint_enabled)),
    'latest_name': rolling_latest_checkpoint_name,
    'previous_name': rolling_previous_checkpoint_name,
    'tmp_name': rolling_tmp_checkpoint_name,
    'keep_previous': True,
}
train_cfg['algorithm']['rollout_epoch'] = int(train_rollout_epoch)
train_cfg['algorithm']['group_size'] = int(train_group_size)
train_cfg['algorithm']['num_group_envs'] = int(train_num_group_envs)
train_cfg.setdefault('actor', {})
train_cfg['actor']['micro_batch_size'] = int(actor_micro_batch_size)
train_cfg['actor']['global_batch_size'] = int(actor_global_batch_size)
train_cfg['actor']['enable_offload'] = bool(int(actor_enable_offload))
if runner_resume_checkpoint_path:
    train_cfg['actor']['resume_checkpoint_path'] = runner_resume_checkpoint_path
    train_cfg.setdefault('actor', {}).setdefault('model', {})
    train_cfg['actor']['model']['lora_path'] = runner_resume_checkpoint_path
save_yaml(train_dst, train_cfg)

eval_cfg = patch_common(load_yaml(eval_src))
save_yaml(eval_dst, eval_cfg)
PY
}

build_task_log_dir() {
  local task_id="$1"
  local input_ckpt="$2"
  local log_dir

  if [[ -n "$input_ckpt" ]]; then
    if [[ "$input_ckpt" =~ task_([0-9]+) ]]; then
      local source_task="${BASH_REMATCH[1]}"
      if [[ "$input_ckpt" =~ global_step_([0-9]+) ]]; then
        local source_step="${BASH_REMATCH[1]}"
        log_dir="./logs/sequential/task_${task_id}_from_task_${source_task}_step_${source_step}_seed${SEED}${VARIANT_LOG_SUFFIX}"
      else
        echo "ERROR: Could not extract global_step from checkpoint path: $input_ckpt" >&2
        return 1
      fi
    else
      echo "ERROR: Could not extract task ID from checkpoint path: $input_ckpt" >&2
      return 1
    fi
  else
    log_dir="./logs/sequential/task_${task_id}_seed${SEED}${VARIANT_LOG_SUFFIX}"
  fi

  if [[ -n "$CONFIG_TAG" ]]; then
    log_dir="$(inject_config_tag_into_log_path "$log_dir" "$CONFIG_TAG")"
  fi

  printf '%s\n' "$REPO_ROOT/${log_dir#./}"
}

cleanup_partial_checkpoints() {
  local log_dir="$1"
  local checkpoint_root="$log_dir/checkpoints"

  rm -rf \
    "$checkpoint_root/$ROLLING_LATEST_CHECKPOINT_NAME" \
    "$checkpoint_root/$ROLLING_PREVIOUS_CHECKPOINT_NAME" \
    "$checkpoint_root/$ROLLING_TMP_CHECKPOINT_NAME"
}

load_partial_checkpoint_metadata() {
  local metadata_path="$1"

  python - "$metadata_path" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as f:
    data = json.load(f)

print(data["global_step"])
print(data["max_steps"])
print(data["actor_checkpoint_path"])
PY
}

ensure_wrapper_eval_checkpoint_alias() {
  local log_dir="$1"
  local actual_step="$2"
  local compat_step="$3"

  if [[ "$actual_step" == "$compat_step" ]]; then
    return 0
  fi

  local checkpoint_root="$log_dir/checkpoints"
  local compat_name="global_step_${compat_step}"
  local actual_name="global_step_${actual_step}"
  local compat_path="$checkpoint_root/$compat_name"

  mkdir -p "$checkpoint_root"

  if [[ -L "$compat_path" ]]; then
    local current_target
    current_target="$(readlink "$compat_path" || true)"
    if [[ "$current_target" != "$actual_name" && "$current_target" != "$checkpoint_root/$actual_name" ]]; then
      rm -f "$compat_path"
    fi
  elif [[ -e "$compat_path" ]]; then
    echo "ERROR: Existing checkpoint compatibility path is not a symlink: $compat_path"
    echo "       Refusing to let wrapper eval read a potentially stale checkpoint."
    return 1
  fi

  if [[ ! -e "$compat_path" ]]; then
    (
      cd "$checkpoint_root" || exit 1
      ln -s "$actual_name" "$compat_name"
    )
  fi
}

resubmit_self() {
  if [[ "$AUTO_RESUBMIT" != "1" ]]; then
    echo "AUTO_RESUBMIT=0; stopping after current stage."
    return 0
  fi

  local sbatch_output
  sbatch_output=$(sbatch --parsable \
    --export=ALL,REPO_ROOT="$REPO_ROOT",CONFIG_NAME="$CONFIG_NAME",SEED="$SEED",LIBERO_TYPE="$LIBERO_TYPE",LIBERO_SUFFIX="$LIBERO_SUFFIX",AUTO_RESUBMIT="$AUTO_RESUBMIT",MAX_RETRIES="$MAX_RETRIES",CONDA_ENV_NAME="$CONDA_ENV_NAME",CONDA_BASE="$CONDA_BASE",STATE_DIR="$STATE_DIR",CONTROLLER_STATE_FILE="$STATE_FILE",MANUAL_CHECKPOINT_PATH="$MANUAL_CHECKPOINT_PATH",ACTOR_GPU_MAP="$ACTOR_GPU_MAP",ROLLOUT_GPU_MAP="$ROLLOUT_GPU_MAP",ENV_GPU_MAP="$ENV_GPU_MAP",TRAIN_GROUP_SIZE="$TRAIN_GROUP_SIZE",TRAIN_NUM_GROUP_ENVS="$TRAIN_NUM_GROUP_ENVS",TRAIN_ROLLOUT_EPOCH="$TRAIN_ROLLOUT_EPOCH",ACTOR_MICRO_BATCH_SIZE="$ACTOR_MICRO_BATCH_SIZE",ACTOR_GLOBAL_BATCH_SIZE="$ACTOR_GLOBAL_BATCH_SIZE",ACTOR_ENABLE_OFFLOAD="$ACTOR_ENABLE_OFFLOAD",PRESERVE_EPISODE_BUDGET="$PRESERVE_EPISODE_BUDGET",EVAL_NUM_ENVS="$EVAL_NUM_ENVS",PRESERVE_EVAL_BUDGET="$PRESERVE_EVAL_BUDGET",RUN_NAMESPACE="$RUN_NAMESPACE",WRAPPER_WORKDIR="$WRAPPER_WORKDIR",TRUE_RESUME_ENABLED="$TRUE_RESUME_ENABLED",ROLLING_PARTIAL_CHECKPOINT_ENABLED="$ROLLING_PARTIAL_CHECKPOINT_ENABLED",PARTIAL_RESUME_EXIT_CODE="$PARTIAL_RESUME_EXIT_CODE",RUNNER_RESUME_SAVE_GRACE_SECONDS="$RUNNER_RESUME_SAVE_GRACE_SECONDS",ROLLING_LATEST_CHECKPOINT_NAME="$ROLLING_LATEST_CHECKPOINT_NAME",ROLLING_PREVIOUS_CHECKPOINT_NAME="$ROLLING_PREVIOUS_CHECKPOINT_NAME",ROLLING_TMP_CHECKPOINT_NAME="$ROLLING_TMP_CHECKPOINT_NAME" \
    "$0")
  local sbatch_status=$?
  if [[ $sbatch_status -ne 0 ]]; then
    echo "ERROR: sbatch resubmission failed"
    return $sbatch_status
  fi
  echo "Resubmitted controller as job: $sbatch_output"
}

retry_or_abort() {
  local stage="$1"
  local task_id="$2"
  local exit_code="$3"

  RETRY_COUNT=$((RETRY_COUNT + 1))
  save_state
  append_history "retry" "$stage" "$task_id" "exit_code=$exit_code retry_count=$RETRY_COUNT"

  if (( RETRY_COUNT <= MAX_RETRIES )) && [[ "$AUTO_RESUBMIT" == "1" ]]; then
    echo "Stage '$stage' failed for task '$task_id' with exit $exit_code. Retrying via resubmission ($RETRY_COUNT/$MAX_RETRIES)."
    resubmit_self || exit "$exit_code"
    exit 0
  fi

  echo "Stage '$stage' failed for task '$task_id' with exit $exit_code. Max retries exceeded."
  exit "$exit_code"
}

RUNNER_RESUME_GLOBAL_STEP=0
RUNNER_RESUME_CHECKPOINT_PATH=""
RUNNER_STOP_UNIX_TIME=0
write_controller_configs || exit 1

JOB_START_EPOCH="$(date +%s)"
JOB_TIME_LIMIT_SECONDS="$(resolve_time_limit_seconds)" || exit 1
RESUBMIT_BUFFER_SECONDS="${RESUBMIT_BUFFER_SECONDS:-1800}"
MIN_REMAINING_SECONDS_TO_CONTINUE="${MIN_REMAINING_SECONDS_TO_CONTINUE:-7200}"

TOTAL_TASKS="${#TASK_IDS[@]}"
CURRENT_TASK_ID=""
if (( TASK_CURSOR < TOTAL_TASKS )); then
  CURRENT_TASK_ID="${TASK_IDS[$TASK_CURSOR]}"
fi

printf '==== TEST_CONTROLLER ====\n'
printf 'Repo root:                 %s\n' "$REPO_ROOT"
printf 'Conda env:                 %s\n' "${CONDA_DEFAULT_ENV:-<unknown>}"
printf 'Base config:               %s\n' "$CONFIG_NAME"
printf 'Controller config:         %s\n' "$CONTROLLER_TRAIN_CONFIG_NAME"
printf 'Suite:                     %s\n' "$SUITE_NAME"
printf 'Task IDs:                  %s\n' "${TASK_IDS[*]}"
printf 'Task cursor:               %s/%s\n' "$TASK_CURSOR" "$TOTAL_TASKS"
printf 'Current task:              %s\n' "${CURRENT_TASK_ID:-<none>}"
printf 'Seed:                      %s\n' "$SEED"
printf 'LIBERO_TYPE:               %s\n' "$LIBERO_TYPE"
printf 'LIBERO_SUFFIX:             %s\n' "${LIBERO_SUFFIX:-<none>}"
printf 'Actor/Rollout/Env GPUs:    %s | %s | %s\n' "$ACTOR_GPU_MAP" "$ROLLOUT_GPU_MAP" "$ENV_GPU_MAP"
printf 'Base train budget:         epochs=%s rollout_epoch=%s group_size=%s num_group_envs=%s\n' "$BASE_BUDGET_EPOCHS" "$BASE_ROLLOUT_EPOCH" "$BASE_GROUP_SIZE" "$BASE_NUM_GROUP_ENVS"
printf 'Scaled train budget:       epochs=%s rollout_epoch=%s group_size=%s num_group_envs=%s\n' "$EFFECTIVE_TRAIN_EPOCHS" "$TRAIN_ROLLOUT_EPOCH" "$TRAIN_GROUP_SIZE" "$TRAIN_NUM_GROUP_ENVS"
printf 'Actor batches:             micro=%s global=%s\n' "$ACTOR_MICRO_BATCH_SIZE" "$ACTOR_GLOBAL_BATCH_SIZE"
printf 'Actor offload:             %s\n' "$ACTOR_ENABLE_OFFLOAD"
printf 'Eval budget:               base_envs=%s base_rollout_epoch=%s scaled_envs=%s scaled_rollout_epoch=%s\n' "$BASE_EVAL_NUM_ENVS" "$BASE_EVAL_ROLLOUT_EPOCH" "$EVAL_NUM_ENVS" "$EFFECTIVE_EVAL_ROLLOUT_EPOCH"
printf 'State file:                %s\n' "$STATE_FILE"
printf 'History file:              %s\n' "$HISTORY_FILE"
printf 'Paths file:                %s\n' "$PATHS_FILE"
printf 'Summary file:              %s\n' "$SUMMARY_FILE"
printf 'Last checkpoint:           %s\n' "${LAST_CHECKPOINT:-<none>}"
printf 'Active task resume:        %s step=%s checkpoint=%s\n' "${ACTIVE_TASK_ID:-<none>}" "${ACTIVE_TASK_STEP:-0}" "${ACTIVE_TASK_CHECKPOINT:-<none>}"
printf 'Generated config dir:      %s\n' "$GENERATED_CONFIG_DIR"
printf 'Run namespace:             %s\n' "$RUN_NAMESPACE"
printf 'Wrapper workdir:           %s\n' "$WRAPPER_WORKDIR"
printf 'RAY_TMPDIR:                %s\n' "$RAY_TMPDIR"
printf 'HF_HOME:                   %s\n' "$HF_HOME"
printf 'Job ID:                    %s\n' "${SLURM_JOB_ID:-manual}"
printf 'Job time limit (seconds):  %s\n' "$JOB_TIME_LIMIT_SECONDS"
printf 'Last task runtime (sec):   %s\n' "${LAST_TASK_RUNTIME_SECONDS:-0}"
printf '\n'

if (( TASK_CURSOR >= TOTAL_TASKS )); then
  echo "Controller already completed all tasks."
  exit 0
fi

while (( TASK_CURSOR < TOTAL_TASKS )); do
  CURRENT_TASK_ID="${TASK_IDS[$TASK_CURSOR]}"
  IS_ACTIVE_TASK_RESUME=0
  RUNNER_RESUME_GLOBAL_STEP=0
  RUNNER_RESUME_CHECKPOINT_PATH=""
  RUNNER_STOP_UNIX_TIME=0

  INPUT_CHECKPOINT=""
  if [[ "$TRUE_RESUME_ENABLED" == "1" ]] && [[ -n "${ACTIVE_TASK_ID:-}" ]]; then
    if [[ "$ACTIVE_TASK_ID" != "$CURRENT_TASK_ID" ]]; then
      echo "ERROR: Active task resume state does not match controller cursor."
      echo "       active_task_id=$ACTIVE_TASK_ID current_task_id=$CURRENT_TASK_ID task_cursor=$TASK_CURSOR"
      exit 1
    fi
    if (( ACTIVE_TASK_STEP <= 0 || ACTIVE_TASK_STEP >= EFFECTIVE_TRAIN_EPOCHS )); then
      echo "ERROR: Invalid active task resume step."
      echo "       active_task_step=$ACTIVE_TASK_STEP effective_train_epochs=$EFFECTIVE_TRAIN_EPOCHS"
      exit 1
    fi
    if [[ -z "${ACTIVE_TASK_LOG_DIR:-}" ]] || [[ ! -d "${ACTIVE_TASK_CHECKPOINT:-}" ]]; then
      echo "ERROR: Active task resume state is incomplete."
      echo "       active_task_log_dir=${ACTIVE_TASK_LOG_DIR:-<none>}"
      echo "       active_task_checkpoint=${ACTIVE_TASK_CHECKPOINT:-<none>}"
      exit 1
    fi
    IS_ACTIVE_TASK_RESUME=1
    EXPECTED_LOG_DIR="$ACTIVE_TASK_LOG_DIR"
    RUNNER_RESUME_GLOBAL_STEP="$ACTIVE_TASK_STEP"
    RUNNER_RESUME_CHECKPOINT_PATH="$ACTIVE_TASK_CHECKPOINT"
  else
    if (( TASK_CURSOR == 0 )) && [[ -n "$MANUAL_CHECKPOINT_PATH" ]]; then
      INPUT_CHECKPOINT="$MANUAL_CHECKPOINT_PATH"
    elif (( TASK_CURSOR > 0 )) && [[ -n "${LAST_CHECKPOINT:-}" ]]; then
      INPUT_CHECKPOINT="$LAST_CHECKPOINT"
    fi
    EXPECTED_LOG_DIR="$(build_task_log_dir "$CURRENT_TASK_ID" "$INPUT_CHECKPOINT")" || exit 1
  fi

  if [[ "$TRUE_RESUME_ENABLED" == "1" ]]; then
    RUNNER_STOP_UNIX_TIME=$((JOB_START_EPOCH + JOB_TIME_LIMIT_SECONDS))
  fi

  write_controller_configs || exit 1
  ensure_wrapper_eval_checkpoint_alias "$EXPECTED_LOG_DIR" "$TRAIN_CHECKPOINT_STEP" "$WRAPPER_EVAL_DEFAULT_STEP" || exit 1

  if (( IS_ACTIVE_TASK_RESUME == 1 )); then
    EXPERIMENT_NAME="$(basename "$EXPECTED_LOG_DIR")"
    if [[ -n "$CONFIG_TAG" ]]; then
      EXPERIMENT_NAME="${EXPERIMENT_NAME}_${CONFIG_TAG}"
    fi
    RUN_CMD=(
      bash "$REPO_ROOT/examples/embodiment/run_embodiment.sh"
      "$CONTROLLER_TRAIN_CONFIG_NAME"
      "env.fixed_task_ids=[${CURRENT_TASK_ID}]"
      "runner.logger.experiment_name=${EXPERIMENT_NAME}"
      "actor.seed=${SEED}"
    )
  else
    RUN_CMD=(bash "$REPO_ROOT/examples/crl_experiment/run_embodiment_sequential.sh" "$CURRENT_TASK_ID" "$INPUT_CHECKPOINT" "" "$CONTROLLER_TRAIN_CONFIG_NAME" "$SEED")
  fi

  if (( IS_ACTIVE_TASK_RESUME == 1 )); then
    echo "Running direct same-task resume for task $CURRENT_TASK_ID from outer epoch step $RUNNER_RESUME_GLOBAL_STEP"
  else
    echo "Running sequential wrapper for task $CURRENT_TASK_ID"
  fi
  printf '  %q' "${RUN_CMD[@]}"
  echo
  echo "  wrapper cwd: $WRAPPER_WORKDIR"
  echo

  TASK_STAGE_START_EPOCH="$(date +%s)"
  (
    cd "$WRAPPER_WORKDIR" || exit 1
    if (( IS_ACTIVE_TASK_RESUME == 1 )); then
      export LOG_DIR="$EXPECTED_LOG_DIR"
    fi
    "${RUN_CMD[@]}"
  )
  RUN_EXIT=$?
  if [[ $RUN_EXIT -eq $PARTIAL_RESUME_EXIT_CODE ]]; then
    TASK_STAGE_END_EPOCH="$(date +%s)"
    LAST_TASK_RUNTIME_SECONDS=$((TASK_STAGE_END_EPOCH - TASK_STAGE_START_EPOCH))

    PARTIAL_METADATA_PATH="${EXPECTED_LOG_DIR}/checkpoints/${ROLLING_LATEST_CHECKPOINT_NAME}/resume_state.json"
    if [[ ! -f "$PARTIAL_METADATA_PATH" ]]; then
      echo "ERROR: Partial resume exit code received but metadata is missing: $PARTIAL_METADATA_PATH"
      retry_or_abort "partial_resume_metadata" "$CURRENT_TASK_ID" 1
    fi

    mapfile -t PARTIAL_METADATA < <(load_partial_checkpoint_metadata "$PARTIAL_METADATA_PATH") || {
      echo "ERROR: Failed to parse partial resume metadata: $PARTIAL_METADATA_PATH"
      retry_or_abort "partial_resume_metadata_parse" "$CURRENT_TASK_ID" 1
    }

    ACTIVE_TASK_ID="$CURRENT_TASK_ID"
    ACTIVE_TASK_STEP="${PARTIAL_METADATA[0]}"
    ACTIVE_TASK_MAX_EPOCHS="${PARTIAL_METADATA[1]}"
    ACTIVE_TASK_CHECKPOINT="${PARTIAL_METADATA[2]}"
    ACTIVE_TASK_LOG_DIR="$EXPECTED_LOG_DIR"
    if (( ACTIVE_TASK_MAX_EPOCHS != EFFECTIVE_TRAIN_EPOCHS )); then
      echo "ERROR: Partial resume metadata does not match the current controller train budget."
      echo "       metadata_max_steps=$ACTIVE_TASK_MAX_EPOCHS effective_train_epochs=$EFFECTIVE_TRAIN_EPOCHS"
      retry_or_abort "partial_resume_metadata_budget" "$CURRENT_TASK_ID" 1
    fi
    RETRY_COUNT=0
    save_state
    save_paths
    append_history "partial_resume" "task_chain" "$CURRENT_TASK_ID" "step=$ACTIVE_TASK_STEP checkpoint=$ACTIVE_TASK_CHECKPOINT runtime_seconds=$LAST_TASK_RUNTIME_SECONDS"

    echo "Saved rolling partial checkpoint for task $CURRENT_TASK_ID at outer epoch step $ACTIVE_TASK_STEP/$ACTIVE_TASK_MAX_EPOCHS"
    if [[ "$AUTO_RESUBMIT" == "1" ]]; then
      append_summary "$CURRENT_TASK_ID" "$LAST_TASK_RUNTIME_SECONDS" "partial_resubmit"
      resubmit_self
      exit 0
    fi
    append_summary "$CURRENT_TASK_ID" "$LAST_TASK_RUNTIME_SECONDS" "partial_stop_no_resubmit"
    echo "AUTO_RESUBMIT=0; stopping after partial task checkpoint."
    exit 0
  elif [[ $RUN_EXIT -ne 0 ]]; then
    retry_or_abort "task_chain" "$CURRENT_TASK_ID" "$RUN_EXIT"
  fi

  if (( IS_ACTIVE_TASK_RESUME == 1 )); then
    CHECKPOINT_LOCATION="${EXPECTED_LOG_DIR#${REPO_ROOT}/}"
    EVAL_CMD=(bash "$REPO_ROOT/examples/crl_experiment/eval_embodiment.sh" "$CHECKPOINT_LOCATION" "" "$CONTROLLER_EVAL_CONFIG_NAME")

    echo "Running evaluation for resumed task $CURRENT_TASK_ID"
    printf '  %q' "${EVAL_CMD[@]}"
    echo

    (
      cd "$REPO_ROOT" || exit 1
      "${EVAL_CMD[@]}"
    )
    EVAL_EXIT=$?
    if [[ $EVAL_EXIT -ne 0 ]]; then
      retry_or_abort "task_eval" "$CURRENT_TASK_ID" "$EVAL_EXIT"
    fi
  fi

  EXPECTED_CHECKPOINT="${EXPECTED_LOG_DIR}/checkpoints/global_step_${TRAIN_CHECKPOINT_STEP}/actor"
  if [[ ! -d "$EXPECTED_CHECKPOINT" ]]; then
    echo "ERROR: Expected checkpoint missing after successful task run: $EXPECTED_CHECKPOINT"
    retry_or_abort "checkpoint_check" "$CURRENT_TASK_ID" 1
  fi

  TASK_STAGE_END_EPOCH="$(date +%s)"
  LAST_TASK_RUNTIME_SECONDS=$((TASK_STAGE_END_EPOCH - TASK_STAGE_START_EPOCH))
  LAST_COMPLETED_TASK="$CURRENT_TASK_ID"
  LAST_CHECKPOINT="$EXPECTED_CHECKPOINT"
  LAST_LOG_DIR="$EXPECTED_LOG_DIR"
  ACTIVE_TASK_ID=""
  ACTIVE_TASK_CHECKPOINT=""
  ACTIVE_TASK_LOG_DIR=""
  ACTIVE_TASK_STEP=0
  ACTIVE_TASK_MAX_EPOCHS=0
  TASK_CURSOR=$((TASK_CURSOR + 1))
  RETRY_COUNT=0
  cleanup_partial_checkpoints "$EXPECTED_LOG_DIR"
  save_state
  save_paths
  append_history "success" "task_chain" "$CURRENT_TASK_ID" "checkpoint=$EXPECTED_CHECKPOINT runtime_seconds=$LAST_TASK_RUNTIME_SECONDS config=$CONTROLLER_TRAIN_CONFIG_NAME"

  if (( TASK_CURSOR >= TOTAL_TASKS )); then
    append_summary "$CURRENT_TASK_ID" "$LAST_TASK_RUNTIME_SECONDS" "completed"
    echo "Controller completed successfully."
    exit 0
  fi

  ELAPSED_JOB_SECONDS=$((TASK_STAGE_END_EPOCH - JOB_START_EPOCH))
  REMAINING_JOB_SECONDS=$((JOB_TIME_LIMIT_SECONDS - ELAPSED_JOB_SECONDS))
  REQUIRED_SECONDS_FOR_NEXT_TASK=$((LAST_TASK_RUNTIME_SECONDS + RESUBMIT_BUFFER_SECONDS))
  if (( REQUIRED_SECONDS_FOR_NEXT_TASK < MIN_REMAINING_SECONDS_TO_CONTINUE )); then
    REQUIRED_SECONDS_FOR_NEXT_TASK=$MIN_REMAINING_SECONDS_TO_CONTINUE
  fi

  echo "Task $CURRENT_TASK_ID finished. Remaining job seconds: $REMAINING_JOB_SECONDS"
  echo "Required seconds to safely start next task: $REQUIRED_SECONDS_FOR_NEXT_TASK"

  if (( REMAINING_JOB_SECONDS >= REQUIRED_SECONDS_FOR_NEXT_TASK )); then
    append_summary "$CURRENT_TASK_ID" "$LAST_TASK_RUNTIME_SECONDS" "continue_same_job"
    echo "Continuing in the same Slurm allocation with next task."
    continue
  fi

  echo "Not enough walltime remains to safely start next task."
  if [[ "$AUTO_RESUBMIT" == "1" ]]; then
    append_summary "$CURRENT_TASK_ID" "$LAST_TASK_RUNTIME_SECONDS" "resubmit"
    resubmit_self
    exit 0
  fi
  append_summary "$CURRENT_TASK_ID" "$LAST_TASK_RUNTIME_SECONDS" "stop_no_resubmit"
  echo "AUTO_RESUBMIT=0; stopping after completed task boundary."
  exit 0
done

echo "Controller completed successfully."
exit 0
