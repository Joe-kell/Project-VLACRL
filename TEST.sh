#!/usr/bin/env bash
#SBATCH --job-name=vlacrl_object_h200
#SBATCH --output=/home/s2758621/Continual_VLA_RL/logs/test-smoke-%j.out
#SBATCH --error=/home/s2758621/Continual_VLA_RL/logs/test-smoke-%j.err
#SBATCH --partition=ICF-Free
#SBATCH --nodelist=herman
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=1400G
#SBATCH --gres=gpu:nvidia_h200:4


set -euo pipefail

# OpenVLA-OFT LIBERO-Object training run using existing repo scripts.
#
# Usage:
#   bash TEST.sh
#
# Optional env overrides:
#   CONFIG_NAME      (default: crl_experiment/libero_object_grpo_openvlaoft_object)
#   TASK_ID          (default: 5)
#   EPOCHS           (default: 1 when LIGHTWEIGHT_SMOKE=1, else 10)
#   SEED             (default: 1234)
#   EXPERIMENT_NAME  (default: rl_openvlaoft_object_h200)
#   MODEL_REPO       (default: Haozhan72/Openvla-oft-SFT-libero-object-traj1)
#   MODEL_DIR        (default: <repo>/model/Openvla-oft-SFT-libero-object-traj1)
#   CONDA_ENV_NAME   (default: vlacrl)
#   CONDA_BASE       (default: $HOME/miniconda3)
#   LIBERO_REPO_PATH (default: <repo>/LIBERO)
#   USE_SMOKE_SCALE  (default: 0; set 1 to run reduced-memory smoke scaling)
#   MATCH_CRL_SAMPLE_BUDGET (default: 1; when smoke scaling is on, auto-increase
#                            runner.max_epochs to match baseline CRL sample budget)
#   BASE_GROUP_SIZE        (default: 8; CRL baseline)
#   BASE_NUM_GROUP_ENVS    (default: 8; CRL baseline)
#   BASE_ROLLOUT_EPOCH     (default: 16; CRL baseline)
#   ACTOR_GPU_MAP    (default: 0 when LIGHTWEIGHT_SMOKE=1, else 0-1)
#   ROLLOUT_GPU_MAP  (default: 1 when LIGHTWEIGHT_SMOKE=1, else 2)
#   ENV_GPU_MAP      (default: 2 when LIGHTWEIGHT_SMOKE=1, else 3)
#   SMOKE_GROUP_SIZE (default: 2)
#   SMOKE_NUM_GROUP_ENVS      (default: 2)
#   SMOKE_MICRO_BATCH_SIZE    (default: 8)
#   SMOKE_GLOBAL_BATCH_SIZE   (default: 32)
#   SMOKE_ROLLOUT_EPOCH       (default: 2)
#   SMOKE_EVAL_ROLLOUT_EPOCH  (default: 2)
#   SMOKE_SAVE_INTERVAL       (default: 1)
#   LIGHTWEIGHT_SMOKE (default: 1; force a very light smoke run that collects
#                      minimal rollout data and executes at least one train step)
#   LIBERO_TYPE       (default: standard; standard|pro|plus)
#   VLA_TYPE          (default: auto-derived from CONFIG_NAME; e.g. openvlaoft)
#   RL_USED           (default: auto-derived from CONFIG_NAME; e.g. grpo)
#   REPRO_BASE_DIR    (default: /home/s2758621/Continual_VLA_RL/logs_libero_plus)

DEFAULT_REPO_ROOT="/home/s2758621/Continual_VLA_RL"

# Under sbatch, the script is executed from /var/spool/... so BASH_SOURCE is not reliable.
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

# Always activate conda env in batch jobs (don't rely on inherited shell state).
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
  echo "       Tried CONDA_BASE=$CONDA_BASE and system conda."
  exit 1
fi

# Pin Hugging Face caches to repo-local paths so batch jobs use expected cache.
export HF_HOME="${HF_HOME:-$REPO_ROOT/model}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-$HF_HUB_OFFLINE}"
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_DATASETS_CACHE"

CONFIG_NAME="${CONFIG_NAME:-crl_experiment/libero_object_grpo_openvlaoft_object}"
TASK_ID="${TASK_ID:-5}"
LIGHTWEIGHT_SMOKE="${LIGHTWEIGHT_SMOKE:-1}"
if [[ "$LIGHTWEIGHT_SMOKE" == "1" ]]; then
  EPOCHS="${EPOCHS:-1}"
else
  EPOCHS="${EPOCHS:-10}"
fi
SEED="${SEED:-1234}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rl_openvlaoft_object_h200}"
MODEL_REPO="${MODEL_REPO:-Haozhan72/Openvla-oft-SFT-libero-object-traj1}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/model/Openvla-oft-SFT-libero-object-traj1}"
export LIBERO_REPO_PATH="${LIBERO_REPO_PATH:-$REPO_ROOT/LIBERO}"
export LIBERO_TYPE="${LIBERO_TYPE:-standard}"
if [[ -z "${USE_SMOKE_SCALE:-}" ]]; then
  if [[ "$LIGHTWEIGHT_SMOKE" == "1" ]]; then
    USE_SMOKE_SCALE="1"
  else
    USE_SMOKE_SCALE="0"
  fi
fi
if [[ "$LIGHTWEIGHT_SMOKE" == "1" ]]; then
  MATCH_CRL_SAMPLE_BUDGET="${MATCH_CRL_SAMPLE_BUDGET:-0}"
else
  MATCH_CRL_SAMPLE_BUDGET="${MATCH_CRL_SAMPLE_BUDGET:-1}"
fi
BASE_GROUP_SIZE="${BASE_GROUP_SIZE:-8}"
BASE_NUM_GROUP_ENVS="${BASE_NUM_GROUP_ENVS:-8}"
BASE_ROLLOUT_EPOCH="${BASE_ROLLOUT_EPOCH:-16}"
if [[ "$LIGHTWEIGHT_SMOKE" == "1" ]]; then
  ACTOR_GPU_MAP="${ACTOR_GPU_MAP:-0}"
  ROLLOUT_GPU_MAP="${ROLLOUT_GPU_MAP:-1}"
  ENV_GPU_MAP="${ENV_GPU_MAP:-2}"
  SMOKE_GROUP_SIZE="${SMOKE_GROUP_SIZE:-1}"
  SMOKE_NUM_GROUP_ENVS="${SMOKE_NUM_GROUP_ENVS:-1}"
  SMOKE_MICRO_BATCH_SIZE="${SMOKE_MICRO_BATCH_SIZE:-1}"
  SMOKE_GLOBAL_BATCH_SIZE="${SMOKE_GLOBAL_BATCH_SIZE:-1}"
  SMOKE_ROLLOUT_EPOCH="${SMOKE_ROLLOUT_EPOCH:-1}"
  SMOKE_EVAL_ROLLOUT_EPOCH="${SMOKE_EVAL_ROLLOUT_EPOCH:-1}"
  SMOKE_SAVE_INTERVAL="${SMOKE_SAVE_INTERVAL:-1}"
else
  ACTOR_GPU_MAP="${ACTOR_GPU_MAP:-0-1}"
  ROLLOUT_GPU_MAP="${ROLLOUT_GPU_MAP:-2}"
  ENV_GPU_MAP="${ENV_GPU_MAP:-3}"
  SMOKE_GROUP_SIZE="${SMOKE_GROUP_SIZE:-2}"
  SMOKE_NUM_GROUP_ENVS="${SMOKE_NUM_GROUP_ENVS:-2}"
  SMOKE_MICRO_BATCH_SIZE="${SMOKE_MICRO_BATCH_SIZE:-8}"
  SMOKE_GLOBAL_BATCH_SIZE="${SMOKE_GLOBAL_BATCH_SIZE:-32}"
  SMOKE_ROLLOUT_EPOCH="${SMOKE_ROLLOUT_EPOCH:-2}"
  SMOKE_EVAL_ROLLOUT_EPOCH="${SMOKE_EVAL_ROLLOUT_EPOCH:-2}"
  SMOKE_SAVE_INTERVAL="${SMOKE_SAVE_INTERVAL:-1}"
fi

if [[ "$CONFIG_NAME" == *openvlaoft* ]]; then
  VLA_TYPE_DEFAULT="openvlaoft"
elif [[ "$CONFIG_NAME" == *openvla* ]]; then
  VLA_TYPE_DEFAULT="openvla"
elif [[ "$CONFIG_NAME" == *simple_cnn* ]]; then
  VLA_TYPE_DEFAULT="simple_cnn"
elif [[ "$CONFIG_NAME" == *pi0* ]]; then
  VLA_TYPE_DEFAULT="pi0"
else
  VLA_TYPE_DEFAULT="unknown_vla"
fi
VLA_TYPE="${VLA_TYPE:-$VLA_TYPE_DEFAULT}"

if [[ "$CONFIG_NAME" == *grpo* ]]; then
  RL_USED_DEFAULT="grpo"
elif [[ "$CONFIG_NAME" == *ppo* ]]; then
  RL_USED_DEFAULT="ppo"
elif [[ "$CONFIG_NAME" == *nft* ]]; then
  RL_USED_DEFAULT="nft"
else
  RL_USED_DEFAULT="unknown_rl"
fi
RL_USED="${RL_USED:-$RL_USED_DEFAULT}"
RUN_NAME="${VLA_TYPE}_${LIBERO_TYPE}_${RL_USED}"
REPRO_BASE_DIR="${REPRO_BASE_DIR:-/home/s2758621/Continual_VLA_RL/logs_libero_plus}"
mkdir -p "$REPRO_BASE_DIR"

# Preserve approximate CRL sample budget when using reduced smoke scaling.
EFFECTIVE_EPOCHS="$EPOCHS"
if [[ "$USE_SMOKE_SCALE" == "1" && "$MATCH_CRL_SAMPLE_BUDGET" == "1" ]]; then
  baseline_samples_per_epoch=$((BASE_GROUP_SIZE * BASE_NUM_GROUP_ENVS * BASE_ROLLOUT_EPOCH))
  smoke_samples_per_epoch=$((SMOKE_GROUP_SIZE * SMOKE_NUM_GROUP_ENVS * SMOKE_ROLLOUT_EPOCH))
  if (( smoke_samples_per_epoch <= 0 )); then
    echo "ERROR: smoke_samples_per_epoch must be > 0, got $smoke_samples_per_epoch"
    exit 1
  fi
  EFFECTIVE_EPOCHS=$(( (EPOCHS * baseline_samples_per_epoch + smoke_samples_per_epoch - 1) / smoke_samples_per_epoch ))
fi

# Force a short Ray temp root to avoid AF_UNIX socket path length failures.
RAY_TMP_BASE_DEFAULT="/tmp/ray_${USER:-unknown}"
export RAY_TMPDIR="${RAY_TMPDIR:-$RAY_TMP_BASE_DEFAULT}"
export TMPDIR="${TMPDIR:-$RAY_TMPDIR}"
export TMP="${TMP:-$RAY_TMPDIR}"
export TEMP="${TEMP:-$RAY_TMPDIR}"
mkdir -p "$RAY_TMPDIR"
chmod 700 "$RAY_TMPDIR" 2>/dev/null || true

 # NCCL / torch distributed debug logs for root-cause capture.
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,ENV}"

# Force training/eval artifact logs into repo logs directory as well.
JOB_TAG="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
export LOG_DIR="${LOG_DIR:-$REPRO_BASE_DIR/${RUN_NAME}/${JOB_TAG}}"
mkdir -p "$LOG_DIR"

report_failure() {
  local exit_code=$?
  echo
  echo "Smoke test failed (exit $exit_code). Collecting debug hints..."
  if [[ -d "$RAY_TMPDIR/session_latest/logs" ]]; then
    mkdir -p "$LOG_DIR/ray"
    cp -a "$RAY_TMPDIR/session_latest/logs" "$LOG_DIR/ray/" 2>/dev/null || true
  fi
  if command -v rg >/dev/null 2>&1; then
    rg -n "Traceback|CUDA|OOM|NCCL|ncclUniqueId|SIGKILL|segfault|fatal|DistBackendError" \
      "$RAY_TMPDIR"/session_latest/logs "$LOG_DIR" 2>/dev/null || true
  else
    grep -R -n -E "Traceback|CUDA|OOM|NCCL|ncclUniqueId|SIGKILL|segfault|fatal|DistBackendError" \
      "$RAY_TMPDIR"/session_latest/logs "$LOG_DIR" 2>/dev/null || true
  fi
  exit "$exit_code"
}
trap report_failure ERR

echo "==== OpenVLA-OFT Smoke Test ===="
echo "Repo root:        $REPO_ROOT"
echo "Config:           $CONFIG_NAME"
echo "LIGHTWEIGHT_SMOKE: $LIGHTWEIGHT_SMOKE"
echo "Task ID:          $TASK_ID"
echo "Epochs (input):   $EPOCHS"
echo "Epochs (effective): $EFFECTIVE_EPOCHS"
echo "Seed:             $SEED"
echo "Experiment name:  $EXPERIMENT_NAME"
echo "Model dir:        $MODEL_DIR"
echo "LIBERO path:      $LIBERO_REPO_PATH"
echo "LIBERO_TYPE:      $LIBERO_TYPE"
echo "VLA_TYPE:         $VLA_TYPE"
echo "RL_USED:          $RL_USED"
echo "RUN_NAME:         $RUN_NAME"
echo "REPRO_BASE_DIR:   $REPRO_BASE_DIR"
echo "Conda env:        ${CONDA_DEFAULT_ENV:-<unknown>}"
echo "HF_HOME:          $HF_HOME"
echo "HF_HUB_CACHE:     $HF_HUB_CACHE"
echo "HF_XET_CACHE:     $HF_XET_CACHE"
echo "RAY_TMPDIR:       $RAY_TMPDIR"
echo "TORCH_DISTRIBUTED_DEBUG: $TORCH_DISTRIBUTED_DEBUG"
echo "NCCL_DEBUG:       $NCCL_DEBUG"
echo "NCCL_ASYNC_ERROR_HANDLING: $NCCL_ASYNC_ERROR_HANDLING"
echo "NCCL_DEBUG_SUBSYS: $NCCL_DEBUG_SUBSYS"
echo "TRANSFORMERS_OFFLINE: $TRANSFORMERS_OFFLINE"
echo "USE_SMOKE_SCALE:  $USE_SMOKE_SCALE"
echo "MATCH_CRL_SAMPLE_BUDGET: $MATCH_CRL_SAMPLE_BUDGET"
if [[ "$USE_SMOKE_SCALE" == "1" ]]; then
  echo "Scaled placement: actor=$ACTOR_GPU_MAP rollout=$ROLLOUT_GPU_MAP env=$ENV_GPU_MAP"
  echo "Scaled rollout:   group_size=$SMOKE_GROUP_SIZE num_group_envs=$SMOKE_NUM_GROUP_ENVS"
  echo "Scaled batches:   micro=$SMOKE_MICRO_BATCH_SIZE global=$SMOKE_GLOBAL_BATCH_SIZE"
  echo "Scaled epochs:    rollout=$SMOKE_ROLLOUT_EPOCH eval_rollout=$SMOKE_EVAL_ROLLOUT_EPOCH save_interval=$SMOKE_SAVE_INTERVAL"
  if [[ "$MATCH_CRL_SAMPLE_BUDGET" == "1" ]]; then
    echo "Baseline sample budget: group_size=$BASE_GROUP_SIZE num_group_envs=$BASE_NUM_GROUP_ENVS rollout_epoch=$BASE_ROLLOUT_EPOCH"
  fi
fi
echo "Run LOG_DIR:      $LOG_DIR"
echo

cp "$0" "$LOG_DIR/TEST.sh.snapshot" 2>/dev/null || true

python - <<'PY'
import hydra, torch
print("Python preflight OK")
print("hydra:", getattr(hydra, "__version__", "unknown"))
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
PY

if [[ ! -d "$LIBERO_REPO_PATH" ]]; then
  echo "ERROR: LIBERO path not found: $LIBERO_REPO_PATH"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model checkpoint not found at $MODEL_DIR"
  echo "Attempting download from Hugging Face: $MODEL_REPO"

  if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: 'hf' CLI not found. Install it, or pre-download model to:"
    echo "       $MODEL_DIR"
    exit 1
  fi

  hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "ERROR: Model download/check failed. Expected file missing:"
  echo "       $MODEL_DIR/config.json"
  exit 1
fi

CMD=(
  bash examples/embodiment/run_embodiment.sh "$CONFIG_NAME"
  "env.fixed_task_ids=[$TASK_ID]"
  "runner.max_epochs=$EFFECTIVE_EPOCHS"
  "runner.logger.logger_backends=[tensorboard]"
  "runner.logger.experiment_name=$EXPERIMENT_NAME"
  "actor.seed=$SEED"
)

if [[ "$USE_SMOKE_SCALE" == "1" ]]; then
  CMD+=(
    # Keep CRL loss/objective logic from config, but scale placement + load for A40 memory.
    "runner.save_interval=$SMOKE_SAVE_INTERVAL"
    "algorithm.rollout_epoch=$SMOKE_ROLLOUT_EPOCH"
    "algorithm.eval_rollout_epoch=$SMOKE_EVAL_ROLLOUT_EPOCH"
    "algorithm.group_size=$SMOKE_GROUP_SIZE"
    "algorithm.num_group_envs=$SMOKE_NUM_GROUP_ENVS"
    "actor.global_batch_size=$SMOKE_GLOBAL_BATCH_SIZE"
    "actor.micro_batch_size=$SMOKE_MICRO_BATCH_SIZE"
    "+cluster.component_placement.actor=$ACTOR_GPU_MAP"
    "+cluster.component_placement.rollout=$ROLLOUT_GPU_MAP"
    "+cluster.component_placement.env=$ENV_GPU_MAP"
  )
fi

if [[ "$LIGHTWEIGHT_SMOKE" == "1" ]]; then
  CMD+=(
    # Force minimal data collection and a minimal train pass.
    "runner.max_epochs=1"
    "runner.save_interval=1"
    "algorithm.rollout_epoch=1"
    "algorithm.eval_rollout_epoch=1"
    "algorithm.group_size=1"
    "algorithm.num_group_envs=1"
    "actor.global_batch_size=1"
    "actor.micro_batch_size=1"
    "env.train.max_episode_steps=8"
    "env.eval.max_episode_steps=8"
    "env.eval.num_envs=1"
    "env.train.video_cfg.save_video=False"
    "env.eval.video_cfg.save_video=False"
  )
fi

echo "Running smoke test command:"
printf '  %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_STATUS="$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null || true)"
REPRO_FILE_BASE="$REPRO_BASE_DIR/$RUN_NAME"
{
  echo "run_name=$RUN_NAME"
  echo "vla_type=$VLA_TYPE"
  echo "libero_type=$LIBERO_TYPE"
  echo "rl_used=$RL_USED"
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_root=$REPO_ROOT"
  echo "git_commit=$GIT_COMMIT"
  echo "config_name=$CONFIG_NAME"
  echo "task_id=$TASK_ID"
  echo "seed=$SEED"
  echo "lightweight_smoke=$LIGHTWEIGHT_SMOKE"
  echo "use_smoke_scale=$USE_SMOKE_SCALE"
  echo "effective_epochs=$EFFECTIVE_EPOCHS"
  echo "log_dir=$LOG_DIR"
  echo "libero_repo_path=$LIBERO_REPO_PATH"
  echo "model_dir=$MODEL_DIR"
  echo "command="
  printf '%q ' "${CMD[@]}"
  echo
  echo "git_status_porcelain_start"
  if [[ -n "$GIT_STATUS" ]]; then
    printf '%s\n' "$GIT_STATUS"
  fi
  echo "git_status_porcelain_end"
} > "$REPRO_FILE_BASE"

echo
echo "Smoke test completed successfully."
echo "Repro summary file: $REPRO_FILE_BASE"
