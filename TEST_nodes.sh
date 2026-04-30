#!/usr/bin/env bash
#SBATCH --job-name=vlacrl_crannog2_object_nodes
#SBATCH --output=/home/s2758621/Continual_VLA_RL/logs/test-nodes-%j.out
#SBATCH --error=/home/s2758621/Continual_VLA_RL/logs/test-nodes-%j.err
#SBATCH --partition=ICF-Free
#SBATCH --time=08:00:00
#SBATCH --nodes=2
#SBATCH --nodelist=crannog[03-04]
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=450G
#SBATCH --gres=gpu:a40:4

set -euo pipefail

# OpenVLA-OFT LIBERO-Object training run on two A40 nodes:
#   - crannog01 + crannog02 (4x A40 each, total 8 GPUs)
#   - default split: actor on one node (4 GPUs), rollout+env on the other node (2+2 GPUs)
#
# Usage:
#   sbatch TEST_nodes.sh
#
# Optional env overrides:
#   CONFIG_NAME      (default: crl_experiment/libero_object_grpo_openvlaoft_object)
#   TASK_ID          (default: 5)
#   EPOCHS           (default: 10)
#   SEED             (default: 1234)
#   EXPERIMENT_NAME  (default: rl_openvlaoft_object_crannog_nodes)
#   MODEL_REPO       (default: Haozhan72/Openvla-oft-SFT-libero-object-traj1)
#   MODEL_DIR        (default: <repo>/model/Openvla-oft-SFT-libero-object-traj1)
#   CONDA_ENV_NAME   (default: vlacrl)
#   CONDA_BASE       (default: $HOME/miniconda3)
#   LIBERO_REPO_PATH (default: <repo>/LIBERO)
#   USE_SMOKE_SCALE  (default: 0; set 1 to run reduced smoke scaling)
#   MATCH_CRL_SAMPLE_BUDGET (default: 1; when smoke scaling is on, auto-increase
#                            runner.max_epochs to match baseline CRL sample budget)
#   BASE_GROUP_SIZE        (default: 8; CRL baseline)
#   BASE_NUM_GROUP_ENVS    (default: 8; CRL baseline)
#   BASE_ROLLOUT_EPOCH     (default: 16; CRL baseline)
#   CLUSTER_NUM_NODES (default: $SLURM_NNODES or 1)
#   MASTER_PORT       (default: derived from job id)
#   RAY_GCS_PORT      (default: MASTER_PORT)
#   RAY_HEAD_NODE_MANAGER_PORT (default: MASTER_PORT+1)
#   RAY_HEAD_OBJECT_MANAGER_PORT (default: MASTER_PORT+2)
#   RAY_HEAD_CLIENT_SERVER_PORT  (default: MASTER_PORT+3)
#   RAY_WORKER_NODE_MANAGER_PORT (default: MASTER_PORT+201)
#   RAY_WORKER_OBJECT_MANAGER_PORT (default: MASTER_PORT+202)
#   RAY_SOCKET_PRECHECK_RETRIES (default: 30)
#   RAY_SOCKET_PRECHECK_SLEEP_SEC (default: 2)
#   RAY_OBJECT_STORE_MEMORY (default: 137438953472 = 128GiB)
#   DISABLE_VIDEO_LOGGING (default: 1; disable train/eval video writing)
#   TRAIN_CAMERA_RES  (default: 256; set 128 for extra render stability)
#   EVAL_CAMERA_RES   (default: 256; set 128 for extra render stability)
#   AUTO_COMPONENT_PLACEMENT (default: 1; auto-map global GPU IDs by Ray node order)
#   ACTOR_GPU_COUNT   (default: 4; GPUs on head node for actor)
#   WORKER_ROLLOUT_GPU_COUNT (default: 3; GPUs on worker node for rollout)
#   WORKER_ENV_GPU_COUNT     (default: 1; GPUs on worker node for env)
#   ACTOR_GPU_MAP    (default: 0-3)
#   ROLLOUT_GPU_MAP  (default: 4-6)
#   ENV_GPU_MAP      (default: 7)
#   SMOKE_GROUP_SIZE (default: 8)
#   SMOKE_NUM_GROUP_ENVS      (default: 12)
#   SMOKE_MICRO_BATCH_SIZE    (default: 48)
#   SMOKE_GLOBAL_BATCH_SIZE   (default: 8448)
#   SMOKE_ROLLOUT_EPOCH       (default: 20)
#   SMOKE_EVAL_ROLLOUT_EPOCH  (default: 2)
#   SMOKE_SAVE_INTERVAL       (default: 25)
#   LORA_PATH        (optional: load LoRA adapter checkpoint dir)
#   RAY_ONLY         (default: 0; set 1 to validate Ray multi-node + placement only)

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
export REPO_ROOT

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
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-1234}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rl_openvlaoft_object_crannog_nodes}"
MODEL_REPO="${MODEL_REPO:-Haozhan72/Openvla-oft-SFT-libero-object-traj1}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/model/Openvla-oft-SFT-libero-object-traj1}"
export LIBERO_REPO_PATH="${LIBERO_REPO_PATH:-$REPO_ROOT/LIBERO}"
USE_SMOKE_SCALE="${USE_SMOKE_SCALE:-0}"
MATCH_CRL_SAMPLE_BUDGET="${MATCH_CRL_SAMPLE_BUDGET:-1}"
BASE_GROUP_SIZE="${BASE_GROUP_SIZE:-8}"
BASE_NUM_GROUP_ENVS="${BASE_NUM_GROUP_ENVS:-8}"
BASE_ROLLOUT_EPOCH="${BASE_ROLLOUT_EPOCH:-16}"
CLUSTER_NUM_NODES="${CLUSTER_NUM_NODES:-${SLURM_NNODES:-2}}"
if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
  DEFAULT_MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))
else
  DEFAULT_MASTER_PORT=29500
fi
MASTER_PORT="${MASTER_PORT:-$DEFAULT_MASTER_PORT}"
JOB_TAG="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-137438953472}"
RAY_GCS_PORT="${RAY_GCS_PORT:-$MASTER_PORT}"
RAY_HEAD_NODE_MANAGER_PORT="${RAY_HEAD_NODE_MANAGER_PORT:-$((MASTER_PORT + 1))}"
RAY_HEAD_OBJECT_MANAGER_PORT="${RAY_HEAD_OBJECT_MANAGER_PORT:-$((MASTER_PORT + 2))}"
RAY_HEAD_CLIENT_SERVER_PORT="${RAY_HEAD_CLIENT_SERVER_PORT:-$((MASTER_PORT + 3))}"
RAY_HEAD_MIN_WORKER_PORT="${RAY_HEAD_MIN_WORKER_PORT:-$((MASTER_PORT + 100))}"
RAY_HEAD_MAX_WORKER_PORT="${RAY_HEAD_MAX_WORKER_PORT:-$((MASTER_PORT + 199))}"
RAY_WORKER_NODE_MANAGER_PORT="${RAY_WORKER_NODE_MANAGER_PORT:-$((MASTER_PORT + 201))}"
RAY_WORKER_OBJECT_MANAGER_PORT="${RAY_WORKER_OBJECT_MANAGER_PORT:-$((MASTER_PORT + 202))}"
RAY_WORKER_MIN_WORKER_PORT="${RAY_WORKER_MIN_WORKER_PORT:-$((MASTER_PORT + 300))}"
RAY_WORKER_MAX_WORKER_PORT="${RAY_WORKER_MAX_WORKER_PORT:-$((MASTER_PORT + 499))}"
RAY_SOCKET_PRECHECK_RETRIES="${RAY_SOCKET_PRECHECK_RETRIES:-30}"
RAY_SOCKET_PRECHECK_SLEEP_SEC="${RAY_SOCKET_PRECHECK_SLEEP_SEC:-2}"
DISABLE_VIDEO_LOGGING="${DISABLE_VIDEO_LOGGING:-1}"
TRAIN_CAMERA_RES="${TRAIN_CAMERA_RES:-256}"
EVAL_CAMERA_RES="${EVAL_CAMERA_RES:-256}"
AUTO_COMPONENT_PLACEMENT="${AUTO_COMPONENT_PLACEMENT:-1}"
ACTOR_GPU_COUNT="${ACTOR_GPU_COUNT:-4}"
WORKER_ROLLOUT_GPU_COUNT="${WORKER_ROLLOUT_GPU_COUNT:-3}"
WORKER_ENV_GPU_COUNT="${WORKER_ENV_GPU_COUNT:-1}"
ACTOR_GPU_MAP="${ACTOR_GPU_MAP:-0-3}"
ROLLOUT_GPU_MAP="${ROLLOUT_GPU_MAP:-4-6}"
ENV_GPU_MAP="${ENV_GPU_MAP:-7}"
SMOKE_GROUP_SIZE="${SMOKE_GROUP_SIZE:-8}"
SMOKE_NUM_GROUP_ENVS="${SMOKE_NUM_GROUP_ENVS:-12}"
SMOKE_MICRO_BATCH_SIZE="${SMOKE_MICRO_BATCH_SIZE:-48}"
SMOKE_GLOBAL_BATCH_SIZE="${SMOKE_GLOBAL_BATCH_SIZE:-8448}"
SMOKE_ROLLOUT_EPOCH="${SMOKE_ROLLOUT_EPOCH:-20}"
SMOKE_EVAL_ROLLOUT_EPOCH="${SMOKE_EVAL_ROLLOUT_EPOCH:-2}"
SMOKE_SAVE_INTERVAL="${SMOKE_SAVE_INTERVAL:-25}"
LORA_PATH="${LORA_PATH:-}"
RAY_ONLY="${RAY_ONLY:-0}"
RAY_COORD_DIR="${RAY_COORD_DIR:-$REPO_ROOT/logs/ray_coord}"
mkdir -p "$RAY_COORD_DIR"
RAY_HEAD_IP_FILE="${RAY_HEAD_IP_FILE:-$RAY_COORD_DIR/ray_head_ip_${JOB_TAG}.txt}"
RAY_HEAD_ADDR_FILE="${RAY_HEAD_ADDR_FILE:-$RAY_COORD_DIR/ray_head_addr_${JOB_TAG}.txt}"
export RAY_HEAD_IP_FILE
export RAY_HEAD_ADDR_FILE

# Python helper blocks read these from environment.
export CONFIG_NAME
export ACTOR_GPU_COUNT
export WORKER_ROLLOUT_GPU_COUNT
export WORKER_ENV_GPU_COUNT

if ! [[ "$CLUSTER_NUM_NODES" =~ ^[0-9]+$ ]] || (( CLUSTER_NUM_NODES < 1 )); then
  echo "ERROR: CLUSTER_NUM_NODES must be a positive integer, got: $CLUSTER_NUM_NODES"
  exit 1
fi

if ! [[ "$MASTER_PORT" =~ ^[0-9]+$ ]]; then
  echo "ERROR: MASTER_PORT must be numeric, got: $MASTER_PORT"
  exit 1
fi

for ray_port in \
  "$RAY_GCS_PORT" \
  "$RAY_HEAD_NODE_MANAGER_PORT" \
  "$RAY_HEAD_OBJECT_MANAGER_PORT" \
  "$RAY_HEAD_CLIENT_SERVER_PORT" \
  "$RAY_HEAD_MIN_WORKER_PORT" \
  "$RAY_HEAD_MAX_WORKER_PORT" \
  "$RAY_WORKER_NODE_MANAGER_PORT" \
  "$RAY_WORKER_OBJECT_MANAGER_PORT" \
  "$RAY_WORKER_MIN_WORKER_PORT" \
  "$RAY_WORKER_MAX_WORKER_PORT"; do
  if ! [[ "$ray_port" =~ ^[0-9]+$ ]] || (( ray_port < 1024 || ray_port > 65535 )); then
    echo "ERROR: invalid Ray port value: $ray_port"
    exit 1
  fi
done

if (( RAY_HEAD_MIN_WORKER_PORT > RAY_HEAD_MAX_WORKER_PORT )); then
  echo "ERROR: RAY_HEAD_MIN_WORKER_PORT > RAY_HEAD_MAX_WORKER_PORT"
  exit 1
fi

if (( RAY_WORKER_MIN_WORKER_PORT > RAY_WORKER_MAX_WORKER_PORT )); then
  echo "ERROR: RAY_WORKER_MIN_WORKER_PORT > RAY_WORKER_MAX_WORKER_PORT"
  exit 1
fi

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

export MASTER_PORT
export CLUSTER_NUM_NODES
export RAY_OBJECT_STORE_MEMORY
export RAY_GCS_PORT
export RAY_HEAD_NODE_MANAGER_PORT
export RAY_HEAD_OBJECT_MANAGER_PORT
export RAY_HEAD_CLIENT_SERVER_PORT
export RAY_HEAD_MIN_WORKER_PORT
export RAY_HEAD_MAX_WORKER_PORT
export RAY_WORKER_NODE_MANAGER_PORT
export RAY_WORKER_OBJECT_MANAGER_PORT
export RAY_WORKER_MIN_WORKER_PORT
export RAY_WORKER_MAX_WORKER_PORT
export RAY_SOCKET_PRECHECK_RETRIES
export RAY_SOCKET_PRECHECK_SLEEP_SEC

# NCCL / torch distributed debug logs for root-cause capture.
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,ENV}"

# Force training/eval artifact logs into repo logs directory as well.
export LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/${EXPERIMENT_NAME}_${JOB_TAG}}"
mkdir -p "$LOG_DIR"

RAY_CLUSTER_STARTED=0
RAY_HEAD_STEP_PID=""
RAY_WORKER_STEP_PID=""
RAY_HEAD_STEP_LOG="${LOG_DIR}/ray_head_step.log"
RAY_WORKER_STEP_LOG="${LOG_DIR}/ray_worker_step.log"
SLURM_HET_SIZE_NUM=0
if [[ "${SLURM_HET_SIZE:-}" =~ ^[0-9]+$ ]]; then
  SLURM_HET_SIZE_NUM="${SLURM_HET_SIZE}"
fi

stop_srun_step() {
  local pid="${1:-}"
  local label="${2:-step}"
  if [[ -z "$pid" ]]; then
    return 0
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping Ray ${label} step (pid=${pid})..."
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

cleanup_ray_cluster() {
  local exit_code=$?
  trap - EXIT

  if [[ -n "$RAY_WORKER_STEP_PID" || -n "$RAY_HEAD_STEP_PID" ]]; then
    echo
    echo "Stopping Ray cluster..."
    stop_srun_step "$RAY_WORKER_STEP_PID" "worker"
    stop_srun_step "$RAY_HEAD_STEP_PID" "head"
  fi

  rm -f "$RAY_HEAD_IP_FILE" "$RAY_HEAD_ADDR_FILE" 2>/dev/null || true
  exit "$exit_code"
}
trap cleanup_ray_cluster EXIT

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
  if [[ -f "$RAY_HEAD_STEP_LOG" ]]; then
    echo "--- Tail: $RAY_HEAD_STEP_LOG ---"
    tail -n 120 "$RAY_HEAD_STEP_LOG" || true
  fi
  if [[ -f "$RAY_WORKER_STEP_LOG" ]]; then
    echo "--- Tail: $RAY_WORKER_STEP_LOG ---"
    tail -n 120 "$RAY_WORKER_STEP_LOG" || true
  fi
  exit "$exit_code"
}
trap report_failure ERR

wait_for_head_port() {
  python - <<'PY'
import os
import socket
import time

head_ip = os.environ["RAY_HEAD_IP"]
head_port = int(os.environ["RAY_GCS_PORT"])
retries = int(os.environ.get("RAY_SOCKET_PRECHECK_RETRIES", "30"))
sleep_sec = float(os.environ.get("RAY_SOCKET_PRECHECK_SLEEP_SEC", "2"))

last_error = ""
for _ in range(retries):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3.0)
    try:
        sock.connect((head_ip, head_port))
        print(f"Head port check OK: {head_ip}:{head_port}")
        raise SystemExit(0)
    except Exception as exc:
        last_error = str(exc)
    finally:
        try:
            sock.close()
        except Exception:
            pass
    time.sleep(sleep_sec)

print(f"Head port check FAILED to {head_ip}:{head_port}: {last_error}")
raise SystemExit(21)
PY
}

start_ray_cluster() {
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: TEST_nodes.sh expects a Slurm allocation. Submit with: sbatch TEST_nodes.sh"
    exit 1
  fi

  rm -f "$RAY_HEAD_IP_FILE" "$RAY_HEAD_ADDR_FILE"

  if (( SLURM_HET_SIZE_NUM >= 2 )); then
    if (( CLUSTER_NUM_NODES > SLURM_HET_SIZE_NUM )); then
      echo "ERROR: CLUSTER_NUM_NODES=$CLUSTER_NUM_NODES exceeds SLURM_HET_SIZE=$SLURM_HET_SIZE_NUM"
      exit 1
    fi
  elif [[ "${SLURM_NNODES:-0}" =~ ^[0-9]+$ ]] && (( CLUSTER_NUM_NODES > SLURM_NNODES )); then
    echo "ERROR: CLUSTER_NUM_NODES=$CLUSTER_NUM_NODES exceeds allocated SLURM_NNODES=$SLURM_NNODES"
    exit 1
  fi

  local -a all_hosts=()
  if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    mapfile -t all_hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  fi

  if (( ${#all_hosts[@]} == 0 )); then
    echo "ERROR: Could not resolve hosts from SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<empty>}"
    exit 1
  fi

  local head_host="${all_hosts[0]}"
  local worker_host=""
  if (( CLUSTER_NUM_NODES > 1 )); then
    if (( ${#all_hosts[@]} < 2 )); then
      echo "ERROR: CLUSTER_NUM_NODES=$CLUSTER_NUM_NODES requires >=2 hosts, got ${#all_hosts[@]}"
      exit 1
    fi
    worker_host="${all_hosts[1]}"
  fi

  if (( SLURM_HET_SIZE_NUM >= 2 )); then
    if [[ -n "${SLURM_JOB_NODELIST_HET_GROUP_0:-}" ]]; then
      head_host="$(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_0}" | head -n 1)"
    fi
    if (( CLUSTER_NUM_NODES > 1 )) && [[ -n "${SLURM_JOB_NODELIST_HET_GROUP_1:-}" ]]; then
      worker_host="$(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_1}" | head -n 1)"
    fi
  fi

  local -a head_srun=()
  local -a worker_srun=()
  if (( SLURM_HET_SIZE_NUM >= 2 )); then
    head_srun=(srun --het-group=0 --nodes=1 --ntasks=1 --ntasks-per-node=1)
    worker_srun=(srun --het-group=1 --nodes=1 --ntasks=1 --ntasks-per-node=1)
  else
    head_srun=(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="$head_host")
    if (( CLUSTER_NUM_NODES > 1 )); then
      worker_srun=(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="$worker_host")
    fi
  fi

  echo "Ray head host:   $head_host"
  if (( CLUSTER_NUM_NODES > 1 )); then
    echo "Ray worker host: $worker_host"
  fi

  echo "Stopping any stale Ray daemons before startup..."
  if (( SLURM_HET_SIZE_NUM >= 2 )); then
    srun --het-group=0 --nodes=1 --ntasks=1 --ntasks-per-node=1 \
      --kill-on-bad-exit=0 \
      bash -lc 'ray stop --force >/dev/null 2>&1 || true' || true
    srun --het-group=1 --nodes=1 --ntasks=1 --ntasks-per-node=1 \
      --kill-on-bad-exit=0 \
      bash -lc 'ray stop --force >/dev/null 2>&1 || true' || true
  else
    srun --nodes="$CLUSTER_NUM_NODES" --ntasks="$CLUSTER_NUM_NODES" --ntasks-per-node=1 \
      --kill-on-bad-exit=0 \
      bash -lc 'ray stop --force >/dev/null 2>&1 || true' || true
  fi

  echo "Starting Ray on $CLUSTER_NUM_NODES node(s) with fixed ports..."

  local peer_ip=""
  if (( CLUSTER_NUM_NODES > 1 )); then
    peer_ip="$(getent ahostsv4 "$worker_host" 2>/dev/null | awk 'NR==1{print $1}')"
  fi
  export RAY_PEER_IP="$peer_ip"

  local head_ip=""
  head_ip="$(
    "${head_srun[@]}" --kill-on-bad-exit=1 bash -lc '
      set -euo pipefail
      if [[ -n "${RAY_PEER_IP:-}" ]] && command -v ip >/dev/null 2>&1; then
        route_src="$(ip route get "$RAY_PEER_IP" 2>/dev/null | awk '"'"'{
          for (i=1; i<=NF; i++) if ($i=="src" && i+1<=NF) { print $(i+1); exit }
        }'"'"')"
        if [[ -n "$route_src" ]]; then
          echo "$route_src"
          exit 0
        fi
      fi
      hostname -I 2>/dev/null | tr " " "\n" | awk '"'"'
        $0 ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ &&
        $0 !~ /^127\./ &&
        $0 !~ /^169\.254\./ &&
        $0 != "" { print; exit }
      '"'"'
    '
  )"
  unset RAY_PEER_IP
  head_ip="$(echo "$head_ip" | tail -n 1 | tr -d '[:space:]')"
  if [[ -z "$head_ip" ]]; then
    echo "ERROR: Could not determine Ray head IPv4 address on host $head_host"
    exit 1
  fi

  local head_addr="${head_ip}:${RAY_GCS_PORT}"
  echo "$head_ip" > "$RAY_HEAD_IP_FILE"
  echo "$head_addr" > "$RAY_HEAD_ADDR_FILE"

  echo "Ray head IP:      $head_ip"
  echo "Ray head address: $head_addr"

  export RAY_HEAD_IP="$head_ip"
  : > "$RAY_HEAD_STEP_LOG"
  "${head_srun[@]}" --kill-on-bad-exit=1 bash -lc '
    set -euo pipefail
    ray stop --force >/dev/null 2>&1 || true
    ray start \
      --head \
      --node-ip-address="$RAY_HEAD_IP" \
      --port="$RAY_GCS_PORT" \
      --node-manager-port="$RAY_HEAD_NODE_MANAGER_PORT" \
      --object-manager-port="$RAY_HEAD_OBJECT_MANAGER_PORT" \
      --ray-client-server-port="$RAY_HEAD_CLIENT_SERVER_PORT" \
      --min-worker-port="$RAY_HEAD_MIN_WORKER_PORT" \
      --max-worker-port="$RAY_HEAD_MAX_WORKER_PORT" \
      --object-store-memory="$RAY_OBJECT_STORE_MEMORY" \
      --temp-dir="$RAY_TMPDIR" \
      --disable-usage-stats \
      --block
  ' > "$RAY_HEAD_STEP_LOG" 2>&1 &
  RAY_HEAD_STEP_PID="$!"

  # Ensure head is alive and accepting TCP before worker join attempts.
  if ! kill -0 "$RAY_HEAD_STEP_PID" 2>/dev/null; then
    echo "ERROR: Ray head step exited prematurely."
    [[ -f "$RAY_HEAD_STEP_LOG" ]] && tail -n 120 "$RAY_HEAD_STEP_LOG" || true
    exit 1
  fi
  wait_for_head_port

  if (( CLUSTER_NUM_NODES > 1 )); then
    : > "$RAY_WORKER_STEP_LOG"
    "${worker_srun[@]}" --kill-on-bad-exit=1 bash -lc '
      set -euo pipefail
      python - <<'"'"'PY'"'"'
import os
import socket
import time

head_ip = os.environ["RAY_HEAD_IP"]
head_port = int(os.environ["RAY_GCS_PORT"])
retries = int(os.environ.get("RAY_SOCKET_PRECHECK_RETRIES", "30"))
sleep_sec = float(os.environ.get("RAY_SOCKET_PRECHECK_SLEEP_SEC", "2"))

last_error = ""
for _ in range(retries):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3.0)
    try:
        sock.connect((head_ip, head_port))
        print(f"Connectivity precheck OK: {head_ip}:{head_port}")
        sock.close()
        raise SystemExit(0)
    except Exception as exc:
        last_error = str(exc)
    finally:
        try:
            sock.close()
        except Exception:
            pass
    time.sleep(sleep_sec)

print(f"Connectivity precheck FAILED to {head_ip}:{head_port}: {last_error}")
raise SystemExit(2)
PY
    '

    "${worker_srun[@]}" --kill-on-bad-exit=1 bash -lc '
      set -euo pipefail
      worker_ip=""
      if command -v ip >/dev/null 2>&1; then
        worker_ip="$(ip route get "$RAY_HEAD_IP" 2>/dev/null | awk '"'"'{
          for (i=1; i<=NF; i++) if ($i=="src" && i+1<=NF) { print $(i+1); exit }
        }'"'"')"
      fi
      if [[ -z "$worker_ip" ]]; then
        worker_ip="$(hostname -I 2>/dev/null | tr " " "\n" | awk '"'"'
          $0 ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ &&
          $0 !~ /^127\./ &&
          $0 !~ /^169\.254\./ &&
          $0 != "" { print; exit }
        '"'"')"
      fi
      if [[ -z "$worker_ip" ]]; then
        echo "ERROR: Could not determine worker node IPv4 address."
        exit 1
      fi

      ray stop --force >/dev/null 2>&1 || true
      ray start \
        --node-ip-address="$worker_ip" \
        --address="$RAY_HEAD_IP:$RAY_GCS_PORT" \
        --node-manager-port="$RAY_WORKER_NODE_MANAGER_PORT" \
        --object-manager-port="$RAY_WORKER_OBJECT_MANAGER_PORT" \
        --min-worker-port="$RAY_WORKER_MIN_WORKER_PORT" \
        --max-worker-port="$RAY_WORKER_MAX_WORKER_PORT" \
        --object-store-memory="$RAY_OBJECT_STORE_MEMORY" \
        --temp-dir="$RAY_TMPDIR" \
        --disable-usage-stats \
        --block
    ' > "$RAY_WORKER_STEP_LOG" 2>&1 &
    RAY_WORKER_STEP_PID="$!"

    if ! kill -0 "$RAY_WORKER_STEP_PID" 2>/dev/null; then
      echo "ERROR: Ray worker step exited prematurely."
      [[ -f "$RAY_WORKER_STEP_LOG" ]] && tail -n 120 "$RAY_WORKER_STEP_LOG" || true
      exit 1
    fi
  fi

  python - <<'PY'
import os
import time

import ray

head = open(os.environ["RAY_HEAD_ADDR_FILE"], "r", encoding="utf-8").read().strip()
target_nodes = int(os.environ.get("CLUSTER_NUM_NODES", "1"))
last = None

for _ in range(30):
    try:
        ray.init(address=head, namespace="RLinf", ignore_reinit_error=True, logging_level="ERROR")
        alive = [n for n in ray.nodes() if n.get("Alive", False)]
        gpus = [int(n.get("Resources", {}).get("GPU", 0)) for n in alive]
        last = (len(alive), gpus)
        if len(alive) >= target_nodes:
            print(f"Ray cluster healthy: alive_nodes={len(alive)} gpu_per_node={gpus}")
            raise SystemExit(0)
    except Exception as exc:
        last = str(exc)
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass
    time.sleep(2)

print(f"Ray cluster validation failed. Last observation: {last}")
raise SystemExit(3)
PY

  ray status --address="${head_addr}" || true
  RAY_CLUSTER_STARTED=1
}

validate_grpo_scaling() {
  export EFFECTIVE_ACTOR_GPU_MAP="$ACTOR_GPU_MAP"
  export EFFECTIVE_ROLLOUT_GPU_MAP="$ROLLOUT_GPU_MAP"
  export EFFECTIVE_ENV_GPU_MAP="$ENV_GPU_MAP"
  export EFFECTIVE_SMOKE_SCALE="$USE_SMOKE_SCALE"
  export EFFECTIVE_SMOKE_GROUP_SIZE="$SMOKE_GROUP_SIZE"
  export EFFECTIVE_SMOKE_NUM_GROUP_ENVS="$SMOKE_NUM_GROUP_ENVS"
  export EFFECTIVE_SMOKE_GLOBAL_BATCH_SIZE="$SMOKE_GLOBAL_BATCH_SIZE"
  export EFFECTIVE_SMOKE_MICRO_BATCH_SIZE="$SMOKE_MICRO_BATCH_SIZE"

  python - <<'PY'
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

repo_root = Path(os.environ["REPO_ROOT"])
config_name = os.environ["CONFIG_NAME"]
cfg_rel = Path("examples/embodiment/config") / f"{config_name}.yaml"
cfg_path = repo_root / cfg_rel

if not cfg_path.exists():
    print(f"ERROR: config file not found: {cfg_path}")
    raise SystemExit(10)

cfg = OmegaConf.load(cfg_path)

if os.environ.get("EFFECTIVE_SMOKE_SCALE", "0") == "1":
    cfg.algorithm.group_size = int(os.environ["EFFECTIVE_SMOKE_GROUP_SIZE"])
    cfg.algorithm.num_group_envs = int(os.environ["EFFECTIVE_SMOKE_NUM_GROUP_ENVS"])
    cfg.actor.global_batch_size = int(os.environ["EFFECTIVE_SMOKE_GLOBAL_BATCH_SIZE"])
    cfg.actor.micro_batch_size = int(os.environ["EFFECTIVE_SMOKE_MICRO_BATCH_SIZE"])

def parse_ids(spec: str):
    ids = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            start, end = int(a), int(b)
            if end < start:
                raise ValueError(f"Invalid range {token}")
            ids.extend(range(start, end + 1))
        else:
            ids.append(int(token))
    return ids

actor_ws = len(parse_ids(os.environ["EFFECTIVE_ACTOR_GPU_MAP"]))
rollout_ws = len(parse_ids(os.environ["EFFECTIVE_ROLLOUT_GPU_MAP"]))
env_ws = len(parse_ids(os.environ["EFFECTIVE_ENV_GPU_MAP"]))

stage_num = int(cfg.rollout.pipeline_stage_num)
algo_num_group_envs = int(cfg.algorithm.num_group_envs)
eval_num_envs = int(cfg.env.eval.num_envs)
group_size = int(cfg.algorithm.group_size)
rollout_epoch = int(cfg.algorithm.rollout_epoch)
global_bsz = int(cfg.actor.global_batch_size)
micro_bsz = int(cfg.actor.micro_batch_size)

errors = []

scale_div = stage_num * env_ws
if scale_div <= 0:
    errors.append(f"Invalid scale divisor stage_num*env_ws={scale_div}")
if algo_num_group_envs % scale_div != 0:
    errors.append(
        f"algorithm.num_group_envs ({algo_num_group_envs}) must be divisible by "
        f"stage_num*env_world_size ({stage_num}*{env_ws}={scale_div})"
    )
if eval_num_envs % scale_div != 0:
    errors.append(
        f"env.eval.num_envs ({eval_num_envs}) must be divisible by "
        f"stage_num*env_world_size ({stage_num}*{env_ws}={scale_div})"
    )

if actor_ws < 1:
    errors.append(f"actor world size must be >=1, got {actor_ws}")
if rollout_ws < 1:
    errors.append(f"rollout world size must be >=1, got {rollout_ws}")
if env_ws < 1:
    errors.append(f"env world size must be >=1, got {env_ws}")

if global_bsz % (micro_bsz * actor_ws) != 0:
    errors.append(
        f"actor.global_batch_size ({global_bsz}) must be divisible by "
        f"actor.micro_batch_size*actor_world_size ({micro_bsz}*{actor_ws}={micro_bsz*actor_ws})"
    )

if rollout_ws % env_ws != 0:
    errors.append(
        f"rollout world size ({rollout_ws}) must be divisible by env world size ({env_ws})"
    )

effective_num_group_envs = algo_num_group_envs // scale_div if scale_div > 0 else -1
per_rank_batch = global_bsz // actor_ws if actor_ws > 0 else -1

if errors:
    print("GRPO scaling preflight FAILED:")
    for e in errors:
        print(f" - {e}")
    raise SystemExit(11)

print("GRPO scaling preflight OK:")
print(f"  actor_ws={actor_ws} rollout_ws={rollout_ws} env_ws={env_ws}")
print(f"  stage_num={stage_num}")
print(f"  algorithm.num_group_envs(raw)={algo_num_group_envs}")
print(f"  algorithm.num_group_envs(effective)={effective_num_group_envs}")
print(f"  env.eval.num_envs(raw)={eval_num_envs}")
print(f"  algorithm.group_size={group_size} algorithm.rollout_epoch={rollout_epoch}")
print(f"  actor.global_batch_size={global_bsz} actor.micro_batch_size={micro_bsz}")
print(f"  per_actor_rank_batch={per_rank_batch}")
print("  note=actor-side rollout split asserts still run during training.")
PY
}

resolve_component_placement() {
  if [[ "$AUTO_COMPONENT_PLACEMENT" != "1" ]]; then
    echo "Auto placement disabled. Using ACTOR_GPU_MAP=$ACTOR_GPU_MAP ROLLOUT_GPU_MAP=$ROLLOUT_GPU_MAP ENV_GPU_MAP=$ENV_GPU_MAP"
    return 0
  fi

  if ! [[ "$ACTOR_GPU_COUNT" =~ ^[0-9]+$ ]] || (( ACTOR_GPU_COUNT < 1 )); then
    echo "ERROR: ACTOR_GPU_COUNT must be >=1, got: $ACTOR_GPU_COUNT"
    exit 1
  fi
  if ! [[ "$WORKER_ROLLOUT_GPU_COUNT" =~ ^[0-9]+$ ]] || (( WORKER_ROLLOUT_GPU_COUNT < 1 )); then
    echo "ERROR: WORKER_ROLLOUT_GPU_COUNT must be >=1, got: $WORKER_ROLLOUT_GPU_COUNT"
    exit 1
  fi
  if ! [[ "$WORKER_ENV_GPU_COUNT" =~ ^[0-9]+$ ]] || (( WORKER_ENV_GPU_COUNT < 1 )); then
    echo "ERROR: WORKER_ENV_GPU_COUNT must be >=1, got: $WORKER_ENV_GPU_COUNT"
    exit 1
  fi

  local placement_lines=""
  placement_lines="$(python - <<'PY'
import os
import socket
import sys

import ray


def ip_sort_key(ip: str):
    try:
        return tuple(int(x) for x in ip.split("."))
    except Exception:
        return (999, ip)


def parse_range(start: int, count: int) -> str:
    if count == 1:
        return str(start)
    return f"{start}-{start + count - 1}"


head_addr_file = os.environ.get("RAY_HEAD_ADDR_FILE", "")
head_ip_file = os.environ.get("RAY_HEAD_IP_FILE", "")
head_addr = ""
head_ip = ""
if head_addr_file and os.path.isfile(head_addr_file):
    head_addr = open(head_addr_file, "r", encoding="utf-8").read().strip()
if head_ip_file and os.path.isfile(head_ip_file):
    head_ip = open(head_ip_file, "r", encoding="utf-8").read().strip()

try:
    ray.init(
        address=head_addr if head_addr else "auto",
        namespace="RLinf",
        ignore_reinit_error=True,
        logging_level="ERROR",
    )
except Exception:
    ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")

alive_nodes = [n for n in ray.nodes() if n.get("Alive", False)]
gpu_nodes = []
for n in alive_nodes:
    resources = n.get("Resources", {})
    num_gpu = int(resources.get("GPU", 0))
    if num_gpu > 0:
        gpu_nodes.append((n.get("NodeManagerAddress", ""), num_gpu))

if not gpu_nodes:
    print("ERROR: No alive GPU nodes detected in Ray cluster.", file=sys.stderr)
    sys.exit(2)

gpu_nodes = sorted(gpu_nodes, key=lambda x: ip_sort_key(x[0]))
offset = 0
node_gpu_map = {}
for ip, count in gpu_nodes:
    node_gpu_map[ip] = (offset, count)
    offset += count

if head_ip not in node_gpu_map:
    # Resolve hostname if file accidentally contains hostname instead of IP.
    try:
        resolved = socket.gethostbyname(head_ip)
        if resolved in node_gpu_map:
            head_ip = resolved
    except Exception:
        pass

if head_ip not in node_gpu_map:
    # Fallback: assume first sorted GPU node is head.
    head_ip = gpu_nodes[0][0]

actor_need = int(os.environ.get("ACTOR_GPU_COUNT", "2"))
rollout_need = int(os.environ.get("WORKER_ROLLOUT_GPU_COUNT", "2"))
env_need = int(os.environ.get("WORKER_ENV_GPU_COUNT", "1"))
worker_need = rollout_need + env_need

head_offset, head_gpus = node_gpu_map[head_ip]
if actor_need > head_gpus:
    print(
        f"ERROR: Actor needs {actor_need} GPUs but head node {head_ip} has {head_gpus}.",
        file=sys.stderr,
    )
    sys.exit(3)

worker_candidates = []
for ip, _ in gpu_nodes:
    if ip == head_ip:
        continue
    off, cnt = node_gpu_map[ip]
    if cnt >= worker_need:
        worker_candidates.append((ip, off, cnt))

if not worker_candidates:
    print(
        f"ERROR: Need worker node with >= {worker_need} GPUs for rollout/env, but none found. GPU nodes: {gpu_nodes}",
        file=sys.stderr,
    )
    sys.exit(4)

worker_ip, worker_offset, worker_gpus = worker_candidates[0]

actor_map = parse_range(head_offset, actor_need)
rollout_map = parse_range(worker_offset, rollout_need)
env_map = parse_range(worker_offset + rollout_need, env_need)

print(f"SET ACTOR_GPU_MAP={actor_map}")
print(f"SET ROLLOUT_GPU_MAP={rollout_map}")
print(f"SET ENV_GPU_MAP={env_map}")
print(f"SET PLACEMENT_HEAD_IP={head_ip}")
print(f"SET PLACEMENT_WORKER_IP={worker_ip}")
print(f"SET PLACEMENT_WORKER_GPUS={worker_gpus}")
PY
)"

  if [[ -z "$placement_lines" ]]; then
    echo "ERROR: Auto placement script returned empty output."
    exit 1
  fi

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    if [[ "$line" == SET\ * ]]; then
      eval "${line#SET }"
    fi
  done <<< "$placement_lines"

  echo "Resolved placement from Ray topology:"
  echo "  head_ip=$PLACEMENT_HEAD_IP worker_ip=$PLACEMENT_WORKER_IP worker_gpus=$PLACEMENT_WORKER_GPUS"
  echo "  actor=$ACTOR_GPU_MAP rollout=$ROLLOUT_GPU_MAP env=$ENV_GPU_MAP"
}

echo "==== OpenVLA-OFT Training Run ===="
echo "Repo root:        $REPO_ROOT"
echo "Config:           $CONFIG_NAME"
echo "Task ID:          $TASK_ID"
echo "Epochs (input):   $EPOCHS"
echo "Epochs (effective): $EFFECTIVE_EPOCHS"
echo "Seed:             $SEED"
echo "Experiment name:  $EXPERIMENT_NAME"
echo "Model dir:        $MODEL_DIR"
echo "LIBERO path:      $LIBERO_REPO_PATH"
echo "Conda env:        ${CONDA_DEFAULT_ENV:-<unknown>}"
echo "SLURM job id:     ${SLURM_JOB_ID:-<none>}"
echo "SLURM node list:  ${SLURM_NODELIST:-<none>}"
echo "SLURM job nodelist: ${SLURM_JOB_NODELIST:-<none>}"
echo "SLURM Nnodes:     ${SLURM_NNODES:-<none>}"
echo "SLURM het size:   ${SLURM_HET_SIZE:-<none>}"
echo "Cluster nodes:    $CLUSTER_NUM_NODES"
echo "MASTER_PORT:      $MASTER_PORT"
echo "RAY_GCS_PORT:     $RAY_GCS_PORT"
echo "RAY_HEAD_NODE_MANAGER_PORT: $RAY_HEAD_NODE_MANAGER_PORT"
echo "RAY_HEAD_OBJECT_MANAGER_PORT: $RAY_HEAD_OBJECT_MANAGER_PORT"
echo "RAY_WORKER_NODE_MANAGER_PORT: $RAY_WORKER_NODE_MANAGER_PORT"
echo "RAY_WORKER_OBJECT_MANAGER_PORT: $RAY_WORKER_OBJECT_MANAGER_PORT"
echo "HF_HOME:          $HF_HOME"
echo "HF_HUB_CACHE:     $HF_HUB_CACHE"
echo "HF_XET_CACHE:     $HF_XET_CACHE"
echo "RAY_TMPDIR:       $RAY_TMPDIR"
echo "RAY_HEAD_IP_FILE: $RAY_HEAD_IP_FILE"
echo "RAY_HEAD_ADDR_FILE: $RAY_HEAD_ADDR_FILE"
echo "RAY_OBJECT_STORE_MEMORY: $RAY_OBJECT_STORE_MEMORY"
echo "TORCH_DISTRIBUTED_DEBUG: $TORCH_DISTRIBUTED_DEBUG"
echo "NCCL_DEBUG:       $NCCL_DEBUG"
echo "NCCL_ASYNC_ERROR_HANDLING: $NCCL_ASYNC_ERROR_HANDLING"
echo "NCCL_DEBUG_SUBSYS: $NCCL_DEBUG_SUBSYS"
echo "TRANSFORMERS_OFFLINE: $TRANSFORMERS_OFFLINE"
echo "USE_SMOKE_SCALE:  $USE_SMOKE_SCALE"
echo "MATCH_CRL_SAMPLE_BUDGET: $MATCH_CRL_SAMPLE_BUDGET"
echo "DISABLE_VIDEO_LOGGING: $DISABLE_VIDEO_LOGGING"
echo "TRAIN_CAMERA_RES: $TRAIN_CAMERA_RES"
echo "EVAL_CAMERA_RES:  $EVAL_CAMERA_RES"
echo "AUTO_COMPONENT_PLACEMENT: $AUTO_COMPONENT_PLACEMENT"
echo "ACTOR_GPU_COUNT:  $ACTOR_GPU_COUNT"
echo "WORKER_ROLLOUT_GPU_COUNT: $WORKER_ROLLOUT_GPU_COUNT"
echo "WORKER_ENV_GPU_COUNT: $WORKER_ENV_GPU_COUNT"
echo "RAY_ONLY:            $RAY_ONLY"
if [[ -n "$LORA_PATH" ]]; then
  echo "LORA_PATH:           $LORA_PATH"
fi
if [[ "$USE_SMOKE_SCALE" == "1" ]]; then
  echo "Requested placement: actor=$ACTOR_GPU_MAP rollout=$ROLLOUT_GPU_MAP env=$ENV_GPU_MAP"
  echo "Scaled rollout:   group_size=$SMOKE_GROUP_SIZE num_group_envs=$SMOKE_NUM_GROUP_ENVS"
  echo "Scaled batches:   micro=$SMOKE_MICRO_BATCH_SIZE global=$SMOKE_GLOBAL_BATCH_SIZE"
  echo "Scaled epochs:    rollout=$SMOKE_ROLLOUT_EPOCH eval_rollout=$SMOKE_EVAL_ROLLOUT_EPOCH save_interval=$SMOKE_SAVE_INTERVAL"
  if [[ "$MATCH_CRL_SAMPLE_BUDGET" == "1" ]]; then
    echo "Baseline sample budget: group_size=$BASE_GROUP_SIZE num_group_envs=$BASE_NUM_GROUP_ENVS rollout_epoch=$BASE_ROLLOUT_EPOCH"
  fi
fi
echo "Run LOG_DIR:      $LOG_DIR"
echo

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

start_ray_cluster
resolve_component_placement
validate_grpo_scaling

if [[ "$RAY_ONLY" == "1" ]]; then
  echo "RAY_ONLY=1 set. Ray cluster and placement validated; exiting before model/training."
  exit 0
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
  "cluster.num_nodes=$CLUSTER_NUM_NODES"
  "env.train.init_params.camera_heights=$TRAIN_CAMERA_RES"
  "env.train.init_params.camera_widths=$TRAIN_CAMERA_RES"
  "env.eval.init_params.camera_heights=$EVAL_CAMERA_RES"
  "env.eval.init_params.camera_widths=$EVAL_CAMERA_RES"
  "+cluster.component_placement.actor=$ACTOR_GPU_MAP"
  "+cluster.component_placement.rollout=$ROLLOUT_GPU_MAP"
  "+cluster.component_placement.env=$ENV_GPU_MAP"
)

if [[ "$DISABLE_VIDEO_LOGGING" == "1" ]]; then
  CMD+=(
    "env.train.video_cfg.save_video=False"
    "env.eval.video_cfg.save_video=False"
  )
fi

if [[ "$USE_SMOKE_SCALE" == "1" ]]; then
  CMD+=(
    # Keep CRL loss/objective logic from config, but scale load for A40 memory.
    "runner.save_interval=$SMOKE_SAVE_INTERVAL"
    "algorithm.rollout_epoch=$SMOKE_ROLLOUT_EPOCH"
    "algorithm.eval_rollout_epoch=$SMOKE_EVAL_ROLLOUT_EPOCH"
    "algorithm.group_size=$SMOKE_GROUP_SIZE"
    "algorithm.num_group_envs=$SMOKE_NUM_GROUP_ENVS"
    "actor.global_batch_size=$SMOKE_GLOBAL_BATCH_SIZE"
    "actor.micro_batch_size=$SMOKE_MICRO_BATCH_SIZE"
  )
fi

if [[ -n "$LORA_PATH" ]]; then
  if [[ ! -d "$LORA_PATH" ]]; then
    echo "ERROR: LORA_PATH does not exist or is not a directory: $LORA_PATH"
    exit 1
  fi
  CMD+=("+actor.model.lora_path=$LORA_PATH")
fi

echo "Running training command:"
printf '  %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"

echo
echo "Training run completed successfully."
