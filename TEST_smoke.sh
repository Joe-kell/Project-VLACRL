#!/usr/bin/env bash
#SBATCH --job-name=vlacrl_object_a40_smoke_launcher
#SBATCH --output=/home/s2758621/Continual_VLA_RL/logs/test-smoke-%j.out
#SBATCH --error=/home/s2758621/Continual_VLA_RL/logs/test-smoke-%j.err
#SBATCH --partition=ICF-Free
#SBATCH --nodelist=crannog03
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=500G
#SBATCH --gres=gpu:a40:4

set -euo pipefail

# Submit smoke tests that execute TEST.sh with lightweight settings.
#
# Default behavior:
#   1) submit standard LIBERO smoke
#   2) submit LIBERO-plus smoke
#
# Optional env overrides:
#   CONDA_ENV_NAME=vlacrl_libplus
#   LIGHTWEIGHT_SMOKE=1
#   RUN_STANDARD=1
#   RUN_PLUS=1
#   CHAIN_PLUS_AFTER_STANDARD=0
#   SBATCH_PARTITION=ICF-Free
#   SBATCH_NODELIST=crannog03
#   SBATCH_NODES=1
#   SBATCH_CPUS_PER_TASK=60
#   SBATCH_MEM=500G
#   SBATCH_GRES=gpu:a40:4
#   ACTOR_GPU_MAP=0-1
#   ROLLOUT_GPU_MAP=2
#   ENV_GPU_MAP=3
#   TARGET_SCRIPT=/home/s2758621/Continual_VLA_RL/TEST.sh
#   EXTRA_EXPORTS="CONFIG_NAME=...,TASK_ID=...,SEED=..."

DEFAULT_REPO_ROOT="/home/s2758621/Continual_VLA_RL"

if [[ -n "${REPO_ROOT:-}" ]] && [[ -d "${REPO_ROOT}" ]]; then
  REPO_ROOT="$REPO_ROOT"
elif [[ -d "${DEFAULT_REPO_ROOT}" ]]; then
  REPO_ROOT="$DEFAULT_REPO_ROOT"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$SCRIPT_DIR"
fi

cd "$REPO_ROOT"

TARGET_SCRIPT="${TARGET_SCRIPT:-$REPO_ROOT/TEST.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlacrl_libplus}"
LIGHTWEIGHT_SMOKE="${LIGHTWEIGHT_SMOKE:-1}"
RUN_STANDARD="${RUN_STANDARD:-1}"
RUN_PLUS="${RUN_PLUS:-1}"
CHAIN_PLUS_AFTER_STANDARD="${CHAIN_PLUS_AFTER_STANDARD:-0}"
SBATCH_PARTITION="${SBATCH_PARTITION:-ICF-Free}"
SBATCH_NODELIST="${SBATCH_NODELIST:-crannog03}"
SBATCH_NODES="${SBATCH_NODES:-1}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-60}"
SBATCH_MEM="${SBATCH_MEM:-500G}"
SBATCH_GRES="${SBATCH_GRES:-gpu:a40:4}"
ACTOR_GPU_MAP="${ACTOR_GPU_MAP:-0-1}"
ROLLOUT_GPU_MAP="${ROLLOUT_GPU_MAP:-2}"
ENV_GPU_MAP="${ENV_GPU_MAP:-3}"
EXTRA_EXPORTS="${EXTRA_EXPORTS:-}"

if [[ ! -f "$TARGET_SCRIPT" ]]; then
  echo "ERROR: target script not found: $TARGET_SCRIPT"
  exit 1
fi

submit_smoke() {
  local libero_type="$1"
  local dependency_job_id="${2:-}"
  local exports="ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},LIBERO_TYPE=${libero_type},LIGHTWEIGHT_SMOKE=${LIGHTWEIGHT_SMOKE},ACTOR_GPU_MAP=${ACTOR_GPU_MAP},ROLLOUT_GPU_MAP=${ROLLOUT_GPU_MAP},ENV_GPU_MAP=${ENV_GPU_MAP}"
  local output
  local cmd=(sbatch)

  if [[ -n "$EXTRA_EXPORTS" ]]; then
    exports="${exports},${EXTRA_EXPORTS}"
  fi

  cmd+=(
    --partition="$SBATCH_PARTITION"
    --nodelist="$SBATCH_NODELIST"
    --nodes="$SBATCH_NODES"
    --cpus-per-task="$SBATCH_CPUS_PER_TASK"
    --mem="$SBATCH_MEM"
    --gres="$SBATCH_GRES"
    --export="$exports"
  )

  if [[ -n "$dependency_job_id" ]]; then
    cmd+=(--dependency="afterok:${dependency_job_id}")
  fi

  cmd+=("$TARGET_SCRIPT")

  output="$("${cmd[@]}")"
  echo "$output" >&2
  echo "$output" | awk '{print $4}'
}

standard_job_id=""
plus_job_id=""

if [[ "$RUN_STANDARD" == "1" ]]; then
  standard_job_id="$(submit_smoke standard)"
fi

if [[ "$RUN_PLUS" == "1" ]]; then
  if [[ "$CHAIN_PLUS_AFTER_STANDARD" == "1" && -n "$standard_job_id" ]]; then
    plus_job_id="$(submit_smoke plus "$standard_job_id")"
  else
    plus_job_id="$(submit_smoke plus)"
  fi
fi

echo "Submitted smoke jobs:"
echo "  standard: ${standard_job_id:-<not-submitted>}"
echo "  plus:     ${plus_job_id:-<not-submitted>}"
echo "SLURM target:"
echo "  partition=${SBATCH_PARTITION} nodelist=${SBATCH_NODELIST} nodes=${SBATCH_NODES}"
echo "  cpus_per_task=${SBATCH_CPUS_PER_TASK} mem=${SBATCH_MEM} gres=${SBATCH_GRES}"
echo "GPU placement exports:"
echo "  ACTOR_GPU_MAP=${ACTOR_GPU_MAP} ROLLOUT_GPU_MAP=${ROLLOUT_GPU_MAP} ENV_GPU_MAP=${ENV_GPU_MAP}"
