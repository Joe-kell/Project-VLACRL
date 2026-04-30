#!/usr/bin/env bash
#SBATCH --job-name=vlacrl_sft_eval
#SBATCH --output=/home/s2758621/Continual_VLA_RL/logs/test-sft-eval-%j.out
#SBATCH --error=/home/s2758621/Continual_VLA_RL/logs/test-sft-eval-%j.err
#SBATCH --partition=ICF-Free
#SBATCH --nodelist=crannog03
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=500G
#SBATCH --gres=gpu:a40:4

set -euo pipefail

# Gate-A SFT base evaluation launcher (no LoRA), LIBERO+ Object only:
#   - LIBERO-Object varied-object task IDs by default:
#       [0,32,79,100,122,127,138,170,210,235]
#
# This script does not download/install model checkpoints.
# It only runs evals for models that are already present on disk.
#
# Optional env overrides:
#   CONDA_ENV_NAME=vlacrl_libplus
#   CONDA_BASE=$HOME/miniconda3
#   LIBERO_TYPE=plus
#   LIBERO_SUFFIX=add
#   SEED=1234
#   REPO_ROOT=/home/s2758621/Continual_VLA_RL
#   RUN_OBJECT=1
#   OBJECT_TASK_IDS=[0,32,79,100,122,127,138,170,210,235]
#   ACTOR_GPU_MAP=0-1
#   ROLLOUT_GPU_MAP=2
#   ENV_GPU_MAP=3
#   EVAL_NUM_ENVS=20
#   EVAL_ROLLOUT_EPOCH=8
#   STRICT_MODEL_CHECK=1
#   SUMMARY_BASE_DIR=/home/s2758621/Continual_VLA_RL/logs_sft_eval

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

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlacrl_libplus}"
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
  echo "       Tried CONDA_BASE=$CONDA_BASE and system conda."
  exit 1
fi

export LIBERO_REPO_PATH="${LIBERO_REPO_PATH:-$REPO_ROOT/LIBERO}"
export LIBERO_TYPE="${LIBERO_TYPE:-plus}"
# LIBERO_SUFFIX controls perturbation family when LIBERO_TYPE=plus.
# `add` emphasizes object-variation tasks (extra/varied objects).
export LIBERO_SUFFIX="${LIBERO_SUFFIX:-add}"
SEED="${SEED:-1234}"

RUN_OBJECT="${RUN_OBJECT:-1}"
# Cross-family LIBERO-Object IDs for varied-object evaluation.
# Keep EVAL_NUM_ENVS divisible by len(OBJECT_TASK_IDS)=10.
OBJECT_TASK_IDS="${OBJECT_TASK_IDS:-[0,32,79,100,122,127,138,170,210,235]}"
STRICT_MODEL_CHECK="${STRICT_MODEL_CHECK:-1}"

ACTOR_GPU_MAP="${ACTOR_GPU_MAP:-0-1}"
ROLLOUT_GPU_MAP="${ROLLOUT_GPU_MAP:-2}"
ENV_GPU_MAP="${ENV_GPU_MAP:-3}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-20}"
EVAL_ROLLOUT_EPOCH="${EVAL_ROLLOUT_EPOCH:-8}"

OBJECT_MODEL_DIR="${OBJECT_MODEL_DIR:-$REPO_ROOT/model/Openvla-oft-SFT-libero-object-traj1}"

# Force a short Ray temp root to avoid AF_UNIX socket path length failures.
RAY_TMP_BASE_DEFAULT="/tmp/ray_${USER:-unknown}"
export RAY_TMPDIR="${RAY_TMPDIR:-$RAY_TMP_BASE_DEFAULT}"
export TMPDIR="${TMPDIR:-$RAY_TMPDIR}"
export TMP="${TMP:-$RAY_TMPDIR}"
export TEMP="${TEMP:-$RAY_TMPDIR}"
mkdir -p "$RAY_TMPDIR"
chmod 700 "$RAY_TMPDIR" 2>/dev/null || true

JOB_TAG="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_BASE_DIR="${SUMMARY_BASE_DIR:-$REPO_ROOT/logs_sft_eval}"
SUMMARY_DIR="${SUMMARY_BASE_DIR}/${JOB_TAG}"
SUMMARY_CSV="${SUMMARY_DIR}/gateA_suite_summary.csv"
mkdir -p "$SUMMARY_DIR"

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "timestamp_utc,job_id,suite,seed,libero_type,task_ids,suite_mean_success,success_once,success_at_end,return,episode_len,task_eval_total,eval_log_path,suite_stdout_log" > "$SUMMARY_CSV"
fi

echo "==== Gate-A SFT Base Eval (LIBERO+ Object) ===="
echo "Repo root:        $REPO_ROOT"
echo "Conda env:        ${CONDA_DEFAULT_ENV:-<unknown>}"
echo "LIBERO_TYPE:      $LIBERO_TYPE"
echo "LIBERO_SUFFIX:    ${LIBERO_SUFFIX:-<none>}"
echo "Seed:             $SEED"
echo "GPU placement:    actor=$ACTOR_GPU_MAP rollout=$ROLLOUT_GPU_MAP env=$ENV_GPU_MAP"
echo "Eval scaling:     env.eval.num_envs=$EVAL_NUM_ENVS algorithm.eval_rollout_epoch=$EVAL_ROLLOUT_EPOCH"
echo "Run suite:        object=$RUN_OBJECT"
echo "Object task IDs:  $OBJECT_TASK_IDS"
echo "Model dirs:"
echo "  object:  $OBJECT_MODEL_DIR"
echo "RAY_TMPDIR:       $RAY_TMPDIR"
echo "Summary dir:      $SUMMARY_DIR"
echo "Summary csv:      $SUMMARY_CSV"
echo "============================="

check_model_dir() {
  local name="$1"
  local dir="$2"
  if [[ -d "$dir" ]]; then
    echo "[$name] model dir OK: $dir"
    return 0
  fi

  echo "[$name] model dir MISSING: $dir"
  if [[ "$STRICT_MODEL_CHECK" == "1" ]]; then
    echo "[$name] aborting because STRICT_MODEL_CHECK=1"
    return 1
  fi

  echo "[$name] skipping because STRICT_MODEL_CHECK=0"
  return 2
}

run_eval_suite() {
  local suite_name="$1"
  local config_name="$2"
  local fixed_task_ids="$3"
  local suite_log="$SUMMARY_DIR/${suite_name}_seed${SEED}.stdout.log"

  echo
  echo "----- Running $suite_name base eval -----"
  echo "[$suite_name] stdout log: $suite_log"
  bash "$REPO_ROOT/examples/embodiment/eval_embodiment.sh" \
    "$config_name" \
    actor.model.is_lora=False \
    runner.only_eval=True \
    "env.fixed_task_ids=$fixed_task_ids" \
    "env.eval.num_envs=$EVAL_NUM_ENVS" \
    "algorithm.eval_rollout_epoch=$EVAL_ROLLOUT_EPOCH" \
    actor.seed="$SEED" \
    "+cluster.component_placement.actor=$ACTOR_GPU_MAP" \
    "+cluster.component_placement.rollout=$ROLLOUT_GPU_MAP" \
    "+cluster.component_placement.env=$ENV_GPU_MAP" \
    2>&1 | tee "$suite_log"

  python - "$suite_log" "$fixed_task_ids" "$suite_name" "$SEED" "$LIBERO_TYPE" "$SUMMARY_CSV" "$JOB_TAG" <<'PY'
import ast
import csv
import datetime as dt
import math
import pathlib
import re
import sys

suite_log = pathlib.Path(sys.argv[1])
task_ids = ast.literal_eval(sys.argv[2])
suite_name = sys.argv[3]
seed = sys.argv[4]
libero_type = sys.argv[5]
summary_csv = pathlib.Path(sys.argv[6])
job_tag = sys.argv[7]

lines = suite_log.read_text(errors="replace").splitlines()

eval_payload = None
for line in reversed(lines):
    idx = line.find("eval_metrics=")
    if idx != -1:
        eval_payload = line[idx + len("eval_metrics="):].strip()
        break

if eval_payload is None:
    raise RuntimeError(
        f"[{suite_name}] Could not find `eval_metrics=` in {suite_log}. "
        "Evaluation likely failed before metric emission."
    )

try:
    metrics = ast.literal_eval(eval_payload)
except Exception as exc:
    raise RuntimeError(
        f"[{suite_name}] Failed to parse eval_metrics payload: {exc}\n"
        f"payload={eval_payload[:500]}"
    ) from exc

task_success_values = []
task_total_values = []
missing_keys = []
for task_id in task_ids:
    task_id = int(task_id)
    success_key = f"eval/env_info/task_{task_id}_success"
    total_key = f"eval/env_info/task_{task_id}_success_total"
    if success_key not in metrics:
        missing_keys.append(success_key)
        continue
    task_success_values.append(float(metrics[success_key]))
    if total_key in metrics:
        task_total_values.append(float(metrics[total_key]))

if missing_keys:
    raise RuntimeError(
        f"[{suite_name}] Missing expected per-task keys: {missing_keys}"
    )

suite_mean_success = sum(task_success_values) / len(task_success_values)
task_eval_total = sum(task_total_values) if task_total_values else float("nan")

success_once = float(metrics.get("eval/env_info/success_once", float("nan")))
success_at_end = float(metrics.get("eval/env_info/success_at_end", float("nan")))
episode_return = float(metrics.get("eval/env_info/return", float("nan")))
episode_len = float(metrics.get("eval/env_info/episode_len", float("nan")))

eval_log_path = ""
for line in lines:
    match = re.search(r"runner\.logger\.log_path=([^ ]+)", line)
    if match:
        eval_log_path = match.group(1).strip("'\"")
        break

timestamp_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
with summary_csv.open("a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            timestamp_utc,
            job_tag,
            suite_name,
            seed,
            libero_type,
            ";".join(str(int(x)) for x in task_ids),
            f"{suite_mean_success:.6f}",
            f"{success_once:.6f}" if not math.isnan(success_once) else "",
            f"{success_at_end:.6f}" if not math.isnan(success_at_end) else "",
            f"{episode_return:.6f}" if not math.isnan(episode_return) else "",
            f"{episode_len:.6f}" if not math.isnan(episode_len) else "",
            f"{task_eval_total:.0f}" if not math.isnan(task_eval_total) else "",
            eval_log_path,
            str(suite_log),
        ]
    )

print(
    f"[{suite_name}] suite_mean_success={suite_mean_success:.6f} "
    f"(tasks={task_ids}) success_once={success_once:.6f} "
    f"success_at_end={success_at_end:.6f}"
)
PY
}

OVERALL_EXIT_CODE=0

if [[ "$RUN_OBJECT" == "1" ]]; then
  if check_model_dir "object" "$OBJECT_MODEL_DIR"; then
    run_eval_suite \
      "libero_object" \
      "crl_experiment/libero_object_grpo_openvlaoft_eval_object" \
      "$OBJECT_TASK_IDS" || OVERALL_EXIT_CODE=$?
  elif [[ $? -eq 1 ]]; then
    exit 1
  fi
fi

echo
echo "Per-suite summary:"
cat "$SUMMARY_CSV"
echo
echo "Gate-A eval script finished with exit code: $OVERALL_EXIT_CODE"
exit "$OVERALL_EXIT_CODE"
