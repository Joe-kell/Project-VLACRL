#! /bin/bash
#
# Setup script for running embodied agent training
#
# Environment Variables (optional overrides):
#   - LIBERO_REPO_PATH: Path to LIBERO repository (defaults to ${REPO_PATH}/LIBERO)
#   - LIBERO_TYPE: `standard` (default) or `plus`
#   - LIBERO_SUFFIX: optional perturbation selector for LIBERO-plus
#
# Note: REPO_PATH is automatically set to the parent directory of examples/
#       If you need to override it, set it before running this script.
set -e -o pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
# NOTE: set LIBERO_REPO_PATH to the path of the LIBERO repo
# Defaults to ${REPO_PATH}/LIBERO if not set
export LIBERO_REPO_PATH="${LIBERO_REPO_PATH:-${REPO_PATH}/LIBERO}"
export LIBERO_TYPE="${LIBERO_TYPE:-standard}"
export LIBERO_SUFFIX="${LIBERO_SUFFIX:-}"
if [ "${LIBERO_TYPE}" = "plus" ]; then
    export LIBERO_PLUS_PATH="${LIBERO_PLUS_PATH:-${REPO_PATH}/third_party/libero_plus}"
    # LIBERO-plus reuses LIBERO_CONFIG_PATH env var for its own config root.
    # Keep it separate from standard LIBERO to avoid reading wrong asset paths.
    export LIBERO_CONFIG_PATH="${LIBERO_PLUS_CONFIG_PATH:-$HOME/.liberoplus}"
    python - <<'PY'
import pathlib
import liberoplus.liberoplus as l_plus

asset_root = pathlib.Path(l_plus.get_libero_path("assets"))
if not asset_root.is_dir():
    raise RuntimeError(
        f"LIBERO_TYPE=plus but assets directory is missing: {asset_root}"
    )
for required in ("scenes", "new_objects", "textures"):
    if not (asset_root / required).is_dir():
        raise RuntimeError(
            f"LIBERO_TYPE=plus assets incomplete: missing '{required}' in {asset_root}"
        )
print(f"[run_embodiment] LIBERO+ assets OK: {asset_root}")
PY
else
    # Standard LIBERO config root.
    export LIBERO_CONFIG_PATH="${LIBERO_REPO_PATH}"
    # BEGIN LIBERO_STANDARD_PATH_NORMALIZATION_HOTFIX
    # Rollback instruction: remove this block through the matching END marker below.
    # Ray workers can run with a different CWD, so relative paths in LIBERO config
    # become brittle. Rewrite non-absolute entries to absolute repo-local paths.
    python - <<'PY'
import os
from pathlib import Path

import yaml

config_dir = Path(os.environ["LIBERO_CONFIG_PATH"]).expanduser().resolve()
config_file = config_dir / "config.yaml"
benchmark_root = (
    Path(os.environ["LIBERO_REPO_PATH"]).expanduser().resolve() / "libero" / "libero"
)

if not benchmark_root.is_dir():
    raise RuntimeError(
        f"LIBERO benchmark root does not exist: {benchmark_root}. "
        "Check LIBERO_REPO_PATH."
    )

config_dir.mkdir(parents=True, exist_ok=True)
cfg = {}
if config_file.exists():
    loaded = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise RuntimeError(
            f"Invalid LIBERO config format at {config_file}. Expected a YAML mapping."
        )
    cfg = loaded

defaults = {
    "benchmark_root": str(benchmark_root),
    "bddl_files": str(benchmark_root / "bddl_files"),
    "init_states": str(benchmark_root / "init_files"),
    "datasets": str((benchmark_root / ".." / "datasets").resolve()),
    "assets": str(benchmark_root / "assets"),
}

for key, absolute_path in defaults.items():
    current = cfg.get(key)
    if isinstance(current, str) and os.path.isabs(current):
        continue
    cfg[key] = absolute_path

config_file.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"[run_embodiment] Standard LIBERO config normalized: {config_file}")
PY
    # END LIBERO_STANDARD_PATH_NORMALIZATION_HOTFIX
fi

export PYTHONPATH=${LIBERO_REPO_PATH}:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

export RAY_DISABLE_IMPORT_WARNING=1
# export RAY_LOG_TO_STDERR=1
# export RAY_BACKEND_LOG_LEVEL=DEBUG 
export RAY_DISABLE_DASHBOARD=1
export PYTORCH_DISTRIBUTED_BACKEND=nccl
 
if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
    CONFIG_PATH="${EMBODIED_PATH}/config/"
else
    # Check if config name contains a path (subdirectory)
    if [[ "$1" == *"/"* ]]; then
        # Extract directory and config name
        CONFIG_DIR=$(dirname "$1")
        CONFIG_NAME=$(basename "$1")
        CONFIG_PATH="${EMBODIED_PATH}/config/${CONFIG_DIR}/"
        shift
    else
        # No subdirectory, use root config directory
        CONFIG_NAME=$1
        CONFIG_PATH="${EMBODIED_PATH}/config/"
	shift
    fi
fi

# Extract config tag from CONFIG_NAME
# If config ends with _openvlaoft or _eval, don't set CONFIG_TAG
# Otherwise, extract the part after the last _
CONFIG_TAG=""
if [[ ! "${CONFIG_NAME}" =~ _openvlaoft$ ]] && [[ ! "${CONFIG_NAME}" =~ _eval$ ]]; then
    if [[ "${CONFIG_NAME}" =~ _([^_/]+)$ ]]; then
        CONFIG_TAG="${BASH_REMATCH[1]}"
    fi
fi

# Build log directory path with config tag if present
# If LOG_DIR is already set (e.g., by crl_experiment scripts), use it as-is
# Otherwise, create default path with config tag if present
LIBERO_VARIANT_LOG_SUFFIX=""
if [ "${LIBERO_TYPE}" != "standard" ]; then
    LIBERO_VARIANT_LOG_SUFFIX="_${LIBERO_TYPE}"
    if [ -n "${LIBERO_SUFFIX}" ]; then
        SAFE_LIBERO_SUFFIX=$(echo "${LIBERO_SUFFIX}" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/_/g')
        LIBERO_VARIANT_LOG_SUFFIX="${LIBERO_VARIANT_LOG_SUFFIX}_${SAFE_LIBERO_SUFFIX}"
    fi
fi

if [ -z "${LOG_DIR}" ]; then
    if [ -n "${CONFIG_TAG}" ]; then
        LOG_DIR="${REPO_PATH}/logs_${CONFIG_TAG}${LIBERO_VARIANT_LOG_SUFFIX}/temp/run_$(date +'%Y%m%d-%H:%M:%S')"
    else
        LOG_DIR="${REPO_PATH}/logs${LIBERO_VARIANT_LOG_SUFFIX}/temp/run_$(date +'%Y%m%d-%H:%M:%S')"
    fi
fi
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${CONFIG_PATH} --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} $@" 
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
