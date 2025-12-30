#!/bin/bash
#
# Launch multiple evaluation jobs with different lora_scale coefficients
#
# Usage: ./launch_lora_scale_evals.sh CHECKPOINT_LOCATION LORA_SCALE_1 [LORA_SCALE_2 ...]
# Example: ./launch_lora_scale_evals.sh logs/bcrl_logit/0.3/task_0 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#
# This script will submit one SLURM job for each lora_scale coefficient provided

if [ $# -lt 2 ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 CHECKPOINT_LOCATION LORA_SCALE_1 [LORA_SCALE_2 ...]"
    echo "Example: $0 logs/bcrl_logit/0.3/task_0 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
    exit 1
fi

CHECKPOINT_LOCATION=$1
shift
LORA_SCALES=("$@")

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
SLURM_SCRIPT="${SCRIPT_DIR}/eval_embodiment_lora_scale.slurm"

# Verify the slurm script exists
if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "ERROR: SLURM script not found at $SLURM_SCRIPT"
    exit 1
fi

# Validate all lora_scale values
for scale in "${LORA_SCALES[@]}"; do
    if ! [[ "$scale" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "ERROR: Invalid lora_scale value: $scale (must be a number)"
        exit 1
    fi
    # Use awk for floating point comparison (more portable than bc)
    if ! awk "BEGIN {exit !($scale >= 0 && $scale <= 1)}"; then
        echo "ERROR: Invalid lora_scale value: $scale (must be between 0.0 and 1.0)"
        exit 1
    fi
done

echo "=========================================="
echo "Launching LoRA Scale Evaluation Jobs"
echo "=========================================="
echo "Checkpoint Location: $CHECKPOINT_LOCATION"
echo "LoRA Scales: ${LORA_SCALES[*]}"
echo "Number of jobs: ${#LORA_SCALES[@]}"
echo "=========================================="
echo ""

# Submit jobs
JOB_IDS=()
for scale in "${LORA_SCALES[@]}"; do
    echo "Submitting job for lora_scale=$scale..."
    JOB_OUTPUT=$(sbatch "$SLURM_SCRIPT" "$CHECKPOINT_LOCATION" "$scale" 2>&1)
    
    if [ $? -eq 0 ]; then
        # Extract job ID from output (format: "Submitted batch job 12345")
        JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+$')
        JOB_IDS+=("$JOB_ID")
        echo "  ✓ Job submitted successfully (Job ID: $JOB_ID)"
    else
        echo "  ✗ Failed to submit job for lora_scale=$scale"
        echo "  Error: $JOB_OUTPUT"
    fi
    echo ""
done

echo "=========================================="
echo "Job Submission Summary"
echo "=========================================="
echo "Total jobs submitted: ${#JOB_IDS[@]}"
if [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "Monitor jobs with: squeue -u \$USER"
    echo "Cancel all jobs with: scancel ${JOB_IDS[*]}"
fi
echo "=========================================="
