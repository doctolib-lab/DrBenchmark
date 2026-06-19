#!/bin/bash
#SBATCH --job-name=drbenchmark_hpo_array
#SBATCH --output=logs/drbenchmark_hpo_array/%A_%a.out    # %A = array job ID, %a = array task ID
#SBATCH --error=logs/drbenchmark_hpo_array/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=ilr@h100
#SBATCH --array=1-17

set -x -e

MODEL_NAME=$1
N_RUNS=${2:-${N_RUNS:-5}}
BASE_SEED=${3:-${BASE_SEED:-42}}
HPO_RETRIES=${4:-${HPO_RETRIES:-3}}
HPO_RETRY_SLEEP=${5:-${HPO_RETRY_SLEEP:-10}}

if [ -z "$MODEL_NAME" ]; then
  echo "Usage: sbatch run_hpo_array.slurm <model_name> [n_runs] [base_seed] [hpo_retries] [hpo_retry_sleep]"
  exit 1
fi

if ! [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID (expected integer 1-17)."
  exit 1
fi

STAGE=$SLURM_ARRAY_TASK_ID
if [ "$STAGE" -lt 1 ] || [ "$STAGE" -gt 17 ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID (expected range 1-17)."
  exit 1
fi

echo "START TIME: $(date)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Selected stage: $STAGE, model=$MODEL_NAME, n_runs=$N_RUNS, base_seed=$BASE_SEED, hpo_retries=$HPO_RETRIES, hpo_retry_sleep=$HPO_RETRY_SLEEP"

# set up environment
module purge
module load arch/h100
# module load pytorch-gpu/py3/2.7.0
# pytorch 2.7.0 doesn't work anymore
module load pytorch-gpu/py3/2.8.0

mkdir -p logs/drbenchmark_hpo

# Force a single stage per array element
export STAGE_MODE=eq
export STAGE
export N_RUNS
export BASE_SEED
export HPO_RETRIES
export HPO_RETRY_SLEEP

# Run the main script
bash run_hpo.sh "$MODEL_NAME"

echo "END TIME: $(date)"
