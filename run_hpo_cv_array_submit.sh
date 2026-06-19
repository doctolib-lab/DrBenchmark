#!/bin/bash
# Helper script to submit the run_hpo_cv_array.slurm job with optional throttling

set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
slurm_script="$script_dir/run_hpo_cv_array.slurm"

task_count=17   # stages 1-17 correspond to dataset tasks (stage 0 is prep)
max_concurrent=${MAX_CONCURRENT:-17}

debug=0
[ "${DEBUG:-0}" = "1" ] && debug=1

if [ "$1" = "--debug" ]; then
  debug=1
  shift
fi

model_name=$1
n_folds=${2:-5}

if [ -z "$model_name" ]; then
  echo "Usage: $0 [--debug] <model_name> [n_folds]"
  echo "  Stage 0 (prep: datasets + model download) is not part of the array; run manually if needed."
  exit 1
fi

if [ "$max_concurrent" -lt 1 ]; then
  echo "Invalid MAX_CONCURRENT=$max_concurrent (must be >=1)."
  exit 1
fi

if [ "$debug" -eq 1 ]; then
  array_spec="3"
  echo "DEBUG mode enabled: submitting only stage 2 (CLISTER)."
else
  if [ "$max_concurrent" -ge "$task_count" ]; then
    array_spec="1-$task_count"
  else
    array_spec="1-$task_count%$max_concurrent"
  fi
fi


echo "Submitting array job: $slurm_script"
echo "  array_spec=$array_spec, max_concurrent=$max_concurrent"
echo "  model_name=$model_name, n_folds=$n_folds"
sbatch --array="$array_spec" "$slurm_script" "$model_name" "$n_folds"

echo "Array job submitted. Monitor with: squeue -u $USER"
