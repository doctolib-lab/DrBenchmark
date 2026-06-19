#!/bin/bash
# Helper script to submit the run_hpo_array.slurm job with optional throttling

set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
slurm_script="$script_dir/run_hpo_array.slurm"

task_count=17   # stages 1-17 correspond to dataset tasks (stage 0 is prep)
max_concurrent=${MAX_CONCURRENT:-10}

debug=0
[ "${DEBUG:-0}" = "1" ] && debug=1

while [[ "$1" == --* ]]; do
  case "$1" in
    --debug) debug=1; shift ;;
    --after)
      dep_job_id=$2
      shift 2
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

model_name=$1
n_runs=${2:-5}
base_seed=${3:-42}
hpo_retries=${4:-${HPO_RETRIES:-3}}
hpo_retry_sleep=${5:-${HPO_RETRY_SLEEP:-10}}

if [ -z "$model_name" ]; then
  echo "Usage: $0 [--debug] [--after <job_id>] <model_name> [n_runs] [base_seed] [hpo_retries] [hpo_retry_sleep]"
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
  array_spec="1-$task_count%$max_concurrent"
fi

# tmp
# array_spec="2"

dep_arg=""
if [ -n "$dep_job_id" ]; then
  dep_arg="--dependency=afterany:$dep_job_id"
fi

echo "Submitting array job: $slurm_script"
echo "  array_spec=$array_spec, max_concurrent=$max_concurrent"
echo "  model_name=$model_name, n_runs=$n_runs, base_seed=$base_seed, hpo_retries=$hpo_retries, hpo_retry_sleep=$hpo_retry_sleep"
[ -n "$dep_arg" ] && echo "  dependency=$dep_arg"
sbatch $dep_arg --array="$array_spec" "$slurm_script" "$model_name" "$n_runs" "$base_seed" "$hpo_retries" "$hpo_retry_sleep"

echo "Array job submitted. Monitor with: squeue -u $USER"

