#!/usr/bin/env bash
# Usage: run_recipe.sh <config.yaml> <model>
# One HPO pass (retried), then N_RUNS-1 seed passes reusing the best hyperparameters.

CONFIG="$1"
MODEL="$2"
N_RUNS="${N_RUNS:-5}"
BASE_SEED="${BASE_SEED:-42}"
HPO_RETRIES="${HPO_RETRIES:-3}"
HPO_RETRY_SLEEP="${HPO_RETRY_SLEEP:-10}"
PYTHON="${PYTHON:-python}"
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run.py"

if [ -z "$CONFIG" ] || [ -z "$MODEL" ]; then
  echo "usage: run_recipe.sh <config.yaml> <model>"
  exit 1
fi

attempt=1
while true; do
  "$PYTHON" "$RUN" --config="$CONFIG" --model="$MODEL"
  status=$?
  if [ "$status" -eq 0 ]; then
    break
  fi
  if [ "$attempt" -ge "$HPO_RETRIES" ]; then
    echo "HPO failed after ${attempt} attempt(s) (exit $status); aborting before training."
    exit "$status"
  fi
  echo "HPO failed (exit $status); retrying $((attempt + 1))/${HPO_RETRIES} in ${HPO_RETRY_SLEEP}s..."
  sleep "$HPO_RETRY_SLEEP"
  attempt=$((attempt + 1))
done

for ((run = 1; run < N_RUNS; run++)); do
  "$PYTHON" "$RUN" --config="$CONFIG" --model="$MODEL" --seed=$((BASE_SEED + run))
done
