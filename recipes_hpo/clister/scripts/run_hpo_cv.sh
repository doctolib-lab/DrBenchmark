#!/usr/bin/env bash
# Apache 2.0

MODEL="$1"
NUM_RUNS="${N_RUNS:-5}"
BASE_SEED="${BASE_SEED:-42}"
HPO_RETRIES="${HPO_RETRIES:-3}"
HPO_RETRY_SLEEP="${HPO_RETRY_SLEEP:-10}"

# First pass runs HPO (or reuses an existing best config if already computed)
attempt=1
while true; do
  python finetuning_bert_regr_hpo_cv_v2.py --config="../yaml/regr_hpo.yaml" --model="$MODEL"
  status=$?
  if [ "$status" -eq 0 ]; then
    break
  fi
  if [ "$attempt" -ge "$HPO_RETRIES" ]; then
    echo "HPO failed after ${attempt} attempt(s) (exit $status); aborting before training."
    exit "$status"
  fi
  echo "HPO failed (exit $status); retrying attempt $((attempt+1))/${HPO_RETRIES} after ${HPO_RETRY_SLEEP}s..."
  sleep "$HPO_RETRY_SLEEP"
  attempt=$((attempt + 1))
done

# Then repeat final training/eval NUM_RUNS-1 times using the best hyperparameters
for ((run=1; run<NUM_RUNS; run++)); do
  SEED=$((BASE_SEED + run))
  python finetuning_bert_regr_hpo_cv_v2.py --config="../yaml/regr_hpo.yaml" --model="$MODEL" --seed="$SEED"
done
