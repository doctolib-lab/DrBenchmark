#!/usr/bin/env bash
# Usage: run_bench.sh <model> [lang|all]
# The recipe tree is the task manifest: every recipes_hpo/<lang>/<corpus>/*_hpo.yaml is a task.
# Set TASK_INDEX=<n> to run only the nth config (SLURM array element).
# Set DEBUG=1 for a smoke run: one HPO trial, one seed, run JSONs kept in runs/debug/.

set -u

MODEL="${1:-}"
SELECTION="${2:-all}"
LANGS="fr es de en nl"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$MODEL" ]; then
  echo "usage: run_bench.sh <model> [lang|all]"
  exit 1
fi

export DISABLE_MLFLOW_INTEGRATION=TRUE
export DISABLE_TENSORBOARD_INTEGRATION=TRUE
export RAY_DISABLE_DASHBOARD=1
export RAY_USAGE_STATS_ENABLED=0
export TOKENIZERS_PARALLELISM=false

[ "$SELECTION" = "all" ] || LANGS="$SELECTION"

configs=""
for lang in $LANGS; do
  for config in "$ROOT"/recipes_hpo/"$lang"/*/*_hpo.yaml; do
    [ -f "$config" ] && configs="$configs$config"$'\n'
  done
done
configs=$(printf '%s' "$configs" | sort)

if [ -z "$configs" ]; then
  echo "no config found for: $LANGS"
  exit 1
fi

echo "$configs" | nl -w2 -s'  '

index=0
while IFS= read -r config; do
  index=$((index + 1))
  if [ -n "${TASK_INDEX:-}" ] && [ "$TASK_INDEX" != "$index" ]; then
    continue
  fi
  echo ">>> [$index] $config"
  bash "$ROOT/recipes_hpo/core/run_recipe.sh" "$config" "$MODEL" || exit $?
done <<< "$configs"
