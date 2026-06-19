#!/bin/bash
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup /bin/bash run_hpo.sh <model> [stage] [stage_mode] &
#   stage_mode: "le" (default, run all stages <= stage) or "eq" (run only stage)
#   stage: stage number to run from/at (default: 1, prep is stage 0, tasks are 1-17)
model_name=$1
stage=${2:-${STAGE:-1}}
stage_mode=${3:-${STAGE_MODE:-le}}

# Set default values for HPO runs
export N_RUNS=${N_RUNS:-5}
export BASE_SEED=${BASE_SEED:-42}
export HPO_RETRIES=${HPO_RETRIES:-3}
export HPO_RETRY_SLEEP=${HPO_RETRY_SLEEP:-10}

if [ -z "$model_name" ]; then
  echo "Usage: $0 <model_name> [stage] [stage_mode]"
  exit 1
fi

compare_stage() {
  local current=$1
  if [ "$stage_mode" = "eq" ]; then
    [ "$stage" -eq "$current" ]
  else
    [ "$stage" -le "$current" ]
  fi
}

# Disable Ray dashboard in HPC environments to avoid startup errors/no GUI
export RAY_DISABLE_DASHBOARD=1
export RAY_USAGE_STATS_ENABLED=0
# Set Ray directory
export RAY_TMPDIR=/lustre/fsn1/projects/rech/ilr/commun/ray
export RAY_RESULTS_DIR=/lustre/fsn1/projects/rech/ilr/commun/ray_results
# Redirect Python tempfile (HF stages each HPO trial's checkpoint via tempfile.TemporaryDirectory
# → defaults to node-local /tmp, which overflows on large models: model+optimizer ≈ 4-5 GB → truncated
# write → "inline_container unexpected pos". JOBSCRATCH is the JZ per-job NVMe scratch; fsn1 is the fallback.
export TMPDIR=${JOBSCRATCH:-/lustre/fsn1/projects/rech/ilr/commun/tmp}
mkdir -p "$TMPDIR"
# Disable MLflow tracking
export DISABLE_MLFLOW_INTEGRATION=TRUE
# Disable TensorBoard tracking
export DISABLE_TENSORBOARD_INTEGRATION=TRUE

set -e

# Download datasets
if compare_stage 0; then
    python download_datasets_locally.py
fi

# Download models
if compare_stage 0; then
  echo "Downloading $model_name"
  model_name_dir=$(echo "$model_name" | tr '[:upper:]' '[:lower:]' | tr '/' '_')
  huggingface-cli download $model_name --local-dir ./models/$model_name_dir
  echo "Saving model in ./models/$model_name_dir"
fi

# # Corpus CAS Task 1 POS
# if compare_stage 1; then
#  pushd recipes_hpo/cas/scripts/
#    echo Corpus CAS Task
#  bash ./run_task_1_hpo.sh "${model_name}"
#  popd
# fi

# Corpus CLISTER Regression
if compare_stage 2; then
 pushd recipes_hpo/clister/scripts/
     echo Corpus CLISTER Task
 bash ./run_hpo.sh "${model_name}"
 popd
fi

# Corpus Diamed CLS
if compare_stage 3; then
 pushd recipes_hpo/diamed/scripts/
     echo Corpus Diamed Task
 bash ./run_hpo.sh "${model_name}"
 popd
fi

# Corpus E3C French_clinical
if compare_stage 4; then
 pushd recipes_hpo/e3c/scripts/
     echo Corpus E3C Task French_clinical
 bash ./run_hpo.sh "${model_name}" French_clinical
 popd
fi

# Corpus E3C French_temporal
if compare_stage 5; then
 pushd recipes_hpo/e3c/scripts/
     echo Corpus E3C Task French_temporal
 bash ./run_hpo.sh "${model_name}" French_temporal
 popd
fi

# # Corpus ESSAI
# if compare_stage 6; then
#   pushd recipes_hpo/essai/scripts/
#       echo Corpus ESSAI
#   bash ./run_task_1_hpo.sh "${model_name}"
#   popd
# fi

# # Corpus FrenchMedMCQA CLS
# if compare_stage 7; then
#  pushd recipes_hpo/frenchmedmcqa/scripts/
#  echo Corpus FrenchMedMCQA CLS
#  bash ./run_task_2_hpo.sh "${model_name}"
#  popd
# fi

# Corpus MantraGSC
# if compare_stage 8; then
#  pushd recipes_hpo/mantragsc/scripts/
#      echo Corpus MantraGSC
#  bash ./run_hpo.sh "${model_name}" all
#  popd
# fi

# Corpus Morfitt
if compare_stage 9; then
 pushd recipes_hpo/morfitt/scripts/
   echo Corpus Morfitt
 bash ./run_hpo.sh "${model_name}"
 popd
fi

# Corpus PXCorpus NER
if compare_stage 10; then
 pushd recipes_hpo/pxcorpus/scripts/
     echo Corpus PXCorpus NER
 bash ./run_task_1_hpo.sh "${model_name}"
 popd
fi

# Corpus PXCorpus CLS
if compare_stage 11; then
 pushd recipes_hpo/pxcorpus/scripts/
     echo Corpus PXCorpus CLS
 bash ./run_task_2_hpo.sh "${model_name}"
 popd
fi

# Corpus QUAERO EMEA
if compare_stage 12; then
 pushd recipes_hpo/quaero/scripts/
 bash ./run_hpo.sh "${model_name}" emea
 popd
fi

# Corpus QUAERO MEDLINE
if compare_stage 13; then
 pushd recipes_hpo/quaero/scripts/
 bash ./run_hpo.sh "${model_name}" medline
 popd
fi

# Corpus DEFT2020 Task 1 Regression
if compare_stage 14; then
 pushd recipes_hpo/deft2020/scripts/
     echo Corpus DEFT2020 Task 1
 bash ./run_task_1_hpo.sh "${model_name}"
 popd
fi

# Corpus DEFT2020 Task 2 CLS
if compare_stage 15; then
 pushd recipes_hpo/deft2020/scripts/
     echo Corpus DEFT2020 Task 2
 bash ./run_task_2_hpo.sh "${model_name}"
 popd
fi

# Corpus DEFT2021 Task 1 NER
if compare_stage 16; then
 pushd recipes_hpo/deft2021/scripts/
     echo Corpus DEFT2021 Task 1
 bash ./run_task_1_hpo.sh "${model_name}"
 popd
fi

# Corpus DEFT2021 Task 2 CLS
if compare_stage 17; then
 pushd recipes_hpo/deft2021/scripts/
     echo Corpus DEFT2021 Task 2
 bash ./run_task_2_hpo.sh "${model_name}"
 popd
fi
