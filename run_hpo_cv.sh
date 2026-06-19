#!/bin/bash
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup /bin/bash run_hpo_cv.sh <model> [stage] [stage_mode] [n_folds] &
#   stage_mode: "le" (default, run all stages <= stage) or "eq" (run only stage)
#   stage: stage number to run from/at (default: 1, prep is stage 0, tasks are 1-17)
model_name=$1
stage=${2:-${STAGE:-1}}
stage_mode=${3:-${STAGE_MODE:-le}}
nfolds=${4:-${N_FOLDS:-5}}

if [ -z "$model_name" ]; then
  echo "Usage: $0 <model_name> [stage] [stage_mode] [n_folds]"
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
#  pushd recipes_hpo_cv/cas/scripts/
#  for fold in `seq 1 1 $nfolds`
#  do
#    echo Corpus CAS Task
#    echo Starting fold ${fold}
#    bash ./run_task_1_hpo_cv.sh "${model_name}" "${fold}"
#  done
#  popd
# fi

# Corpus CLISTER Regression
if compare_stage 2; then
 pushd recipes_hpo_cv/clister/scripts/
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus CLISTER Task
     echo Starting fold ${fold}
     bash ./run_hpo_cv.sh "${model_name}" "${fold}"
 done
 popd
fi

# Corpus Diamed CLS
if compare_stage 3; then
 pushd recipes_hpo_cv/diamed/scripts/
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus Diamed Task
     echo Starting fold ${fold}
     bash ./run_hpo_cv.sh "${model_name}" "${fold}"
 done
 popd
fi

# Corpus E3C French_clinical
if compare_stage 4; then
 pushd recipes_hpo_cv/e3c/scripts/
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus E3C Task French_clinical
     echo Starting fold ${fold}
     bash ./run_hpo_cv.sh "${model_name}" French_clinical "${fold}"
 done
 popd
fi

# Corpus E3C French_temporal
if compare_stage 5; then
 pushd recipes_hpo_cv/e3c/scripts/
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus E3C Task French_temporal
     echo Starting fold ${fold}
     bash ./run_hpo_cv.sh "${model_name}" French_temporal "${fold}"
 done
 popd
fi

# # Corpus ESSAI
# if compare_stage 6; then
#   pushd recipes_hpo_cv/essai/scripts/
#   for fold in `seq 1 1 $nfolds`
#   do
#       echo Corpus ESSAI
#       echo Starting fold ${fold}
#       bash ./run_task_1_hpo_cv.sh "${model_name}" "${fold}"
#   done
#   popd
# fi

# # Corpus FrenchMedMCQA CLS
# if compare_stage 7; then
#  pushd recipes_hpo_cv/frenchmedmcqa/scripts/
#  for fold in `seq 1 1 $nfolds`
#  do
#      echo Corpus FrenchMedMCQA CLS
#      echo Starting fold ${fold}
#      bash ./run_task_2_hpo_cv.sh "${model_name}" "${fold}"
#  done
#  popd
# fi

# Corpus MantraGSC
if compare_stage 8; then
 pushd recipes_hpo_cv/mantragsc/scripts/
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus MantraGSC
     echo Starting fold ${fold}
     bash ./run_hpo_cv.sh "${model_name}" all "${fold}"
 done
 popd
fi

# Corpus Morfitt
if compare_stage 9; then
 pushd recipes_hpo_cv/morfitt/scripts/
 for fold in `seq 1 1 $nfolds`
 do
   echo Corpus Morfitt
   echo Starting fold ${fold}
   bash ./run_hpo_cv.sh "${model_name}" "${fold}"
 done
 popd
fi

# Corpus PXCorpus NER
if compare_stage 10; then
 pushd recipes_hpo_cv/pxcorpus/scripts/
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus PXCorpus NER
     echo Starting fold ${fold}
     bash ./run_task_1_hpo_cv.sh "${model_name}" "${fold}"
 done
 popd
fi

# Corpus PXCorpus CLS
if compare_stage 11; then
 pushd recipes_hpo_cv/pxcorpus/scripts/
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus PXCorpus CLS
     echo Starting fold ${fold}
     bash ./run_task_2_hpo_cv.sh "${model_name}" "${fold}"
 done
 popd
fi

# Corpus QUAERO EMEA
if compare_stage 12; then
 pushd recipes_hpo_cv/quaero/scripts/
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus QUAERO EMEA
     echo Starting fold ${fold}
     bash ./run_hpo_cv.sh "${model_name}" emea "${fold}"
 done
 popd
fi

# Corpus QUAERO MEDLINE
if compare_stage 13; then
 pushd recipes_hpo_cv/quaero/scripts/
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus QUAERO MEDLINE
     echo Starting fold ${fold}
     bash ./run_hpo_cv.sh "${model_name}" medline "${fold}"
 done
 popd
fi

# Corpus DEFT2020 Task 1 Regression
if compare_stage 14; then
 pushd recipes_hpo_cv/deft2020/scripts/
 for fold in `seq 1 1 $nfolds`; do
     echo Corpus DEFT2020 Task 1
     echo Starting fold ${fold}
     bash ./run_task_1_hpo_cv.sh "${model_name}" "${fold}"
 done
 popd
fi

# Corpus DEFT2020 Task 2 CLS
if compare_stage 15; then
 pushd recipes_hpo_cv/deft2020/scripts/
 for fold in `seq 1 1 $nfolds`; do
     echo Corpus DEFT2020 Task 2
     echo Starting fold ${fold}
     bash ./run_task_2_hpo_cv.sh "${model_name}" "${fold}"
 done
 popd
fi

# Corpus DEFT2021 Task 1 NER
if compare_stage 16; then
 pushd recipes_hpo_cv/deft2021/scripts/
 for fold in `seq 1 1 $nfolds`; do
     echo Corpus DEFT2021 Task 1
     echo Starting fold ${fold}
     bash ./run_task_1_hpo_cv.sh "${model_name}" "${fold}"
 done
 popd
fi

# Corpus DEFT2021 Task 2 CLS
if compare_stage 17; then
 pushd recipes_hpo_cv/deft2021/scripts/
 for fold in `seq 1 1 $nfolds`; do
     echo Corpus DEFT2021 Task 2
     echo Starting fold ${fold}
     bash ./run_task_2_hpo_cv.sh "${model_name}" "${fold}"
 done
 popd
fi
