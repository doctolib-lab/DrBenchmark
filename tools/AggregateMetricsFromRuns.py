#!/usr/bin/env python3
"""
=============================================================================
AggregateMetricsFromRuns.py
=============================================================================

This script computes aggregation metrics (Z-Score, Min-Max, Rank, Bootstrap 
Pairwise Win Rate) directly on individual run results, rather than on 
pre-averaged means.

This approach is more statistically rigorous because:
  1. It preserves variance information from individual runs
  2. It allows computing proper confidence intervals on aggregated scores
  3. Bootstrap win probability gives statistically valid pairwise comparisons

Input:
  - results.json: Contains individual run results per model/task/metric
    Format: data[model][task][metric] = [run1_value, run2_value, ...]

Output:
  - LaTeX tables with aggregated metrics and confidence intervals
  - JSON file with detailed aggregation results

Usage:
  python AggregateMetricsFromRuns.py [run_label] [results_json_path]

Examples:
  python AggregateMetricsFromRuns.py recipes_hpo
  python AggregateMetricsFromRuns.py recipes_hpo ./stats/recipes_hpo/results.json
=============================================================================
"""

import os
import sys
import json
import statistics
import random
import numpy as np
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Any


# =============================================================================
# CONFIGURATION
# =============================================================================

# Take only first N runs
REQUIRED_NUM_RUNS = 5

# Random seed for reproducibility of bootstrap sampling
RANDOM_SEED = 42

# Number of bootstrap samples for pairwise win probability estimation
NUM_BOOTSTRAP_SAMPLES = 10000

# Confidence level for confidence intervals (e.g., 0.95 for 95% CI)
CONFIDENCE_LEVEL = 0.95


# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

# Optional arguments:
#   argv[1]: run label (e.g., "recipes" or "recipes_hpo")
#   argv[2]: explicit path to results JSON (individual runs, not averaged)
run_label = sys.argv[1] if len(sys.argv) > 1 else "recipes_hpo"

if len(sys.argv) > 2:
    results_file = sys.argv[2]
else:
    results_file = f"./stats/{run_label}/results.json"
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

print(f"Loading individual run results from: {results_file}")


# =============================================================================
# MODEL MAPPING
# =============================================================================
# Maps internal model paths to display names for tables

mapping = {
    # OSS models
    ####################################################################################################
    # french general
    # "flaubert_flaubert_base_uncased": "FlauBERT",
    "almanach_camembert-base": "CamemBERT",
    "almanach_camembert-large": "CamemBERT-Large",
    # "almanach_camemberta-base": "CamemBERTa",
    # "almanach_moderncamembert-cv2-base": "ModernCamemBERT",
    "almanach_moderncamembert-base": "ModernCamemBERT",
    # french medical
    "dr-bert_drbert-7gb": "DrBERT",  # good
    "dr-bert_drbert-7gb-large": "DrBERT-Large",
    # "dr-bert_drbert-4gb-cp-pubmedbert": "DrBERT CP PubMedBERT",
    "almanach_camembert-bio-base": "CamemBERT-BIO",
    "jknafou_transbert-bio-fr": "TransBERT-BIO",  # good
    "almanach_moderncamembert-bio-base": "ModernCamemBERT-Bio",
    "almanach_moderncamembert-bio-large": "ModernCamemBERT-Bio-Large",
    # english medical
    # "microsoft_biomednlp-pubmedbert-base-uncased-abstract-fulltext": "PubMedBERT",
    # "microsoft_biomednlp-pubmedbert-base-uncased-abstract": "OLD-PubMedBERT",
    "dmis-lab_biobert-v1.1": "Biobert-v1.1",
    "dmis-lab_biobert-large-cased-v1.1": "Biobert-Large-v1.1",
    "thomas-sounack_bioclinical-modernbert-base": "BioClinical-ModernBERT",
    "thomas-sounack_bioclinical-modernbert-large": "BioClinical-ModernBERT-Large",
    "almanach_modernbert-bio-base": "ModernBERT-Bio",
    "almanach_modernbert-bio-large": "ModernBERT-Bio-Large",
    #
    # # doctobert exp 1 (nachos)
    ####################################################################################################
    # # "doctobert_dynamic_mlm": "DoctoBERT-Nachos-Scratch",  # silu one
    # # "doctobert_phase_1": "DoctoBERT p1",
    # # "doctobert_phase_2": "DoctoBERT p2",
    # # "doctobert_phase_2_gelu": "DoctoBERT p2 gelu",
    # # "doctobert_dynamic_mlm_gelu": "DoctoBERT-Nachos-Scratch",  # good, gelu one
    # # "ct-moderncamembert-decay-dynamic": "DoctoBERT-Nachos-ModernCamemBERT",
    # # "ct-bio-clinical-modernbert-decay-dynamic": "DoctoBERT-Nachos-BioClinicalModernBERT",
    # # "cp-moderncamembert-base-p1-54b-dynamic": "ModernCamembert Short context",
    # # "doctobert-exp-nachos-base-pretrain_750_128b_512_toks-lr-decay_8B_dynamic_mlm_prob-hf": "DoctoBERT Short Context",
    # # doctobert exp 2 (fineweb2)
    # # "doctobert-exp-fineweb2-base-pretrain-30b": "DoctoBERT-FW2-Unfiltered",
    # # "doctobert-exp-fineweb2-base-pretrain-30b-edu2": "DoctoBERT-FW2-Edu2",
    # # "doctobert-exp-fineweb2-base-pretrain-30b-edu3": "DoctoBERT-FW2-Edu3",
    # # "doctobert-exp-fineweb2-base-pretrain-30b-clinical-edu2-ep9": "DoctoBERT-FW2-Edu2-BioCli",
    # # "doctobert-exp-fineweb2-base-pretrain-30b-clinical": "DoctoBERT-FW2-BioCli",  # good
    # # "doctobert-exp-fineweb2-base-pretrain-30b-clinical-cs": "DoctoBERT-BioCli-cs",
    # # "doctobert-exp-fineweb2-base-pretrain-15b-clinical-vocab50k-ep3": "DoctoBERT-FW2-BioCli-Vocab50k", # good
    # # doctobert exp 3 (final fr) v1: p1 on edu-2
    # # "doctobert-03-fr-base-pretrain-102b-hf-ep0-ba3923-rank0": "DoctoBERT-P1-EP0-BA3923",
    # # "doctobert-03-fr-base-pretrain-102b-hf-ep0-ba7843-rank0": "DoctoBERT-P1-EP0-BA7843",
    # # "doctobert-03-fr-base-pretrain-102b-hf-ep1-ba11234-rank0": "DoctoBERT-P1-EP1-BA11234",
    # # "doctobert-03-fr-base-pretrain-102b-hf-ep2-ba14731-rank0": "DoctoBERT-P1-EP2-BA14731",
    # # "doctobert-03-fr-base-pretrain-102b-hf-ep2-ba18799-rank0": "DoctoBERT-P1-EP2-BA18799",
    # # "doctobert-03-fr-base-pretrain-102b-hf-ep3-ba22825-rank0": "DoctoBERT-P1-EP3-BA22825",  # good
    # # "doctobert-03-fr-base-pretrain-102b-context-extension-22b-hf-ep1-ba4728-rank0": "DoctoBERT-P2-EP1-BA4728",  # good
    # # "doctobert-03-fr-base-pretrain-102b-context-extension-22b-lr-decay-32b-dynamic-30-15-hf-ep2-ba6783-rank0": "DoctoBERT-P3-EP2-BA6783",
    # "doctobert-03-fr-base-pretrain-102b-context-extension-22b-lr-decay-32b-dynamic-30-15-hf-ep2-ba6783-rank0": "DoctoBERT-v1-P3",  # good
    # # doctobert exp 3 (final fr) v1.1: p1 on edu-2 + selected domains
    # # "doctobert-03-fr-base-pretrain-58b-hf-ep3-ba12930-rank0": "DoctoBERT-58B-P1-EP3-BA12930",
    # # "doctobert-03-fr-base-pretrain-66b-hf-ep0-ba2545-rank0": "DoctoBERT-66B-P1-EP0-BA2545",
    # # "doctobert-03-fr-base-pretrain-66b-hf-ep0-ba4876-rank0": "DoctoBERT-66B-P1-EP0-BA4876",
    # # "doctobert-03-fr-base-pretrain-66b-hf-ep1-ba7208-rank0": "DoctoBERT-66B-P1-EP1-BA7208",
    # # "doctobert-03-fr-base-pretrain-66b-hf-ep1-ba9327-rank0": "DoctoBERT-66B-P1-EP1-BA9327",
    # # "doctobert-03-fr-base-pretrain-66b-hf-ep2-ba12506-rank0": "DoctoBERT-66B-P1-EP2-BA12506",
    # # "doctobert-03-fr-base-pretrain-66b-hf-ep3-ba14625-rank0": "DoctoBERT-66B-P1-EP3-BA14625", # "p1-selected-66b",
    # # "doctobert-03-fr-base-pretrain-66b-hf-ep3-ba16472-rank0": "DoctoBERT-66B-P1-EP3-BA16472",
    # # "doctobert-03-fr-base-pretrain-66b-hf-ep3-ba18379-rank0": "DoctoBERT-66B-P1-EP3-BA18379",
    # # "doctobert-03-fr-base-pretrain-66b-hf-ep4-ba20075-rank0": "DoctoBERT-66B-P1-EP4-BA20075",
    # # "doctobert-03-fr-base-pretrain-66b-hf-ep4-ba21983-rank0": "DoctoBERT-66B-P1-EP4-BA21983", # "p1-selected-66b",
    # # "doctobert-03-fr-base-pretrain-66b-context-extension-16b-hf-ep2-ba3561-rank0": "DoctoBERT-66B-P2-EP2-BA3561",
    # # "doctobert-03-fr-base-pretrain-66b-context-extension-16b-lr-decay-32b-dynamic-30-15-hf-ep2-ba6783-rank0": "p3-edu4-selected-32b",
    # # "doctobert-fr-base-pretrain-66b-context-extension-16b-lr-decay-dynamic-30-15-edu5-selected-24b-hf-ep2-ba5087-rank0": "p3-edu5-selected-24b",
    # # "doctobert-fr-base-pretrain-66b-context-extension-16b-lr-decay-dynamic-30-15-edu4-biomedical-22b-hf-ep3-ba4664-rank0": "p3-edu4-biomedical-22b",
    # # "doctobert-fr-base-pretrain-66b-context-extension-16b-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "p3-edu4-clinical-21b",
    # # "doctobert-fr-base-pretrain-66b-context-extension-16b-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "DoctoBERT-P3-v1.1",  # good
    # # "doctobert-fr-base-pretrain-66b-context-extension-16b-lr-decay-dynamic-30-15-edu4-more-strict-selected-22b-hf-ep2-ba4664-rank0": "p3-edu4-more-strict-selected-22b",
    # # doctobert exp 3 (final fr) v1.2: p1 on edu-4 + selected domains
    # # "doctobert-fr-base-pretrain-52b-hf-ep1-ba4453-rank0": "DoctoBERT-52B-P1-EP1-BA4453",
    # # "doctobert-fr-base-pretrain-52b-hf-ep2-ba8268-rank0": "DoctoBERT-52B-P1-EP2-BA8268",
    # # "doctobert-fr-base-pretrain-52b-hf-ep2-ba11659-rank0": "DoctoBERT-52B-P1-EP2-BA11659", # "p1-selected-52b",
    # # only biomedical, clinical, biomedical+clinical
    # # "doctobert-fr-base-pretrain-biomedical-19b-hf-ep3-ba4241-rank0": "DoctoBERT-P1-Bio",
    # # "doctobert-fr-base-pretrain-biomedical-clinical-35b-hf-ep3-ba7843-rank0": "DoctoBERT-P1-BioCli",
    # # "doctobert-fr-base-pretrain-clinical-16b-hf-ep1-ba1485-rank0": "DoctoBERT-16B-P1-EP1-BA1485",
    # # "doctobert-fr-base-pretrain-clinical-16b-hf-ep2-ba2757-rank0": "DoctoBERT-16B-P1-EP2-BA2757",
    # # "doctobert-fr-base-pretrain-clinical-16b-hf-ep3-ba3605-rank0": "DoctoBERT-P1-Cli", # "p1-clinical-16b",
    # # "doctobert-fr-base-pretrain-clinical-16b-hf-ep4-ba5022-rank0": "DoctoBERT-16B-P1-EP4-BA5022",
    # # "doctobert-fr-base-pretrain-clinical-16b-hf-ep5-ba6294-rank0": "DoctoBERT-16B-P1-EP5-BA6294",
    # # "doctobert-fr-base-pretrain-clinical-16b-hf-ep6-ba7142-rank0": "DoctoBERT-16B-P1-EP6-BA7142",
    # # doctobert exp 3 (final fr) v1.3 (rework v1.1): p1 on edu-2 + selected domains
    # # "doctobert-fr-base-pretrain-100b-hf-ep1-ba5724-rank0": "DoctoBERT-100B-P1-EP1-BA5724",
    # # "doctobert-fr-base-pretrain-100b-hf-ep2-ba9963-rank0": "DoctoBERT-100B-P1-EP2-BA9963",
    # # "doctobert-fr-base-pretrain-100b-hf-ep3-ba14202-rank0": "DoctoBERT-100B-P1-EP3-BA14202",
    # # "doctobert-fr-base-pretrain-100b-hf-ep4-ba18440-rank0": "DoctoBERT-100B-P1-EP4-BA18440",
    # # "doctobert-fr-base-pretrain-100b-hf-ep4-ba21831-rank0": "DoctoBERT-100B-P1-EP4-BA21831",
    # # re-evaluate converted 66B intermediate checkpoints
    # # "doctobert-fr-base-pretrain-66b-ep0-ba4876-context-extension-8b-hf-ep1-ba1781-rank0": "66B-EP0-BA4876-P2-EP1-BA1781",
    # # "doctobert-fr-base-pretrain-66b-ep1-ba9327-context-extension-8b-hf-ep1-ba1781-rank0": "66B-EP1-BA9327-P2-EP1-BA1781",
    # # "doctobert-fr-base-pretrain-66b-ep3-ba14625-context-extension-8b-hf-ep1-ba1781-rank0": "66B-EP3-BA14625-P2-EP1-BA1781",
    # # p2
    # # "doctobert-fr-base-pretrain-100b-ep1-ba5724-context-extension-8b-hf-ep1-ba1781-rank0": "100B-EP1-BA5724-P2-EP1-BA1781",
    # # "doctobert-fr-base-pretrain-100b-ep2-ba9963-context-extension-8b-hf-ep1-ba1781-rank0": "100B-EP2-BA9963-P2-EP1-BA1781",
    # # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-hf-ep0-ba424-rank0": "100B-EP3-BA14202-P2-EP0-BA424",
    # # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-hf-ep0-ba848-rank0": "100B-EP3-BA14202-P2-EP0-BA848",
    # # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-hf-ep1-ba1781-rank0": "100B-EP3-BA14202-P2-EP1-BA1781",
    # # p3
    # # "doctobert-fr-base-pretrain-100b-ep1-ba5724-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "100B-BA5724-P2-BA1781-P3-BA4452",
    # # "doctobert-fr-base-pretrain-100b-ep2-ba9963-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "100B-BA9963-P2-BA1781-P3-BA4452",
    # # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep0-ba424-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "100B-BA14202-P2-BA424-P3-BA4452",
    # # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep0-ba848-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "100B-BA14202-P2-BA848-P3-BA4452",
    # # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep1-ba1696-rank0": "100B-BA14202-P2-BA1781-P3-BA1696",
    # # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep2-ba2968-rank0": "100B-BA14202-P2-BA1781-P3-BA2968",
    # # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "100B-BA14202-P2-BA1781-P3-BA4452",
    # # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "DoctoBERT-P3-Cli",  # good
    # # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "DoctoBERT-P3-v1.1-rw",  # good
    # # "doctobert-fr-base-pretrain-100b-ep3-ba14202-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "100B-BA14202-P3-Clinical",
    # # "doctobert-fr-base-pretrain-100b-ep3-ba14202-lr-decay-dynamic-30-15-edu4-more-strict-selected-33b-hf-ep3-ba6995-rank0": "100B-BA14202-P3-Strict",
    # # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-more-strict-selected-33b-hf-ep3-ba6995-rank0": "DoctoBERT-P3-Str",
    # # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-selected-48b-hf-ep3-ba10175-rank0": "DoctoBERT-P3-Sel",
    # # "doctobert-fr-base-pretrain-transbert-23b-hf-ep1-ba2333-rank0": "DoctoBERT-P1-TCB-BA2333",
    # # "doctobert-fr-base-pretrain-transbert-23b-hf-ep2-ba4029-rank0": "DoctoBERT-P1-TCB-BA4029",
    # # "doctobert-fr-base-pretrain-transbert-23b-hf-ep3-ba5512-rank0": "DoctoBERT-P1-TCB-BA5512",  # good
    # # "doctobert-fr-base-pretrain-transbert-23b-hf-ep4-ba7311-rank0": "DoctoBERT-P1-TCB-BA7311",
    # # "doctobert-fr-base-pretrain-transbert-23b-hf-ep5-ba9007-rank0": "DoctoBERT-P1-TCB-BA9007",
    # # "doctobert-fr-base-pretrain-transbert-23b-hf-ep6-ba10278-rank0": "DoctoBERT-P1-TCB-BA10278",
    # # "doctobert-fr-base-pretrain-transbert-23b-hf-ep7-ba11974-rank0": "DoctoBERT-P1-TCB-BA11974",
    # # "doctobert-fr-base-pretrain-transbert-23b-hf-ep8-ba13669-rank0": "DoctoBERT-P1-TCB-BA13669",
    # # "doctobert-fr-base-pretrain-transbert-23b-hf-ep9-ba15789-rank0": "DoctoBERT-P1-TCB-BA15789",  # good
    # # "doctobert-fr-base-pretrain-transbert-23b-bias-hf-ep3-ba5512-rank0": "DoctoBERT-P1-TCB-Bias",  # good
    # # "doctobert-fr-base-pretrain-transbert-23b-bias-cls-hf-ep3-ba5512-rank0": "DoctoBERT-P1-TCB-Bias-CLS",
    # # "doctobert-fr-base-pretrain-transbert-23b-bias-no-cls-hf-ep3-ba5512-rank0": "DoctoBERT-P1-TCB-Bias-CLS-No-Bias",  # good
    # # "doctobert-fr-base-pretrain-transbert-23b-full-attn-hf-ep3-ba5512-rank0": "DoctoBERT-P1-TCB-Full-Attn",
    # # "doctobert-fr-base-pretrain-transbert-23b-shallow-hf-ep3-ba5512-rank0": "DoctoBERT-P1-TCB-Shallow",
    # # "doctobert-fr-base-pretrain-transbert-70b-linear-hf-ep2-ba3392-rank0": "DoctoBERT-P1-TCB-Lin-BA3392",
    # # "doctobert-fr-base-pretrain-transbert-70b-linear-hf-ep4-ba6359-rank0": "DoctoBERT-P1-TCB-Lin-BA6359",
    # # "doctobert-fr-base-pretrain-transbert-70b-linear-hf-ep6-ba9326-rank0": "DoctoBERT-P1-TCB-Lin-BA9326",
    # # "doctobert-fr-base-pretrain-transbert-70b-linear-hf-ep8-ba12294-rank0": "DoctoBERT-P1-TCB-Lin-BA12294",
    # # "doctobert-fr-base-pretrain-transbert-70b-linear-hf-ep9-ba14837-rank0": "DoctoBERT-P1-TCB-Lin-BA14837",
    # # "doctobert-fr-base-pretrain-transbert-70b-linear-wo-pack-hf-ep9-ba46372-rank0": "DoctoBERT-P1-TCB-Lin-WoPack",  # good
    # # "doctobert-fr-base-pretrain-transbert-70b-linear-wo-pack-mask15-hf-ep9-ba46372-rank0": "DoctoBERT-P1-TCB-Lin-WoPack-Mask15",
    # # "doctobert-fr-base-pretrain-transbert-70b-linear-wo-pack-roberta-hf-ep10-ba47307-rank0": "DoctoBERT-P1-TCB-Lin-WoPack-RoBERTa",  # good
    #
    # # exp fineweb-2
    ####################################################################################################
    # # baseline
    # # "doctobert-exp-fineweb2-base-pretrain-nachos-hf-ep121-ba4367-rank0": "Nachos-Old",
    # "doctobert-exp-fineweb2-base-pretrain-nachos-new-hf-ep10-ba4368-rank0": "Nachos-1.3B-EP10",
    # "doctobert-exp-fineweb2-base-pretrain-transcorpus-hf-ep2-ba4368-rank0": "Transcorpus-5.2B-EP2",
    # # "doctobert-exp-fineweb2-base-pretrain-synthesized-hf-ep15-ba4368-rank0": "Synthesized-777M-EP15",
    # # filtering
    # "doctobert-exp-fineweb2-base-pretrain-unfiltered-hf-ep1-ba4368-rank0": "FW2-Unfiltered-7.2B-EP1",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-hf-ep11-ba4368-rank0": "FW2-Biocli-1.2B-EP11",
    # "doctobert-exp-fineweb2-base-pretrain-edu2-hf-ep3-ba4368-rank0": "FW2-Edu2-4.7B-EP3",
    # "doctobert-exp-fineweb2-base-pretrain-edu4-hf-ep8-ba4369-rank0": "FW2-Edu4-1.6B-EP8",
    # "doctobert-exp-fineweb2-base-pretrain-medterm01-hf-ep5-ba4368-rank0": "FW2-Medterm01-2.5B-EP5",
    # "doctobert-exp-fineweb2-base-pretrain-medterm02-hf-ep18-ba4368-rank0": "FW2-Medterm02-762M-EP18",
    # # filtering combined (intersection)
    # # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-hf-ep6-ba1401-rank0": "FW2-Biocli-Medterm01-702M-EP6",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-hf-ep20-ba4368-rank0": "FW2-Biocli-Medterm01-702M-EP20",  # good
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm02-hf-ep52-ba4368-rank0": "FW2-Biocli-Medterm02-264M-EP52",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-edu4-hf-ep24-ba4368-rank0": "FW2-Biocli-Edu4-587M-EP24",
    # "doctobert-exp-fineweb2-base-pretrain-medterm01-edu4-hf-ep15-ba4369-rank0": "FW2-Medterm01-Edu4-933M-EP15",
    # # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-100b-hf-ep100-ba21835-rank0": "FW2-Biocli-Medterm01-702M-EP100-100B",
    # # filtering combined (union)
    # # "doctobert-exp-fineweb2-base-pretrain-biocli-edu4-union-hf-ep6-ba4368-rank0": "FW2-Biocli-Edu4-Union-2.2B-EP6",
    # # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-union-hf-ep4-ba4368-rank0": "FW2-Biocli-Medterm01-Union-3.0B-EP4",
    # # "doctobert-exp-fineweb2-base-pretrain-medterm01-edu4-union-hf-ep4-ba4368-rank0": "FW2-Medterm01-Edu4-Union-3.2B-EP4",
    #
    # # rewritten
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-unrewritten-hf-ep3-ba4368-rank0": "Rewritten-Raw-3.6B-EP3",
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-hf-ep7-ba4368-rank0": "Rewritten-1.8B-EP7",
    # # # rewritten combined
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-medterm01-hf-ep8-ba4368-rank0": "Rewritten-Medterm01-1.5B-EP8",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-medterm02-hf-ep12-ba4368-rank0": "Rewritten-Medterm02-973M-EP12",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-biocli-hf-ep26-ba4368-rank0": "Rewritten-Biocli-475M-EP26",
    # # # rewritten by model
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-gptoss-hf-ep5-ba1093-rank0": "Rewritten-GPT-OSS-543M-EP5",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-medgemma-hf-ep5-ba1093-rank0": "Rewritten-Medgemma-587M-EP5",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-qwen3-30b-hf-ep11-ba1093-rank0": "Rewritten-Qwen3-30B-321M-EP11",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-qwen3-next-hf-ep10-ba1093-rank0": "Rewritten-Qwen3-Next-80B-300M-EP10",
    # # # filtering + rewritten
    # "doctobert-exp-fineweb2-base-pretrain-medterm02-plus-rewritten-hf-ep5-ba4368-rank0": "FW2-Medterm02-Plus-Rewritten-2.5B-EP5",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-hf-ep5-ba4368-rank0": "FW2-Biocli-Medterm01-Plus-Rewritten-2.5B-EP5",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-biocli-medterm01-hf-ep11-ba4368-rank0": "FW2-Biocli-Medterm01-Plus-Rewritten-Biocli-Medterm01-1.1B-EP11",
    # "doctobert-exp-fineweb2-base-pretrain-medterm01-edu4-plus-rewritten-hf-ep4-ba4368-rank0": "FW2-Medterm01-Edu4-Plus-Rewritten-2.6B-EP4",
    # # filtering + rewritten (longer training)
    # # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-40b-hf-ep10-ba8734-rank0": "FW2-Biocli-Medterm01-Plus-Rewritten-40B-2.5B-EP10",
    # # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-80b-hf-ep21-ba17467-rank0": "FW2-Biocli-Medterm01-Plus-Rewritten-80B-2.5B-EP21",
    # # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-biocli-medterm01-40b-hf-ep23-ba8734-rank0": "FW2-Biocli-Medterm01-Plus-Rewritten-Biocli-Medterm01-40B-1.1B-EP23",
    # # filtering + synth v2
    # "doctobert-exp-fineweb2-base-pretrain-raw-20b-1m-med01-edu4-plus-synth-v2-hf-ep14-ba4368-rank0": "Mixed-20B-RawMed01Edu4+SynthV2-830M-EP14",
    #
    # # Roberta
    # "doctobert-exp-fineweb2-base-pretrain-medterm02-plus-rewritten-roberta-hf-ep6-ba13337-rank0": "FW2-Medterm02-Plus-Rewritten-Roberta-2.5B-EP6",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-adamw-hf-ep28-ba11107-rank0": "FW2-Biocli-Medterm01-702M-EP28-Roberta-AdamW",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-hf-ep28-ba11107-rank0": "FW2-Biocli-Medterm01-702M-EP28-Roberta-StableAdamW",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-wd001-hf-ep28-ba11107-rank0": "FW2-Biocli-Medterm01-702M-EP28-Roberta-StableAdamW-WD001",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-roberta-hf-ep6-ba13379-rank0": "FW2-Biocli-Medterm01-Plus-Rewritten-2.5B-EP6-Roberta",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-100b-hf-ep37-ba14440-rank0": "FW2-Biocli-Medterm01-702M-EP37-Roberta-100B",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-100b-hf-ep72-ba27768-rank0": "FW2-Biocli-Medterm01-702M-EP72-Roberta-100B",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-100b-hf-ep107-ba41096-rank0": "FW2-Biocli-Medterm01-702M-EP107-Roberta-100B",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-100b-hf-ep144-ba55535-rank0": "FW2-Biocli-Medterm01-702M-EP144-Roberta-100B",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-200b-hf-ep289-ba111070-rank0": "FW2-Biocli-Medterm01-702M-EP289-Roberta-200B",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-200b-hf-ep289-ba111070-rank0": "Ablation-Roberta-FW2-Biocli-Medterm01-702M-EP289",
    # modernbert ablation
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-wo-bs-warmup-hf-ep20-ba4240-rank0": "FW2-Biocli-Medterm01-702M-EP20-NoBSWarmup",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-linear-hf-ep20-ba4368-rank0": "FW2-Biocli-Medterm01-702M-EP20-Linear",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-linear-15-hf-ep20-ba4368-rank0": "FW2-Biocli-Medterm01-702M-EP20-Linear-15",
    #
    # 05 exp fineweb-2 rewritten; rewriting v3/v4; 100k scale
    ####################################################################################################
    # # raw data
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-v3-raw-unfiltered-hf-ep19-ba220-rank0": "Raw-Unfiltered-38M-EP19",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v3-raw-hf-ep26-ba220-rank0": "Raw-28M-EP26",
    # # "doctobert-exp-fineweb2-base-pretrain-raw-100k-med01-edu4-hf-ep210-ba220-rank0": "Raw-Med01-Edu4-5M-EP210",  # filtered
    # # diff rewriting approaches
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v3-hf-ep52-ba220-rank0": "Rewritten-V3-13M-EP52",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v3-b-hf-ep52-ba220-rank0": "Rewritten-V3-13M-EP52-b",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v3-c-hf-ep52-ba220-rank0": "Rewritten-V3-13M-EP52-c",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v3-1-hf-ep53-ba221-rank0": "Rewritten-V3.1-11M-EP53",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v3-1-b-hf-ep53-ba220-rank0": "Rewritten-V3.1-11M-EP53-b",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v3-1-c-hf-ep53-ba220-rank0": "Rewritten-V3.1-11M-EP53-c",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v3-2-hf-ep53-ba220-rank0": "Rewritten-V3.2-11M-EP53",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v3-2-b-hf-ep53-ba220-rank0": "Rewritten-V3.2-11M-EP53-b",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v3-2-c-hf-ep53-ba220-rank0": "Rewritten-V3.2-11M-EP53-c",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-hf-ep35-ba220-rank0": "Rewritten-V4-22M-EP35",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-hf-ep52-ba220-rank0": "Rewritten-V4.1-14M-EP52",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-b-hf-ep52-ba220-rank0": "Rewritten-V4.1-14M-EP52-b",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-c-hf-ep52-ba220-rank0": "Rewritten-V4.1-14M-EP52-c",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-plus-dedup-hf-ep52-ba220-rank0": "Rewritten-V4.1.5-15M-EP52",
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-dedup-hf-ep52-ba220-rank0": "Rewritten-V4.2-15M-EP52",
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-mga-official-qwen3-5-35b-hf-ep13-ba220-rank0": "Rewritten-MGA-Official-50M-EP13",
    # # apply filtering to rewritten data
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-dedup-med01-hf-ep105-ba220-rank0": "Rewritten-V4.2-Med01-8M-EP105",
    # # multiple samplings
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v3-1-3seed-hf-ep17-ba220-rank0": "Rewritten-V3.1-3seed-36M-EP17",  # combine from 3 runs
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-hf-ep15-ba220-rank0": "Rewritten-V4.2-3Samplings-43M-EP15",
    # # multiple samplings + raw
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v3-1-3seed-plus-raw-hf-ep10-ba220-rank0": "Rewritten-V3.1-3seed-Plus-Raw-64M-EP10",
    # # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-3seed-plus-raw-hf-ep9-ba220-rank0": "Rewritten-V4.1-3seed-Plus-Raw-73M-EP9",
    # # rewritten + raw
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v3-1-plus-raw-hf-ep17-ba220-rank0": "Rewritten-V3.1-Plus-Raw-41M-EP17",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-plus-raw-hf-ep16-ba220-rank0": "Rewritten-V4.1-Plus-Raw-43M-EP16",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-dedup-plus-raw-hf-ep16-ba220-rank0": "Rewritten-V4.2-Plus-Raw-43M-EP16",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-dedup-plus-raw-unfiltered-hf-ep13-ba220-rank0": "Rewritten-V4.2-Plus-Raw-Unfiltered-54M-EP13",
    # # rewritten + raw-filtered
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v3-1-plus-raw-med01-edu4-hf-ep42-ba220-rank0": "Rewritten-V3.1-Plus-Raw-Med01-Edu4-17M-EP42",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-plus-raw-med01-edu4-hf-ep35-ba220-rank0": "Rewritten-V4.1-Plus-Raw-Med01-Edu4-20M-EP35",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-plus-dedup-plus-raw-med01-edu4-hf-ep35-ba220-rank0": "Rewritten-V4.1.5-Plus-Raw-Med01-Edu4-20M-EP35",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-dedup-plus-raw-med01-edu4-hf-ep35-ba220-rank0": "Rewritten-V4.2-Plus-Raw-Med01-Edu4-20M-EP35",
    # # ablation on rewriting model
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-3m1n-qwen3-5-35b-hf-ep52-ba220-rank0": "Rewritten-V4.2-Qwen3.5-35B-15M-EP52",  # rerun of prev run
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-3m1n-qwen3-5-122b-hf-ep42-ba220-rank0": "Rewritten-V4.2-Qwen3.5-122B-16M-EP42",
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-3m1n-medgemma-27b-hf-ep30-ba220-rank0": "Rewritten-V4.2-MedGemma-27B-21M-EP30",
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-3m1n-gemma-4-26b-hf-ep42-ba220-rank0": "Rewritten-V4.2-Gemma-4-26B-16M-EP42",
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-3m1n-gpt-oss-120b-hf-ep42-ba220-rank0": "Rewritten-V4.2-GPT-OSS-120B-17M-EP42",
    #
    # 05 exp fineweb-2 rewritten; 1M-doc V4.2 scale-up; 20B-token training budget
    ####################################################################################################
    # # raw
    # "doctobert-exp-fineweb2-base-pretrain-raw-20b-1m-unfiltered-hf-ep35-ba4368-rank0": "Raw-20B-Unfiltered-392M-EP35",
    # # "doctobert-exp-fineweb2-base-pretrain-raw-20b-1m-stage2-filtered-hf-ep46-ba4368-rank0": "Raw-20B-V4.2-Filtered-304M-EP46",
    # # "doctobert-exp-fineweb2-base-pretrain-raw-20b-1m-med01-edu4-hf-ep282-ba4368-rank0": "Raw-20B-Med01-Edu4-53M-EP282",
    # # rewritten
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen3-5-35b-hf-ep81-ba4368-rank0": "Rewritten-20B-V4.2-Qwen3.5-35B-158M-EP81",
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-gemma-4-26b-hf-ep83-ba4368-rank0": "Rewritten-20B-V4.2-Gemma-4-26B-158M-EP83",
    # # rewritten filtered
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen3-5-35b-med01-hf-ep146-ba4367-rank0": "Rewritten-20B-V4.2-Qwen3.5-35B-Med01-90M-EP146",
    # # rewritten multiple models
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-plus-gemma-hf-ep41-ba4368-rank0": "Rewritten-20B-V4.2-Qwen+Gemma-316M-EP41",
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-half-plus-gemma-half-hf-ep83-ba4368-rank0": "Rewritten-20B-V4.2-Qwen-Half+Gemma-Half-158M-EP83",
    # # rewritten + raw
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-plus-raw-unfiltered-hf-ep24-ba4368-rank0": "Rewritten-20B-V4.2-Qwen+Raw-Unfiltered-550M-EP24",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-plus-raw-stage2-filtered-hf-ep29-ba4368-rank0": "Rewritten-20B-V4.2-Qwen+Raw-Filtered-462M-EP29",
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-plus-raw-med01-edu4-hf-ep63-ba4368-rank0": "Rewritten-20B-V4.2-Qwen+Raw-Med01-Edu4-211M-EP63",
    # # rewritten multiple models + raw
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-plus-gemma-plus-raw-med01-edu4-hf-ep35-ba4368-rank0": "Rewritten-20B-V4.2-Qwen+Gemma+Raw-Med01-Edu4-369M-EP35",
    # # "doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-half-plus-gemma-half-plus-raw-med01-edu4-hf-ep63-ba4368-rank0": "Rewritten-20B-V4.2-Qwen-Half+Gemma-Half+Raw-Med01-Edu4-211M-EP63",
    # # rewritten + raw; diff ratio
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-plus-raw-med01-edu4-raw33-hf-ep86-ba4368-rank0": "Rewritten-20B-V4.2-Qwen+Raw-Med01-Edu4-1to2-157M-EP86",
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-plus-raw-med01-edu4-raw50-hf-ep132-ba4368-rank0": "Rewritten-20B-V4.2-Qwen+Raw-Med01-Edu4-1to1-105M-EP132",
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-plus-raw-med01-edu4-raw67-hf-ep179-ba4370-rank0": "Rewritten-20B-V4.2-Qwen+Raw-Med01-Edu4-2to1-79M-EP179",
    #
    # final — DoctoBERT FR v2 (07_fr_v2)
    ####################################################################################################
    # v1
    "doctobert-03-fr-base-pretrain-102b-context-extension-22b-lr-decay-32b-dynamic-30-15-hf-ep2-ba6783-rank0": "DB-v1-P3",  # good
    # roberta
    # ablation
    "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-200b-hf-ep289-ba111070-rank0": "Ablation-Roberta-FW2-Biocli-Medterm01-702M-EP289",
    "doctobert-exp-base-pretrain-roberta-fw2-med01-edu4-hf-ep249-ba93146-rank0": "Ablation-Roberta-FW2-Med01-Edu4-934M-EP249",
    "doctobert-exp-base-pretrain-roberta-fw2-med01-edu4-plus-rewritten-hf-ep794-ba136637-rank0": "Ablation-Roberta-FW2-Med01-Edu4+Rewritten-210M-EP794",
    # p1
    "doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-hf-ep32-ba92308-rank0": "DB-v2-RoBERTa-P1-200B-FineMed-Med01-Edu4-3.81B-EP32",
    "doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-hf-ep14-ba106710-rank0": "DB-v2-RoBERTa-P1-200B-FineMed-Med01-Edu4+Rewritten-8.32B-EP14",
    # p2 continual pretraining
    "doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-biocli-rwquality-hf-ep19-ba47690-rank0": "DB-v2-RoBERTa-P1-200B-P2-100B-FineMed+Rewritten-Med01-Edu4-BioCli-3.03B-EP19",
    "doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-biocli-rwquality-200b-hf-ep39-ba95379-rank0": "DB-v2-RoBERTa-P1-200B-P2-200B-FineMed+Rewritten-Med01-Edu4-BioCli-3.03B-EP39",
    "doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-biocli-plus-rewritten-biocli-hf-ep16-ba50085-rank0": "DB-v2-RoBERTa-P1-200B-P2-100B-FineMed-Med01-Edu4-BioCli+Rewritten-BioCli-3.56B-EP16",
    # longer training p1 500b + p2 500b/200b
    "doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-500b-new-hf-ep36-ba266775-rank0": "DB-v2-RoBERTa-P1-500B-FineMed+Rewritten-Med01-Edu4-8.32B-EP36",
    # "doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-biocli-rwquality-500b-cp-hf-ep47-ba114455-rank0": "DB-v2-RoBERTa-P1-500B-P2-300B-FineMed+Rewritten-Med01-Edu4-BioCli-3.03B-EP47",
    "doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-biocli-rwquality-200b-cp-hf-ep39-ba95379-rank0": "DB-v2-RoBERTa-P1-500B-P2-200B-FineMed+Rewritten-Med01-Edu4-BioCli-3.03B-EP39",  # final
    # longer training p1 1t + p2 2OOb
    "doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-1t-lr1e4-hf-ep73-ba533549-rank0": "DB-v2-RoBERTa-P1-1T-FineMed+Rewritten-Med01-Edu4-8.32B-EP73",
    # modernbert
    # p1 pretraining
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-hf-ep16-ba21833-rank0": "DB-v2-P1-FineMed-Med01-Edu4-3.81B-EP16",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-hf-ep7-ba21832-rank0": "DB-v2-P1-FineMed-Med01-Edu4+Rewritten-8.32B-EP7",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-plus-n-t-hf-ep5-ba21832-rank0": "DB-v2-P1-FineMed-Med01-Edu4+Rewritten+NACHOS+Transcorpus-12.86B-EP5",
    # p1 (200B extensions)
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-200b-hf-ep33-ba43665-rank0": "DB-v2-P1-FineMed-Med01-Edu4-200B-3.81B-EP33",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-hf-ep15-ba43663-rank0": "DB-v2-P1-FineMed-Med01-Edu4+Rewritten-200B-8.32B-EP15",
    # p2 context-extension
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-hf-ep5-ba4242-rank0": "DB-v2-P2-FineMed-Med01-Edu4+Rewritten-2.53B-EP5",
    # p3 annealing
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-hf-ep3-ba4241-rank0": "DB-v2-P3-FineMed-Med01-Edu4-3.81B-EP3",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-plus-rewritten-hf-ep1-ba4240-rank0": "DB-v2-P3-FineMed-Med01-Edu4+Rewritten-8.32B-EP1",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-plus-rewritten-biocli-hf-ep4-ba4240-rank0": "DB-v2-P3-FineMed+Rewritten-Med01-Edu4-BioCli-3.03B-EP4",  # final
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-plus-rewritten-biocli-dyn-mlm-hf-ep4-ba4240-rank0": "DB-v2-P3-FineMed+Rewritten-Med01-Edu4-BioCli-DynMLM-3.03B-EP4",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med02-edu4-plus-rewritten-rwquality-hf-ep5-ba4240-rank0": "DB-v2-P3-FineMed+Rewritten-Med02-Edu4-2.54B-EP5",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-biocli-plus-rewritten-biocli-hf-ep3-ba4240-rank0": "DB-v2-P3-FineMed-Med01-Edu4-BioCli+Rewritten-BioCli-3.56B-EP3",
    # p3 100B / 50B on FineMed+Rewritten-Med01-Edu4-BioCli
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-100b-med01-edu4-plus-rewritten-biocli-hf-ep21-ba21200-rank0": "DB-v2-P3-FineMed+Rewritten-Med01-Edu4-BioCli-100B-3.03B-EP21",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-50b-med01-edu4-plus-rewritten-biocli-hf-ep10-ba10600-rank0": "DB-v2-P3-FineMed+Rewritten-Med01-Edu4-BioCli-50B-3.03B-EP10",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-50b-med01-edu4-plus-rewritten-biocli-mlm03-hf-ep10-ba10600-rank0": "DB-v2-P3-FineMed+Rewritten-Med01-Edu4-BioCli-MLM03-50B-3.03B-EP10",
    # p1/p2/p3 with lr decay and rewarmup
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-hf-ep15-ba43663-rank0": "DB-v2-P1-Rewarmup-FineMed-Med01-Edu4+Rewritten-200B-8.32B-EP15",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-p2-20b-hf-ep5-ba4242-rank0": "DB-v2-P2-Rewarmup-FineMed-Med01-Edu4+Rewritten-20B-2.53B-EP5",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-p2-20b-p3-exp-50b-med01-edu4-plus-rewritten-biocli-hf-ep4-ba4240-rank0": "DB-v2-P3-Rewarmup-FineMed+Rewritten-Med01-Edu4-BioCli-3.03B-EP4",
    # p1/p2/p3 share lr decay
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-126b-p2-20b-hf-ep5-ba4242-rank0": "DB-v2-P2-ContDecay-FineMed+Rewritten-Med01-Edu4-20B-2.53B-EP5",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-126b-p2-20b-p3-general-54b-resume2-hf-ep2-ba6359-rank0": "DB-v2-P3-ContDecay-FineMed+Rewritten-Med01-Edu4-General-8.32B-EP2",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-126b-p2-20b-p3-biocli-54b-resume2-hf-ep2-ba2544-rank0": "DB-v2-P3-ContDecay-FineMed+Rewritten-Med01-Edu4-BioCli-3.03B-EP2",
    # shorter p1/p2, longer p3
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-hf-ep3-ba9115-rank0": "DB-v2-P1-LongerP3-FineMed-Med01-Edu4+Rewritten-42B-8.32B-EP3",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-p2-10b-hf-ep2-ba2121-rank0": "DB-v2-P2-LongerP3-FineMed-Med01-Edu4+Rewritten-10B-2.53B-EP2",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-p2-10b-p3-general-100b-hf-ep7-ba21197-rank0": "DB-v2-P3-LongerP3-General-FineMed+Rewritten-100B-8.32B-EP7",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-p2-10b-p3-biocli-100b-hf-ep21-ba21200-rank0": "DB-v2-P3-LongerP3-BioCli-FineMed+Rewritten-100B-3.03B-EP21",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-p2-10b-p3-biocli-20b-hf-ep4-ba4240-rank0": "DB-v2-P3-LongerP3-BioCli-FineMed+Rewritten-20B-3.03B-EP4",
    # long-context p1/p2, lr decayed from 8e-4 -> 3e-4 -> 0
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-3e4-hf-ep15-ba44288-rank0": "DB-v2-P1-8192FS-Decay3e4-FineMed-Med01-Edu4+Rewritten-200B-8.32B-EP15",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-3e4-biocli-20b-hf-ep4-ba4240-rank0": "DB-v2-P2-8192FS-Decay3e4-BioCli-20B-3.03B-EP4",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-3e4-biocli-50b-hf-ep10-ba10600-rank0": "DB-v2-P2-8192FS-Decay3e4-BioCli-50B-3.03B-EP10",
    # long-context p1/p2, lr decayed from 8e-4 -> 0
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-0-hf-ep15-ba43665-rank0": "DB-v2-P1-8192FS-Decay0-FineMed-Med01-Edu4+Rewritten-200B-8.32B-EP15",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-0-biocli-br150b-2e4-hf-ep4-ba4240-rank0": "DB-v2-P2-8192FS-Decay0-BioCli-20B-3.03B-EP4",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-0-biocli-br150b-2e4-50b-hf-ep10-ba10600-rank0": "DB-v2-P2-8192FS-Decay0-BioCli-50B-3.03B-EP10",
    # large model
    "doctobert-fr-v2-large-pretrain-finemed-med01-edu4-plus-rewritten-hf-ep7-ba22458-rank0": "DB-v2-Large-P1-FineMed-Med01-Edu4+Rewritten-100B-8.32B-EP7",
    "doctobert-fr-v2-large-pretrain-finemed-med01-edu4-plus-rewritten-p2-20b-hf-ep5-ba3638-rank0": "DB-v2-Large-P2-FineMed-Med01-Edu4+Rewritten-20B-2.53B-EP5",
    "doctobert-fr-v2-large-pretrain-finemed-med01-edu4-plus-rewritten-p2-20b-p3-biocli-20b-hf-ep4-ba3636-rank0": "DB-v2-Large-P3-BioCli-20B-3.03B-EP4",
    # base P3 short-doc (1024-split) biocli — rework of DB-v2-P3-…-BioCli-3.03B-EP4 (0.680)
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-plus-rewritten-biocli-shortdoc1024-hf-ep4-ba4240-rank0": "DB-v2-P3-FineMed+Rewritten-Med01-Edu4-BioCli-ShortDoc1024-3.03B-EP4",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-plus-rewritten-biocli-mixdoc1024doc8192-hf-ep2-ba4240-rank0": "DB-v2-P3-FineMed+Rewritten-Med01-Edu4-BioCli-MixDoc-6.05B-EP2",
    "doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-50b-med01-edu4-plus-rewritten-biocli-mixdoc1024doc8192-hf-ep5-ba10599-rank0": "DB-v2-P3-FineMed+Rewritten-Med01-Edu4-BioCli-MixDoc-6.05B-EP5",
}


# =============================================================================
# TASK CONFIGURATION
# =============================================================================

# List of tasks to include in aggregation
# Format: "corpus|task_type|fewshot"
tasks = [
    # "mantragsc|ner|1.0",
    'quaero|ner-emea|1.0',
    'quaero|ner-medline|1.0',
    'e3c|ner-clinical|1.0',
    'e3c|ner-temporal|1.0',
    'morfitt|cls|1.0',
    'clister|regr|1.0',
    'deft2020|regr|1.0',
    'deft2020|cls|1.0',
    'deft2021|cls|1.0',
    'deft2021|ner|1.0',
    'diamed|cls|1.0',
    'pxcorpus|ner|1.0',
    'pxcorpus|cls|1.0',
]

# Optional: Filter to specific task prefixes for aggregation
# Set to None to include all tasks
# tasks_filter = None
# 8 selected tasks (no low snr + low corr, no regr)
# tasks_filter = [
#     # 'mantragsc',  # didn't run for all models
#     'quaero',
#     'e3c',
#     'morfitt',
#     # 'clister',  # regr task
#     # 'deft2020',
#     # 'deft2020|regr', # regr task
#     # 'deft2020|cls', # low snr + low corr
#     'deft2021',
#     'diamed',
#     # 'pxcorpus', # low snr + low corr
# ]
# 6 tasks correlated with quaero|ner-medline
# tasks_filter = [
#     'quaero|ner-medline',
#     'e3c',
#     'morfitt',
#     'diamed',
#     'deft2021|ner',
# ]
tasks_filter = [
    'quaero',
    'e3c',
    'morfitt',
    'diamed',
    'deft2021|ner',
    #'deft2021|cls',  # incoherent with others
    # 'clister|regr',
    # 'deft2020|regr',
]


# =============================================================================
# LOAD DATA
# =============================================================================

with open(results_file, "r") as f:
    raw_data = json.load(f)

# Build model list from mapping
models = ["../../../models/" + m.lower().replace("/", "_") for m in mapping.keys()]

# Filter to models that exist in the data
models = [m for m in models if m in raw_data]

print(f"Found {len(models)} models in data")
for m in models:
    display_name = mapping.get(m.replace("../../../models/", ""), m)
    print(f"  - {display_name}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_primary_metric_name(task_key: str) -> str:
    """
    Determine the primary metric name for a given task.
    
    Args:
        task_key: Task identifier in format "corpus|task_type|fewshot"
    
    Returns:
        Name of the primary metric for this task type
    """
    if "|ner" in task_key or "|pos" in task_key:
        return "overall_f1"
    elif "|cls" in task_key:
        return "weighted_f1"
    elif "|regr" in task_key:
        # For regression, we use EDRM as primary metric
        # (could also use spearman_correlation_coef)
        return "edrm"
    elif "|mcqa" in task_key:
        return "hamming_score"
    else:
        raise ValueError(f"Unknown task type in: {task_key}")


def extract_run_values(data: Dict, model: str, task: str) -> List[float]:
    """
    Extract individual run values for a model on a task.
    
    Ensures exactly REQUIRED_NUM_RUNS runs per model/task:
      - If more runs exist, takes the first REQUIRED_NUM_RUNS
      - If fewer runs exist, raises an error
    
    Args:
        data: Raw data dictionary
        model: Model path
        task: Task key
    
    Returns:
        List of metric values from individual runs (exactly REQUIRED_NUM_RUNS values)
    
    Raises:
        ValueError: If model has fewer than REQUIRED_NUM_RUNS runs for the task
    """
    if model not in data or task not in data[model]:
        return []
    
    metric_name = get_primary_metric_name(task)
    
    if metric_name not in data[model][task]:
        return []
    
    runs = data[model][task][metric_name]
    
    if len(runs) < REQUIRED_NUM_RUNS:
        model_display = model.replace("../../../models/", "")
        raise ValueError(
            f"Model '{model_display}' has only {len(runs)} runs for task '{task}', "
            f"but {REQUIRED_NUM_RUNS} runs are required."
        )
    
    # Take first REQUIRED_NUM_RUNS runs if more exist
    return runs[:REQUIRED_NUM_RUNS]


def filter_tasks(tasks: List[str], task_filter: Optional[List[str]]) -> List[str]:
    """
    Filter tasks based on prefix filter.
    
    Args:
        tasks: List of all tasks
        task_filter: List of prefixes to include (or None for all)
    
    Returns:
        Filtered list of tasks
    """
    if task_filter is None:
        return tasks
    
    filtered = []
    for task in tasks:
        for prefix in task_filter:
            if task.startswith(prefix):
                filtered.append(task)
                break
    return filtered


def compute_per_task_model_means(
    data: Dict,
    models: List[str],
    tasks: List[str],
    task_filter: Optional[List[str]] = None
) -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Compute per-task model means (averaging over runs) as the first level of
    two-level aggregation.

    Args:
        data: Raw data with individual runs
        models: List of model paths
        tasks: List of task keys
        task_filter: Optional prefix filter for tasks

    Returns:
        Tuple of:
          - filtered_tasks: List of tasks after filtering
          - means_dict: model -> task -> mean over runs
          - se_dict: model -> task -> SE of mean over runs
    """
    filtered_tasks = filter_tasks(tasks, task_filter)

    means_dict: Dict[str, Dict[str, float]] = {}
    se_dict: Dict[str, Dict[str, float]] = {}

    # Track missing model/task combinations
    missing: List[Tuple[str, str]] = []

    for model in models:
        means_dict[model] = {}
        se_dict[model] = {}
        display_name = mapping.get(model.replace("../../../models/", ""), model)
        for task in filtered_tasks:
            runs = extract_run_values(data, model, task)
            if runs:
                mean_val = np.mean(runs)
                se_val = np.std(runs, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else 0.0
                means_dict[model][task] = mean_val
                se_dict[model][task] = se_val
            else:
                missing.append((display_name, task))

    if missing:
        msg = "Missing model/task evaluations:\n"
        for display_name, task in missing:
            msg += f"  {display_name} — missing: {task}\n"
        msg += f"Total: {len(missing)} missing evaluations"
        raise ValueError(msg)

    return filtered_tasks, means_dict, se_dict


def compute_raw_scores_on_runs(
    data: Dict,
    models: List[str],
    tasks: List[str],
    task_filter: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Compute raw (non-normalized) scores using two-level aggregation.

    For each model:
      1. Per task: average runs to get one mean per task
      2. Average task-level means across tasks

    Note: Raw averaging across heterogeneous tasks (NER F1, regression EDRM, etc.)
    is not ideal since scales differ. Prefer normalized metrics (z-score, min-max).

    Args:
        data: Raw data with individual runs
        models: List of model paths
        tasks: List of task keys
        task_filter: Optional prefix filter for tasks

    Returns:
        Dictionary with raw score results per model
    """
    filtered_tasks, means_dict, _ = compute_per_task_model_means(data, models, tasks, task_filter)

    results = {}

    for model in models:
        display_name = mapping.get(model.replace("../../../models/", ""), model)

        # Collect per-task means (one value per task)
        task_means = [means_dict[model][t] for t in filtered_tasks if t in means_dict[model]]

        if task_means:
            mean_raw = np.mean(task_means)
            std_raw = np.std(task_means, ddof=1) if len(task_means) > 1 else 0.0
            se_raw = std_raw / np.sqrt(len(task_means)) if len(task_means) > 0 else 0.0
            ci_lower, ci_upper = compute_bootstrap_ci_mean(task_means, CONFIDENCE_LEVEL)

            results[display_name] = {
                "mean_raw": mean_raw,
                "std_raw": std_raw,
                "se_raw": se_raw,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_values": len(task_means),
                "display_name": display_name
            }
        else:
            results[display_name] = {
                "mean_raw": 0.0,
                "std_raw": 0.0,
                "se_raw": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "n_values": 0,
                "display_name": display_name
            }

    return results


def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for a list of values using percentile method.
    
    Args:
        values: List of values
        confidence: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) < 2:
        return (values[0] if values else 0.0, values[0] if values else 0.0)
    
    alpha = 1 - confidence
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    return (np.percentile(values, lower_percentile), np.percentile(values, upper_percentile))


def compute_bootstrap_ci_mean(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = NUM_BOOTSTRAP_SAMPLES,
    seed: int = RANDOM_SEED
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean using BCa method.
    
    BCa (Bias-Corrected and Accelerated) corrects for bias and skewness
    in the bootstrap distribution, providing more accurate CIs than
    simple percentile method.
    
    Args:
        values: List of values
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (lower_bound, upper_bound) for the mean
    """
    from scipy.stats import bootstrap
    
    if len(values) < 2:
        return (values[0] if values else 0.0, values[0] if values else 0.0)
    
    rng = np.random.default_rng(seed)
    
    bootstrap_result = bootstrap(
        data=[values],
        statistic=np.mean,
        n_resamples=n_bootstrap,
        vectorized=True,
        method="BCa",
        random_state=rng,
        confidence_level=confidence,
    )
    
    lower_bound = bootstrap_result.confidence_interval.low
    upper_bound = bootstrap_result.confidence_interval.high
    
    # Handle NaN cases (can occur with degenerate data)
    mean_val = np.mean(values)
    if np.isnan(lower_bound):
        lower_bound = mean_val
    if np.isnan(upper_bound):
        upper_bound = mean_val
    
    return (lower_bound, upper_bound)


# =============================================================================
# Z-SCORE NORMALIZATION (TWO-LEVEL AGGREGATION)
# =============================================================================

def compute_zscore_on_runs(
    data: Dict,
    models: List[str],
    tasks: List[str],
    task_filter: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Compute Z-Score normalization using two-level aggregation.

    Level 1: Per task, average runs to get one mean per model per task.
    Level 2: Z-score normalize model means within each task, then aggregate
             across tasks (one z-score per model per task).

    Args:
        data: Raw data with individual runs
        models: List of model paths
        tasks: List of task keys
        task_filter: Optional prefix filter for tasks

    Returns:
        Dictionary with z-score results per model
    """
    print("\n" + "=" * 60)
    print("Computing Z-Score normalization (two-level)...")
    print("=" * 60)

    filtered_tasks, means_dict, _ = compute_per_task_model_means(data, models, tasks, task_filter)
    print(f"Using {len(filtered_tasks)} tasks for aggregation")

    print("\n" + "-" * 60)
    print("Per-task normalization:")
    print("-" * 60)

    # model -> list of z-scores (one per task)
    model_zscores: Dict[str, List[float]] = {m: [] for m in models}
    # model -> task -> z-score
    model_task_zscores: Dict[str, Dict[str, float]] = {m: {} for m in models}

    for task in filtered_tasks:
        # Collect model means for this task
        task_model_means = []
        for model in models:
            if task in means_dict[model]:
                task_model_means.append(means_dict[model][task])

        if len(task_model_means) < 2:
            print(f"  Skipping {task}: insufficient data")
            continue

        # Z-score normalize using model means (not individual runs)
        task_mean = np.mean(task_model_means)
        task_std = np.std(task_model_means, ddof=1)

        if task_std == 0:
            print(f"  Skipping {task}: zero standard deviation")
            continue

        print(f"  {task}: {len(task_model_means)} models, "
              f"task_mean={task_mean:.4f}, task_std={task_std:.4f}")

        for model in models:
            if task not in means_dict[model]:
                continue
            z = (means_dict[model][task] - task_mean) / task_std
            model_zscores[model].append(z)
            model_task_zscores[model][task] = z

    # Aggregate z-scores per model (one value per task)
    print("\n" + "-" * 60)
    print("Aggregating z-scores per model:")
    print("-" * 60)

    results = {}

    for model in models:
        zscores = model_zscores[model]
        display_name = mapping.get(model.replace("../../../models/", ""), model)

        if not zscores:
            print(f"  Warning: No z-scores for {display_name}")
            continue

        mean_z = np.mean(zscores)
        std_z = np.std(zscores, ddof=1) if len(zscores) > 1 else 0.0
        se = std_z / np.sqrt(len(zscores)) if len(zscores) > 0 else 0.0
        ci_lower, ci_upper = compute_bootstrap_ci_mean(zscores, CONFIDENCE_LEVEL)

        results[display_name] = {
            "mean_zscore": mean_z,
            "std_zscore": std_z,
            "se_zscore": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_values": len(zscores),
            "per_task_zscores": {
                task: z for task, z in model_task_zscores[model].items()
            }
        }

    for display_name, res in sorted(results.items(), key=lambda x: x[1]["mean_zscore"], reverse=True):
        print(f"  {display_name}: mean_z={res['mean_zscore']:.4f} ± SE {res['se_zscore']:.4f} "
              f"(95% CI: [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}], n={res['n_values']})")

    return results


# =============================================================================
# MIN-MAX NORMALIZATION (TWO-LEVEL AGGREGATION)
# =============================================================================

def compute_minmax_on_runs(
    data: Dict,
    models: List[str],
    tasks: List[str],
    task_filter: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Compute Min-Max normalization using two-level aggregation.

    Level 1: Per task, average runs to get one mean per model per task.
    Level 2: Min-max normalize model means within each task to [0, 1],
             then aggregate across tasks.

    Args:
        data: Raw data with individual runs
        models: List of model paths
        tasks: List of task keys
        task_filter: Optional prefix filter for tasks

    Returns:
        Dictionary with min-max normalized results per model
    """
    print("\n" + "=" * 60)
    print("Computing Min-Max normalization (two-level)...")
    print("=" * 60)

    filtered_tasks, means_dict, _ = compute_per_task_model_means(data, models, tasks, task_filter)
    print(f"Using {len(filtered_tasks)} tasks for aggregation")

    print("\n" + "-" * 60)
    print("Per-task normalization:")
    print("-" * 60)

    # model -> list of normalized values (one per task)
    model_normalized: Dict[str, List[float]] = {m: [] for m in models}
    # model -> task -> normalized value
    model_task_normalized: Dict[str, Dict[str, float]] = {m: {} for m in models}

    for task in filtered_tasks:
        # Collect model means for this task
        task_model_means = {m: means_dict[m][task] for m in models if task in means_dict[m]}

        if len(task_model_means) < 2:
            print(f"  Skipping {task}: insufficient data")
            continue

        task_min = min(task_model_means.values())
        task_max = max(task_model_means.values())

        if task_max == task_min:
            print(f"  Skipping {task}: min equals max")
            continue

        print(f"  {task}: {len(task_model_means)} models, "
              f"task_min={task_min:.4f}, task_max={task_max:.4f}")

        for model, mean_val in task_model_means.items():
            normalized = (mean_val - task_min) / (task_max - task_min)
            model_normalized[model].append(normalized)
            model_task_normalized[model][task] = normalized

    # Aggregate normalized values per model
    print("\n" + "-" * 60)
    print("Aggregating normalized values per model:")
    print("-" * 60)

    results = {}

    for model in models:
        normalized = model_normalized[model]
        display_name = mapping.get(model.replace("../../../models/", ""), model)

        if not normalized:
            continue

        mean_norm = np.mean(normalized)
        std_norm = np.std(normalized, ddof=1) if len(normalized) > 1 else 0.0
        se = std_norm / np.sqrt(len(normalized)) if len(normalized) > 0 else 0.0
        ci_lower, ci_upper = compute_bootstrap_ci_mean(normalized, CONFIDENCE_LEVEL)

        results[display_name] = {
            "mean_normalized": mean_norm,
            "std_normalized": std_norm,
            "se_normalized": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_values": len(normalized),
            "per_task_normalized": {
                task: val for task, val in model_task_normalized[model].items()
            }
        }

    for display_name, res in sorted(results.items(), key=lambda x: x[1]["mean_normalized"], reverse=True):
        print(f"  {display_name}: mean_norm={res['mean_normalized']:.4f} ± SE {res['se_normalized']:.4f} "
              f"(95% CI: [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}], n={res['n_values']})")

    return results


# =============================================================================
# RANK-BASED AGGREGATION (TWO-LEVEL AGGREGATION)
# =============================================================================

def compute_rank_on_runs(
    data: Dict,
    models: List[str],
    tasks: List[str],
    task_filter: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Compute rank-based aggregation using two-level aggregation.

    Level 1: Per task, average runs to get one mean per model per task.
    Level 2: Rank model means within each task (1=best, N_models=worst),
             then average ranks across tasks.

    Lower average rank = better model.

    Args:
        data: Raw data with individual runs
        models: List of model paths
        tasks: List of task keys
        task_filter: Optional prefix filter for tasks

    Returns:
        Dictionary with rank results per model
    """
    print("\n" + "=" * 60)
    print("Computing Rank-based aggregation (two-level)...")
    print("=" * 60)

    filtered_tasks, means_dict, _ = compute_per_task_model_means(data, models, tasks, task_filter)
    print(f"Using {len(filtered_tasks)} tasks for aggregation")

    print("\n" + "-" * 60)
    print("Per-task ranking:")
    print("-" * 60)

    # model -> list of ranks (one per task)
    model_ranks: Dict[str, List[float]] = {m: [] for m in models}
    # model -> task -> rank
    model_task_ranks: Dict[str, Dict[str, float]] = {m: {} for m in models}

    for task in filtered_tasks:
        # Collect model means for this task
        task_model_means = {m: means_dict[m][task] for m in models if task in means_dict[m]}

        if len(task_model_means) < 2:
            print(f"  Skipping {task}: insufficient data")
            continue

        # Rank model means (higher value = better = rank 1)
        sorted_models = sorted(task_model_means.items(), key=lambda x: x[1], reverse=True)
        n_models_in_task = len(sorted_models)

        # Assign ranks with tie handling (average rank for ties)
        ranks: Dict[str, float] = {}
        i = 0
        while i < n_models_in_task:
            j = i
            while j < n_models_in_task and sorted_models[j][1] == sorted_models[i][1]:
                j += 1
            avg_rank = (i + 1 + j) / 2
            for k in range(i, j):
                ranks[sorted_models[k][0]] = avg_rank
            i = j

        print(f"  {task}: {n_models_in_task} models, rank range: 1-{n_models_in_task}")

        for model, rank in ranks.items():
            model_ranks[model].append(rank)
            model_task_ranks[model][task] = rank

    # Aggregate ranks per model
    print("\n" + "-" * 60)
    print("Aggregating ranks per model:")
    print("-" * 60)

    results = {}

    for model in models:
        ranks_list = model_ranks[model]
        display_name = mapping.get(model.replace("../../../models/", ""), model)

        if not ranks_list:
            print(f"  Warning: No ranks for {display_name}")
            continue

        mean_rank = np.mean(ranks_list)
        std_rank = np.std(ranks_list, ddof=1) if len(ranks_list) > 1 else 0.0
        se = std_rank / np.sqrt(len(ranks_list)) if len(ranks_list) > 0 else 0.0
        ci_lower, ci_upper = compute_bootstrap_ci_mean(ranks_list, CONFIDENCE_LEVEL)

        results[display_name] = {
            "mean_rank": mean_rank,
            "std_rank": std_rank,
            "se_rank": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_values": len(ranks_list),
            "per_task_ranks": {
                task: rank for task, rank in model_task_ranks[model].items()
            }
        }

    for display_name, res in sorted(results.items(), key=lambda x: x[1]["mean_rank"]):
        print(f"  {display_name}: mean_rank={res['mean_rank']:.2f} ± SE {res['se_rank']:.2f} "
              f"(95% CI: [{res['ci_lower']:.2f}, {res['ci_upper']:.2f}], n={res['n_values']})")

    return results


# =============================================================================
# TASK-LEVEL PAIRWISE WIN PROBABILITY
# =============================================================================

def compute_bootstrap_win_probability(
    data: Dict,
    models: List[str],
    tasks: List[str],
    task_filter: Optional[List[str]] = None,
    n_bootstrap: int = NUM_BOOTSTRAP_SAMPLES,
    seed: int = RANDOM_SEED
) -> Dict[str, Dict]:
    """
    Compute pairwise win probability using task-level paired comparison.

    For each pair of models (A, B):
      1. Per task: compare A's mean vs B's mean (averaged over runs)
      2. Count task wins across all tasks
      3. win_prob = (wins + 0.5 * ties) / n_tasks
      4. CI via bootstrap over tasks (resample which tasks to include)

    Then aggregate across all opponents to get overall win rate per model.

    Args:
        data: Raw data with individual runs
        models: List of model paths
        tasks: List of task keys
        task_filter: Optional prefix filter for tasks
        n_bootstrap: Number of bootstrap samples for CI
        seed: Random seed for reproducibility

    Returns:
        Dictionary with pairwise win probabilities and model summaries
    """
    print("\n" + "=" * 60)
    print("Computing Task-Level Pairwise Win Probability...")
    print("=" * 60)

    np.random.seed(seed)

    filtered_tasks, means_dict, _ = compute_per_task_model_means(data, models, tasks, task_filter)

    # Get display names
    model_names = {
        m: mapping.get(m.replace("../../../models/", ""), m)
        for m in models
    }

    pairwise_results: Dict[Tuple[str, str], Dict] = {}
    model_win_summary: Dict[str, Dict] = {
        model_names[m]: {"wins": 0, "total": 0, "task_win_probs": []}
        for m in models
    }

    model_pairs = list(combinations(models, 2))
    print(f"Comparing {len(model_pairs)} model pairs across {len(filtered_tasks)} tasks")

    for model_a, model_b in model_pairs:
        name_a = model_names[model_a]
        name_b = model_names[model_b]

        pair_key = (name_a, name_b)
        pairwise_results[pair_key] = {
            "tasks": {},
            "overall_win_prob_a": 0.0,
            "overall_win_prob_a_ci": (0.0, 0.0),
        }

        # Collect per-task outcomes: 1.0 = A wins, 0.0 = B wins, 0.5 = tie
        task_outcomes_a = []

        for task in filtered_tasks:
            mean_a = means_dict[model_a].get(task)
            mean_b = means_dict[model_b].get(task)

            if mean_a is None or mean_b is None:
                continue

            if mean_a > mean_b:
                outcome = 1.0
            elif mean_b > mean_a:
                outcome = 0.0
            else:
                outcome = 0.5

            pairwise_results[pair_key]["tasks"][task] = {
                "win_prob_a": 1.0 if mean_a > mean_b else (0.0 if mean_b > mean_a else 0.5),
                "mean_a": mean_a,
                "mean_b": mean_b,
            }

            task_outcomes_a.append(outcome)

        if task_outcomes_a:
            overall_win_prob = np.mean(task_outcomes_a)

            # Bootstrap CI over tasks
            rng = np.random.default_rng(seed)
            bootstrap_win_probs = []
            outcomes_arr = np.array(task_outcomes_a)
            for _ in range(n_bootstrap):
                resample_idx = rng.integers(0, len(outcomes_arr), size=len(outcomes_arr))
                bootstrap_win_probs.append(np.mean(outcomes_arr[resample_idx]))

            ci_lower = float(np.percentile(bootstrap_win_probs, (1 - CONFIDENCE_LEVEL) / 2 * 100))
            ci_upper = float(np.percentile(bootstrap_win_probs, (1 + CONFIDENCE_LEVEL) / 2 * 100))

            pairwise_results[pair_key]["overall_win_prob_a"] = overall_win_prob
            pairwise_results[pair_key]["overall_win_prob_a_ci"] = (ci_lower, ci_upper)
            pairwise_results[pair_key]["n_tasks"] = len(task_outcomes_a)

            model_win_summary[name_a]["task_win_probs"].append(overall_win_prob)
            model_win_summary[name_b]["task_win_probs"].append(1 - overall_win_prob)

            # Count significant wins (CI doesn't overlap 0.5)
            model_win_summary[name_a]["total"] += 1
            model_win_summary[name_b]["total"] += 1
            if ci_lower > 0.5:
                model_win_summary[name_a]["wins"] += 1
            elif ci_upper < 0.5:
                model_win_summary[name_b]["wins"] += 1

    # Compute overall statistics per model
    for model_name, summary in model_win_summary.items():
        if summary["task_win_probs"]:
            summary["mean_win_prob"] = np.mean(summary["task_win_probs"])
            summary["std_win_prob"] = np.std(summary["task_win_probs"], ddof=1) if len(summary["task_win_probs"]) > 1 else 0.0
            se = summary["std_win_prob"] / np.sqrt(len(summary["task_win_probs"]))
            summary["se_win_prob"] = se
            summary["ci_win_prob"] = compute_bootstrap_ci_mean(summary["task_win_probs"], CONFIDENCE_LEVEL)
            summary["n_comparisons"] = len(summary["task_win_probs"])
        else:
            summary["mean_win_prob"] = 0.5
            summary["std_win_prob"] = 0.0
            summary["se_win_prob"] = 0.0
            summary["ci_win_prob"] = (0.5, 0.5)
            summary["n_comparisons"] = 0

    # Print summary
    print("\nModel Win Probability Summary (vs all other models):")
    sorted_models = sorted(model_win_summary.items(),
                          key=lambda x: x[1]["mean_win_prob"],
                          reverse=True)
    for model_name, summary in sorted_models:
        print(f"  {model_name}: win_prob={summary['mean_win_prob']:.3f} ± SE {summary['se_win_prob']:.3f} "
              f"(95% CI: [{summary['ci_win_prob'][0]:.3f}, {summary['ci_win_prob'][1]:.3f}], "
              f"significant_wins={summary['wins']}/{summary['total']})")

    return {
        "pairwise": {f"{k[0]} vs {k[1]}": v for k, v in pairwise_results.items()},
        "model_summary": model_win_summary
    }


# =============================================================================
# TASK FILTERING AND ANALYSIS
# =============================================================================

def compute_task_discriminability(
    data: Dict,
    models: List[str],
    tasks: List[str],
    task_filter: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Compute per-task discriminability metrics for filtering.
    
    Metrics computed:
      - variance_between: Variance of model means (between-model variance)
      - variance_within: Average within-model variance (run-to-run noise)
      - snr: Signal-to-noise ratio = variance_between / variance_within
             Higher SNR means model differences are reliable, not just noise
      - range: Max - Min of model means
      - cv: Coefficient of variation (std / mean) of model means
      - mean_score: Average score across all models
      - std_score: Std of model means
    
    The SNR metric is similar to an ANOVA F-statistic and indicates whether
    between-model differences are larger than within-model (run-to-run) noise.
    
    Args:
        data: Raw data with individual runs
        models: List of model paths
        tasks: List of task keys
        task_filter: Optional prefix filter for tasks
    
    Returns:
        Dictionary: task -> {variance_between, variance_within, snr, range, cv, ...}
    """
    print("\n" + "=" * 60)
    print("Computing Task Discriminability Metrics...")
    print("=" * 60)
    
    filtered_tasks = filter_tasks(tasks, task_filter)
    task_stats = {}
    
    for task in filtered_tasks:
        model_means = []
        model_variances = []  # Within-model variances
        all_runs_per_model = []
        
        for model in models:
            runs = extract_run_values(data, model, task)
            if runs:
                model_means.append(np.mean(runs))
                all_runs_per_model.append(runs)
                # Within-model variance (run-to-run variance for this model)
                if len(runs) > 1:
                    model_variances.append(np.var(runs, ddof=1))
        
        if len(model_means) >= 2:
            # Between-model variance: how much do model means differ?
            variance_between = np.var(model_means, ddof=1)
            std_score = np.std(model_means, ddof=1)
            mean_score = np.mean(model_means)
            score_range = max(model_means) - min(model_means)
            cv = std_score / mean_score if mean_score > 0 else 0
            
            # Within-model variance: average run-to-run variance across models
            # This represents the "noise" level
            if model_variances:
                variance_within = np.mean(model_variances)
            else:
                variance_within = 0.0
            
            # Signal-to-noise ratio
            # High SNR = model differences are large relative to run-to-run noise
            # Low SNR = model differences might just be noise
            if variance_within > 0:
                snr = variance_between / variance_within
            else:
                # If no within-model variance (single run per model or all identical),
                # set SNR to inf if there's between-model variance, else 0
                snr = float('inf') if variance_between > 0 else 0.0
            
            task_stats[task] = {
                "variance_between": variance_between,
                "variance_within": variance_within,
                "snr": snr,
                "range": score_range,
                "cv": cv,
                "n_models": len(model_means),
                "mean_score": mean_score,
                "std_score": std_score,
                "min_score": min(model_means),
                "max_score": max(model_means),
                # Keep old name for backward compatibility
                "variance": variance_between,
            }
    
    # Print results sorted by SNR (descending) - most reliable discriminating tasks first
    print("\nTask Discriminability (sorted by SNR, descending):")
    print("-" * 100)
    print(f"{'Task':<35} {'Var(between)':>12} {'Var(within)':>12} {'SNR':>10} {'Range':>10} {'Mean':>10} {'N':>5}")
    print("-" * 100)
    
    for task, stats in sorted(task_stats.items(), key=lambda x: x[1]["snr"], reverse=True):
        snr_str = f"{stats['snr']:.2f}" if stats['snr'] != float('inf') else "inf"
        print(f"{task:<35} {stats['variance_between']:>12.6f} {stats['variance_within']:>12.6f} "
              f"{snr_str:>10} {stats['range']:>10.4f} {stats['mean_score']:>10.4f} {stats['n_models']:>5}")
    
    return task_stats


def compute_cross_task_correlation(
    data: Dict,
    models: List[str],
    tasks: List[str],
    task_filter: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise Pearson correlation between tasks based on model performance.
    
    For each pair of tasks, compute correlation of model scores across models.
    High correlation suggests redundant tasks.
    
    Args:
        data: Raw data with individual runs
        models: List of model paths
        tasks: List of task keys
        task_filter: Optional prefix filter for tasks
    
    Returns:
        Tuple of (correlation_matrix, task_names)
    """
    print("\n" + "=" * 60)
    print("Computing Cross-Task Correlation Matrix...")
    print("=" * 60)
    
    filtered_tasks = filter_tasks(tasks, task_filter)
    
    # Build matrix: rows = models, columns = tasks
    # Each cell = mean score of that model on that task
    task_model_scores = {}
    
    for task in filtered_tasks:
        scores = []
        for model in models:
            runs = extract_run_values(data, model, task)
            if runs:
                scores.append(np.mean(runs))
            else:
                scores.append(np.nan)
        task_model_scores[task] = scores
    
    # Filter out tasks with too many NaNs
    valid_tasks = []
    for task, scores in task_model_scores.items():
        non_nan_count = sum(1 for s in scores if not np.isnan(s))
        if non_nan_count >= 3:  # Need at least 3 models for meaningful correlation
            valid_tasks.append(task)
    
    n_tasks = len(valid_tasks)
    corr_matrix = np.zeros((n_tasks, n_tasks))
    
    for i, task_i in enumerate(valid_tasks):
        for j, task_j in enumerate(valid_tasks):
            scores_i = np.array(task_model_scores[task_i])
            scores_j = np.array(task_model_scores[task_j])
            
            # Mask NaN values
            valid_mask = ~np.isnan(scores_i) & ~np.isnan(scores_j)
            
            if valid_mask.sum() >= 3:
                corr, _ = np.corrcoef(scores_i[valid_mask], scores_j[valid_mask])[0, 1], None
                corr_matrix[i, j] = corr if not np.isnan(corr) else 0
            else:
                corr_matrix[i, j] = 0 if i != j else 1
    
    # Print correlation summary
    print("\nCross-Task Correlation Matrix:")
    print("-" * 60)
    
    # Find highly correlated pairs (excluding diagonal)
    high_corr_pairs = []
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            if abs(corr_matrix[i, j]) > 0.7:
                high_corr_pairs.append((valid_tasks[i], valid_tasks[j], corr_matrix[i, j]))
    
    if high_corr_pairs:
        print("\nHighly correlated task pairs (|r| > 0.7) - potentially redundant:")
        for t1, t2, r in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {t1} <-> {t2}: r = {r:.3f}")
    else:
        print("\nNo highly correlated task pairs found (|r| > 0.7)")
    
    # Compute average correlation per task (excluding self-correlation)
    # Low average correlation could indicate:
    #   - Unique signal (good if high SNR) 
    #   - Noisy/low-quality task (bad if low SNR)
    print("\n" + "-" * 60)
    print("Per-task average correlation with other tasks:")
    print("(Low correlation + Low SNR → potentially low-quality task)")
    print("-" * 60)
    
    task_avg_corr = {}
    for i, task in enumerate(valid_tasks):
        # Average absolute correlation with all other tasks
        other_corrs = [abs(corr_matrix[i, j]) for j in range(n_tasks) if i != j]
        if other_corrs:
            avg_corr = np.mean(other_corrs)
            max_corr = np.max(other_corrs)
            min_corr = np.min(other_corrs)
            task_avg_corr[task] = {
                "avg_abs_corr": avg_corr,
                "max_abs_corr": max_corr,
                "min_abs_corr": min_corr,
            }
    
    # Print sorted by average correlation (ascending - lowest first = potential outliers)
    print(f"\n{'Task':<35} {'Avg|r|':>10} {'Max|r|':>10} {'Min|r|':>10}")
    print("-" * 70)
    for task, stats in sorted(task_avg_corr.items(), key=lambda x: x[1]["avg_abs_corr"]):
        print(f"{task:<35} {stats['avg_abs_corr']:>10.3f} {stats['max_abs_corr']:>10.3f} {stats['min_abs_corr']:>10.3f}")
    
    # Identify potential outlier tasks (low correlation with others)
    low_corr_threshold = 0.3
    outlier_tasks = [t for t, s in task_avg_corr.items() if s["avg_abs_corr"] < low_corr_threshold]
    if outlier_tasks:
        print(f"\nPotential outlier tasks (avg |r| < {low_corr_threshold}):")
        for task in outlier_tasks:
            print(f"  - {task}: avg|r| = {task_avg_corr[task]['avg_abs_corr']:.3f}")
        print("  → Check SNR: if low SNR, consider excluding; if high SNR, task may measure unique ability")
    
    return corr_matrix, valid_tasks, task_avg_corr


def plot_cross_task_correlation(
    corr_matrix: np.ndarray,
    task_names: List[str],
    output_path: str,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Plot a 2D heatmap of cross-task correlations.
    
    Args:
        corr_matrix: Correlation matrix from compute_cross_task_correlation
        task_names: List of task names
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with diverging colormap (blue-white-red)
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    
    im = ax.imshow(corr_matrix, cmap=cmap, norm=norm, aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom", fontsize=10)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(task_names)))
    ax.set_yticks(np.arange(len(task_names)))
    
    # Format task names for display (shorter labels)
    short_names = [t.replace("|1.0", "").replace("|", "\n") for t in task_names]
    ax.set_xticklabels(short_names, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(short_names, fontsize=8)
    
    # Add correlation values as text
    for i in range(len(task_names)):
        for j in range(len(task_names)):
            val = corr_matrix[i, j]
            # Choose text color based on background
            text_color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                   color=text_color, fontsize=7)
    
    ax.set_title("Cross-Task Correlation Matrix\n(based on model performance)", fontsize=12)
    ax.set_xlabel("Task", fontsize=10)
    ax.set_ylabel("Task", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nCorrelation heatmap saved to: {output_path}")


def filter_tasks_by_discriminability(
    task_stats: Dict[str, Dict],
    min_range: float = 0.02,
    min_cv: float = 0.01,
    min_snr: Optional[float] = None
) -> List[str]:
    """
    Filter tasks based on discriminability thresholds.
    
    Args:
        task_stats: Output from compute_task_discriminability
        min_range: Minimum score range to keep task (default: 0.02 = 2%)
        min_cv: Minimum coefficient of variation to keep task (default: 0.01)
        min_snr: Minimum signal-to-noise ratio to keep task (default: None = no filter)
                 Recommended: min_snr=1.0 means between-model variance > within-model variance
    
    Returns:
        List of task names that pass the filter
    """
    kept_tasks = []
    removed_tasks = []
    
    for task, stats in task_stats.items():
        passes_range = stats["range"] >= min_range
        passes_cv = stats["cv"] >= min_cv
        passes_snr = (min_snr is None) or (stats.get("snr", float('inf')) >= min_snr)
        
        if passes_range and passes_cv and passes_snr:
            kept_tasks.append(task)
        else:
            removed_tasks.append((task, stats["range"], stats["cv"], stats.get("snr", 0)))
    
    if removed_tasks:
        snr_msg = f" or SNR < {min_snr}" if min_snr is not None else ""
        print(f"\nTasks removed by discriminability filter (range < {min_range} or CV < {min_cv}{snr_msg}):")
        for task, r, cv, snr in removed_tasks:
            snr_str = f"{snr:.2f}" if snr != float('inf') else "inf"
            print(f"  - {task}: range={r:.4f}, cv={cv:.4f}, snr={snr_str}")
    
    print(f"\nKept {len(kept_tasks)}/{len(task_stats)} tasks after discriminability filtering")
    
    return kept_tasks


def filter_tasks_by_correlation(
    corr_matrix: np.ndarray,
    task_names: List[str],
    task_stats: Dict[str, Dict],
    max_corr: float = 0.9
) -> List[str]:
    """
    Filter out redundant tasks based on high cross-correlation.
    
    When two tasks are highly correlated, keep the one with higher discriminability.
    
    Args:
        corr_matrix: Correlation matrix from compute_cross_task_correlation
        task_names: List of task names
        task_stats: Output from compute_task_discriminability
        max_corr: Maximum allowed correlation (default: 0.9)
    
    Returns:
        List of task names after removing redundant tasks
    """
    n_tasks = len(task_names)
    tasks_to_remove = set()
    
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            if abs(corr_matrix[i, j]) > max_corr:
                task_i = task_names[i]
                task_j = task_names[j]
                
                # Keep the task with higher variance (more discriminating)
                var_i = task_stats.get(task_i, {}).get("variance", 0)
                var_j = task_stats.get(task_j, {}).get("variance", 0)
                
                if var_i >= var_j:
                    tasks_to_remove.add(task_j)
                    print(f"  Removing {task_j} (corr={corr_matrix[i, j]:.3f} with {task_i}, lower variance)")
                else:
                    tasks_to_remove.add(task_i)
                    print(f"  Removing {task_i} (corr={corr_matrix[i, j]:.3f} with {task_j}, lower variance)")
    
    kept_tasks = [t for t in task_names if t not in tasks_to_remove]
    
    print(f"\nKept {len(kept_tasks)}/{n_tasks} tasks after correlation filtering (max_corr={max_corr})")
    
    return kept_tasks


# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

def generate_combined_latex_table(
    zscore_results: Dict,
    minmax_results: Dict,
    rank_results: Dict,
    bootstrap_results: Dict,
    sort_by: str = "mean_win_prob"
) -> str:
    """
    Generate a combined LaTeX table with all aggregation metrics.

    Columns: Model | Z-Score | Min-Max | Avg Rank | Win Prob

    Args:
        zscore_results: Results from compute_zscore_on_runs
        minmax_results: Results from compute_minmax_on_runs
        rank_results: Results from compute_rank_on_runs
        bootstrap_results: Results from compute_bootstrap_win_probability
        sort_by: Metric to sort by ('mean_win_prob', 'mean_zscore', 'mean_normalized', 'mean_rank')
    
    Returns:
        LaTeX table as string
    """
    # Get all model names
    all_models = (set(zscore_results.keys()) | set(minmax_results.keys()) | 
                  set(rank_results.keys()) | set(bootstrap_results["model_summary"].keys()))
    
    # Create sortable list with sorting key
    model_data = []
    for model in all_models:
        z_data = zscore_results.get(model, {})
        m_data = minmax_results.get(model, {})
        r_data = rank_results.get(model, {})
        b_data = bootstrap_results["model_summary"].get(model, {})
        
        if sort_by == "mean_zscore":
            sort_val = z_data.get("mean_zscore", float("-inf"))
        elif sort_by == "mean_normalized":
            sort_val = m_data.get("mean_normalized", float("-inf"))
        elif sort_by == "mean_rank":
            # For rank, lower is better, so negate for sorting
            sort_val = -r_data.get("mean_rank", float("inf"))
        elif sort_by == "mean_win_prob":
            sort_val = b_data.get("mean_win_prob", float("-inf"))
        else:
            sort_val = 0
        
        model_data.append((model, z_data, m_data, r_data, b_data, sort_val))
    
    # Sort by chosen metric (descending = best first)
    model_data.sort(key=lambda x: x[5], reverse=True)
    
    # Find best and second-best for each metric
    z_scores = [(m[0], m[1].get("mean_zscore", float("-inf"))) for m in model_data if m[1]]
    m_scores = [(m[0], m[2].get("mean_normalized", float("-inf"))) for m in model_data if m[2]]
    # For rank, lower is better
    r_scores = [(m[0], m[3].get("mean_rank", float("inf"))) for m in model_data if m[3]]
    w_scores = [(m[0], m[4].get("mean_win_prob", float("-inf"))) for m in model_data if m[4]]
    
    z_sorted = sorted(z_scores, key=lambda x: x[1], reverse=True)
    m_sorted = sorted(m_scores, key=lambda x: x[1], reverse=True)
    r_sorted = sorted(r_scores, key=lambda x: x[1], reverse=False)  # Lower is better
    w_sorted = sorted(w_scores, key=lambda x: x[1], reverse=True)
    
    z_best = z_sorted[0][0] if z_sorted else None
    z_second = z_sorted[1][0] if len(z_sorted) > 1 else None
    m_best = m_sorted[0][0] if m_sorted else None
    m_second = m_sorted[1][0] if len(m_sorted) > 1 else None
    r_best = r_sorted[0][0] if r_sorted else None
    r_second = r_sorted[1][0] if len(r_sorted) > 1 else None
    w_best = w_sorted[0][0] if w_sorted else None
    w_second = w_sorted[1][0] if len(w_sorted) > 1 else None
    
    # Build LaTeX table
    lines = []
    lines.append(r"\begin{table*}[t!]")
    lines.append(r"\small")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{|l|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Model} & \textbf{Z-Score} & \textbf{Min-Max} & \textbf{Avg Rank} & \textbf{Win Prob} \\")
    lines.append(r"\hline")
    
    for model, z_data, m_data, r_data, b_data, _ in model_data:
        # Z-Score column (show mean ± SE)
        if z_data:
            z_mean = z_data["mean_zscore"]
            z_se = z_data["se_zscore"]
            z_str = f"{z_mean:.3f}$\\pm${z_se:.3f}"
            if model == z_best:
                z_str = f"\\textbf{{{z_str}}}"
            elif model == z_second:
                z_str = f"\\underline{{{z_str}}}"
        else:
            z_str = "-"

        # Min-Max column (show mean ± SE)
        if m_data:
            m_mean = m_data["mean_normalized"]
            m_se = m_data["se_normalized"]
            m_str = f"{m_mean:.3f}$\\pm${m_se:.3f}"
            if model == m_best:
                m_str = f"\\textbf{{{m_str}}}"
            elif model == m_second:
                m_str = f"\\underline{{{m_str}}}"
        else:
            m_str = "-"

        # Rank column (lower is better, show mean ± SE)
        if r_data:
            r_mean = r_data["mean_rank"]
            r_se = r_data["se_rank"]
            r_str = f"{r_mean:.2f}$\\pm${r_se:.2f}"
            if model == r_best:
                r_str = f"\\textbf{{{r_str}}}"
            elif model == r_second:
                r_str = f"\\underline{{{r_str}}}"
        else:
            r_str = "-"

        # Win Prob column (show mean ± SE)
        if b_data:
            w_mean = b_data["mean_win_prob"]
            w_se = b_data.get("se_win_prob", 0)
            w_str = f"{w_mean:.3f}$\\pm${w_se:.3f}"
            if model == w_best:
                w_str = f"\\textbf{{{w_str}}}"
            elif model == w_second:
                w_str = f"\\underline{{{w_str}}}"
        else:
            w_str = "-"

        # Escape underscores in model name for LaTeX
        model_escaped = model.replace("_", r"\_")
        lines.append(f"{model_escaped} & {z_str} & {m_str} & {r_str} & {w_str} \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Aggregated benchmark results using two-level aggregation. "
                 r"Per task, runs are averaged; then model means are normalized and aggregated across tasks. "
                 r"Values show mean $\pm$ SE. "
                 r"Avg Rank shows average rank across tasks (lower is better). "
                 r"Win Prob shows task-level pairwise win probability vs all other models. "
                 r"Best model in bold, second best underlined.}")
    lines.append(r"\label{table:aggregation_runs}")
    lines.append(r"\end{table*}")
    
    return "\n".join(lines)


def generate_raw_metrics_latex_table(
    data: Dict,
    models: List[str],
    tasks: List[str],
    model_mapping: Dict[str, str]
) -> str:
    """
    Generate a LaTeX table with raw (non-normalized) metrics per task.
    
    This table shows the actual metric values (mean ± std) for each model on each task,
    similar to GenerateLatexTableFromResultsJSON.py but computed from individual runs.
    
    Columns: Dataset | Task | Model1 | Model2 | ...
    
    Args:
        data: Raw data with individual runs
        models: List of model paths
        tasks: List of task keys
        model_mapping: Dictionary mapping model paths to display names
    
    Returns:
        LaTeX table as string
    """
    from collections import defaultdict
    
    # Organize tasks by corpus
    corpus_tasks = defaultdict(list)
    for t in tasks:
        parts = t.split("|")
        if len(parts) >= 2:
            corpus = parts[0]
            task_type = parts[1]
            corpus_tasks[corpus].append((task_type, t))
    
    # Get display names for models
    model_names = []
    for m in models:
        name = model_mapping.get(m.replace("../../../models/", ""), m)
        model_names.append(name)
    
    output = []
    
    # Table header
    output.append(r"\begin{table*}[t!]")
    output.append(r"\tiny")
    output.append(r"\centering")
    col_spec = "|l|l|" + "c|" * len(models)
    output.append(f"\\begin{{tabular}}{{{col_spec}}}")
    output.append(r"\hline")
    
    # Header row with model names
    header_models = " & ".join([f"\\textbf{{{name.replace('_', '-')}}}" for name in model_names])
    output.append(f"\\textbf{{Dataset}} & \\textbf{{Task}} & {header_models} \\\\ ")
    
    # Process each corpus
    for corpus in corpus_tasks:
        task_list = corpus_tasks[corpus]
        num_rows = len(task_list)
        corpus_written = False
        
        for idx, (task_type, tkey) in enumerate(task_list):
            # Check if task exists for all models
            task_exists_for_all = all(
                m in data and tkey in data[m] 
                for m in models
            )
            if not task_exists_for_all:
                continue
            
            runs_metrics = []
            
            # Determine metric type and compute values
            if "regr" in tkey:
                # Regression tasks: show EDRM / Spearman
                edrm_values, spear_values = [], []
                edrm_stds, spear_stds = [], []
                
                for m in models:
                    edrm_runs = data[m][tkey].get('edrm', [])
                    spear_runs = data[m][tkey].get('spearman_correlation_coef', [])
                    
                    if edrm_runs:
                        edrm_avg = statistics.mean(edrm_runs)
                        edrm_std = statistics.stdev(edrm_runs) if len(edrm_runs) > 1 else 0.0
                    else:
                        edrm_avg, edrm_std = 0.0, 0.0
                    
                    if spear_runs:
                        spear_avg = statistics.mean(spear_runs)
                        spear_std = statistics.stdev(spear_runs) if len(spear_runs) > 1 else 0.0
                    else:
                        spear_avg, spear_std = 0.0, 0.0
                    
                    edrm_values.append(edrm_avg)
                    edrm_stds.append(edrm_std)
                    spear_values.append(spear_avg)
                    spear_stds.append(spear_std)
                
                # Find best and second best
                edrm_sorted = sorted(range(len(edrm_values)), key=lambda i: edrm_values[i], reverse=True)
                spear_sorted = sorted(range(len(spear_values)), key=lambda i: spear_values[i], reverse=True)
                edrm_best = edrm_sorted[0]
                edrm_second = edrm_sorted[1] if len(edrm_sorted) > 1 else None
                spear_best = spear_sorted[0]
                spear_second = spear_sorted[1] if len(spear_sorted) > 1 else None
                
                for i in range(len(models)):
                    edrm_part = f"{round(edrm_values[i]*100, 2)}$\\pm${round(edrm_stds[i]*100, 2)}"
                    spear_part = f"{round(spear_values[i]*100, 2)}$\\pm${round(spear_stds[i]*100, 2)}"
                    
                    if i == edrm_best:
                        edrm_part = f"\\textbf{{{edrm_part}}}"
                    elif i == edrm_second:
                        edrm_part = f"\\underline{{{edrm_part}}}"
                    
                    if i == spear_best:
                        spear_part = f"\\textbf{{{spear_part}}}"
                    elif i == spear_second:
                        spear_part = f"\\underline{{{spear_part}}}"
                    
                    runs_metrics.append(f"{edrm_part} / {spear_part}")
            
            elif "mcqa" in tkey:
                # MCQA tasks: show Hamming / Exact
                hamming_values, exact_values = [], []
                hamming_stds, exact_stds = [], []
                
                for m in models:
                    hamming_runs = data[m][tkey].get('hamming_score', [])
                    exact_runs = data[m][tkey].get('exact_match', [])
                    
                    if hamming_runs:
                        hamming_avg = statistics.mean(hamming_runs)
                        hamming_std = statistics.stdev(hamming_runs) if len(hamming_runs) > 1 else 0.0
                    else:
                        hamming_avg, hamming_std = 0.0, 0.0
                    
                    if exact_runs:
                        exact_avg = statistics.mean(exact_runs)
                        exact_std = statistics.stdev(exact_runs) if len(exact_runs) > 1 else 0.0
                    else:
                        exact_avg, exact_std = 0.0, 0.0
                    
                    hamming_values.append(hamming_avg)
                    hamming_stds.append(hamming_std)
                    exact_values.append(exact_avg)
                    exact_stds.append(exact_std)
                
                # Find best and second best
                hamming_sorted = sorted(range(len(hamming_values)), key=lambda i: hamming_values[i], reverse=True)
                exact_sorted = sorted(range(len(exact_values)), key=lambda i: exact_values[i], reverse=True)
                hamming_best = hamming_sorted[0]
                hamming_second = hamming_sorted[1] if len(hamming_sorted) > 1 else None
                exact_best = exact_sorted[0]
                exact_second = exact_sorted[1] if len(exact_sorted) > 1 else None
                
                for i in range(len(models)):
                    hamming_part = f"{round(hamming_values[i]*100, 2)}$\\pm${round(hamming_stds[i]*100, 2)}"
                    exact_part = f"{round(exact_values[i]*100, 2)}$\\pm${round(exact_stds[i]*100, 2)}"
                    
                    if i == hamming_best:
                        hamming_part = f"\\textbf{{{hamming_part}}}"
                    elif i == hamming_second:
                        hamming_part = f"\\underline{{{hamming_part}}}"
                    
                    if i == exact_best:
                        exact_part = f"\\textbf{{{exact_part}}}"
                    elif i == exact_second:
                        exact_part = f"\\underline{{{exact_part}}}"
                    
                    runs_metrics.append(f"{hamming_part} / {exact_part}")
            
            else:
                # NER, POS, CLS tasks: show single metric (F1)
                metric_values = []
                metric_stds = []
                
                for m in models:
                    # Determine which metric to use
                    if "|ner" in tkey or "|pos" in tkey:
                        metric_runs = data[m][tkey].get('overall_f1', [])
                    else:  # cls
                        metric_runs = data[m][tkey].get('weighted_f1', [])
                    
                    if metric_runs:
                        avg = statistics.mean(metric_runs)
                        std = statistics.stdev(metric_runs) if len(metric_runs) > 1 else 0.0
                    else:
                        avg, std = 0.0, 0.0
                    
                    metric_values.append(avg)
                    metric_stds.append(std)
                
                # Find best and second best
                sorted_idx = sorted(range(len(metric_values)), key=lambda i: metric_values[i], reverse=True)
                best_idx = sorted_idx[0]
                second_idx = sorted_idx[1] if len(sorted_idx) > 1 else None
                
                for i in range(len(models)):
                    metric_str = f"{round(metric_values[i]*100, 2)}$\\pm${round(metric_stds[i]*100, 2)}"
                    
                    if i == best_idx:
                        metric_str = f"\\textbf{{{metric_str}}}"
                    elif i == second_idx:
                        metric_str = f"\\underline{{{metric_str}}}"
                    
                    runs_metrics.append(metric_str)
            
            # Build LaTeX row
            corpus_label = ""
            if not corpus_written:
                corpus_label = f"\\hline\n\n\\multirow{{{num_rows}}}{{*}}{{{corpus.replace('_', '-').upper()}}}"
                corpus_written = True
            
            metrics_str = " & ".join(runs_metrics)
            if corpus_label:
                output.append(f"{corpus_label} & {task_type.upper()} & {metrics_str} \\\\ ")
            else:
                output.append(f" & {task_type.upper()} & {metrics_str} \\\\ ")
    
    # Table footer
    output.append(r"\hline")
    output.append(r"\end{tabular}")
    output.append(r"\caption{Performance on biomedical tasks in French. "
                  r"Results shown as mean$\pm$std computed from individual runs. "
                  r"Best model in bold, second best underlined.}")
    output.append(r"\label{table:raw_results}")
    output.append(r"\end{table*}")

    return "\n".join(output)


def generate_raw_metrics_latex_table_transposed(
    data: Dict,
    models: List[str],
    tasks: List[str],
    model_mapping: Dict[str, str]
) -> str:
    """
    Transposed version of generate_raw_metrics_latex_table.

    Rows: models. Columns: tasks (grouped by corpus). Best/second-best are
    highlighted per column (per task), like the original.
    """
    from collections import defaultdict

    corpus_tasks = defaultdict(list)
    for t in tasks:
        parts = t.split("|")
        if len(parts) >= 2:
            corpus_tasks[parts[0]].append((parts[1], t))

    # Keep only tasks that exist for all models, preserving corpus order
    ordered_columns = []  # list of (corpus, task_type, tkey)
    for corpus, task_list in corpus_tasks.items():
        for task_type, tkey in task_list:
            if all(m in data and tkey in data[m] for m in models):
                ordered_columns.append((corpus, task_type, tkey))

    # Display names for models (rows)
    model_names = [
        model_mapping.get(m.replace("../../../models/", ""), m)
        for m in models
    ]

    # Compute per-task formatted cell strings (one per model), with bold/underline
    def _format_with_rank(values, stds, model_idx):
        sorted_idx = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
        best = sorted_idx[0]
        second = sorted_idx[1] if len(sorted_idx) > 1 else None
        s = f"{round(values[model_idx]*100, 2)}$\\pm${round(stds[model_idx]*100, 2)}"
        if model_idx == best:
            return f"\\textbf{{{s}}}"
        if model_idx == second:
            return f"\\underline{{{s}}}"
        return s

    cells_per_task = []  # list aligned with ordered_columns, each is list[len(models)]
    for corpus, task_type, tkey in ordered_columns:
        if "regr" in tkey:
            edrm_vals, spear_vals, edrm_stds, spear_stds = [], [], [], []
            for m in models:
                edrm_runs = data[m][tkey].get('edrm', [])
                spear_runs = data[m][tkey].get('spearman_correlation_coef', [])
                edrm_vals.append(statistics.mean(edrm_runs) if edrm_runs else 0.0)
                edrm_stds.append(statistics.stdev(edrm_runs) if len(edrm_runs) > 1 else 0.0)
                spear_vals.append(statistics.mean(spear_runs) if spear_runs else 0.0)
                spear_stds.append(statistics.stdev(spear_runs) if len(spear_runs) > 1 else 0.0)
            cells = []
            for i in range(len(models)):
                edrm_part = _format_with_rank(edrm_vals, edrm_stds, i)
                spear_part = _format_with_rank(spear_vals, spear_stds, i)
                cells.append(f"{edrm_part} / {spear_part}")
            cells_per_task.append(cells)

        elif "mcqa" in tkey:
            ham_vals, ex_vals, ham_stds, ex_stds = [], [], [], []
            for m in models:
                ham_runs = data[m][tkey].get('hamming_score', [])
                ex_runs = data[m][tkey].get('exact_match', [])
                ham_vals.append(statistics.mean(ham_runs) if ham_runs else 0.0)
                ham_stds.append(statistics.stdev(ham_runs) if len(ham_runs) > 1 else 0.0)
                ex_vals.append(statistics.mean(ex_runs) if ex_runs else 0.0)
                ex_stds.append(statistics.stdev(ex_runs) if len(ex_runs) > 1 else 0.0)
            cells = []
            for i in range(len(models)):
                ham_part = _format_with_rank(ham_vals, ham_stds, i)
                ex_part = _format_with_rank(ex_vals, ex_stds, i)
                cells.append(f"{ham_part} / {ex_part}")
            cells_per_task.append(cells)

        else:
            vals, stds = [], []
            for m in models:
                if "|ner" in tkey or "|pos" in tkey:
                    runs = data[m][tkey].get('overall_f1', [])
                else:
                    runs = data[m][tkey].get('weighted_f1', [])
                vals.append(statistics.mean(runs) if runs else 0.0)
                stds.append(statistics.stdev(runs) if len(runs) > 1 else 0.0)
            cells_per_task.append([_format_with_rank(vals, stds, i) for i in range(len(models))])

    # Group columns by corpus for the multi-column header
    corpus_spans = []  # list of (corpus, span)
    if ordered_columns:
        cur = ordered_columns[0][0]
        span = 0
        for corpus, _, _ in ordered_columns:
            if corpus == cur:
                span += 1
            else:
                corpus_spans.append((cur, span))
                cur = corpus
                span = 1
        corpus_spans.append((cur, span))

    output = []
    output.append(r"\begin{table*}[t!]")
    output.append(r"\tiny")
    output.append(r"\centering")
    col_spec = "|l|" + "c|" * len(ordered_columns)
    output.append(f"\\begin{{tabular}}{{{col_spec}}}")
    output.append(r"\hline")

    # Header row 1: corpus groups
    corpus_header_parts = [r"\textbf{Model}"]
    for corpus, span in corpus_spans:
        corpus_header_parts.append(
            f"\\multicolumn{{{span}}}{{c|}}{{\\textbf{{{corpus.replace('_', '-').upper()}}}}}"
        )
    output.append(" & ".join(corpus_header_parts) + r" \\ ")
    output.append(r"\cline{2-" + str(len(ordered_columns) + 1) + "}")

    # Header row 2: task types
    task_header_parts = [""]
    for _, task_type, _ in ordered_columns:
        task_header_parts.append(f"\\textbf{{{task_type.upper()}}}")
    output.append(" & ".join(task_header_parts) + r" \\ ")
    output.append(r"\hline")

    # Body: one row per model
    for i, name in enumerate(model_names):
        row = [name.replace('_', '-')] + [cells_per_task[c][i] for c in range(len(ordered_columns))]
        output.append(" & ".join(row) + r" \\ ")

    output.append(r"\hline")
    output.append(r"\end{tabular}")
    output.append(r"\caption{Performance on biomedical tasks in French (models on rows). "
                  r"Results shown as mean$\pm$std computed from individual runs. "
                  r"Best model per task in bold, second best underlined.}")
    output.append(r"\label{table:raw_results_transposed}")
    output.append(r"\end{table*}")

    return "\n".join(output)


def generate_pairwise_latex_table(bootstrap_results: Dict) -> str:
    """
    Generate a LaTeX table showing pairwise win probabilities between all model pairs.
    
    Args:
        bootstrap_results: Results from compute_bootstrap_win_probability
    
    Returns:
        LaTeX table as string
    """
    model_names = list(bootstrap_results["model_summary"].keys())
    n_models = len(model_names)
    
    # Build matrix of win probabilities
    win_matrix = np.zeros((n_models, n_models))
    np.fill_diagonal(win_matrix, 0.5)  # Diagonal is 0.5 (tie with self)
    
    for pair_key, pair_data in bootstrap_results["pairwise"].items():
        # Parse "Model A vs Model B"
        parts = pair_key.split(" vs ")
        if len(parts) != 2:
            continue
        name_a, name_b = parts
        
        if name_a in model_names and name_b in model_names:
            idx_a = model_names.index(name_a)
            idx_b = model_names.index(name_b)
            win_prob_a = pair_data["overall_win_prob_a"]
            win_matrix[idx_a][idx_b] = win_prob_a
            win_matrix[idx_b][idx_a] = 1 - win_prob_a
    
    # Build LaTeX table
    lines = []
    lines.append(r"\begin{table*}[t!]")
    lines.append(r"\tiny")
    lines.append(r"\centering")
    
    col_spec = "|l|" + "c|" * n_models
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")
    
    # Header row
    header = r"\textbf{Model}"
    for name in model_names:
        name_escaped = name.replace("_", r"\_")
        header += f" & \\textbf{{{name_escaped}}}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\hline")
    
    # Data rows
    for i, row_name in enumerate(model_names):
        row_name_escaped = row_name.replace("_", r"\_")
        row = row_name_escaped
        for j in range(n_models):
            val = win_matrix[i][j]
            if i == j:
                row += " & -"
            elif val > 0.55:
                row += f" & \\textbf{{{val:.2f}}}"
            elif val < 0.45:
                row += f" & \\textit{{{val:.2f}}}"
            else:
                row += f" & {val:.2f}"
        row += r" \\"
        lines.append(row)
    
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Pairwise win probabilities (row model vs column model). "
                 r"Values $>$ 0.55 in bold, values $<$ 0.45 in italics.}")
    lines.append(r"\label{table:pairwise_win}")
    lines.append(r"\end{table*}")
    
    return "\n".join(lines)


def print_raw_metrics_table_transposed(
    data: Dict,
    models: List[str],
    tasks: List[str],
    model_mapping: Dict[str, str],
) -> str:
    """
    Plaintext counterpart to generate_raw_metrics_latex_table_transposed.

    Models on rows, tasks on columns, two-row header (corpus group + task type).
    Cells: mean±std (×100). For regr → EDRM/Spearman; for mcqa → Hamming/Exact.
    Best per column marked with a trailing "*", second-best with "^".
    """
    from collections import defaultdict

    corpus_tasks = defaultdict(list)
    for t in tasks:
        parts = t.split("|")
        if len(parts) >= 2:
            corpus_tasks[parts[0]].append((parts[1], t))

    ordered_columns = []  # (corpus, task_type, tkey)
    for corpus, task_list in corpus_tasks.items():
        for task_type, tkey in task_list:
            if all(m in data and tkey in data[m] for m in models):
                ordered_columns.append((corpus, task_type, tkey))

    model_names = [
        model_mapping.get(m.replace("../../../models/", ""), m)
        for m in models
    ]

    def _mark(values, stds, i):
        order = sorted(range(len(values)), key=lambda k: values[k], reverse=True)
        best = order[0]
        second = order[1] if len(order) > 1 else None
        s = f"{values[i]*100:.2f}±{stds[i]*100:.2f}"
        if i == best:
            return s + "*"
        if i == second:
            return s + "^"
        return s

    cells_per_task = []  # list aligned with ordered_columns; each is list[len(models)]
    for corpus, task_type, tkey in ordered_columns:
        if "regr" in tkey:
            v1, v2, s1, s2 = [], [], [], []
            for m in models:
                e = data[m][tkey].get('edrm', [])
                p = data[m][tkey].get('spearman_correlation_coef', [])
                v1.append(statistics.mean(e) if e else 0.0)
                s1.append(statistics.stdev(e) if len(e) > 1 else 0.0)
                v2.append(statistics.mean(p) if p else 0.0)
                s2.append(statistics.stdev(p) if len(p) > 1 else 0.0)
            cells_per_task.append([f"{_mark(v1, s1, i)} / {_mark(v2, s2, i)}" for i in range(len(models))])
        elif "mcqa" in tkey:
            v1, v2, s1, s2 = [], [], [], []
            for m in models:
                h = data[m][tkey].get('hamming_score', [])
                ex = data[m][tkey].get('exact_match', [])
                v1.append(statistics.mean(h) if h else 0.0)
                s1.append(statistics.stdev(h) if len(h) > 1 else 0.0)
                v2.append(statistics.mean(ex) if ex else 0.0)
                s2.append(statistics.stdev(ex) if len(ex) > 1 else 0.0)
            cells_per_task.append([f"{_mark(v1, s1, i)} / {_mark(v2, s2, i)}" for i in range(len(models))])
        else:
            vals, stds = [], []
            for m in models:
                if "|ner" in tkey or "|pos" in tkey:
                    runs = data[m][tkey].get('overall_f1', [])
                else:
                    runs = data[m][tkey].get('weighted_f1', [])
                vals.append(statistics.mean(runs) if runs else 0.0)
                stds.append(statistics.stdev(runs) if len(runs) > 1 else 0.0)
            cells_per_task.append([_mark(vals, stds, i) for i in range(len(models))])

    # Group columns by corpus for the multi-column header
    corpus_spans = []  # list of (corpus, span)
    if ordered_columns:
        cur = ordered_columns[0][0]
        span = 0
        for corpus, _, _ in ordered_columns:
            if corpus == cur:
                span += 1
            else:
                corpus_spans.append((cur, span))
                cur = corpus
                span = 1
        corpus_spans.append((cur, span))

    # Column widths: model column + one per task
    model_col_w = max([len("Model")] + [len(n) for n in model_names])
    task_widths = []
    for c, (_, task_type, _) in enumerate(ordered_columns):
        cell_w = max(len(task_type), max(len(cells_per_task[c][i]) for i in range(len(models))) if models else 0)
        task_widths.append(cell_w)

    # Ensure each corpus group is at least as wide as the corpus name (expand last col in group if needed)
    idx = 0
    for corpus, span in corpus_spans:
        group_cols = task_widths[idx:idx + span]
        group_w = sum(group_cols) + 3 * (span - 1)  # " | " between cols
        cname = corpus.upper()
        if len(cname) > group_w:
            extra = len(cname) - group_w
            task_widths[idx + span - 1] += extra
        idx += span

    sep = " | "
    total_width = model_col_w + len(sep) + sum(task_widths) + len(sep) * len(task_widths) - (len(sep) if not task_widths else 0)

    lines = []
    lines.append("\n" + "=" * total_width)
    lines.append("RAW METRICS PER TASK (mean ± std, ×100; * best per task, ^ second best)")
    lines.append("=" * total_width)

    # Header row 1: model header + corpus groups (centered over their span)
    parts = [f"{'':<{model_col_w}}"]
    idx = 0
    for corpus, span in corpus_spans:
        group_cols = task_widths[idx:idx + span]
        group_w = sum(group_cols) + 3 * (span - 1)
        parts.append(f"{corpus.upper():^{group_w}}")
        idx += span
    lines.append(sep.join(parts))

    # Header row 2: model + per-task type
    parts = [f"{'Model':<{model_col_w}}"]
    for c, (_, task_type, _) in enumerate(ordered_columns):
        parts.append(f"{task_type.upper():<{task_widths[c]}}")
    lines.append(sep.join(parts))
    lines.append("-" * total_width)

    # Body
    for i, name in enumerate(model_names):
        parts = [f"{name:<{model_col_w}}"]
        for c in range(len(ordered_columns)):
            parts.append(f"{cells_per_task[c][i]:<{task_widths[c]}}")
        lines.append(sep.join(parts))

    lines.append("=" * total_width)
    return "\n".join(lines)


def print_summary_table(
    raw_results: Optional[Dict] = None,
    zscore_results: Optional[Dict] = None,
    minmax_results: Optional[Dict] = None,
    rank_results: Optional[Dict] = None,
    bootstrap_results: Optional[Dict] = None,
    sort_by: str = "mean_win_prob"
) -> str:
    """
    Print a formatted summary table to console.

    Args:
        raw_results: Results from compute_raw_scores_on_runs (or None to skip column)
        zscore_results: Results from compute_zscore_on_runs (or None to skip column)
        minmax_results: Results from compute_minmax_on_runs (or None to skip column)
        rank_results: Results from compute_rank_on_runs (or None to skip column)
        bootstrap_results: Results from compute_bootstrap_win_probability (or None to skip column)
        sort_by: Metric to sort by ('mean_win_prob', 'mean_normalized', 'mean_zscore', 'mean_rank', 'mean_raw')

    Returns:
        Formatted string table
    """
    all_models = set()
    if raw_results is not None:
        all_models |= set(raw_results.keys())
    if zscore_results is not None:
        all_models |= set(zscore_results.keys())
    if minmax_results is not None:
        all_models |= set(minmax_results.keys())
    if rank_results is not None:
        all_models |= set(rank_results.keys())

    model_data = []
    for model in all_models:
        if sort_by == "mean_win_prob":
            val = bootstrap_results["model_summary"].get(model, {}).get("mean_win_prob", float("-inf")) if bootstrap_results else float("-inf")
        elif sort_by == "mean_normalized":
            val = minmax_results.get(model, {}).get("mean_normalized", float("-inf")) if minmax_results else float("-inf")
        elif sort_by == "mean_zscore":
            val = zscore_results.get(model, {}).get("mean_zscore", float("-inf")) if zscore_results else float("-inf")
        elif sort_by == "mean_rank":
            val = -(rank_results.get(model, {}).get("mean_rank", float("inf")) if rank_results else float("inf"))
        elif sort_by == "mean_raw":
            val = raw_results.get(model, {}).get("mean_raw", float("-inf")) if raw_results else float("-inf")
        else:
            val = 0
        model_data.append((model, val))

    model_data.sort(key=lambda x: x[1], reverse=True)
    
    # Build headers and row data dynamically based on which results are provided
    headers = ["Model"]
    if raw_results is not None:
        headers.append("Raw Score")
    if zscore_results is not None:
        headers.append("Z-Score")
    if minmax_results is not None:
        headers.append("Min-Max")
    if rank_results is not None:
        headers.append("Avg Rank")
    if bootstrap_results is not None:
        headers.append("Win Prob")
    
    # First pass: compute all formatted strings to determine column widths
    rows = []
    for model, _ in model_data:
        display_name = mapping.get(model.replace("../../../models/", ""), model)
        row = [display_name]
        
        if raw_results is not None:
            raw_data = raw_results.get(model, {})
            raw_str = f"{raw_data.get('mean_raw', 0)*100:.2f}±{raw_data.get('se_raw', 0)*100:.2f}" if "mean_raw" in raw_data else "-"
            if "mean_raw" in raw_data:
                raw_str += f" [{raw_data.get('ci_lower', 0)*100:.2f},{raw_data.get('ci_upper', 0)*100:.2f}]"
            row.append(raw_str)

        if zscore_results is not None:
            z_data = zscore_results.get(model, {})
            z_str = f"{z_data.get('mean_zscore', 0):.3f}±{z_data.get('se_zscore', 0):.3f}" if z_data else "-"
            if z_data:
                z_str += f" [{z_data.get('ci_lower', 0):.2f},{z_data.get('ci_upper', 0):.2f}]"
            row.append(z_str)

        if minmax_results is not None:
            m_data = minmax_results.get(model, {})
            m_str = f"{m_data.get('mean_normalized', 0)*100:.2f}±{m_data.get('se_normalized', 0)*100:.2f}" if m_data else "-"
            if m_data:
                m_str += f" [{m_data.get('ci_lower', 0)*100:.2f},{m_data.get('ci_upper', 0)*100:.2f}]"
            row.append(m_str)

        if rank_results is not None:
            r_data = rank_results.get(model, {})
            r_str = f"{r_data.get('mean_rank', 0):.2f}±{r_data.get('se_rank', 0):.2f}" if r_data else "-"
            row.append(r_str)

        if bootstrap_results is not None:
            b_data = bootstrap_results["model_summary"].get(model, {})
            w_str = f"{b_data.get('mean_win_prob', 0)*100:.2f}±{b_data.get('se_win_prob', 0)*100:.2f}" if b_data else "-"
            row.append(w_str)
        
        rows.append(tuple(row))
    
    # Calculate column widths (max of header and all values)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))
    
    # Build table
    lines = []
    total_width = sum(col_widths) + 3 * len(col_widths) - 1  # account for " | " separators
    
    lines.append("\n" + "=" * total_width)
    lines.append("AGGREGATION SUMMARY (two-level: mean over runs per task, then aggregate across tasks; values show mean ± SE)")
    lines.append("=" * total_width)
    
    # Header row
    header_parts = [f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers))]
    lines.append(" | ".join(header_parts))
    lines.append("-" * total_width)
    
    # Data rows
    for row in rows:
        row_parts = [f"{row[i]:<{col_widths[i]}}" for i in range(len(row))]
        lines.append(" | ".join(row_parts))
    
    lines.append("=" * total_width)
    
    return "\n".join(lines)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("AGGREGATE METRICS FROM INDIVIDUAL RUNS")
    print("=" * 80)
    print(f"Run label: {run_label}")
    print(f"Results file: {results_file}")
    print(f"Number of bootstrap samples: {NUM_BOOTSTRAP_SAMPLES}")
    print(f"Confidence level: {CONFIDENCE_LEVEL * 100}%")
    
    # Create output directory
    output_dir = f"./stats/{run_label}"
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # Task Filtering Analysis
    # =========================================================================
    do_task_filtering = False

    if do_task_filtering:
        # Compute task discriminability metrics
        task_discriminability = compute_task_discriminability(
            data=raw_data,
            models=models,
            tasks=tasks,
            task_filter=tasks_filter
        )
        
        # Compute cross-task correlation
        corr_matrix, corr_task_names, task_avg_corr = compute_cross_task_correlation(
            data=raw_data,
            models=models,
            tasks=tasks,
            task_filter=tasks_filter
        )
        
        # Plot cross-task correlation heatmap
        plot_cross_task_correlation(
            corr_matrix=corr_matrix,
            task_names=corr_task_names,
            output_path=f"{output_dir}/cross_task_correlation.png"
        )
        
        # Print combined task quality summary
        print("\n" + "=" * 100)
        print("TASK QUALITY SUMMARY (SNR + Correlation)")
        print("=" * 100)
        print("Interpretation:")
        print("  - High SNR + High Avg|r|  → Good reliable task, consistent with benchmark")
        print("  - High SNR + Low Avg|r|   → Unique task measuring different ability (keep it)")
        print("  - Low SNR + High Avg|r|   → Noisy but aligned with benchmark (consider keeping)")
        print("  - Low SNR + Low Avg|r|    → Potentially low-quality task (consider removing)")
        print("-" * 100)
        print(f"{'Task':<35} {'SNR':>10} {'Avg|r|':>10} {'Range':>10} {'Assessment':<25}")
        print("-" * 100)
        
        for task in corr_task_names:
            disc = task_discriminability.get(task, {})
            corr = task_avg_corr.get(task, {})
            
            snr = disc.get("snr", 0)
            avg_corr = corr.get("avg_abs_corr", 0)
            score_range = disc.get("range", 0)
            
            # Categorize task quality
            if snr >= 1.0 and avg_corr >= 0.4:
                assessment = "Good (reliable, aligned)"
            elif snr >= 1.0 and avg_corr < 0.4:
                assessment = "Unique (reliable, different)"
            elif snr < 1.0 and avg_corr >= 0.4:
                assessment = "Noisy but aligned"
            else:
                assessment = "CHECK: Low SNR + Low corr"
            
            snr_str = f"{snr:.2f}" if snr != float('inf') else "inf"
            print(f"{task:<35} {snr_str:>10} {avg_corr:>10.3f} {score_range:>10.4f} {assessment:<25}")
        
        print("=" * 100)
        
        # Optional: Filter tasks by discriminability
        # Uncomment to apply filtering:
        # discriminating_tasks = filter_tasks_by_discriminability(
        #     task_discriminability,
        #     min_range=0.02,   # Minimum score range across models
        #     min_cv=0.01,      # Minimum coefficient of variation
        #     min_snr=1.0,      # Minimum signal-to-noise ratio (between-model var / within-model var)
        # )
        # uncorrelated_tasks = filter_tasks_by_correlation(
        #     corr_matrix, corr_task_names, task_discriminability, max_corr=0.9
        # )

        exit(0)  # Exit after task filtering analysis for now (uncomment to proceed to aggregation)
    
    # =========================================================================
    # Compute all aggregations
    # =========================================================================
    
    zscore_results = compute_zscore_on_runs(
        data=raw_data,
        models=models,
        tasks=tasks,
        task_filter=tasks_filter
    )
    
    minmax_results = compute_minmax_on_runs(
        data=raw_data,
        models=models,
        tasks=tasks,
        task_filter=tasks_filter
    )
    
    rank_results = compute_rank_on_runs(
        data=raw_data,
        models=models,
        tasks=tasks,
        task_filter=tasks_filter
    )
    
    bootstrap_results = compute_bootstrap_win_probability(
        data=raw_data,
        models=models,
        tasks=tasks,
        task_filter=tasks_filter,
        n_bootstrap=NUM_BOOTSTRAP_SAMPLES,
        seed=RANDOM_SEED
    )
    
    raw_score_results = compute_raw_scores_on_runs(
        data=raw_data,
        models=models,
        tasks=tasks,
        task_filter=tasks_filter
    )

    # Print summary to console (sorted by win prob, most robust metric)
    summary = print_summary_table(
        raw_score_results,
        # zscore_results,
        None,
        minmax_results,
        rank_results,
        bootstrap_results,
        sort_by="mean_win_prob",
    )
    print(summary)

    # # Print raw per-task metrics (models on rows, tasks on columns)
    # raw_console = print_raw_metrics_table_transposed(
    #     data=raw_data,
    #     models=models,
    #     tasks=tasks,
    #     model_mapping=mapping,
    # )
    # print(raw_console)

    # # Generate and save combined LaTeX table (sorted by win prob, most robust metric)
    # combined_latex = generate_combined_latex_table(
    #     zscore_results, minmax_results, rank_results, bootstrap_results,
    #     sort_by="mean_win_prob"
    # )
    # combined_file = f"{output_dir}/table_aggregation_runs.tex"
    # with open(combined_file, "w") as f:
    #     f.write(combined_latex)
    # print(f"\nCombined aggregation table saved to: {combined_file}")
    
    # # Generate and save pairwise win probability table
    # pairwise_latex = generate_pairwise_latex_table(bootstrap_results)
    # pairwise_file = f"{output_dir}/table_pairwise_win.tex"
    # with open(pairwise_file, "w") as f:
    #     f.write(pairwise_latex)
    # print(f"Pairwise win probability table saved to: {pairwise_file}")
    
    # Generate and save raw (non-normalized) metrics table
    # Filter tasks if task_filter is specified
    filtered_tasks = filter_tasks(tasks, tasks_filter)
    raw_latex = generate_raw_metrics_latex_table(
        data=raw_data,
        models=models,
        tasks=tasks,
        # tasks=filtered_tasks,  # print all tasks
        model_mapping=mapping
    )
    raw_file = f"{output_dir}/table_raw_metrics.tex"
    with open(raw_file, "w") as f:
        f.write(raw_latex)
    print(f"Raw metrics table saved to: {raw_file}")

    # Transposed: models on rows, tasks on columns
    raw_latex_t = generate_raw_metrics_latex_table_transposed(
        data=raw_data,
        models=models,
        # tasks=tasks,
        tasks=filtered_tasks,  # print all tasks
        model_mapping=mapping,
    )
    raw_file_t = f"{output_dir}/table_raw_metrics_transposed.tex"
    with open(raw_file_t, "w") as f:
        f.write(raw_latex_t)
    print(f"Raw metrics table (transposed) saved to: {raw_file_t}")
    
    # Save full results as JSON
    full_results = {
        "metadata": {
            "run_label": run_label,
            "results_file": results_file,
            "n_bootstrap": NUM_BOOTSTRAP_SAMPLES,
            "confidence_level": CONFIDENCE_LEVEL,
            "random_seed": RANDOM_SEED,
            "tasks_used": filter_tasks(tasks, tasks_filter),
            "models": list(mapping.values()),
        },
        # "task_analysis": {
        #     "discriminability": task_discriminability,
        #     "cross_task_correlation": {
        #         "task_names": corr_task_names,
        #         "correlation_matrix": corr_matrix.tolist(),
        #         "per_task_avg_correlation": task_avg_corr,
        #     },
        # },
        "zscore": zscore_results,
        "minmax": minmax_results,
        "rank": rank_results,
        "bootstrap_pairwise": bootstrap_results,
    }
    
    json_file = f"{output_dir}/aggregation_from_runs.json"
    with open(json_file, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"Full aggregation results saved to: {json_file}")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
