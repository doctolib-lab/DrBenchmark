import os
import sys
from collections import defaultdict
import json
import pandas as pd

# Optional arguments:
#   argv[1]: run label (e.g., "recipes" or "recipes_hpo")
#   argv[2]: explicit path to overall_averaged_metrics JSON
run_label = sys.argv[1] if len(sys.argv) > 1 else "recipes"
if len(sys.argv) > 2:
    stats_file = sys.argv[2]
else:
    stats_file = f"./stats/{run_label}/overall_averaged_metrics.json"
    if not os.path.exists(stats_file):
        legacy = f"./stats/overall_averaged_metrics_{run_label}.json"
        if os.path.exists(legacy):
            stats_file = legacy
        else:
            raise FileNotFoundError(f"Stats file not found: {stats_file}")

with open(stats_file, "r") as f:
    data = json.load(f)

# Model mapping (unchanged)
mapping = {
    # open source french general
    # "almanach_camemberta-base": "CamemBERTa",
    # "camembert-base": "CamemBERT",
    # "flaubert_flaubert_base_uncased": "FlauBERT",
    # "almanach_moderncamembert-cv2-base": "ModernCamemBERT",
    # open source french medical
    # "almanach_camembert-bio-base": "CamemBERT-BIO",
    # "dr-bert_drbert-7gb": "DrBERT 7GB",  # good
    # "dr-bert_drbert-4gb-cp-pubmedbert": "DrBERT CP PubMedBERT",
    # open source english medical
    # "microsoft_biomednlp-pubmedbert-base-uncased-abstract-fulltext": "PubMedBERT",
    # "microsoft_biomednlp-pubmedbert-base-uncased-abstract": "OLD-PubMedBERT",
    # "thomas-sounack_bioclinical-modernbert-base": "Bio clinical ModernBERT",
    # doctobert exp 1 (nachos)
    # "doctobert_dynamic_mlm": "DoctoBERT Nachos",  # good
    # "doctobert_phase_1": "DoctoBERT p1",
    # "doctobert_phase_2": "DoctoBERT p2",
    # "doctobert_phase_2_gelu": "DoctoBERT p2 gelu",
    # "doctobert_dynamic_mlm_gelu": "DoctoBERT p3 gelu",
    # "ct-moderncamembert-decay-dynamic": "ModernCamemBERT NACHOS",
    # "ct-bio-clinical-modernbert-decay-dynamic": "Bio Clinical ModernBERT NACHOS",
    # "cp-moderncamembert-base-p1-54b-dynamic": "ModernCamembert Short context",
    # "doctobert-exp-nachos-base-pretrain_750_128b_512_toks-lr-decay_8B_dynamic_mlm_prob-hf": "DoctoBERT Short Context",
    # doctobert exp 2 (fineweb2)
    # "doctobert-exp-fineweb2-base-pretrain-30b": "DoctoBERT 30b",
    # "doctobert-exp-fineweb2-base-pretrain-30b-clinical": "DoctoBERT 30b clinical",  # good
    # "doctobert-exp-fineweb2-base-pretrain-30b-edu2": "DoctoBERT 30b edu 2",
    # "doctobert-exp-fineweb2-base-pretrain-30b-edu3": "DoctoBERT 30b edu 3",
    # "doctobert-exp-fineweb2-base-pretrain-30b-clinical-cs": "DoctoBERT 30b clinical cs",
    # "doctobert-exp-fineweb2-base-pretrain-30b-clinical-edu2-ep9": "DoctoBERT 30b clinical edu 2",
    # "doctobert-exp-fineweb2-base-pretrain-15b-clinical-vocab50k-ep3": "DoctoBERT Fineweb2 15B Clinical Vocab50k",
    # doctobert exp 3 (final fr)
    "doctobert-03-fr-base-pretrain-102b-hf-ep0-ba3923-rank0": "EP0 BA3923",
    "doctobert-03-fr-base-pretrain-102b-hf-ep0-ba7843-rank0": "EP0 BA7843",
    "doctobert-03-fr-base-pretrain-102b-hf-ep1-ba11234-rank0": "EP1 BA11234",
    "doctobert-03-fr-base-pretrain-102b-hf-ep2-ba14731-rank0": "EP2 BA14731",
    "doctobert-03-fr-base-pretrain-102b-hf-ep2-ba18799-rank0": "EP2 BA18799",
    "doctobert-03-fr-base-pretrain-102b-hf-ep3-ba22825-rank0": "EP3 BA22825",
}

models = ["../../../models/" + m.lower().replace("/","_") for m in mapping.keys()]

# Get all unique tasks present for all models
# tasks = []
# for model in models:
#     if model in data and data[model]:
#         tasks = list(data[model].keys())
#         break

# if not tasks:
#     print("Error: No tasks found in any model data")
#     exit()

tasks = [
    'cas|pos|1.0',
    # 'cas|cls|1.0',
    # 'cas|ner-spec|1.0',
    # 'cas|ner-neg|1.0',
    'essai|pos|1.0',
    # 'essai|cls|1.0',
    # 'essai|ner-spec|1.0',
    # 'essai|ner-neg|1.0',
    'quaero|ner-emea|1.0',
    'quaero|ner-medline|1.0',
    'e3c|ner-clinical|1.0',
    'e3c|ner-temporal|1.0',
    'morfitt|cls|1.0',
    'frenchmedmcqa|mcqa|1.0',
    'frenchmedmcqa|cls|1.0',
    'clister|regr|1.0',
    'deft2020|regr|1.0',
    'deft2020|cls|1.0',
    'deft2021|cls|1.0',
    'deft2021|ner|1.0',
    'diamed|cls|1.0',
    'pxcorpus|ner|1.0',
    'pxcorpus|cls|1.0',
]

# Organize tasks by corpus
corpus_tasks = defaultdict(list)
for t in tasks:
    corpus, task, fewshot = t.split("|")
    corpus_tasks[corpus].append((task, t))

data_to_plot = []
for corpus in corpus_tasks:
    task_list = corpus_tasks[corpus]
    num_rows = len(task_list)
    corpus_written = False
    for idx, (task, tkey) in enumerate(task_list):
        # Skip tasks that don't exist for all models
        task_exists_for_all = all(m in data and tkey in data[m] for m in models)
        if not task_exists_for_all:
            continue

        runs_metrics = []

        if tkey.find("deft2020|regr") != -1 or tkey.find("clister|regr") != -1:
            # Regression (EDRM / Spearman)
            edrm_values, spear_values = [], []
            for m in models:
                edrm_avg = data[m][tkey]['edrm']['avg']
                edrm_std = data[m][tkey]['edrm']['std']
                spear_avg = data[m][tkey]['spearman_correlation_coef']['avg']
                spear_std = data[m][tkey]['spearman_correlation_coef']['std']
                edrm_values.append(edrm_avg)
                spear_values.append(spear_avg)
                data_to_plot.append({
                    "model": mapping.get(m.replace("../../../models/",""), m),
                    "corpus": corpus,
                    "task_type": task.split("-")[0],
                    "task_sub": f"{task}-edrm",
                    "task": f"{corpus}-{task}-edrm",
                    "avg": edrm_avg,
                    "std": edrm_std,
                })
                data_to_plot.append({
                    "model": mapping.get(m.replace("../../../models/",""), m),
                    "corpus": corpus,
                    "task_type": task.split("-")[0],
                    "task_sub": f"{task}-spear",
                    "task": f"{corpus}-{task}-spear",
                    "avg": spear_avg,
                    "std": spear_std,
                })
        elif tkey.find("frenchmedmcqa|mcqa") != -1:
            # MCQA (Hamming / Exact)
            hamming_values, exact_values = [], []
            for m in models:
                hamming_avg = data[m][tkey]['hamming_score']['avg']
                hamming_std = data[m][tkey]['hamming_score']['std']
                exact_avg = data[m][tkey]['exact_match']['avg']
                exact_std = data[m][tkey]['exact_match']['std']
                hamming_values.append(hamming_avg)
                exact_values.append(exact_avg)
                data_to_plot.append({
                    "model": mapping.get(m.replace("../../../models/",""), m),
                    "corpus": corpus,
                    "task_type": task.split("-")[0],
                    "task_sub": f"{task}-hamming",
                    "task": f"{corpus}-{task}-hamming",
                    "avg": hamming_avg,
                    "std": hamming_std,
                })
                data_to_plot.append({
                    "model": mapping.get(m.replace("../../../models/",""), m),
                    "corpus": corpus,
                    "task_type": task.split("-")[0],
                    "task_sub": f"{task}-exact",
                    "task": f"{corpus}-{task}-exact",
                    "avg": exact_avg,
                    "std": exact_std,
                })
        else:
            # Single metric (NER, POS, CLS)
            comparison_values = []
            for m in models:
                if tkey.find("|ner") != -1 or tkey.find("|pos") != -1:
                    f1_avg = data[m][tkey]['overall_f1']['avg']
                    f1_std = data[m][tkey]['overall_f1']['std']
                    metric = f"{round(f1_avg*100,2)}$\\pm${round(f1_std*100,2)}"
                    comparison_values.append(f1_avg)
                    data_to_plot.append({
                        "model": mapping.get(m.replace("../../../models/",""), m),
                        "corpus": corpus,
                        "task_type": task.split("-")[0],
                        "task_sub": f"{task}-f1",
                        #"task": f"{corpus}-{task}-overall_f1",
                        "task": f"{corpus}-{task}-f1",
                        "avg": f1_avg,
                        "std": f1_std,
                    })
                else:
                    wf1_avg = data[m][tkey]['weighted_f1']['avg']
                    wf1_std = data[m][tkey]['weighted_f1']['std']
                    metric = f"{round(wf1_avg*100,2)}$\\pm${round(wf1_std*100,2)}"
                    comparison_values.append(wf1_avg)
                    data_to_plot.append({
                        "model": mapping.get(m.replace("../../../models/",""), m),
                        "corpus": corpus,
                        "task_type": task.split("-")[0],
                        "task_sub": f"{task}-weighted_f1",
                        "task": f"{corpus}-{task}-weighted_f1",
                        "avg": wf1_avg,
                        "std": wf1_std,
                    })

df = pd.DataFrame(data_to_plot)
df.head()