import os
import sys
from collections import defaultdict
import json

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

# f_in = open("./models.txt","r")
# models = ["../../../models/" + m.lower().replace("/","_") for m in f_in.read().split("\n") if m.strip()]
# f_in.close()

# Model mapping (unchanged)
mapping = {
    # open source french general
    # "almanach_camemberta-base": "CamemBERTa",
    # "camembert-base": "CamemBERT",
    # "flaubert_flaubert_base_uncased": "FlauBERT",
    # "almanach_camembert-base": "CamemBERT",
    # "almanach_camemberta-base": "CamemBERTa",
    # "almanach_moderncamembert-cv2-base": "ModernCamemBERT",
    # open source french medical
    "almanach_camembert-bio-base": "CamemBERT-BIO",
    "dr-bert_drbert-7gb": "DrBERT",  # good
    "jknafou_transbert-bio-fr": "TransBERT-BIO",  # good
    # "dr-bert_drbert-4gb-cp-pubmedbert": "DrBERT CP PubMedBERT",
    # open source english medical
    # "microsoft_biomednlp-pubmedbert-base-uncased-abstract-fulltext": "PubMedBERT",
    # "microsoft_biomednlp-pubmedbert-base-uncased-abstract": "OLD-PubMedBERT",
    # "dmis-lab_biobert-v1.1": "Biobert-v1.1",
    # "thomas-sounack_bioclinical-modernbert-base": "BioClinical-ModernBERT",
    # doctobert exp 1 (nachos)
    # "doctobert_dynamic_mlm": "DoctoBERT-Nachos-Scratch",  # silu one
    # "doctobert_phase_1": "DoctoBERT p1",
    # "doctobert_phase_2": "DoctoBERT p2",
    # "doctobert_phase_2_gelu": "DoctoBERT p2 gelu",
    # "doctobert_dynamic_mlm_gelu": "DoctoBERT-Nachos-Scratch",  # good, gelu one
    # "ct-moderncamembert-decay-dynamic": "DoctoBERT-Nachos-ModernCamemBERT",
    # "ct-bio-clinical-modernbert-decay-dynamic": "DoctoBERT-Nachos-BioClinicalModernBERT",
    # "cp-moderncamembert-base-p1-54b-dynamic": "ModernCamembert Short context",
    # "doctobert-exp-nachos-base-pretrain_750_128b_512_toks-lr-decay_8B_dynamic_mlm_prob-hf": "DoctoBERT Short Context",
    # doctobert exp 2 (fineweb2)
    # "doctobert-exp-fineweb2-base-pretrain-30b": "DoctoBERT-FW2-Unfiltered",
    # "doctobert-exp-fineweb2-base-pretrain-30b-edu2": "DoctoBERT-FW2-Edu2",
    # "doctobert-exp-fineweb2-base-pretrain-30b-edu3": "DoctoBERT-FW2-Edu3",
    # "doctobert-exp-fineweb2-base-pretrain-30b-clinical-edu2-ep9": "DoctoBERT-FW2-Edu2-BioCli",
    # "doctobert-exp-fineweb2-base-pretrain-30b-clinical": "DoctoBERT-FW2-BioCli",  # good
    # "doctobert-exp-fineweb2-base-pretrain-30b-clinical-cs": "DoctoBERT-BioCli-cs",
    # "doctobert-exp-fineweb2-base-pretrain-15b-clinical-vocab50k-ep3": "DoctoBERT-FW2-BioCli-Vocab50k", # good
    # doctobert exp 3 (final fr) v1: p1 on edu-2
    # "doctobert-03-fr-base-pretrain-102b-hf-ep0-ba3923-rank0": "DoctoBERT-P1-EP0-BA3923",
    # "doctobert-03-fr-base-pretrain-102b-hf-ep0-ba7843-rank0": "DoctoBERT-P1-EP0-BA7843",
    # "doctobert-03-fr-base-pretrain-102b-hf-ep1-ba11234-rank0": "DoctoBERT-P1-EP1-BA11234",
    # "doctobert-03-fr-base-pretrain-102b-hf-ep2-ba14731-rank0": "DoctoBERT-P1-EP2-BA14731",
    # "doctobert-03-fr-base-pretrain-102b-hf-ep2-ba18799-rank0": "DoctoBERT-P1-EP2-BA18799",
    # "doctobert-03-fr-base-pretrain-102b-hf-ep3-ba22825-rank0": "DoctoBERT-P1-EP3-BA22825",  # good
    # "doctobert-03-fr-base-pretrain-102b-context-extension-22b-hf-ep1-ba4728-rank0": "DoctoBERT-P2-EP1-BA4728",  # good
    # "doctobert-03-fr-base-pretrain-102b-context-extension-22b-lr-decay-32b-dynamic-30-15-hf-ep2-ba6783-rank0": "DoctoBERT-P3-EP2-BA6783",
    # "doctobert-03-fr-base-pretrain-102b-context-extension-22b-lr-decay-32b-dynamic-30-15-hf-ep2-ba6783-rank0": "DoctoBERT-P3-v1.0",  # good
    # doctobert exp 3 (final fr) v1.1: p1 on edu-2 + selected domains
    # "doctobert-03-fr-base-pretrain-58b-hf-ep3-ba12930-rank0": "DoctoBERT-58B-P1-EP3-BA12930",
    # "doctobert-03-fr-base-pretrain-66b-hf-ep0-ba2545-rank0": "DoctoBERT-66B-P1-EP0-BA2545",
    # "doctobert-03-fr-base-pretrain-66b-hf-ep0-ba4876-rank0": "DoctoBERT-66B-P1-EP0-BA4876",
    # "doctobert-03-fr-base-pretrain-66b-hf-ep1-ba7208-rank0": "DoctoBERT-66B-P1-EP1-BA7208",
    # "doctobert-03-fr-base-pretrain-66b-hf-ep1-ba9327-rank0": "DoctoBERT-66B-P1-EP1-BA9327",
    # "doctobert-03-fr-base-pretrain-66b-hf-ep2-ba12506-rank0": "DoctoBERT-66B-P1-EP2-BA12506",
    # "doctobert-03-fr-base-pretrain-66b-hf-ep3-ba14625-rank0": "DoctoBERT-66B-P1-EP3-BA14625", # "p1-selected-66b",    # "doctobert-03-fr-base-pretrain-66b-hf-ep3-ba16472-rank0": "DoctoBERT-66B-P1-EP3-BA16472",
    # "doctobert-03-fr-base-pretrain-66b-hf-ep3-ba18379-rank0": "DoctoBERT-66B-P1-EP3-BA18379",
    # "doctobert-03-fr-base-pretrain-66b-hf-ep4-ba20075-rank0": "DoctoBERT-66B-P1-EP4-BA20075",
    # "doctobert-03-fr-base-pretrain-66b-hf-ep4-ba21983-rank0": "DoctoBERT-66B-P1-EP4-BA21983", # "p1-selected-66b",    # "doctobert-03-fr-base-pretrain-66b-context-extension-16b-hf-ep2-ba3561-rank0": "DoctoBERT-66B-P2-EP2-BA3561",
    # "doctobert-03-fr-base-pretrain-66b-context-extension-16b-lr-decay-32b-dynamic-30-15-hf-ep2-ba6783-rank0": "p3-edu4-selected-32b",
    # "doctobert-fr-base-pretrain-66b-context-extension-16b-lr-decay-dynamic-30-15-edu5-selected-24b-hf-ep2-ba5087-rank0": "p3-edu5-selected-24b",
    # "doctobert-fr-base-pretrain-66b-context-extension-16b-lr-decay-dynamic-30-15-edu4-biomedical-22b-hf-ep3-ba4664-rank0": "p3-edu4-biomedical-22b",
    # "doctobert-fr-base-pretrain-66b-context-extension-16b-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "p3-edu4-clinical-21b",
    # "doctobert-fr-base-pretrain-66b-context-extension-16b-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "DoctoBERT-P3-v1.1",  # good
    # "doctobert-fr-base-pretrain-66b-context-extension-16b-lr-decay-dynamic-30-15-edu4-more-strict-selected-22b-hf-ep2-ba4664-rank0": "p3-edu4-more-strict-selected-22b",
    # doctobert exp 3 (final fr) v1.2: p1 on edu-4 + selected domains
    # "doctobert-fr-base-pretrain-52b-hf-ep1-ba4453-rank0": "DoctoBERT-52B-P1-EP1-BA4453",
    # "doctobert-fr-base-pretrain-52b-hf-ep2-ba8268-rank0": "DoctoBERT-52B-P1-EP2-BA8268",
    # "doctobert-fr-base-pretrain-52b-hf-ep2-ba11659-rank0": "DoctoBERT-52B-P1-EP2-BA11659", # "p1-selected-52b",    # only biomedical, clinical, biomedical+clinical
    # "doctobert-fr-base-pretrain-biomedical-19b-hf-ep3-ba4241-rank0": "DoctoBERT-P1-Bio",
    # "doctobert-fr-base-pretrain-biomedical-clinical-35b-hf-ep3-ba7843-rank0": "DoctoBERT-P1-BioCli",
    # "doctobert-fr-base-pretrain-clinical-16b-hf-ep1-ba1485-rank0": "DoctoBERT-16B-P1-EP1-BA1485",
    # "doctobert-fr-base-pretrain-clinical-16b-hf-ep2-ba2757-rank0": "DoctoBERT-16B-P1-EP2-BA2757",
    # "doctobert-fr-base-pretrain-clinical-16b-hf-ep3-ba3605-rank0": "DoctoBERT-P1-Cli", # "p1-clinical-16b",    # "doctobert-fr-base-pretrain-clinical-16b-hf-ep4-ba5022-rank0": "DoctoBERT-16B-P1-EP4-BA5022",
    # "doctobert-fr-base-pretrain-clinical-16b-hf-ep5-ba6294-rank0": "DoctoBERT-16B-P1-EP5-BA6294",
    # "doctobert-fr-base-pretrain-clinical-16b-hf-ep6-ba7142-rank0": "DoctoBERT-16B-P1-EP6-BA7142",
    # doctobert exp 3 (final fr) v1.3 (rework v1.1): p1 on edu-2 + selected domains
    # "doctobert-fr-base-pretrain-100b-hf-ep1-ba5724-rank0": "DoctoBERT-100B-P1-EP1-BA5724",
    # "doctobert-fr-base-pretrain-100b-hf-ep2-ba9963-rank0": "DoctoBERT-100B-P1-EP2-BA9963",
    # "doctobert-fr-base-pretrain-100b-hf-ep3-ba14202-rank0": "DoctoBERT-100B-P1-EP3-BA14202",
    # "doctobert-fr-base-pretrain-100b-hf-ep4-ba18440-rank0": "DoctoBERT-100B-P1-EP4-BA18440",
    # "doctobert-fr-base-pretrain-100b-hf-ep4-ba21831-rank0": "DoctoBERT-100B-P1-EP4-BA21831",
    # re-evaluate converted 66B intermediate checkpoints
    # "doctobert-fr-base-pretrain-66b-ep0-ba4876-context-extension-8b-hf-ep1-ba1781-rank0": "66B-EP0-BA4876-P2-EP1-BA1781",
    # "doctobert-fr-base-pretrain-66b-ep1-ba9327-context-extension-8b-hf-ep1-ba1781-rank0": "66B-EP1-BA9327-P2-EP1-BA1781",
    # "doctobert-fr-base-pretrain-66b-ep3-ba14625-context-extension-8b-hf-ep1-ba1781-rank0": "66B-EP3-BA14625-P2-EP1-BA1781",
    # p2
    # "doctobert-fr-base-pretrain-100b-ep1-ba5724-context-extension-8b-hf-ep1-ba1781-rank0": "100B-EP1-BA5724-P2-EP1-BA1781",
    # "doctobert-fr-base-pretrain-100b-ep2-ba9963-context-extension-8b-hf-ep1-ba1781-rank0": "100B-EP2-BA9963-P2-EP1-BA1781",
    # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-hf-ep0-ba424-rank0": "100B-EP3-BA14202-P2-EP0-BA424",
    # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-hf-ep0-ba848-rank0": "100B-EP3-BA14202-P2-EP0-BA848",
    # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-hf-ep1-ba1781-rank0": "100B-EP3-BA14202-P2-EP1-BA1781",
    # p3
    # "doctobert-fr-base-pretrain-100b-ep1-ba5724-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "100B-BA5724-P2-BA1781-P3-BA4452",
    # "doctobert-fr-base-pretrain-100b-ep2-ba9963-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "100B-BA9963-P2-BA1781-P3-BA4452",
    # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep0-ba424-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "100B-BA14202-P2-BA424-P3-BA4452",
    # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep0-ba848-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "100B-BA14202-P2-BA848-P3-BA4452",
    # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep1-ba1696-rank0": "100B-BA14202-P2-BA1781-P3-BA1696",
    # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep2-ba2968-rank0": "100B-BA14202-P2-BA1781-P3-BA2968",
    # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "100B-BA14202-P2-BA1781-P3-BA4452",
    # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "DoctoBERT-P3-Cli",  # good
    # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "DoctoBERT-P3-v1.1-rw",  # good
    # "doctobert-fr-base-pretrain-100b-ep3-ba14202-lr-decay-dynamic-30-15-edu4-clinical-21b-hf-ep3-ba4452-rank0": "100B-BA14202-P3-Clinical",
    # "doctobert-fr-base-pretrain-100b-ep3-ba14202-lr-decay-dynamic-30-15-edu4-more-strict-selected-33b-hf-ep3-ba6995-rank0": "100B-BA14202-P3-Strict",
    # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-more-strict-selected-33b-hf-ep3-ba6995-rank0": "DoctoBERT-P3-Str",
    # "doctobert-fr-base-pretrain-100b-ep3-ba14202-context-extension-8b-ep1-ba1781-lr-decay-dynamic-30-15-edu4-selected-48b-hf-ep3-ba10175-rank0": "DoctoBERT-P3-Sel",
    # "doctobert-fr-base-pretrain-transbert-23b-hf-ep1-ba2333-rank0": "DoctoBERT-P1-TCB-BA2333",
    # "doctobert-fr-base-pretrain-transbert-23b-hf-ep2-ba4029-rank0": "DoctoBERT-P1-TCB-BA4029",
    # "doctobert-fr-base-pretrain-transbert-23b-hf-ep3-ba5512-rank0": "DoctoBERT-P1-TCB-BA5512",  # good
    # "doctobert-fr-base-pretrain-transbert-23b-hf-ep4-ba7311-rank0": "DoctoBERT-P1-TCB-BA7311",
    # "doctobert-fr-base-pretrain-transbert-23b-hf-ep5-ba9007-rank0": "DoctoBERT-P1-TCB-BA9007",
    # "doctobert-fr-base-pretrain-transbert-23b-hf-ep6-ba10278-rank0": "DoctoBERT-P1-TCB-BA10278",
    # "doctobert-fr-base-pretrain-transbert-23b-hf-ep7-ba11974-rank0": "DoctoBERT-P1-TCB-BA11974",
    # "doctobert-fr-base-pretrain-transbert-23b-hf-ep8-ba13669-rank0": "DoctoBERT-P1-TCB-BA13669",
    # "doctobert-fr-base-pretrain-transbert-23b-hf-ep9-ba15789-rank0": "DoctoBERT-P1-TCB-BA15789",  # good
    # "doctobert-fr-base-pretrain-transbert-23b-bias-hf-ep3-ba5512-rank0": "DoctoBERT-P1-TCB-Bias",  # good
    # "doctobert-fr-base-pretrain-transbert-23b-bias-cls-hf-ep3-ba5512-rank0": "DoctoBERT-P1-TCB-Bias-CLS",
    # "doctobert-fr-base-pretrain-transbert-23b-bias-no-cls-hf-ep3-ba5512-rank0": "DoctoBERT-P1-TCB-Bias-CLS-No-Bias",  # good
    # "doctobert-fr-base-pretrain-transbert-23b-full-attn-hf-ep3-ba5512-rank0": "DoctoBERT-P1-TCB-Full-Attn",
    # "doctobert-fr-base-pretrain-transbert-23b-shallow-hf-ep3-ba5512-rank0": "DoctoBERT-P1-TCB-Shallow",
    # "doctobert-fr-base-pretrain-transbert-70b-linear-hf-ep2-ba3392-rank0": "DoctoBERT-P1-TCB-Lin-BA3392",
    # "doctobert-fr-base-pretrain-transbert-70b-linear-hf-ep4-ba6359-rank0": "DoctoBERT-P1-TCB-Lin-BA6359",
    # "doctobert-fr-base-pretrain-transbert-70b-linear-hf-ep6-ba9326-rank0": "DoctoBERT-P1-TCB-Lin-BA9326",
    # "doctobert-fr-base-pretrain-transbert-70b-linear-hf-ep8-ba12294-rank0": "DoctoBERT-P1-TCB-Lin-BA12294",
    # "doctobert-fr-base-pretrain-transbert-70b-linear-hf-ep9-ba14837-rank0": "DoctoBERT-P1-TCB-Lin-BA14837",
    # "doctobert-fr-base-pretrain-transbert-70b-linear-wo-pack-hf-ep9-ba46372-rank0": "DoctoBERT-P1-TCB-Lin-WoPack",  # good
    # "doctobert-fr-base-pretrain-transbert-70b-linear-wo-pack-mask15-hf-ep9-ba46372-rank0": "DoctoBERT-P1-TCB-Lin-WoPack-Mask15",
    # "doctobert-fr-base-pretrain-transbert-70b-linear-wo-pack-roberta-hf-ep10-ba47307-rank0": "DoctoBERT-P1-TCB-Lin-WoPack-RoBERTa",  # good
    # exp fineweb-2
    # baseline
    # "doctobert-exp-fineweb2-base-pretrain-nachos-hf-ep121-ba4367-rank0": "Nachos-Old",
    # "doctobert-exp-fineweb2-base-pretrain-nachos-new-hf-ep10-ba4368-rank0": "Nachos-1.3B-EP10",
    # "doctobert-exp-fineweb2-base-pretrain-transcorpus-hf-ep2-ba4368-rank0": "Transcorpus-5.2B-EP2",
    # "doctobert-exp-fineweb2-base-pretrain-synthesized-hf-ep15-ba4368-rank0": "Synthesized-777M-EP15",
    # filtering
    # "doctobert-exp-fineweb2-base-pretrain-unfiltered-hf-ep1-ba4368-rank0": "FW2-Unfiltered-7.2B-EP1",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-hf-ep11-ba4368-rank0": "FW2-Biocli-1.2B-EP11",
    # "doctobert-exp-fineweb2-base-pretrain-edu2-hf-ep3-ba4368-rank0": "FW2-Edu2-4.7B-EP3",
    # "doctobert-exp-fineweb2-base-pretrain-edu4-hf-ep8-ba4369-rank0": "FW2-Edu4-1.6B-EP8",
    # "doctobert-exp-fineweb2-base-pretrain-medterm01-hf-ep5-ba4368-rank0": "FW2-Medterm01-2.5B-EP5",
    # "doctobert-exp-fineweb2-base-pretrain-medterm02-hf-ep18-ba4368-rank0": "FW2-Medterm02-762M-EP18",
    # filtering combined
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-hf-ep6-ba1401-rank0": "FW2-Biocli-Medterm01-702M-EP6",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-hf-ep20-ba4368-rank0": "FW2-Biocli-Medterm01-702M-EP20",  # good
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm02-hf-ep52-ba4368-rank0": "FW2-Biocli-Medterm02-264M-EP52",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-edu4-hf-ep24-ba4368-rank0": "FW2-Biocli-Edu4-587M-EP24",
    # "doctobert-exp-fineweb2-base-pretrain-medterm01-edu4-hf-ep15-ba4369-rank0": "FW2-Medterm01-Edu4-933M-EP15",
    # # rewritten
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-unrewritten-hf-ep3-ba4368-rank0": "Rewritten-Raw-3.6B-EP3",
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-hf-ep7-ba4368-rank0": "Rewritten-1.8B-EP7",
    # # rewritten combined
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-medterm01-hf-ep8-ba4368-rank0": "Rewritten-Medterm01-1.5B-EP8",
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-medterm02-hf-ep12-ba4368-rank0": "Rewritten-Medterm02-973M-EP12",
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-biocli-hf-ep26-ba4368-rank0": "Rewritten-Biocli-475M-EP26",
    # # rewritten by model
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-gptoss-hf-ep5-ba1093-rank0": "Rewritten-GPT-OSS-543M-EP5",
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-medgemma-hf-ep5-ba1093-rank0": "Rewritten-Medgemma-587M-EP5",
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-qwen3-30b-hf-ep11-ba1093-rank0": "Rewritten-Qwen3-30B-321M-EP11",
    # "doctobert-exp-fineweb2-base-pretrain-rewritten-qwen3-next-hf-ep10-ba1093-rank0": "Rewritten-Qwen3-Next-80B-300M-EP10",
    # # filtering + rewritten
    # "doctobert-exp-fineweb2-base-pretrain-medterm02-plus-rewritten-hf-ep5-ba4368-rank0": "FW2-Medterm02-Plus-Rewritten-2.5B-EP5",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-hf-ep5-ba4368-rank0": "FW2-Biocli-Medterm01-Plus-Rewritten-2.5B-EP5",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-biocli-medterm01-hf-ep11-ba4368-rank0": "FW2-Biocli-Medterm01-Plus-Rewritten-Biocli-Medterm01-1.1B-EP11",
    # filtering + rewritten (longer training)
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-40b-hf-ep10-ba8734-rank0": "FW2-Biocli-Medterm01-Plus-Rewritten-40B-2.5B-EP10",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-80b-hf-ep21-ba17467-rank0": "FW2-Biocli-Medterm01-Plus-Rewritten-80B-2.5B-EP21",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-biocli-medterm01-40b-hf-ep23-ba8734-rank0": "FW2-Biocli-Medterm01-Plus-Rewritten-Biocli-Medterm01-40B-1.1B-EP23",
    #
    # "doctobert-exp-fineweb2-base-pretrain-medterm02-plus-rewritten-roberta-hf-ep6-ba13337-rank0": "FW2-Medterm02-Plus-Rewritten-Roberta-2.5B-EP6",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-adamw-hf-ep28-ba11107-rank0": "FW2-Biocli-Medterm01-702M-EP28-Roberta-AdamW",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-hf-ep28-ba11107-rank0": "FW2-Biocli-Medterm01-702M-EP28-Roberta-StableAdamW",
    # new
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-wo-bs-warmup-hf-ep20-ba4240-rank0": "FW2-Biocli-Medterm01-702M-EP20-NoBSWarmup",
    # "doctobert-exp-fineweb2-base-pretrain-medterm01-edu4-plus-rewritten-hf-ep4-ba4368-rank0": "FW2-Medterm01-Edu4-Plus-Rewritten-2.6B-EP4",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-wd001-hf-ep28-ba11107-rank0": "FW2-Biocli-Medterm01-702M-EP28-Roberta-WD001",
    # "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-roberta-hf-ep6-ba13379-rank0": "FW2-Biocli-Medterm01-Plus-Rewritten-2.5B-EP6-Roberta",
    "doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-100b-hf-ep144-ba55535-rank0": "FW2-Biocli-Medterm01-702M-EP144-Roberta-100B",
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
    # 'cas|pos|1.0',
    # 'cas|cls|1.0',
    # 'cas|ner-spec|1.0',
    # 'cas|ner-neg|1.0',
    # 'essai|pos|1.0',
    # 'essai|cls|1.0',
    # 'essai|ner-spec|1.0',
    # 'essai|ner-neg|1.0',
    "mantragsc|ner|1.0",
    'quaero|ner-emea|1.0',
    'quaero|ner-medline|1.0',
    'e3c|ner-clinical|1.0',
    'e3c|ner-temporal|1.0',
    'morfitt|cls|1.0',
    # 'frenchmedmcqa|mcqa|1.0',
    # 'frenchmedmcqa|cls|1.0',
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

output = []

output.append("""
\\begin{table*}[t!]
\\tiny
\\centering
\\begin{tabular}{|l|l|%s}
\\hline
""" % ("c|"*len(models)))

output.append("\\textbf{Dataset} & \\textbf{Task} & " + " & ".join(["\\textbf{" + mapping.get(m.replace("../../../models/",""),m) + "}" for m in models]) + " \\\\ ")

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
            edrm_best_idx = sorted(range(len(edrm_values)), key=lambda i: edrm_values[i], reverse=True)[0]
            edrm_second_idx = sorted(range(len(edrm_values)), key=lambda i: edrm_values[i], reverse=True)[1] if len(edrm_values) > 1 else None
            spear_best_idx = sorted(range(len(spear_values)), key=lambda i: spear_values[i], reverse=True)[0]
            spear_second_idx = sorted(range(len(spear_values)), key=lambda i: spear_values[i], reverse=True)[1] if len(spear_values) > 1 else None
            for i, m in enumerate(models):
                edrm_avg = data[m][tkey]['edrm']['avg']
                edrm_std = data[m][tkey]['edrm']['std']
                spear_avg = data[m][tkey]['spearman_correlation_coef']['avg']
                spear_std = data[m][tkey]['spearman_correlation_coef']['std']
                edrm_part = f"{round(edrm_avg*100,2)}$\\pm${round(edrm_std*100,2)}"
                spear_part = f"{round(spear_avg*100,2)}$\\pm${round(spear_std*100,2)}"
                if i == edrm_best_idx:
                    edrm_part = f"\\textbf{{{edrm_part}}}"
                elif i == edrm_second_idx:
                    edrm_part = f"\\underline{{{edrm_part}}}"
                if i == spear_best_idx:
                    spear_part = f"\\textbf{{{spear_part}}}"
                elif i == spear_second_idx:
                    spear_part = f"\\underline{{{spear_part}}}"
                metric = f"{edrm_part} / {spear_part}"
                runs_metrics.append(metric)
        elif tkey.find("frenchmedmcqa|mcqa") != -1:
            # MCQA (Hamming / Exact)
            hamming_values, exact_values = [], []
            for m in models:
                hamming_avg = data[m][tkey]['hamming_score']['avg']
                exact_avg = data[m][tkey]['exact_match']['avg']
                hamming_values.append(hamming_avg)
                exact_values.append(exact_avg)
            hamming_best_idx = sorted(range(len(hamming_values)), key=lambda i: hamming_values[i], reverse=True)[0]
            hamming_second_idx = sorted(range(len(hamming_values)), key=lambda i: hamming_values[i], reverse=True)[1] if len(hamming_values) > 1 else None
            exact_best_idx = sorted(range(len(exact_values)), key=lambda i: exact_values[i], reverse=True)[0]
            exact_second_idx = sorted(range(len(exact_values)), key=lambda i: exact_values[i], reverse=True)[1] if len(exact_values) > 1 else None
            for i, m in enumerate(models):
                hamming_avg = data[m][tkey]['hamming_score']['avg']
                hamming_std = data[m][tkey]['hamming_score']['std']
                exact_avg = data[m][tkey]['exact_match']['avg']
                exact_std = data[m][tkey]['exact_match']['std']
                hamming_part = f"{round(hamming_avg*100,2)}$\\pm${round(hamming_std*100,2)}"
                exact_part = f"{round(exact_avg*100,2)}$\\pm${round(exact_std*100,2)}"
                if i == hamming_best_idx:
                    hamming_part = f"\\textbf{{{hamming_part}}}"
                elif i == hamming_second_idx:
                    hamming_part = f"\\underline{{{hamming_part}}}"
                if i == exact_best_idx:
                    exact_part = f"\\textbf{{{exact_part}}}"
                elif i == exact_second_idx:
                    exact_part = f"\\underline{{{exact_part}}}"
                metric = f"{hamming_part} / {exact_part}"
                runs_metrics.append(metric)
        else:
            # Single metric (NER, POS, CLS)
            comparison_values = []
            for m in models:
                if tkey.find("|ner") != -1 or tkey.find("|pos") != -1:
                    f1_avg = data[m][tkey]['overall_f1']['avg']
                    f1_std = data[m][tkey]['overall_f1']['std']
                    metric = f"{round(f1_avg*100,2)}$\\pm${round(f1_std*100,2)}"
                    comparison_values.append(f1_avg)
                else:
                    wf1_avg = data[m][tkey]['weighted_f1']['avg']
                    wf1_std = data[m][tkey]['weighted_f1']['std']
                    metric = f"{round(wf1_avg*100,2)}$\\pm${round(wf1_std*100,2)}"
                    comparison_values.append(wf1_avg)
                runs_metrics.append(metric)
            sorted_idx = sorted(range(len(comparison_values)), key=lambda i: comparison_values[i], reverse=True)
            best_idx = sorted_idx[0]
            second_idx = sorted_idx[1] if len(sorted_idx) > 1 else None
            for i in range(len(runs_metrics)):
                if i == best_idx:
                    runs_metrics[i] = f"\\textbf{{{runs_metrics[i]}}}"
                elif i == second_idx:
                    runs_metrics[i] = f"\\underline{{{runs_metrics[i]}}}"

        # LaTeX line construction
        corpus_label = ""
        if not corpus_written:
            corpus_label = f"\\hline\n\n\\multirow{{{num_rows}}}{{*}}{{{corpus.replace('_','-').upper()}}}"
            corpus_written = True
        line = f"{corpus_label} & {task.upper()} & " + " & ".join(runs_metrics) + " \\\\ "
        output.append(line if corpus_label else f" & {task.upper()} & " + " & ".join(runs_metrics) + " \\\\ ")

output.append("""
\\hline
\\end{tabular}
\\caption{Performance of the baselines on the set of biomedical tasks in French. Results shown as mean$\\pm$std. Best model in bold and second best is underlined.}
\\label{table:results}
\\end{table*}
""")
# print("\n".join(output))

output_dir = f"./stats/{run_label}"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/table.tex"
with open(output_file, "w") as f:
    f.write("\n".join(output))
print(f"Table saved to {output_file}")

# =============================================================================
# Aggregation Metrics
# =============================================================================
from aggregation_metrics import (
    compute_all_aggregations,
    extract_primary_metric,
    print_aggregation_summary,
    generate_aggregation_latex_table,
    print_zscore_table,
    print_minmax_table,
    print_rank_table,
    generate_zscore_latex_table,
    generate_minmax_latex_table,
    generate_rank_latex_table
)

print("\n" + "=" * 80)
print("Computing aggregation metrics...")
print("=" * 80)

# Define baseline model for anchored normalization
baseline_scores = None

# baseline_model_key = "doctobert-exp-fineweb2-base-pretrain-nachos-new-hf-ep10-ba4368-rank0"
# baseline_model_path = "../../../models/" + baseline_model_key.lower().replace("/", "_")

# # Extract baseline scores from the baseline model
# baseline_scores = {}
# if baseline_model_path in data:
#     for task in tasks:
#         if task in data[baseline_model_path]:
#             baseline_scores[task] = extract_primary_metric(data[baseline_model_path][task], task)
#     print(f"Using '{mapping.get(baseline_model_key, baseline_model_key)}' as baseline reference")
# else:
#     print(f"Warning: Baseline model not found in data, using standard normalization")
#     baseline_scores = None

# tasks_filter = None
tasks_filter = [
    'quaero',
    'e3c',
    'morfitt',
    # 'clister|regr',
    # 'deft2020|regr',
    # 'deft2020|cls',
    'deft2021',
    'diamed',
    # 'pxcorpus',
]

# Compute all aggregations
aggregation_results = compute_all_aggregations(
    data=data,
    models=models,
    tasks=tasks,
    model_mapping=mapping,
    baseline_scores=baseline_scores,
    tasks_filter=tasks_filter,
)

# Print overall summary to console
summary = print_aggregation_summary(aggregation_results, sort_by='avg_rank')
print(summary)

# Print detailed tables for each normalization method
# print("\n")
# print(print_zscore_table(aggregation_results))
# print("\n")
# print(print_minmax_table(aggregation_results))
# print("\n")
# print(print_rank_table(aggregation_results))

# Generate and save combined LaTeX table
agg_latex = generate_aggregation_latex_table(aggregation_results, sort_by='avg_rank')
agg_latex_file = f"{output_dir}/table_aggregation.tex"
with open(agg_latex_file, "w") as f:
    f.write(agg_latex)
print(f"\nAggregation table saved to {agg_latex_file}")

# # Generate and save Z-Score LaTeX table
# zscore_latex = generate_zscore_latex_table(aggregation_results)
# zscore_latex_file = f"{output_dir}/table_zscore.tex"
# with open(zscore_latex_file, "w") as f:
#     f.write(zscore_latex)
# print(f"Z-Score table saved to {zscore_latex_file}")

# # Generate and save MinMax LaTeX table
# minmax_latex = generate_minmax_latex_table(aggregation_results)
# minmax_latex_file = f"{output_dir}/table_minmax.tex"
# with open(minmax_latex_file, "w") as f:
#     f.write(minmax_latex)
# print(f"MinMax table saved to {minmax_latex_file}")

# # Generate and save Rank-based LaTeX table
# rank_latex = generate_rank_latex_table(aggregation_results)
# rank_latex_file = f"{output_dir}/table_rank.tex"
# with open(rank_latex_file, "w") as f:
#     f.write(rank_latex)
# print(f"Rank table saved to {rank_latex_file}")

# Save full aggregation results as JSON
agg_json_file = f"{output_dir}/overall_averaged_metrics_aggregation.json"
with open(agg_json_file, "w") as f:
    json.dump(aggregation_results, f, indent=2)
print(f"Aggregation results saved to {agg_json_file}")
