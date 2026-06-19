#!/bin/bash
# Helper script to submit the run_hpo_array.slurm job for all models

set -e

model_names=(
    # ===================== OSS baselines (see models.txt / download_models_locally.py) =====================
    # base
    # almanach_camembert-base
    # almanach_camemberta-base
    # almanach_camembert-bio-base
    # almanach_moderncamembert-cv2-base
    # almanach_moderncamembert-base
    # almanach_moderncamembert-bio-base
    # almanach_modernbert-bio-base
    # thomas-sounack_bioclinical-modernbert-base
    # dmis-lab_biobert-v1.1
    # large
    # almanach_camembert-large
    # almanach_moderncamembert-bio-large
    # almanach_modernbert-bio-large
    # thomas-sounack_bioclinical-modernbert-large
    # dmis-lab_biobert-large-cased-v1.1
    # dr-bert_drbert-7gb-large

    # ===================== Exp 1: FW2 source + filtering ablations (~4.4B-tok) =====================
    # source corpora
    # doctobert-exp-fineweb2-base-pretrain-unfiltered-hf-ep1-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-nachos-hf-ep121-ba4367-rank0
    # doctobert-exp-fineweb2-base-pretrain-nachos-new-hf-ep10-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-transcorpus-hf-ep2-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-synthesized-hf-ep15-ba4368-rank0
    # single-axis filters
    # doctobert-exp-fineweb2-base-pretrain-biocli-hf-ep11-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-edu2-hf-ep3-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-edu4-hf-ep8-ba4369-rank0
    # doctobert-exp-fineweb2-base-pretrain-medterm01-hf-ep5-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-medterm02-hf-ep18-ba4368-rank0
    # combined filters
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-hf-ep6-ba1401-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-hf-ep20-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm02-hf-ep52-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-edu4-hf-ep24-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-medterm01-edu4-hf-ep15-ba4369-rank0
    # union variants
    # doctobert-exp-fineweb2-base-pretrain-biocli-edu4-union-hf-ep6-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-union-hf-ep4-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-medterm01-edu4-union-hf-ep4-ba4368-rank0
    # rewritten (by source filter)
    # doctobert-exp-fineweb2-base-pretrain-rewritten-hf-ep7-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-unrewritten-hf-ep3-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-medterm01-hf-ep8-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-medterm02-hf-ep12-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-biocli-hf-ep26-ba4368-rank0
    # rewritten (by rewriter LLM)
    # doctobert-exp-fineweb2-base-pretrain-rewritten-gptoss-hf-ep5-ba1093-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-medgemma-hf-ep5-ba1093-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-qwen3-30b-hf-ep11-ba1093-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-qwen3-next-hf-ep10-ba1093-rank0
    # filter + rewritten
    # doctobert-exp-fineweb2-base-pretrain-medterm02-plus-rewritten-hf-ep5-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-medterm01-edu4-plus-rewritten-hf-ep4-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-hf-ep5-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-biocli-medterm01-hf-ep11-ba4368-rank0
    # filter + rewritten (longer training: 40B / 80B)
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-40b-hf-ep10-ba8734-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-80b-hf-ep21-ba17467-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-biocli-medterm01-40b-hf-ep23-ba8734-rank0
    # RoBERTa variants (optimizer / schedule / scale)
    # doctobert-exp-fineweb2-base-pretrain-medterm02-plus-rewritten-roberta-hf-ep6-ba13337-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-plus-rewritten-roberta-hf-ep6-ba13379-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-adamw-hf-ep28-ba11107-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-hf-ep28-ba11107-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-wd001-hf-ep28-ba11107-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-100b-hf-ep37-ba14440-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-100b-hf-ep72-ba27768-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-100b-hf-ep107-ba41096-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-100b-hf-ep144-ba55535-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-roberta-stableadamw-200b-hf-ep289-ba111070-rank0
    # schedule / scale variants
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-wo-bs-warmup-hf-ep20-ba4240-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-linear-hf-ep20-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-linear-15-hf-ep20-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-biocli-medterm01-100b-hf-ep100-ba21835-rank0

    # ===================== Exp 2: rewriting-recipe ablations (~220-batch) =====================
    # prompt versions v3 / v4
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v3-hf-ep52-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v3-raw-hf-ep26-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v3-raw-unfiltered-hf-ep19-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v3-b-hf-ep52-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v3-c-hf-ep52-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-hf-ep35-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v3-1-hf-ep53-ba221-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v3-1-b-hf-ep53-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v3-1-c-hf-ep53-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v3-2-hf-ep53-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v3-2-b-hf-ep53-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v3-2-c-hf-ep53-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-hf-ep52-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-b-hf-ep52-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-c-hf-ep52-ba220-rank0
    # + raw mix / 3-seed
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v3-1-3seed-hf-ep17-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v3-1-plus-raw-hf-ep17-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-plus-raw-hf-ep16-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v3-1-3seed-plus-raw-hf-ep10-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-3seed-plus-raw-hf-ep9-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-plus-hf-ep21-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-plus-plus-raw-hf-ep11-ba220-rank0
    # v4.2 + raw / dedup / med01
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-hf-ep15-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-plus-raw-hf-ep9-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v3-1-plus-raw-med01-edu4-hf-ep42-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-plus-raw-med01-edu4-hf-ep35-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-plus-dedup-hf-ep52-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-dedup-hf-ep52-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-1-plus-dedup-plus-raw-med01-edu4-hf-ep35-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-dedup-plus-raw-med01-edu4-hf-ep35-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-dedup-med01-hf-ep105-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-dedup-plus-raw-hf-ep16-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-dedup-plus-raw-unfiltered-hf-ep13-ba220-rank0
    # rewriter-LLM ablation (V4.2 3m1n x 5 LLMs + mga_official + raw-only)
    # doctobert-exp-fineweb2-base-pretrain-rewritten-mga-official-qwen3-5-35b-hf-ep13-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-3m1n-qwen3-5-35b-hf-ep52-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-3m1n-qwen3-5-122b-hf-ep42-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-3m1n-medgemma-27b-hf-ep30-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-3m1n-gemma-4-26b-hf-ep42-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-v4-2-3m1n-gpt-oss-120b-hf-ep42-ba220-rank0
    # doctobert-exp-fineweb2-base-pretrain-raw-100k-med01-edu4-hf-ep210-ba220-rank0
    # 1M-doc V4.2 scale-up (20B-tok)
    # doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen3-5-35b-hf-ep81-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-gemma-4-26b-hf-ep83-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-raw-20b-1m-stage2-filtered-hf-ep46-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-raw-20b-1m-unfiltered-hf-ep35-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-raw-20b-1m-med01-edu4-hf-ep282-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-raw-20b-1m-med01-edu4-plus-synth-v2-hf-ep14-ba4368-rank0
    # 1M-doc V4.2 merge variants (20B-tok)
    # doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-plus-gemma-hf-ep41-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-plus-raw-stage2-filtered-hf-ep29-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-plus-raw-med01-edu4-hf-ep63-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-plus-gemma-plus-raw-med01-edu4-hf-ep35-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-half-plus-gemma-half-hf-ep83-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-half-plus-gemma-half-plus-raw-med01-edu4-hf-ep63-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen3-5-35b-med01-hf-ep146-ba4367-rank0
    # raw:rewritten word-ratio sweep
    # doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-plus-raw-med01-edu4-raw33-hf-ep86-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-plus-raw-med01-edu4-raw50-hf-ep132-ba4368-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-plus-raw-med01-edu4-raw67-hf-ep179-ba4370-rank0
    # doctobert-exp-fineweb2-base-pretrain-rewritten-20b-v4-2-1m-qwen-plus-raw-unfiltered-hf-ep24-ba4368-rank0

    # ===================== DoctoBERT-fr v2 final lineage =====================
    # --- base: P1 pretrain ---
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-hf-ep16-ba21833-rank0
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-hf-ep7-ba21832-rank0
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-plus-n-t-hf-ep5-ba21832-rank0
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-hf-ep3-ba9115-rank0   # WSD P1 ep3 ablation (~42B)
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-200b-hf-ep33-ba43665-rank0
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-hf-ep15-ba43663-rank0
    # --- base: P2 context-extension (8192) of -plus-rewritten-200b ---
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-hf-ep5-ba4242-rank0
    # --- base: P3 anneal (8192), branched from -plus-rewritten-200b-p2-20b ---
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-hf-ep3-ba4241-rank0
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-plus-rewritten-hf-ep1-ba4240-rank0
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-plus-rewritten-biocli-hf-ep4-ba4240-rank0
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-plus-rewritten-biocli-dyn-mlm-hf-ep4-ba4240-rank0
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med02-edu4-plus-rewritten-rwquality-hf-ep5-ba4240-rank0  # stricter med02 + rewriting-quality filter
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-biocli-plus-rewritten-biocli-hf-ep3-ba4240-rank0  # biocli x biocli (no rwquality)
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-50b-med01-edu4-plus-rewritten-biocli-hf-ep10-ba10600-rank0  # 50B, mlm=0.15
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-50b-med01-edu4-plus-rewritten-biocli-mlm03-hf-ep10-ba10600-rank0  # 50B, mlm=0.30
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-100b-med01-edu4-plus-rewritten-biocli-hf-ep21-ba21200-rank0  # 100B
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-plus-rewritten-biocli-shortdoc1024-hf-ep4-ba4240-rank0  # short-doc (1024)
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-plus-rewritten-biocli-mixdoc1024doc8192-hf-ep2-ba4240-rank0  # mix-doc (1024+8192), 20B
    doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-50b-med01-edu4-plus-rewritten-biocli-mixdoc1024doc8192-hf-ep5-ba10599-rank0    # mix-doc (1024+8192), 50B  [active]
    # --- base: idea2 P2 ctx-ext 10B from WSD ep3, then P3 anneals ---
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-p2-10b-hf-ep2-ba2121-rank0
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-p2-10b-p3-general-100b-hf-ep7-ba21197-rank0
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-p2-10b-p3-biocli-100b-hf-ep21-ba21200-rank0
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-p2-10b-p3-biocli-20b-hf-ep4-ba4240-rank0
    # --- ModernBERT 200B-decay lineage (linear_decay_with_warmup p1) ---
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-hf-ep15-ba43663-rank0
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-p2-20b-hf-ep5-ba4242-rank0
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-p2-20b-p3-exp-50b-med01-edu4-plus-rewritten-biocli-hf-ep4-ba4240-rank0
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-126b-p2-20b-hf-ep5-ba4242-rank0
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-126b-p2-20b-p3-biocli-54b-resume2-hf-ep2-ba2544-rank0
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-126b-p2-20b-p3-general-54b-resume2-hf-ep2-ba6359-rank0
    # --- ModernBERT 8192 from-scratch (200B decay trunk + biocli cooldowns) ---
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-3e4-hf-ep15-ba44288-rank0   # stage1: general decay 8e-4->3e-4
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-3e4-biocli-20b-hf-ep4-ba4240-rank0   # stage2: biocli cooldown, 20B
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-3e4-biocli-50b-hf-ep10-ba10600-rank0   # stage2: biocli cooldown, 50B
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-0-hf-ep15-ba43665-rank0   # single-slope full decay 8e-4->0
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-0-biocli-br150b-2e4-hf-ep4-ba4240-rank0   # biocli branch @150B, 2e-4->0, 20B
    # doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-0-biocli-br150b-2e4-50b-hf-ep10-ba10600-rank0   # biocli branch @150B, 2e-4->0, 50B
    # --- RoBERTa stream (vocab_32680, max_seq_len=512) ---
    # doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-hf-ep32-ba92308-rank0
    # doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-hf-ep14-ba106710-rank0
    # doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-biocli-rwquality-hf-ep19-ba47690-rank0   # +biocli/rwquality, 100B
    # doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-biocli-rwquality-200b-hf-ep39-ba95379-rank0
    # doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-biocli-rwquality-200b-cp-hf-ep39-ba95379-rank0
    # doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-biocli-plus-rewritten-biocli-hf-ep16-ba50085-rank0   # biocli x biocli, 100B
    # doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-biocli-rwquality-500b-cp-hf-ep47-ba114455-rank0   # 500B continual (pre-spike ckpt)
    # doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-500b-new-hf-ep36-ba266775-rank0
    # doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-1t-lr1e4-hf-ep73-ba533549-rank0   # 1T intermediate ckpt
    # RoBERTa fw2-only ablations (DrBERT tokenizer)
    # doctobert-exp-base-pretrain-roberta-fw2-med01-edu4-hf-ep249-ba93146-rank0
    # doctobert-exp-base-pretrain-roberta-fw2-med01-edu4-plus-rewritten-hf-ep794-ba136637-rank0
    # --- ModernBERT LARGE ---
    # doctobert-fr-v2-large-pretrain-finemed-med01-edu4-plus-rewritten-hf-ep7-ba22458-rank0   # P1 (~113B tok)
    # doctobert-fr-v2-large-pretrain-finemed-med01-edu4-plus-rewritten-p2-20b-hf-ep5-ba3638-rank0   # P2 ctx-ext 8192
    # doctobert-fr-v2-large-pretrain-finemed-med01-edu4-plus-rewritten-p2-20b-p3-biocli-20b-hf-ep4-ba3636-rank0   # P3 biocli anneal
)


prev_job_id=""
for model_name in "${model_names[@]}"; do
    after_arg=()
    [ -n "$prev_job_id" ] && after_arg=(--after "$prev_job_id")

    prev_job_id=$(bash run_hpo_array_submit.sh "${after_arg[@]}" "$model_name" | tee /dev/stderr | grep -oP 'Submitted batch job \K\d+')
done

echo "All jobs submitted"
