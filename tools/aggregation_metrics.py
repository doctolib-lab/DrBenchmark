"""
Aggregation metrics for multi-task model evaluation.

This module provides various normalization and rank-based methods to aggregate
model performance across multiple tasks with heterogeneous metrics.

Methods implemented:
- Z-Score standardization
- Min-Max (Range) normalization
- Rank-based aggregation (average rank, Borda count)
- Pairwise win counts
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def extract_primary_metric(task_data: dict, task_key: str) -> float:
    """
    Extract the primary metric value for a given task.
    
    Args:
        task_data: Dictionary containing metric results for the task
        task_key: Task identifier string (e.g., 'quaero|ner-emea|1.0')
    
    Returns:
        Primary metric value (avg) for the task
    """
    if "regr" in task_key:
        # For regression tasks, use EDRM as primary metric
        return task_data['edrm']['avg']
    elif "mcqa" in task_key:
        # For MCQA, use exact match as primary
        return task_data['exact_match']['avg']
    elif "|ner" in task_key or "|pos" in task_key:
        # For NER/POS, use overall F1
        return task_data['overall_f1']['avg']
    else:
        # For classification, use weighted F1
        return task_data['weighted_f1']['avg']


def build_score_matrix(
    data: dict,
    models: List[str],
    tasks: List[str]
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build a score matrix from the data dictionary.
    
    Args:
        data: Full results dictionary
        models: List of model paths
        tasks: List of task keys
    
    Returns:
        Tuple of (score_matrix, valid_models, valid_tasks)
        score_matrix shape: (n_models, n_tasks)
    """
    valid_tasks = []
    valid_models = models
    
    # Filter tasks that exist for all models
    for task in tasks:
        if all(m in data and task in data[m] for m in models):
            valid_tasks.append(task)
    
    # Build matrix
    n_models = len(valid_models)
    n_tasks = len(valid_tasks)
    score_matrix = np.zeros((n_models, n_tasks))
    
    for i, model in enumerate(valid_models):
        for j, task in enumerate(valid_tasks):
            score_matrix[i, j] = extract_primary_metric(data[model][task], task)
    
    return score_matrix, valid_models, valid_tasks


# =============================================================================
# Z-Score Standardization
# =============================================================================

def zscore_aggregation(
    score_matrix: np.ndarray,
    models: List[str],
    tasks: List[str]
) -> Dict[str, Dict]:
    """
    Compute z-score standardized aggregation across tasks.
    
    For each task, convert scores to z-scores (subtract mean, divide by std),
    then average z-scores across tasks for each model.
    
    Args:
        score_matrix: Shape (n_models, n_tasks)
        models: List of model names
        tasks: List of task names
    
    Returns:
        Dictionary with model names as keys, containing:
        - 'zscore_avg': Mean z-score across tasks
        - 'zscore_per_task': Dict of z-scores per task
    """
    n_models, n_tasks = score_matrix.shape
    
    # Compute z-scores per task (column-wise)
    task_means = score_matrix.mean(axis=0)
    task_stds = score_matrix.std(axis=0, ddof=0)
    
    # Avoid division by zero for tasks with no variance
    task_stds = np.where(task_stds == 0, 1, task_stds)
    
    zscores = (score_matrix - task_means) / task_stds
    
    # Aggregate: mean z-score per model
    mean_zscores = zscores.mean(axis=1)
    
    results = {}
    for i, model in enumerate(models):
        results[model] = {
            'zscore_avg': float(mean_zscores[i]),
            'zscore_std': float(zscores[i].std()),
            'zscore_per_task': {tasks[j]: float(zscores[i, j]) for j in range(n_tasks)}
        }
    
    return results


# =============================================================================
# Min-Max Normalization
# =============================================================================

def minmax_aggregation(
    score_matrix: np.ndarray,
    models: List[str],
    tasks: List[str],
    custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Dict]:
    """
    Compute min-max normalized aggregation across tasks.
    
    For each task, rescale scores to [0, 1] range using:
    NormScore = (score - min) / (max - min)
    
    Args:
        score_matrix: Shape (n_models, n_tasks)
        models: List of model names
        tasks: List of task names
        custom_bounds: Optional dict mapping task names to (low, high) bounds.
                       If not provided, uses observed min/max per task.
    
    Returns:
        Dictionary with model names as keys, containing:
        - 'minmax_avg': Mean normalized score across tasks
        - 'minmax_per_task': Dict of normalized scores per task
    """
    n_models, n_tasks = score_matrix.shape
    
    normalized = np.zeros_like(score_matrix)
    
    for j, task in enumerate(tasks):
        col = score_matrix[:, j]
        
        if custom_bounds and task in custom_bounds:
            low, high = custom_bounds[task]
        else:
            low, high = col.min(), col.max()
        
        # Avoid division by zero
        if high - low == 0:
            normalized[:, j] = 1.0  # All models equal, assign max normalized
        else:
            normalized[:, j] = (col - low) / (high - low)
    
    # Aggregate: mean normalized score per model
    mean_normalized = normalized.mean(axis=1)
    
    results = {}
    for i, model in enumerate(models):
        results[model] = {
            'minmax_avg': float(mean_normalized[i]),
            'minmax_std': float(normalized[i].std()),
            'minmax_per_task': {tasks[j]: float(normalized[i, j]) for j in range(n_tasks)}
        }
    
    return results


def minmax_with_baseline_aggregation(
    score_matrix: np.ndarray,
    models: List[str],
    tasks: List[str],
    baseline_scores: Dict[str, float],
    ceiling_scores: Optional[Dict[str, float]] = None
) -> Dict[str, Dict]:
    """
    Compute relative improvement over a baseline model.
    
    For each task:
    RelativeImprovement = (score - baseline) / baseline * 100  (as percentage)
    
    This measures "how much better/worse is this model compared to baseline?"
    - Positive values: model is better than baseline
    - Negative values: model is worse than baseline
    - Zero: model equals baseline
    
    Args:
        score_matrix: Shape (n_models, n_tasks)
        models: List of model names
        tasks: List of task names
        baseline_scores: Dict mapping task names to baseline model scores
        ceiling_scores: Not used (kept for API compatibility)
    
    Returns:
        Dictionary with model names as keys, containing relative improvement scores
    """
    n_models, n_tasks = score_matrix.shape
    
    relative_improvement = np.zeros_like(score_matrix)
    
    for j, task in enumerate(tasks):
        col = score_matrix[:, j]
        baseline = baseline_scores.get(task, 0.0)
        
        # Avoid division by zero
        if baseline == 0:
            # If baseline is 0, use absolute difference instead
            relative_improvement[:, j] = col * 100
        else:
            # Percentage change from baseline
            relative_improvement[:, j] = (col - baseline) / baseline * 100
    
    mean_relative = relative_improvement.mean(axis=1)
    
    results = {}
    for i, model in enumerate(models):
        results[model] = {
            'baseline_norm_avg': float(mean_relative[i]),
            'baseline_norm_std': float(relative_improvement[i].std()),
            'baseline_norm_per_task': {tasks[j]: float(relative_improvement[i, j]) for j in range(n_tasks)}
        }
    
    return results


# =============================================================================
# Rank-Based Aggregation
# =============================================================================

def rank_aggregation(
    score_matrix: np.ndarray,
    models: List[str],
    tasks: List[str]
) -> Dict[str, Dict]:
    """
    Compute rank-based aggregation across tasks.
    
    For each task, rank models (1 = best), then compute mean/median rank.
    
    Args:
        score_matrix: Shape (n_models, n_tasks)
        models: List of model names
        tasks: List of task names
    
    Returns:
        Dictionary with model names as keys, containing:
        - 'avg_rank': Mean rank across tasks (lower is better)
        - 'median_rank': Median rank across tasks
        - 'rank_per_task': Dict of ranks per task
        - 'borda_score': Borda count score (higher is better)
    """
    n_models, n_tasks = score_matrix.shape
    
    # Compute ranks per task (column-wise), higher score = lower (better) rank
    ranks = np.zeros_like(score_matrix)
    
    for j in range(n_tasks):
        col = score_matrix[:, j]
        # argsort gives indices that would sort ascending, we want descending
        # so best (highest) score gets rank 1
        order = np.argsort(-col)  # descending order indices
        ranks_for_task = np.empty_like(order)
        ranks_for_task[order] = np.arange(1, n_models + 1)
        ranks[:, j] = ranks_for_task
    
    # Aggregate ranks
    avg_ranks = ranks.mean(axis=1)
    median_ranks = np.median(ranks, axis=1)
    
    # Borda count: for each task, model at rank r gets (n_models - r) points
    borda_scores = (n_models - ranks).sum(axis=1)
    
    results = {}
    for i, model in enumerate(models):
        results[model] = {
            'avg_rank': float(avg_ranks[i]),
            'median_rank': float(median_ranks[i]),
            'rank_std': float(ranks[i].std()),
            'borda_score': float(borda_scores[i]),
            'rank_per_task': {tasks[j]: int(ranks[i, j]) for j in range(n_tasks)}
        }
    
    return results


def pairwise_wins(
    score_matrix: np.ndarray,
    models: List[str],
    tasks: List[str]
) -> Dict[str, Dict]:
    """
    Compute pairwise win counts between models.
    
    For each pair of models, count on how many tasks model A > model B.
    
    Args:
        score_matrix: Shape (n_models, n_tasks)
        models: List of model names
        tasks: List of task names
    
    Returns:
        Dictionary with model names as keys, containing:
        - 'total_wins': Total number of pairwise wins
        - 'win_rate': Win rate (wins / total comparisons)
        - 'wins_against': Dict of win counts against each other model
    """
    n_models, n_tasks = score_matrix.shape
    
    results = {}
    
    for i, model_a in enumerate(models):
        wins_against = {}
        total_wins = 0
        
        for k, model_b in enumerate(models):
            if i == k:
                continue
            
            # Count tasks where model_a > model_b
            wins = int(np.sum(score_matrix[i, :] > score_matrix[k, :]))
            wins_against[model_b] = wins
            total_wins += wins
        
        total_comparisons = (n_models - 1) * n_tasks
        
        results[model_a] = {
            'total_wins': total_wins,
            'win_rate': total_wins / total_comparisons if total_comparisons > 0 else 0,
            'wins_against': wins_against
        }
    
    return results


# =============================================================================
# Combined Aggregation Report
# =============================================================================

def compute_all_aggregations(
    data: dict,
    models: List[str],
    tasks: List[str],
    model_mapping: Optional[Dict[str, str]] = None,
    baseline_scores: Optional[Dict[str, float]] = None,
    tasks_filter: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Compute all aggregation metrics for a set of models and tasks.
    
    Args:
        data: Full results dictionary
        models: List of model paths
        tasks: List of task keys
        model_mapping: Optional dict mapping model paths to display names
        baseline_scores: Optional baseline scores for baseline-relative normalization
        tasks_filter: Optional list of task patterns to include. Supports:
                      - Exact match: "quaero|ner-emea|1.0"
                      - Partial match: "ner" (matches any task containing "ner")
                      - Task type: "|cls|" (matches classification tasks)
                      If None, all tasks are included.
    
    Returns:
        Dictionary containing all aggregation results
    """
    # Filter tasks if specified
    if tasks_filter:
        filtered_tasks = []
        for task in tasks:
            for pattern in tasks_filter:
                if pattern in task:
                    filtered_tasks.append(task)
                    break
        tasks = filtered_tasks
        if not tasks:
            raise ValueError(f"No tasks matched the filter patterns: {tasks_filter}")
    
    # Build score matrix
    score_matrix, valid_models, valid_tasks = build_score_matrix(data, models, tasks)
    
    if len(valid_tasks) == 0:
        raise ValueError("No valid tasks found for the given models")
    
    # Compute all aggregations
    zscore_results = zscore_aggregation(score_matrix, valid_models, valid_tasks)
    minmax_results = minmax_aggregation(score_matrix, valid_models, valid_tasks)
    rank_results = rank_aggregation(score_matrix, valid_models, valid_tasks)
    pairwise_results = pairwise_wins(score_matrix, valid_models, valid_tasks)
    
    # Optionally compute baseline-relative normalization
    baseline_results = None
    if baseline_scores:
        baseline_results = minmax_with_baseline_aggregation(
            score_matrix, valid_models, valid_tasks, baseline_scores
        )
    
    # Combine into unified report
    combined = {
        'metadata': {
            'n_models': len(valid_models),
            'n_tasks': len(valid_tasks),
            'tasks': valid_tasks
        },
        'models': {}
    }
    
    for model in valid_models:
        display_name = model_mapping.get(model.replace("../../../models/", ""), model) if model_mapping else model
        
        combined['models'][model] = {
            'display_name': display_name,
            'zscore': zscore_results[model],
            'minmax': minmax_results[model],
            'rank': rank_results[model],
            'pairwise': pairwise_results[model]
        }
        
        if baseline_results:
            combined['models'][model]['baseline_norm'] = baseline_results[model]
    
    return combined


def print_aggregation_summary(
    aggregation_results: Dict,
    sort_by: str = 'zscore_avg'
) -> str:
    """
    Generate a formatted summary of aggregation results (models as rows, metrics as columns).
    
    Args:
        aggregation_results: Output from compute_all_aggregations
        sort_by: Metric to sort by ('zscore_avg', 'minmax_avg', 'avg_rank', 'borda_score')
    
    Returns:
        Formatted string summary
    """
    models_data = aggregation_results['models']
    n_tasks = aggregation_results['metadata']['n_tasks']
    
    # Check if baseline normalization is available
    has_baseline = any('baseline_norm' in data for data in models_data.values())
    
    # Prepare data for sorting
    rows = []
    for model, data in models_data.items():
        row = {
            'model': model,
            'display_name': data['display_name'],
            'zscore_avg': data['zscore']['zscore_avg'],
            'zscore_std': data['zscore']['zscore_std'],
            'minmax_avg': data['minmax']['minmax_avg'],
            'minmax_std': data['minmax']['minmax_std'],
            'avg_rank': data['rank']['avg_rank'],
            'rank_std': data['rank']['rank_std'],
            'borda_score': data['rank']['borda_score'],
            'win_rate': data['pairwise']['win_rate']
        }
        if has_baseline and 'baseline_norm' in data:
            row['baseline_avg'] = data['baseline_norm']['baseline_norm_avg']
            row['baseline_std'] = data['baseline_norm']['baseline_norm_std']
        rows.append(row)
    
    # Sort (for rank, lower is better; for others, higher is better)
    reverse = sort_by != 'avg_rank'
    rows.sort(key=lambda x: x[sort_by], reverse=reverse)
    
    # Build summary string
    lines = []
    width = 140 if has_baseline else 120
    lines.append("=" * width)
    lines.append(f"AGGREGATION SUMMARY (sorted by {sort_by})")
    lines.append(f"Number of tasks: {n_tasks}")
    lines.append("=" * width)
    lines.append("")
    
    if has_baseline:
        header = f"{'Rank':<5} {'Model':<25} {'Z-Score (avg±std)':>20} {'MinMax (avg±std)':>20} {'Δ% vs Baseline':>18} {'Avg Rank (±std)':>18} {'Win%':>8}"
    else:
        header = f"{'Rank':<5} {'Model':<25} {'Z-Score (avg±std)':>20} {'MinMax (avg±std)':>20} {'Avg Rank (±std)':>18} {'Win%':>8}"
    lines.append(header)
    lines.append("-" * width)
    
    for idx, row in enumerate(rows, 1):
        zscore_str = f"{row['zscore_avg']:.3f}±{row['zscore_std']:.3f}"
        minmax_str = f"{row['minmax_avg']:.3f}±{row['minmax_std']:.3f}"
        rank_str = f"{row['avg_rank']:.2f}±{row['rank_std']:.2f}"
        if has_baseline:
            # Show relative improvement as percentage with sign
            sign = "+" if row['baseline_avg'] >= 0 else ""
            baseline_str = f"{sign}{row['baseline_avg']:.2f}%±{row['baseline_std']:.2f}"
            line = f"{idx:<5} {row['display_name']:<25} {zscore_str:>20} {minmax_str:>20} {baseline_str:>18} {rank_str:>18} {row['win_rate']*100:>7.1f}%"
        else:
            line = f"{idx:<5} {row['display_name']:<25} {zscore_str:>20} {minmax_str:>20} {rank_str:>18} {row['win_rate']*100:>7.1f}%"
        lines.append(line)
    
    lines.append("=" * width)
    
    return "\n".join(lines)


def generate_aggregation_latex_table(
    aggregation_results: Dict,
    sort_by: str = 'avg_rank'
) -> str:
    """
    Generate a LaTeX table of aggregation results (models as rows, metrics as columns).
    
    Args:
        aggregation_results: Output from compute_all_aggregations
        sort_by: Metric to sort by
    
    Returns:
        LaTeX table string
    """
    models_data = aggregation_results['models']
    n_tasks = aggregation_results['metadata']['n_tasks']
    
    # Check if baseline normalization is available
    has_baseline = any('baseline_norm' in data for data in models_data.values())
    
    # Prepare data for sorting
    rows = []
    for model, data in models_data.items():
        row = {
            'display_name': data['display_name'],
            'zscore_avg': data['zscore']['zscore_avg'],
            'zscore_std': data['zscore']['zscore_std'],
            'minmax_avg': data['minmax']['minmax_avg'],
            'minmax_std': data['minmax']['minmax_std'],
            'avg_rank': data['rank']['avg_rank'],
            'rank_std': data['rank']['rank_std'],
            'borda_score': data['rank']['borda_score'],
            'win_rate': data['pairwise']['win_rate']
        }
        if has_baseline and 'baseline_norm' in data:
            row['baseline_avg'] = data['baseline_norm']['baseline_norm_avg']
            row['baseline_std'] = data['baseline_norm']['baseline_norm_std']
        rows.append(row)
    
    # Sort
    reverse = sort_by != 'avg_rank'
    rows.sort(key=lambda x: x[sort_by], reverse=reverse)
    
    # Find best and second best values for highlighting
    best_zscore = max(r['zscore_avg'] for r in rows)
    second_zscore = sorted([r['zscore_avg'] for r in rows], reverse=True)
    second_zscore = second_zscore[1] if len(second_zscore) > 1 else None
    
    best_minmax = max(r['minmax_avg'] for r in rows)
    second_minmax = sorted([r['minmax_avg'] for r in rows], reverse=True)
    second_minmax = second_minmax[1] if len(second_minmax) > 1 else None
    
    best_rank = min(r['avg_rank'] for r in rows)
    second_rank = sorted([r['avg_rank'] for r in rows])
    second_rank = second_rank[1] if len(second_rank) > 1 else None
    
    best_winrate = max(r['win_rate'] for r in rows)
    second_winrate = sorted([r['win_rate'] for r in rows], reverse=True)
    second_winrate = second_winrate[1] if len(second_winrate) > 1 else None
    
    if has_baseline:
        best_baseline = max(r['baseline_avg'] for r in rows)
        second_baseline = sorted([r['baseline_avg'] for r in rows], reverse=True)
        second_baseline = second_baseline[1] if len(second_baseline) > 1 else None
    
    # Build LaTeX
    lines = []
    lines.append("\\begin{table}[t!]")
    lines.append("\\centering")
    lines.append("\\small")
    if has_baseline:
        lines.append("\\begin{tabular}{|l|c|c|c|c|c|}")
        lines.append("\\hline")
        lines.append("\\textbf{Model} & \\textbf{Z-Score} & \\textbf{MinMax} & \\textbf{$\\Delta$\\% vs Ref} & \\textbf{Avg Rank} & \\textbf{Win\\%} \\\\")
    else:
        lines.append("\\begin{tabular}{|l|c|c|c|c|}")
        lines.append("\\hline")
        lines.append("\\textbf{Model} & \\textbf{Z-Score} & \\textbf{MinMax} & \\textbf{Avg Rank} & \\textbf{Win\\%} \\\\")
    lines.append("\\hline")
    
    for row in rows:
        # Z-Score with std
        zscore_str = f"{row['zscore_avg']:.2f}$\\pm${row['zscore_std']:.2f}"
        if row['zscore_avg'] == best_zscore:
            zscore_str = f"\\textbf{{{zscore_str}}}"
        elif second_zscore and row['zscore_avg'] == second_zscore:
            zscore_str = f"\\underline{{{zscore_str}}}"
        
        # MinMax with std
        minmax_str = f"{row['minmax_avg']:.2f}$\\pm${row['minmax_std']:.2f}"
        if row['minmax_avg'] == best_minmax:
            minmax_str = f"\\textbf{{{minmax_str}}}"
        elif second_minmax and row['minmax_avg'] == second_minmax:
            minmax_str = f"\\underline{{{minmax_str}}}"
        
        # Baseline with std (if available) - show as percentage with sign
        if has_baseline:
            sign = "+" if row['baseline_avg'] >= 0 else ""
            baseline_str = f"{sign}{row['baseline_avg']:.2f}\\%$\\pm${row['baseline_std']:.2f}"
            if row['baseline_avg'] == best_baseline:
                baseline_str = f"\\textbf{{{baseline_str}}}"
            elif second_baseline and row['baseline_avg'] == second_baseline:
                baseline_str = f"\\underline{{{baseline_str}}}"
        
        # Avg Rank with std
        rank_str = f"{row['avg_rank']:.2f}$\\pm${row['rank_std']:.2f}"
        if row['avg_rank'] == best_rank:
            rank_str = f"\\textbf{{{rank_str}}}"
        elif second_rank and row['avg_rank'] == second_rank:
            rank_str = f"\\underline{{{rank_str}}}"
        
        # Win%
        winrate_str = f"{row['win_rate']*100:.1f}\\%"
        if row['win_rate'] == best_winrate:
            winrate_str = f"\\textbf{{{winrate_str}}}"
        elif second_winrate and row['win_rate'] == second_winrate:
            winrate_str = f"\\underline{{{winrate_str}}}"
        
        if has_baseline:
            lines.append(f"{row['display_name']} & {zscore_str} & {minmax_str} & {baseline_str} & {rank_str} & {winrate_str} \\\\")
        else:
            lines.append(f"{row['display_name']} & {zscore_str} & {minmax_str} & {rank_str} & {winrate_str} \\\\")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    if has_baseline:
        lines.append(f"\\caption{{Aggregated performance metrics across {n_tasks} tasks. Z-Score and MinMax: higher is better. $\\Delta$\\% vs Ref: relative improvement over baseline model. Avg Rank: lower is better. Best in bold, second underlined.}}")
    else:
        lines.append(f"\\caption{{Aggregated performance metrics across {n_tasks} tasks. Z-Score and MinMax: higher is better. Avg Rank: lower is better. Best in bold, second underlined.}}")
    lines.append("\\label{table:aggregation}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


# =============================================================================
# Individual Table Generation Functions
# =============================================================================

def print_zscore_table(aggregation_results: Dict) -> str:
    """
    Print Z-Score per-task table to console (tasks as rows, models as columns).
    
    Args:
        aggregation_results: Output from compute_all_aggregations
    
    Returns:
        Formatted string table
    """
    models_data = aggregation_results['models']
    tasks = aggregation_results['metadata']['tasks']
    
    # Sort models by zscore_avg (descending)
    sorted_models = sorted(
        models_data.items(),
        key=lambda x: x[1]['zscore']['zscore_avg'],
        reverse=True
    )
    model_names = [data['display_name'] for _, data in sorted_models]
    
    lines = []
    lines.append("=" * 120)
    lines.append("Z-SCORE STANDARDIZATION (per-task z-scores, higher = better relative performance)")
    lines.append("=" * 120)
    
    # Header with model names
    header = f"{'Task':<25} " + " ".join([f"{m:>15}" for m in model_names])
    lines.append(header)
    lines.append("-" * 120)
    
    # Each row is a task
    for task in tasks:
        task_short = task.split("|")[0][:10] + "|" + task.split("|")[1]
        values = [models_data[m]['zscore']['zscore_per_task'].get(task, 0) for m, _ in sorted_models]
        row = f"{task_short:<25} " + " ".join([f"{v:>15.3f}" for v in values])
        lines.append(row)
    
    lines.append("-" * 120)
    # Average row with std
    avg_row = f"{'AVG±STD':<25} "
    for model, data in sorted_models:
        avg = data['zscore']['zscore_avg']
        std = data['zscore']['zscore_std']
        avg_row += f"{avg:>7.3f}±{std:<6.3f} "
    lines.append(avg_row)
    
    lines.append("=" * 120)
    return "\n".join(lines)


def print_minmax_table(aggregation_results: Dict) -> str:
    """
    Print MinMax per-task table to console (tasks as rows, models as columns).
    
    Args:
        aggregation_results: Output from compute_all_aggregations
    
    Returns:
        Formatted string table
    """
    models_data = aggregation_results['models']
    tasks = aggregation_results['metadata']['tasks']
    
    # Sort models by minmax_avg (descending)
    sorted_models = sorted(
        models_data.items(),
        key=lambda x: x[1]['minmax']['minmax_avg'],
        reverse=True
    )
    model_names = [data['display_name'] for _, data in sorted_models]
    
    lines = []
    lines.append("=" * 120)
    lines.append("MIN-MAX NORMALIZATION (scaled to [0,1], higher = better)")
    lines.append("=" * 120)
    
    # Header with model names
    header = f"{'Task':<25} " + " ".join([f"{m:>15}" for m in model_names])
    lines.append(header)
    lines.append("-" * 120)
    
    # Each row is a task
    for task in tasks:
        task_short = task.split("|")[0][:10] + "|" + task.split("|")[1]
        values = [models_data[m]['minmax']['minmax_per_task'].get(task, 0) for m, _ in sorted_models]
        row = f"{task_short:<25} " + " ".join([f"{v:>15.3f}" for v in values])
        lines.append(row)
    
    lines.append("-" * 120)
    # Average row with std
    avg_row = f"{'AVG±STD':<25} "
    for model, data in sorted_models:
        avg = data['minmax']['minmax_avg']
        std = data['minmax']['minmax_std']
        avg_row += f"{avg:>7.3f}±{std:<6.3f} "
    lines.append(avg_row)
    
    lines.append("=" * 120)
    return "\n".join(lines)


def print_rank_table(aggregation_results: Dict) -> str:
    """
    Print Rank-based per-task table to console (tasks as rows, models as columns).
    
    Args:
        aggregation_results: Output from compute_all_aggregations
    
    Returns:
        Formatted string table
    """
    models_data = aggregation_results['models']
    tasks = aggregation_results['metadata']['tasks']
    
    # Sort models by avg_rank (ascending - lower is better)
    sorted_models = sorted(
        models_data.items(),
        key=lambda x: x[1]['rank']['avg_rank'],
        reverse=False
    )
    model_names = [data['display_name'] for _, data in sorted_models]
    
    lines = []
    lines.append("=" * 120)
    lines.append("RANK-BASED AGGREGATION (rank per task, 1=best, lower avg = better)")
    lines.append("=" * 120)
    
    # Header with model names
    header = f"{'Task':<25} " + " ".join([f"{m:>15}" for m in model_names])
    lines.append(header)
    lines.append("-" * 120)
    
    # Each row is a task
    for task in tasks:
        task_short = task.split("|")[0][:10] + "|" + task.split("|")[1]
        values = [models_data[m]['rank']['rank_per_task'].get(task, 0) for m, _ in sorted_models]
        row = f"{task_short:<25} " + " ".join([f"{v:>15d}" for v in values])
        lines.append(row)
    
    lines.append("-" * 120)
    # Summary rows
    avg_row = f"{'AVG RANK±STD':<25} "
    for model, data in sorted_models:
        avg = data['rank']['avg_rank']
        std = data['rank']['rank_std']
        avg_row += f"{avg:>7.2f}±{std:<6.2f} "
    lines.append(avg_row)
    
    borda_row = f"{'BORDA SCORE':<25} "
    for model, data in sorted_models:
        borda = data['rank']['borda_score']
        borda_row += f"{borda:>15.0f} "
    lines.append(borda_row)
    
    win_row = f"{'WIN RATE':<25} "
    for model, data in sorted_models:
        win_rate = data['pairwise']['win_rate'] * 100
        win_row += f"{win_rate:>14.1f}% "
    lines.append(win_row)
    
    lines.append("=" * 120)
    return "\n".join(lines)


def generate_zscore_latex_table(aggregation_results: Dict) -> str:
    """
    Generate LaTeX table for Z-Score results (tasks as rows, models as columns).
    
    Args:
        aggregation_results: Output from compute_all_aggregations
    
    Returns:
        LaTeX table string
    """
    models_data = aggregation_results['models']
    tasks = aggregation_results['metadata']['tasks']
    n_tasks = len(tasks)
    
    # Sort models by zscore_avg (descending)
    sorted_models = sorted(
        models_data.items(),
        key=lambda x: x[1]['zscore']['zscore_avg'],
        reverse=True
    )
    n_models = len(sorted_models)
    
    # Find best per task
    best_per_task = {}
    second_per_task = {}
    for task in tasks:
        values = [(m, d['zscore']['zscore_per_task'].get(task, float('-inf'))) for m, d in models_data.items()]
        sorted_vals = sorted(values, key=lambda x: x[1], reverse=True)
        best_per_task[task] = sorted_vals[0][0] if sorted_vals else None
        second_per_task[task] = sorted_vals[1][0] if len(sorted_vals) > 1 else None
    
    best_avg = max(d['zscore']['zscore_avg'] for d in models_data.values())
    second_avg_val = sorted([d['zscore']['zscore_avg'] for d in models_data.values()], reverse=True)
    second_avg = second_avg_val[1] if len(second_avg_val) > 1 else None
    
    lines = []
    lines.append("\\begin{table*}[t!]")
    lines.append("\\tiny")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{|l|" + "c|" * n_models + "}")
    lines.append("\\hline")
    
    # Header with model names
    model_headers = " & ".join([f"\\textbf{{{data['display_name']}}}" for _, data in sorted_models])
    lines.append(f"\\textbf{{Task}} & {model_headers} \\\\ ")
    lines.append("\\hline")
    
    # Each row is a task
    for task in tasks:
        task_label = task.replace("|", "-").replace("_", "-")
        
        task_strs = []
        for model, data in sorted_models:
            val = data['zscore']['zscore_per_task'].get(task, 0)
            val_str = f"{val:.2f}"
            if model == best_per_task.get(task):
                val_str = f"\\textbf{{{val_str}}}"
            elif model == second_per_task.get(task):
                val_str = f"\\underline{{{val_str}}}"
            task_strs.append(val_str)
        
        lines.append(f"{task_label} & " + " & ".join(task_strs) + " \\\\ ")
    
    lines.append("\\hline")
    
    # Average row with std
    avg_strs = []
    for model, data in sorted_models:
        avg = data['zscore']['zscore_avg']
        std = data['zscore']['zscore_std']
        avg_str = f"{avg:.2f}$\\pm${std:.2f}"
        if avg == best_avg:
            avg_str = f"\\textbf{{{avg_str}}}"
        elif second_avg and avg == second_avg:
            avg_str = f"\\underline{{{avg_str}}}"
        avg_strs.append(avg_str)
    
    lines.append(f"\\textbf{{AVG}} & " + " & ".join(avg_strs) + " \\\\ ")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{Z-Score standardized performance across {n_tasks} tasks. Higher values indicate better relative performance. Best in bold, second underlined.}}")
    lines.append("\\label{table:zscore}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_minmax_latex_table(aggregation_results: Dict) -> str:
    """
    Generate LaTeX table for MinMax results (tasks as rows, models as columns).
    
    Args:
        aggregation_results: Output from compute_all_aggregations
    
    Returns:
        LaTeX table string
    """
    models_data = aggregation_results['models']
    tasks = aggregation_results['metadata']['tasks']
    n_tasks = len(tasks)
    
    # Sort models by minmax_avg (descending)
    sorted_models = sorted(
        models_data.items(),
        key=lambda x: x[1]['minmax']['minmax_avg'],
        reverse=True
    )
    n_models = len(sorted_models)
    
    # Find best per task
    best_per_task = {}
    second_per_task = {}
    for task in tasks:
        values = [(m, d['minmax']['minmax_per_task'].get(task, float('-inf'))) for m, d in models_data.items()]
        sorted_vals = sorted(values, key=lambda x: x[1], reverse=True)
        best_per_task[task] = sorted_vals[0][0] if sorted_vals else None
        second_per_task[task] = sorted_vals[1][0] if len(sorted_vals) > 1 else None
    
    best_avg = max(d['minmax']['minmax_avg'] for d in models_data.values())
    second_avg_val = sorted([d['minmax']['minmax_avg'] for d in models_data.values()], reverse=True)
    second_avg = second_avg_val[1] if len(second_avg_val) > 1 else None
    
    lines = []
    lines.append("\\begin{table*}[t!]")
    lines.append("\\tiny")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{|l|" + "c|" * n_models + "}")
    lines.append("\\hline")
    
    # Header with model names
    model_headers = " & ".join([f"\\textbf{{{data['display_name']}}}" for _, data in sorted_models])
    lines.append(f"\\textbf{{Task}} & {model_headers} \\\\ ")
    lines.append("\\hline")
    
    # Each row is a task
    for task in tasks:
        task_label = task.replace("|", "-").replace("_", "-")
        
        task_strs = []
        for model, data in sorted_models:
            val = data['minmax']['minmax_per_task'].get(task, 0)
            val_str = f"{val:.2f}"
            if model == best_per_task.get(task):
                val_str = f"\\textbf{{{val_str}}}"
            elif model == second_per_task.get(task):
                val_str = f"\\underline{{{val_str}}}"
            task_strs.append(val_str)
        
        lines.append(f"{task_label} & " + " & ".join(task_strs) + " \\\\ ")
    
    lines.append("\\hline")
    
    # Average row with std
    avg_strs = []
    for model, data in sorted_models:
        avg = data['minmax']['minmax_avg']
        std = data['minmax']['minmax_std']
        avg_str = f"{avg:.2f}$\\pm${std:.2f}"
        if avg == best_avg:
            avg_str = f"\\textbf{{{avg_str}}}"
        elif second_avg and avg == second_avg:
            avg_str = f"\\underline{{{avg_str}}}"
        avg_strs.append(avg_str)
    
    lines.append(f"\\textbf{{AVG}} & " + " & ".join(avg_strs) + " \\\\ ")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{Min-Max normalized performance across {n_tasks} tasks. Scores scaled to [0,1] range. Best in bold, second underlined.}}")
    lines.append("\\label{table:minmax}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_rank_latex_table(aggregation_results: Dict) -> str:
    """
    Generate LaTeX table for Rank-based results (tasks as rows, models as columns).
    
    Args:
        aggregation_results: Output from compute_all_aggregations
    
    Returns:
        LaTeX table string
    """
    models_data = aggregation_results['models']
    tasks = aggregation_results['metadata']['tasks']
    n_tasks = len(tasks)
    
    # Sort models by avg_rank (ascending - lower is better)
    sorted_models = sorted(
        models_data.items(),
        key=lambda x: x[1]['rank']['avg_rank'],
        reverse=False
    )
    n_models = len(sorted_models)
    
    # Find best per task (lowest rank = 1)
    best_per_task = {}
    for task in tasks:
        values = [(m, d['rank']['rank_per_task'].get(task, float('inf'))) for m, d in models_data.items()]
        sorted_vals = sorted(values, key=lambda x: x[1])
        best_per_task[task] = sorted_vals[0][0] if sorted_vals else None
    
    best_avg_rank = min(d['rank']['avg_rank'] for d in models_data.values())
    second_avg_val = sorted([d['rank']['avg_rank'] for d in models_data.values()])
    second_avg_rank = second_avg_val[1] if len(second_avg_val) > 1 else None
    best_borda = max(d['rank']['borda_score'] for d in models_data.values())
    best_winrate = max(d['pairwise']['win_rate'] for d in models_data.values())
    
    lines = []
    lines.append("\\begin{table*}[t!]")
    lines.append("\\tiny")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{|l|" + "c|" * n_models + "}")
    lines.append("\\hline")
    
    # Header with model names
    model_headers = " & ".join([f"\\textbf{{{data['display_name']}}}" for _, data in sorted_models])
    lines.append(f"\\textbf{{Task}} & {model_headers} \\\\ ")
    lines.append("\\hline")
    
    # Each row is a task
    for task in tasks:
        task_label = task.replace("|", "-").replace("_", "-")
        
        task_strs = []
        for model, data in sorted_models:
            val = data['rank']['rank_per_task'].get(task, 0)
            val_str = f"{val}"
            if model == best_per_task.get(task):
                val_str = f"\\textbf{{{val_str}}}"
            task_strs.append(val_str)
        
        lines.append(f"{task_label} & " + " & ".join(task_strs) + " \\\\ ")
    
    lines.append("\\hline")
    
    # Average Rank row with std
    avg_strs = []
    for model, data in sorted_models:
        avg = data['rank']['avg_rank']
        std = data['rank']['rank_std']
        avg_str = f"{avg:.2f}$\\pm${std:.2f}"
        if avg == best_avg_rank:
            avg_str = f"\\textbf{{{avg_str}}}"
        elif second_avg_rank and avg == second_avg_rank:
            avg_str = f"\\underline{{{avg_str}}}"
        avg_strs.append(avg_str)
    lines.append(f"\\textbf{{AVG RANK}} & " + " & ".join(avg_strs) + " \\\\ ")
    
    # Borda score row
    borda_strs = []
    for model, data in sorted_models:
        borda = data['rank']['borda_score']
        borda_str = f"{borda:.0f}"
        if borda == best_borda:
            borda_str = f"\\textbf{{{borda_str}}}"
        borda_strs.append(borda_str)
    lines.append(f"\\textbf{{BORDA}} & " + " & ".join(borda_strs) + " \\\\ ")
    
    # Win rate row
    win_strs = []
    for model, data in sorted_models:
        win_rate = data['pairwise']['win_rate'] * 100
        win_str = f"{win_rate:.1f}\\%"
        if data['pairwise']['win_rate'] == best_winrate:
            win_str = f"\\textbf{{{win_str}}}"
        win_strs.append(win_str)
    lines.append(f"\\textbf{{WIN\\%}} & " + " & ".join(win_strs) + " \\\\ ")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{Rank-based aggregation across {n_tasks} tasks. Rank 1 = best on each task. Lower average rank is better. Borda score and Win\\% higher is better.}}")
    lines.append("\\label{table:rank}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)
