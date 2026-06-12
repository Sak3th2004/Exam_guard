"""Gini Impurity analysis for answer distribution forensics.

Gini coefficient measures inequality in answer distributions.
In exam fraud detection:
- Organic answers have moderate Gini (students choose differently)
- Copy rings show very LOW Gini (everyone picks the same wrong answer)
- Random guessing shows very HIGH Gini (nearly uniform)

Gini = 1 - Σ(p_i²)  where p_i = proportion choosing option i
"""

from __future__ import annotations

import numpy as np


def gini_impurity(counts: np.ndarray) -> float:
    """Compute Gini impurity for an answer distribution.
    
    Returns value in [0, 1-1/k] where k = number of options.
    Low = concentrated (one option dominates). High = uniform.
    """
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return float(1.0 - np.sum(probs ** 2))


def per_question_gini(answers: np.ndarray, n_options: int = 4) -> list[float]:
    """Compute Gini impurity per question."""
    n_students, n_questions = answers.shape
    ginis = []
    for q in range(n_questions):
        counts = np.bincount(answers[:, q].astype(int), minlength=n_options)
        ginis.append(gini_impurity(counts))
    return ginis


def detect_gini_anomalies(
    answers: np.ndarray, 
    n_options: int = 4,
    low_threshold: float = 0.2,
    high_threshold: float = 0.74,
) -> dict:
    """Detect questions with abnormal Gini impurity.
    
    Low Gini: everyone picked the same answer (possible leak or trivial question)
    High Gini: nearly uniform distribution (possible random guessing in a cohort)
    
    Returns summary stats and flagged question indices.
    """
    ginis = per_question_gini(answers, n_options)
    
    # Max Gini for k options = 1 - 1/k
    max_gini = 1.0 - 1.0 / n_options
    
    low_gini_qs = [i for i, g in enumerate(ginis) if g < low_threshold]
    high_gini_qs = [i for i, g in enumerate(ginis) if g > high_threshold]
    
    return {
        "gini_values": [round(g, 4) for g in ginis],
        "mean_gini": round(float(np.mean(ginis)), 4),
        "std_gini": round(float(np.std(ginis)), 4),
        "max_possible_gini": round(max_gini, 4),
        "low_gini_questions": low_gini_qs,
        "high_gini_questions": high_gini_qs,
        "n_low_gini": len(low_gini_qs),
        "n_high_gini": len(high_gini_qs),
    }
