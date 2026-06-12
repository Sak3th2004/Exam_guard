"""Shannon Entropy and Information Theory utilities for exam forensics.

Shannon entropy H = -Σ p(x) × log₂(p(x))

High entropy = diverse/random answers (normal)
Low entropy = concentrated answers (potential coordination)

Relative entropy (KL divergence) measures how much an observed
distribution differs from an expected distribution.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def shannon_entropy(counts: np.ndarray) -> float:
    """Compute Shannon entropy in bits for a count distribution."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]  # avoid log(0)
    return float(-np.sum(probs * np.log2(probs)))


def normalized_entropy(counts: np.ndarray) -> float:
    """Entropy normalized to [0, 1] where 1 = maximum entropy (uniform)."""
    k = len(counts)
    if k <= 1:
        return 0.0
    h = shannon_entropy(counts)
    h_max = np.log2(k)
    return float(h / h_max) if h_max > 0 else 0.0


def kl_divergence(observed: np.ndarray, expected: np.ndarray) -> float:
    """Compute KL divergence D_KL(P || Q) in bits.
    
    P = observed distribution, Q = expected distribution.
    Smoothed to avoid division by zero.
    """
    p = observed.astype(float) + 1e-10
    q = expected.astype(float) + 1e-10
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log2(p / q)))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence — symmetric version of KL divergence.
    
    JSD(P || Q) = 0.5 × KL(P || M) + 0.5 × KL(Q || M)
    where M = 0.5 × (P + Q)
    
    Returns value in [0, 1] (bits).
    """
    p_norm = p.astype(float) + 1e-10
    q_norm = q.astype(float) + 1e-10
    p_norm /= p_norm.sum()
    q_norm /= q_norm.sum()
    m = 0.5 * (p_norm + q_norm)
    return float(0.5 * kl_divergence(p_norm, m) + 0.5 * kl_divergence(q_norm, m))


def per_center_entropy(
    answers: np.ndarray, 
    center_ids: list[str], 
    n_options: int = 4
) -> dict[str, dict]:
    """Compute entropy statistics per exam center.
    
    Returns dict mapping center_id → {mean_entropy, min_entropy, n_students}.
    Low mean_entropy centers may indicate coordinated fraud.
    """
    results = {}
    unique_centers = sorted(set(center_ids))
    
    for cid in unique_centers:
        mask = np.array([c == cid for c in center_ids])
        center_answers = answers[mask]
        n_students = center_answers.shape[0]
        
        if n_students < 5:
            continue
        
        n_questions = center_answers.shape[1]
        entropies = []
        
        for q in range(n_questions):
            counts = np.bincount(center_answers[:, q].astype(int), minlength=n_options)
            entropies.append(normalized_entropy(counts))
        
        results[cid] = {
            "mean_entropy": round(float(np.mean(entropies)), 4),
            "min_entropy": round(float(np.min(entropies)), 4),
            "max_entropy": round(float(np.max(entropies)), 4),
            "std_entropy": round(float(np.std(entropies)), 4),
            "n_students": n_students,
            "n_low_entropy_questions": sum(1 for e in entropies if e < 0.3),
        }
    
    return results
