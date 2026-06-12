"""Zipf's Law Analysis Utility — used by Benford engine.

Zipf's Law states that the frequency of a word (or element) is
inversely proportional to its rank in a frequency table:
    freq(r) ∝ 1/r^s  (s ≈ 1)

Applied to exam forensics:
- Wrong answer choices should follow Zipf distribution if organic
- Flat distributions indicate coordinated answer sharing
- Power-law deviation flags systematic manipulation

Reference: Zipf, G.K. (1949). Human Behavior and the Principle of Least Effort.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def zipf_law(rank: np.ndarray, s: float, c: float) -> np.ndarray:
    """Zipf distribution: f(r) = c / r^s"""
    return c / np.power(rank, s)


def compute_zipf_deviation(answer_column: np.ndarray, n_options: int = 4) -> dict:
    """Analyze whether wrong answer frequencies follow Zipf's Law.
    
    Args:
        answer_column: 1D array of student answers for one question
        n_options: Number of answer options
    
    Returns:
        Dict with Zipf exponent, R² fit, and deviation metric
    """
    # Count frequencies of each option
    counts = np.bincount(answer_column.astype(int), minlength=n_options)
    
    if counts.sum() == 0:
        return {"zipf_s": 0, "r_squared": 0, "is_organic": True}
    
    # Sort descending for rank-frequency analysis
    sorted_counts = np.sort(counts)[::-1]
    nonzero = sorted_counts[sorted_counts > 0]
    
    if len(nonzero) < 2:
        return {"zipf_s": 0, "r_squared": 0, "is_organic": True}
    
    ranks = np.arange(1, len(nonzero) + 1, dtype=float)
    freqs = nonzero.astype(float)
    
    try:
        # Fit Zipf's law: f(r) = c / r^s
        popt, _ = curve_fit(zipf_law, ranks, freqs, p0=[1.0, freqs[0]], maxfev=1000)
        s_param = float(popt[0])
        
        # Compute R² (goodness of fit)
        predicted = zipf_law(ranks, *popt)
        ss_res = np.sum((freqs - predicted) ** 2)
        ss_tot = np.sum((freqs - np.mean(freqs)) ** 2)
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-10)
        
        # Organic answers typically have s ∈ [0.5, 2.0] with R² > 0.8
        is_organic = 0.3 < s_param < 3.0 and r_squared > 0.6
        
        return {
            "zipf_s": round(s_param, 4),
            "r_squared": round(max(0, r_squared), 4),
            "is_organic": is_organic,
        }
    except (RuntimeError, ValueError):
        return {"zipf_s": 0, "r_squared": 0, "is_organic": True}


def analyze_answer_distribution(answers: np.ndarray, n_options: int = 4) -> dict:
    """Analyze overall answer distribution for forensic anomalies.
    
    Checks:
    1. Per-question Zipf conformity
    2. Shannon entropy of answer choices
    3. Chi-squared uniformity test
    """
    n_students, n_questions = answers.shape
    
    zipf_results = []
    entropies = []
    uniform_deviations = []
    
    for q in range(n_questions):
        col = answers[:, q]
        
        # Zipf analysis
        zipf = compute_zipf_deviation(col, n_options)
        zipf_results.append(zipf)
        
        # Shannon entropy
        counts = np.bincount(col.astype(int), minlength=n_options)
        probs = counts / max(counts.sum(), 1)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        entropies.append(float(entropy))
        
        # Chi-squared test for uniformity
        expected = np.full(n_options, counts.sum() / n_options)
        chi2, p = stats.chisquare(counts, expected)
        uniform_deviations.append(float(p))
    
    n_organic = sum(1 for z in zipf_results if z["is_organic"])
    avg_entropy = np.mean(entropies) if entropies else 0
    max_entropy = np.log2(n_options)
    
    return {
        "n_questions_analyzed": n_questions,
        "organic_questions_pct": round(n_organic / max(n_questions, 1), 4),
        "avg_entropy": round(float(avg_entropy), 4),
        "max_entropy": round(float(max_entropy), 4),
        "entropy_ratio": round(float(avg_entropy / max(max_entropy, 1e-10)), 4),
        "avg_zipf_s": round(float(np.mean([z["zipf_s"] for z in zipf_results])), 4),
        "avg_zipf_r2": round(float(np.mean([z["r_squared"] for z in zipf_results])), 4),
        "uniform_questions": sum(1 for p in uniform_deviations if p > 0.05),
        "non_uniform_questions": sum(1 for p in uniform_deviations if p <= 0.05),
    }
