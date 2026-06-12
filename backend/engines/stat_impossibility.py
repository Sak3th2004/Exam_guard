"""Engine 2: Statistical Impossibility Prover.

Algorithms: Binomial Probability + Bonferroni Correction
Mathematically PROVES that copying was not coincidence.

For each flagged pair:
  1. Compute per-question match probability (Herfindahl index)
  2. P(X >= M matches) via binomial survival function
  3. Apply Bonferroni correction for multiple testing
  4. If P < α_corrected → STATISTICALLY IMPOSSIBLE
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
from scipy import stats

from engines.base_engine import BaseEngine, EngineOutput

logger = logging.getLogger(__name__)

# Bonferroni significance level
ALPHA = 0.05
# Maximum pairs to evaluate (for performance)
MAX_PAIRS_EVALUATE = 50_000


class StatImpossibilityEngine(BaseEngine):
    """Prove statistical impossibility of observed matching patterns."""

    def __init__(self):
        super().__init__(engine_name="stat_impossibility", requires_gpu=False)

    async def _run_analysis(
        self,
        answers: np.ndarray,
        answer_key: Optional[np.ndarray],
        student_ids: Optional[list[str]],
        center_ids: Optional[list[str]],
        timing_data: Optional[np.ndarray],
        question_texts: Optional[list[str]],
        ground_truth: Optional[dict[str, Any]],
        **kwargs: Any,
    ) -> EngineOutput:
        n_students, n_questions = answers.shape
        n_options = int(answers.max()) + 1

        if student_ids is None:
            student_ids = [f"STU_{i:06d}" for i in range(n_students)]

        # ── Step 1: Per-question match probability (Herfindahl index) ──
        self.report_progress(10, "Computing per-question match probabilities (Herfindahl index)")

        p_match_per_q = np.zeros(n_questions)
        for q in range(n_questions):
            col = answers[:, q]
            counts = np.bincount(col, minlength=n_options)
            freqs = counts / n_students
            # Herfindahl: sum of squared frequencies
            p_match_per_q[q] = float(np.sum(freqs ** 2))

        p_avg = float(np.mean(p_match_per_q))
        logger.info(f"Average per-question match probability: {p_avg:.4f}")

        # ── Step 2: Identify candidate pairs (from copy ring results or top similar) ──
        self.report_progress(20, "Identifying high-similarity pairs for testing")

        # Get candidate pairs from kwargs (passed from E1) or compute top pairs
        candidate_pairs = kwargs.get("candidate_pairs", None)

        if candidate_pairs is None:
            # Compute pairwise matches for a sample
            sample_size = min(5000, n_students)
            sample_indices = np.random.choice(n_students, size=sample_size, replace=False)
            candidate_pairs = []

            for idx_i in range(len(sample_indices)):
                for idx_j in range(idx_i + 1, len(sample_indices)):
                    i = sample_indices[idx_i]
                    j = sample_indices[idx_j]
                    matches = int((answers[i] == answers[j]).sum())
                    if matches > n_questions * p_avg * 1.8:  # Stricter threshold
                        candidate_pairs.append((i, j, matches))

                if idx_i % 500 == 0 and idx_i > 0:
                    pct = 20 + int(30 * idx_i / len(sample_indices))
                    self.report_progress(pct, f"Sampling pairs: {idx_i}/{len(sample_indices)}")

        # Also include pairs from copy ring if available
        copy_ring_pairs = kwargs.get("copy_ring_edges", [])
        for edge in copy_ring_pairs:
            i, j = edge.get("i", -1), edge.get("j", -1)
            if 0 <= i < n_students and 0 <= j < n_students:
                matches = int((answers[i] == answers[j]).sum())
                candidate_pairs.append((i, j, matches))

        # Deduplicate
        seen = set()
        unique_pairs = []
        for pair in candidate_pairs:
            key = (min(pair[0], pair[1]), max(pair[0], pair[1]))
            if key not in seen:
                seen.add(key)
                unique_pairs.append(pair)

        # Limit
        if len(unique_pairs) > MAX_PAIRS_EVALUATE:
            unique_pairs.sort(key=lambda p: p[2], reverse=True)
            unique_pairs = unique_pairs[:MAX_PAIRS_EVALUATE]

        logger.info(f"Evaluating {len(unique_pairs)} candidate pairs")

        # ── Step 3: Binomial testing with Bonferroni ──
        self.report_progress(50, f"Running binomial tests on {len(unique_pairs)} pairs")

        # Bonferroni correction
        n_total_possible_pairs = n_students * (n_students - 1) // 2
        alpha_corrected = ALPHA / max(n_total_possible_pairs, 1)

        impossible_pairs: list[dict[str, Any]] = []
        flagged_ids: set[str] = set()

        for idx, (i, j, total_matches) in enumerate(unique_pairs):
            if isinstance(i, (int, np.integer)) and isinstance(j, (int, np.integer)):
                a_i = answers[int(i)]
                a_j = answers[int(j)]

                total_matches = int((a_i == a_j).sum())

                # Wrong answer matches
                matching_wrong = 0
                if answer_key is not None:
                    for q in range(n_questions):
                        if a_i[q] == a_j[q] and a_i[q] != answer_key[q]:
                            matching_wrong += 1

                # Expected matches under independence
                expected_matches = p_avg * n_questions

                # Binomial survival function: P(X >= M)
                p_value = float(stats.binom.sf(total_matches - 1, n_questions, p_avg))

                is_impossible = p_value < alpha_corrected

                if is_impossible:
                    # Human-readable probability
                    if p_value > 0:
                        log_p = math.log10(p_value)
                        human_readable = _format_probability(p_value)
                    else:
                        log_p = -300
                        human_readable = "Probability too small to compute — effectively zero"

                    pair_data = {
                        "student_a": student_ids[int(i)],
                        "student_b": student_ids[int(j)],
                        "matching_total": total_matches,
                        "matching_wrong": matching_wrong,
                        "total_questions": n_questions,
                        "expected_matches": round(expected_matches, 1),
                        "p_value": p_value,
                        "log10_p_value": round(log_p, 1),
                        "alpha_corrected": alpha_corrected,
                        "is_impossible": True,
                        "human_readable": human_readable,
                    }
                    impossible_pairs.append(pair_data)
                    flagged_ids.add(student_ids[int(i)])
                    flagged_ids.add(student_ids[int(j)])

            if idx % 1000 == 0 and idx > 0:
                pct = 50 + int(40 * idx / max(len(unique_pairs), 1))
                self.report_progress(pct, f"Binomial tests: {idx}/{len(unique_pairs)}")

        # Sort by p-value (most extreme first)
        impossible_pairs.sort(key=lambda p: p["p_value"])

        return EngineOutput(
            engine_name=self.engine_name,
            flagged_count=len(flagged_ids),
            flagged_student_ids=list(flagged_ids),
            result_data={
                "impossible_pairs": len(impossible_pairs),
                "pairs": impossible_pairs[:200],  # Cap for JSON size
                "p_avg_match": round(p_avg, 4),
                "alpha_corrected": alpha_corrected,
                "n_total_possible_pairs": n_total_possible_pairs,
                "n_evaluated": len(unique_pairs),
            },
            summary={
                "impossible_pairs": len(impossible_pairs),
                "flagged": len(flagged_ids),
                "most_extreme_p": impossible_pairs[0]["p_value"] if impossible_pairs else None,
                "most_extreme_readable": impossible_pairs[0]["human_readable"] if impossible_pairs else "None",
            },
        )


def _format_probability(p_value: float) -> str:
    """Convert p-value to human-readable probability string."""
    if p_value <= 0 or p_value > 1:
        return "Statistically impossible"

    inverse = 1.0 / max(p_value, 1e-300)

    if inverse < 1000:
        return f"1 in {inverse:.0f}"
    elif inverse < 1e6:
        return f"1 in {inverse/1000:.1f} thousand"
    elif inverse < 1e9:
        return f"1 in {inverse/1e6:.1f} million"
    elif inverse < 1e12:
        return f"1 in {inverse/1e9:.1f} billion"
    elif inverse < 1e15:
        return f"1 in {inverse/1e12:.1f} trillion"
    else:
        exponent = int(math.log10(inverse))
        mantissa = inverse / (10 ** exponent)
        comparisons = {
            15: "more unlikely than winning the lottery twice in a row",
            20: "more unlikely than being struck by lightning 3 times",
            30: "more unlikely than shuffling a deck into the same order twice",
            44: "more unlikely than randomly selecting a specific atom from all atoms in the solar system",
        }
        comparison = ""
        for threshold, text in sorted(comparisons.items()):
            if exponent >= threshold:
                comparison = f" — {text}"
        return f"1 in {mantissa:.1f} × 10^{exponent}{comparison}"
