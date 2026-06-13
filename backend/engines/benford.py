"""Engine 9 (Bonus): Benford's Law Analysis.

Algorithm: First-digit frequency distribution test
Tests whether exam score distributions follow Benford's Law.
Significant deviation indicates potential data manipulation.

Benford's Law states that in naturally occurring datasets,
the leading digit d (1-9) appears with probability:
  P(d) = log10(1 + 1/d)

This means digit 1 appears ~30.1% of the time, while digit 9
appears only ~4.6%. Manipulated data tends to show uniform
distributions or suspicious spikes.

Reference: Benford, F. (1938). "The law of anomalous numbers."
Widely used in forensic accounting (Nigrini, 2012).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from scipy import stats

from engines.base_engine import BaseEngine, EngineOutput

logger = logging.getLogger(__name__)


class BenfordEngine(BaseEngine):
    """Detect score manipulation using Benford's Law."""

    def __init__(self):
        super().__init__(engine_name="benford_law", requires_gpu=False)

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

        if student_ids is None:
            student_ids = [f"STU_{i:06d}" for i in range(n_students)]

        self.report_progress(10, "Computing student scores for Benford analysis")

        # ── Step 1: Compute scores ──
        if answer_key is not None:
            scores = (answers == answer_key[np.newaxis, :]).sum(axis=1)
        else:
            scores = answers.sum(axis=1)

        # Filter out zeros (can't have a leading digit of 0)
        nonzero_scores = scores[scores > 0]

        if len(nonzero_scores) < 50:
            return EngineOutput(
                engine_name=self.engine_name,
                status="skipped",
                result_data={"message": "Too few nonzero scores for Benford analysis"},
                summary={"applicable": False},
            )

        # Check if data spans enough orders of magnitude for Benford to apply
        score_range_ratio = float(nonzero_scores.max()) / float(nonzero_scores.min()) if nonzero_scores.min() > 0 else 1
        if score_range_ratio < 10:
            return EngineOutput(
                engine_name=self.engine_name,
                status="complete",
                result_data={
                    "message": "Score range too narrow for Benford analysis",
                    "score_range_ratio": round(score_range_ratio, 2),
                    "conforms_to_benford": True,
                },
                summary={"applicable": False, "conforms": True, "flagged": 0},
            )

        self.report_progress(30, "Extracting first-digit distribution")

        first_digits = np.array([int(str(abs(int(s)))[0]) for s in nonzero_scores])

        observed_counts = np.array([np.sum(first_digits == d) for d in range(1, 10)])
        observed_freq = observed_counts / observed_counts.sum()

        benford_freq = np.array([np.log10(1 + 1/d) for d in range(1, 10)])

        self.report_progress(50, "Running chi-squared goodness-of-fit test")

        expected_counts = benford_freq * observed_counts.sum()
        chi2_stat, chi2_pvalue = stats.chisquare(observed_counts, f_exp=expected_counts)

        self.report_progress(60, "Computing KL divergence")

        obs_smooth = observed_freq + 1e-10
        obs_smooth /= obs_smooth.sum()
        kl_divergence = float(np.sum(obs_smooth * np.log(obs_smooth / benford_freq)))

        self.report_progress(70, "Analyzing per-center Benford conformity")

        flagged_centers = []
        center_results = {}

        if center_ids:
            unique_centers = list(set(center_ids))
            for cid in unique_centers:
                center_mask = np.array([c == cid for c in center_ids])
                center_scores = scores[center_mask]
                center_nonzero = center_scores[center_scores > 0]

                if len(center_nonzero) < 30:
                    continue

                center_range = float(center_nonzero.max()) / float(center_nonzero.min()) if center_nonzero.min() > 0 else 1
                if center_range < 10:
                    continue

                center_digits = np.array([int(str(abs(int(s)))[0]) for s in center_nonzero])
                center_counts = np.array([np.sum(center_digits == d) for d in range(1, 10)])

                if center_counts.sum() < 30:
                    continue

                center_expected = benford_freq * center_counts.sum()
                c_chi2, c_pval = stats.chisquare(center_counts, f_exp=center_expected)

                center_results[cid] = {
                    "chi2": round(float(c_chi2), 2),
                    "p_value": float(c_pval),
                    "n_students": int(center_mask.sum()),
                    "conforms": c_pval > 0.05,
                }

                if c_pval < 0.001 and kl_divergence > 0.1:
                    flagged_centers.append(cid)

        self.report_progress(85, "Flagging students from Benford-anomalous centers")

        flagged_ids = []
        if flagged_centers and center_ids:
            for i, sid in enumerate(student_ids):
                if i < len(center_ids) and center_ids[i] in flagged_centers:
                    flagged_ids.append(sid)

        conforms = chi2_pvalue > 0.05
        severity = "low" if chi2_pvalue > 0.1 else "medium" if chi2_pvalue > 0.01 else "high"

        self.report_progress(100, f"Benford analysis complete — {'conforms' if conforms else 'deviation detected'}")

        return EngineOutput(
            engine_name=self.engine_name,
            flagged_count=len(flagged_ids),
            flagged_student_ids=flagged_ids,
            result_data={
                "conforms_to_benford": conforms,
                "chi2_statistic": round(float(chi2_stat), 3),
                "chi2_p_value": float(chi2_pvalue),
                "kl_divergence": round(kl_divergence, 6),
                "severity": severity,
                "observed_distribution": {str(d): round(float(observed_freq[d-1]), 4) for d in range(1, 10)},
                "expected_benford": {str(d): round(float(benford_freq[d-1]), 4) for d in range(1, 10)},
                "flagged_centers": flagged_centers,
                "center_analysis": center_results,
                "n_scores_analyzed": int(len(nonzero_scores)),
            },
            summary={
                "conforms": conforms,
                "chi2_p_value": round(float(chi2_pvalue), 4),
                "kl_divergence": round(kl_divergence, 6),
                "flagged_centers": len(flagged_centers),
                "flagged": len(flagged_ids),
            },
        )
