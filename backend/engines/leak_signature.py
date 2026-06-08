"""Engine 4: Leak Signature Recognizer.

Algorithms: Difficulty Curve Inversion + IRT 2PL Person-Fit (lz*)
Detects the distinctive "fingerprint" of a paper leak.

Key Insight:
  Normal student:  Easy Qs → high accuracy, Hard Qs → low accuracy
  Leaked student:  BOTH easy AND hard → high accuracy (flat curve)

Pipeline:
  1. Compute per-question difficulty
  2. Sort questions into quartiles Q1(easy)...Q4(hard)
  3. Per student, compute accuracy in each quartile
  4. gradient = Q4_accuracy - Q1_accuracy (normal < -0.20, suspicious > -0.05)
  5. IRT 2PL person-fit lz* statistic for misfit detection
  6. Flag students with BOTH flat gradient AND high IRT misfit
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

from engines.base_engine import BaseEngine, EngineOutput

logger = logging.getLogger(__name__)

GRADIENT_THRESHOLD = -0.05
IRT_MISFIT_THRESHOLD = 2.0


class LeakSignatureEngine(BaseEngine):
    """Detect paper leak signatures via difficulty curve analysis."""

    def __init__(self):
        super().__init__(engine_name="leak_signature", requires_gpu=False)

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

        if answer_key is None:
            return EngineOutput(
                engine_name=self.engine_name,
                status="skipped",
                result_data={"message": "Answer key required for leak detection"},
            )

        correct_matrix = (answers == answer_key[np.newaxis, :])
        scores = correct_matrix.sum(axis=1)

        # ── Step 1: Question difficulty ──
        self.report_progress(10, "Computing question difficulties")

        # difficulty = 1 - correct_rate (harder questions have higher difficulty)
        q_correct_rate = correct_matrix.mean(axis=0)
        q_difficulty = 1.0 - q_correct_rate

        # ── Step 2: Sort into quartiles ──
        self.report_progress(15, "Sorting questions into difficulty quartiles")

        quartile_boundaries = np.percentile(q_difficulty, [25, 50, 75])
        q_quartile = np.zeros(n_questions, dtype=int)
        for q in range(n_questions):
            if q_difficulty[q] <= quartile_boundaries[0]:
                q_quartile[q] = 0  # Q1 (Easy)
            elif q_difficulty[q] <= quartile_boundaries[1]:
                q_quartile[q] = 1  # Q2
            elif q_difficulty[q] <= quartile_boundaries[2]:
                q_quartile[q] = 2  # Q3
            else:
                q_quartile[q] = 3  # Q4 (Hard)

        # ── Step 3: Per-student accuracy by quartile ──
        self.report_progress(20, "Computing per-student accuracy by difficulty quartile")

        quartile_accuracy = np.zeros((n_students, 4))
        for qrt in range(4):
            q_mask = (q_quartile == qrt)
            n_in_quartile = q_mask.sum()
            if n_in_quartile > 0:
                quartile_accuracy[:, qrt] = correct_matrix[:, q_mask].sum(axis=1) / n_in_quartile

        # ── Step 4: Gradient analysis ──
        self.report_progress(35, "Computing difficulty gradients")

        # gradient = Q4_accuracy - Q1_accuracy
        gradients = quartile_accuracy[:, 3] - quartile_accuracy[:, 0]

        # Normal students have negative gradient (worse on hard questions)
        # Leaked students have flat/positive gradient
        gradient_flagged = gradients > GRADIENT_THRESHOLD

        # ── Step 5: IRT 2PL Person-Fit ──
        self.report_progress(45, "Computing IRT 2PL person-fit statistics")

        # Estimate question parameters from data
        # discrimination: approximate from point-biserial correlation
        discriminations = np.ones(n_questions)
        difficulties_irt = np.zeros(n_questions)

        for q in range(n_questions):
            # Point-biserial correlation as discrimination proxy
            r_pb = np.corrcoef(correct_matrix[:, q].astype(float), scores.astype(float))[0, 1]
            discriminations[q] = max(0.5, min(3.0, abs(r_pb) * 2.5))
            # Difficulty from correct rate (logit transform)
            p = max(0.01, min(0.99, q_correct_rate[q]))
            difficulties_irt[q] = -np.log(p / (1 - p))

        # Estimate theta for each student via MLE
        self.report_progress(55, "Estimating IRT ability parameters (θ)")

        thetas = np.zeros(n_students)
        misfit_scores = np.zeros(n_students)

        # Vectorized theta estimation (faster than per-student optimization)
        for i in range(n_students):
            response = correct_matrix[i].astype(float)

            # MLE for theta
            def neg_log_likelihood(theta):
                p = expit(discriminations * (theta - difficulties_irt))
                p = np.clip(p, 1e-8, 1 - 1e-8)
                ll = response * np.log(p) + (1 - response) * np.log(1 - p)
                return -ll.sum()

            result = minimize_scalar(neg_log_likelihood, bounds=(-4, 4), method="bounded")
            thetas[i] = result.x

            # Person-fit lz* statistic
            p_hat = expit(discriminations * (thetas[i] - difficulties_irt))
            p_hat = np.clip(p_hat, 1e-8, 1 - 1e-8)

            # Log-likelihood
            ll_obs = (response * np.log(p_hat) + (1 - response) * np.log(1 - p_hat)).sum()
            # Expected log-likelihood
            ll_exp = (p_hat * np.log(p_hat) + (1 - p_hat) * np.log(1 - p_hat)).sum()
            # Variance
            var_ll = (p_hat * (1 - p_hat) * (np.log(p_hat / (1 - p_hat))) ** 2).sum()

            if var_ll > 0:
                misfit_scores[i] = (ll_obs - ll_exp) / np.sqrt(max(var_ll, 1e-8))
            else:
                misfit_scores[i] = 0

            if i % 10000 == 0 and i > 0:
                pct = 55 + int(25 * i / n_students)
                self.report_progress(pct, f"IRT estimation: {i:,}/{n_students:,}")

        # ── Step 6: Combined flagging ──
        self.report_progress(85, "Identifying students with leak signatures")

        # Flag students with BOTH flat gradient AND high IRT misfit
        misfit_flagged = np.abs(misfit_scores) > IRT_MISFIT_THRESHOLD
        combined_flagged = gradient_flagged & misfit_flagged

        # Also flag pure gradient outliers with very flat curves
        very_flat = gradients > 0.0
        combined_flagged = combined_flagged | very_flat

        flagged_indices = np.where(combined_flagged)[0]
        flagged_ids = [student_ids[i] for i in flagged_indices]

        # ── Determine leaked questions ──
        # Questions where flagged students perform much better than expected
        if len(flagged_indices) > 0:
            flagged_accuracy = correct_matrix[flagged_indices].mean(axis=0)
            normal_mask = ~combined_flagged
            normal_accuracy = correct_matrix[normal_mask].mean(axis=0) if normal_mask.any() else q_correct_rate

            # Questions with >20% higher accuracy for flagged students
            accuracy_gap = flagged_accuracy - normal_accuracy
            leaked_qs = np.where(accuracy_gap > 0.20)[0]
            leaked_questions_str = f"Q{leaked_qs[0]+1}-Q{leaked_qs[-1]+1} ({len(leaked_qs)} questions)" if len(leaked_qs) > 0 else "None identified"
        else:
            leaked_questions_str = "None identified"
            normal_accuracy = q_correct_rate

        # ── Difficulty curve data for visualization ──
        normal_curve = []
        flagged_curve = []
        national_avg = []
        quartile_labels = ["Q1 (Easy)", "Q2", "Q3", "Q4 (Hard)"]

        for qrt in range(4):
            national_avg.append(round(float(quartile_accuracy[:, qrt].mean()), 3))
            if combined_flagged.any():
                flagged_curve.append(round(float(quartile_accuracy[combined_flagged, qrt].mean()), 3))
            else:
                flagged_curve.append(0)
            normal_curve.append(round(float(quartile_accuracy[~combined_flagged, qrt].mean()), 3))

        return EngineOutput(
            engine_name=self.engine_name,
            flagged_count=len(flagged_ids),
            flagged_student_ids=flagged_ids,
            result_data={
                "leak_detected": len(flagged_ids) > 0,
                "leaked_group_size": len(flagged_ids),
                "leaked_questions": leaked_questions_str,
                "difficulty_curve_data": {
                    "quartiles": quartile_labels,
                    "normal_accuracy": normal_curve,
                    "flagged_accuracy": flagged_curve,
                    "national_average": national_avg,
                },
                "gradient_stats": {
                    "mean_gradient": round(float(gradients.mean()), 4),
                    "flagged_mean_gradient": round(float(gradients[combined_flagged].mean()), 4) if combined_flagged.any() else 0,
                    "normal_mean_gradient": round(float(gradients[~combined_flagged].mean()), 4),
                    "threshold": GRADIENT_THRESHOLD,
                },
                "irt_stats": {
                    "mean_misfit": round(float(misfit_scores.mean()), 3),
                    "flagged_mean_misfit": round(float(misfit_scores[combined_flagged].mean()), 3) if combined_flagged.any() else 0,
                    "threshold": IRT_MISFIT_THRESHOLD,
                },
            },
            summary={
                "leak_detected": len(flagged_ids) > 0,
                "flagged": len(flagged_ids),
                "leaked_questions": leaked_questions_str,
                "normal_gradient": round(float(gradients[~combined_flagged].mean()), 3) if (~combined_flagged).any() else 0,
                "flagged_gradient": round(float(gradients[combined_flagged].mean()), 3) if combined_flagged.any() else 0,
            },
        )
