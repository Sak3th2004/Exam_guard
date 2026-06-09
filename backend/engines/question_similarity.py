"""Engine 8: Sentence Transformer — Question Similarity / Reuse Detector.

Model: all-MiniLM-L6-v2 (HuggingFace, pretrained)
Detects recycled/paraphrased questions from previous year papers.

Pipeline:
  1. Embed all current exam questions using sentence-transformers
  2. Intra-exam: cosine similarity matrix, flag > 0.85
  3. Cross-exam: compare against reference bank (if provided)

NOTE: Only runs when question text is available. Gracefully skips otherwise.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from engines.base_engine import BaseEngine, EngineOutput

logger = logging.getLogger(__name__)

INTRA_SIMILARITY_THRESHOLD = 0.85
CROSS_SIMILARITY_THRESHOLD = 0.80


class QuestionSimilarityEngine(BaseEngine):
    """Detect question reuse and duplication using sentence embeddings."""

    def __init__(self):
        super().__init__(engine_name="question_similarity", requires_gpu=True)

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
        if question_texts is None or len(question_texts) == 0:
            return EngineOutput(
                engine_name=self.engine_name,
                status="skipped",
                result_data={"applicable": False, "message": "Question text not available"},
                summary={"applicable": False, "message": "No question text provided"},
            )

        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError as e:
            return EngineOutput(
                engine_name=self.engine_name,
                status="failed",
                error_message=f"sentence-transformers not installed: {e}",
            )

        self.report_progress(10, "Loading sentence transformer model (all-MiniLM-L6-v2)")

        try:
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            logger.info(f"Sentence transformer loaded on {device}")
        except Exception as e:
            return EngineOutput(
                engine_name=self.engine_name,
                status="failed",
                error_message=f"Failed to load model: {e}",
            )

        # ── Embed current questions ──
        self.report_progress(30, f"Embedding {len(question_texts)} questions")

        current_embeddings = model.encode(
            question_texts, batch_size=64, show_progress_bar=False
        )

        # ── Intra-exam similarity ──
        self.report_progress(50, "Computing intra-exam similarity matrix")

        sim_matrix = cosine_similarity(current_embeddings)
        intra_duplicates = []

        for i in range(len(question_texts)):
            for j in range(i + 1, len(question_texts)):
                if sim_matrix[i][j] > INTRA_SIMILARITY_THRESHOLD:
                    intra_duplicates.append({
                        "q_a": i + 1,
                        "q_b": j + 1,
                        "similarity": round(float(sim_matrix[i][j]), 3),
                        "text_a": question_texts[i][:150],
                        "text_b": question_texts[j][:150],
                        "type": "intra_exam_duplicate",
                    })

        # ── Cross-exam similarity (if reference bank provided) ──
        self.report_progress(70, "Checking cross-exam similarity")

        cross_reuses = []
        reference_bank = kwargs.get("reference_bank", None)

        if reference_bank and len(reference_bank) > 0:
            ref_embeddings = model.encode(reference_bank, batch_size=64, show_progress_bar=False)
            cross_sim = cosine_similarity(current_embeddings, ref_embeddings)

            for i in range(len(question_texts)):
                max_sim_idx = int(np.argmax(cross_sim[i]))
                max_sim = float(cross_sim[i][max_sim_idx])
                if max_sim > CROSS_SIMILARITY_THRESHOLD:
                    cross_reuses.append({
                        "current_q": i + 1,
                        "reference_q": max_sim_idx + 1,
                        "similarity": round(max_sim, 3),
                        "current_text": question_texts[i][:150],
                        "reference_text": reference_bank[max_sim_idx][:150],
                        "type": "cross_exam_reuse",
                    })

        # Combine all flagged questions
        flagged_questions = intra_duplicates + cross_reuses

        return EngineOutput(
            engine_name=self.engine_name,
            flagged_count=len(flagged_questions),
            flagged_student_ids=[],  # This engine flags questions, not students
            result_data={
                "applicable": True,
                "model": "all-MiniLM-L6-v2",
                "intra_duplicates": len(intra_duplicates),
                "cross_exam_reuses": len(cross_reuses),
                "flagged_questions": flagged_questions,
                "total_questions_analyzed": len(question_texts),
                "device": device,
            },
            summary={
                "applicable": True,
                "intra_duplicates": len(intra_duplicates),
                "cross_reuses": len(cross_reuses),
                "total_flagged": len(flagged_questions),
                "device": device,
            },
        )
