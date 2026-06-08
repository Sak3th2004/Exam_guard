"""Abstract base class for all detection engines.

Provides:
  - Standardized interface (analyze method)
  - Progress callback support
  - Timing instrumentation
  - Error handling and graceful degradation
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Type alias for progress callback: (percentage: int, message: str) -> None
ProgressCallback = Callable[[int, str], None]


@dataclass
class EngineOutput:
    """Standardized output from any detection engine."""
    engine_name: str
    status: str = "complete"  # complete | failed | skipped
    flagged_count: int = 0
    flagged_student_ids: list[str] = field(default_factory=list)
    result_data: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    error_message: Optional[str] = None


class BaseEngine(ABC):
    """Abstract base class for detection engines.

    All 8 detection engines inherit from this class.
    Subclasses must implement `_run_analysis`.
    """

    def __init__(self, engine_name: str, requires_gpu: bool = False):
        self.engine_name = engine_name
        self.requires_gpu = requires_gpu
        self._progress_callback: Optional[ProgressCallback] = None

    def set_progress_callback(self, callback: ProgressCallback) -> None:
        """Set the progress callback for real-time updates."""
        self._progress_callback = callback

    def report_progress(self, percentage: int, message: str) -> None:
        """Report progress to the orchestrator via callback."""
        if self._progress_callback:
            self._progress_callback(percentage, message)
        logger.debug(f"[{self.engine_name}] {percentage}% — {message}")

    async def analyze(
        self,
        answers: np.ndarray,
        answer_key: Optional[np.ndarray] = None,
        student_ids: Optional[list[str]] = None,
        center_ids: Optional[list[str]] = None,
        timing_data: Optional[np.ndarray] = None,
        question_texts: Optional[list[str]] = None,
        ground_truth: Optional[dict[str, Any]] = None,
        progress_callback: Optional[ProgressCallback] = None,
        **kwargs: Any,
    ) -> EngineOutput:
        """Run the detection engine with timing and error handling.

        Args:
            answers: (n_students, n_questions) answer matrix
            answer_key: (n_questions,) correct answers
            student_ids: List of student identifiers
            center_ids: List of center assignments per student
            timing_data: Optional (n_students, n_questions) response times
            question_texts: Optional list of question text strings
            ground_truth: Optional ground truth for supervised engines
            progress_callback: Optional callback for progress updates
            **kwargs: Engine-specific parameters

        Returns:
            EngineOutput with results, flagged entities, and metadata
        """
        if progress_callback:
            self._progress_callback = progress_callback

        self.report_progress(0, f"Starting {self.engine_name}")
        start_time = time.time()

        try:
            output = await self._run_analysis(
                answers=answers,
                answer_key=answer_key,
                student_ids=student_ids,
                center_ids=center_ids,
                timing_data=timing_data,
                question_texts=question_texts,
                ground_truth=ground_truth,
                **kwargs,
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            output.duration_ms = elapsed_ms
            output.status = "complete"

            self.report_progress(100, f"{self.engine_name} complete — {output.flagged_count} flagged")
            logger.info(
                f"[{self.engine_name}] Complete in {elapsed_ms}ms — "
                f"{output.flagged_count} flagged"
            )
            return output

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[{self.engine_name}] Failed after {elapsed_ms}ms: {e}", exc_info=True)
            self.report_progress(100, f"{self.engine_name} failed: {str(e)[:100]}")

            return EngineOutput(
                engine_name=self.engine_name,
                status="failed",
                duration_ms=elapsed_ms,
                error_message=str(e),
            )

    @abstractmethod
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
        """Implement the actual detection logic.

        Subclasses must override this method.
        """
        ...
