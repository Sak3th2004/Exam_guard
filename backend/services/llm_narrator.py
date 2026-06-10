"""Local LLM Forensic Narrator — Ollama integration with template fallback.

Transforms structured detection data into human-readable forensic narratives.
CRITICAL: LLM writes REPORT only. NEVER used for detection.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)


async def generate_narrative(section: str, data: dict[str, Any]) -> str:
    """Generate forensic narrative for a report section using Ollama LLM."""

    system_prompt = (
        "You are a forensic examination analyst writing an official investigation report. "
        "Write in formal, precise language suitable for legal proceedings. State findings as "
        "facts supported by the data provided. Do not speculate. Use specific numbers and "
        "statistics. Each paragraph should be 3-5 sentences."
    )

    prompts = {
        "executive_summary": (
            f"Write an executive summary for this examination integrity analysis:\n"
            f"- Total students analyzed: {data.get('total_students', 0):,}\n"
            f"- Total centers: {data.get('total_centers', 0)}\n"
            f"- Overall integrity score: {data.get('integrity_score', 0)}/100\n"
            f"- Total flagged students: {data.get('total_flagged', 0):,}\n"
            f"- Engines that detected fraud: {data.get('engines_with_findings', 'N/A')}\n"
            f"Write 2 paragraphs summarizing key findings."
        ),
        "copy_ring": (
            f"Write a forensic finding about detected copying clusters:\n"
            f"- Clusters found: {data.get('clusters_found', 0)}\n"
            f"- Total students in clusters: {data.get('flagged_count', 0)}\n"
            f"- Largest cluster: {data.get('largest_cluster_size', 0)} students\n"
            f"- Average wrong answer agreement: {data.get('avg_waa', 0)}\n"
            f"Write 2 paragraphs explaining the evidence."
        ),
        "center_anomaly": (
            f"Write a forensic finding about anomalous examination centers:\n"
            f"- Anomalous centers: {data.get('anomalous_count', 0)}\n"
            f"- Most anomalous center: {data.get('worst_center', 'N/A')}\n"
            f"Write 2 paragraphs suitable for an investigation report."
        ),
        "leak_signature": (
            f"Write a forensic finding about detected paper leak signatures:\n"
            f"- Students with leak signature: {data.get('leaked_group_size', 0)}\n"
            f"- Questions likely leaked: {data.get('leaked_questions', 'N/A')}\n"
            f"- Normal difficulty gradient: {data.get('normal_gradient', 0)}\n"
            f"- Leaked group gradient: {data.get('leaked_gradient', 0)}\n"
            f"Write 2 paragraphs explaining what the difficulty curve inversion means."
        ),
    }

    prompt = prompts.get(section, f"Summarize this forensic data:\n{json.dumps(data, indent=2)}")

    try:
        async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT) as client:
            response = await client.post(settings.OLLAMA_URL, json={
                "model": settings.OLLAMA_MODEL,
                "prompt": f"{system_prompt}\n\n{prompt}",
                "stream": False,
                "options": {"temperature": 0.3, "top_p": 0.9},
            })
            if response.status_code == 200:
                return response.json().get("response", _template_fallback(section, data))
    except Exception as e:
        logger.warning(f"Ollama unavailable ({e}), using template fallback")

    return _template_fallback(section, data)


def _template_fallback(section: str, data: dict[str, Any]) -> str:
    """Generate report text without LLM — ensures report always works."""
    templates = {
        "executive_summary": (
            f"This forensic analysis examined {data.get('total_students', 0):,} students across "
            f"{data.get('total_centers', 0)} examination centers. The overall examination integrity "
            f"score is {data.get('integrity_score', 0)}/100, indicating "
            f"{'significant' if data.get('integrity_score', 100) < 50 else 'moderate' if data.get('integrity_score', 100) < 75 else 'minor'} "
            f"fraud indicators. A total of {data.get('total_flagged', 0):,} students were flagged "
            f"by one or more detection engines.\n\n"
            f"The analysis employed 8 independent detection engines spanning classical statistics, "
            f"deep learning, and ensemble methods. Detection was performed entirely through mathematical "
            f"and machine learning algorithms — no AI was used for fraud determination. The following "
            f"report presents findings organized by detection methodology with supporting evidence."
        ),
        "copy_ring": (
            f"Copy ring analysis identified {data.get('clusters_found', 0)} distinct clusters of "
            f"students exhibiting statistically significant answer similarity. A total of "
            f"{data.get('flagged_count', 0)} students were found within these clusters. The largest "
            f"cluster contained {data.get('largest_cluster_size', 0)} students with an average wrong "
            f"answer agreement rate of {data.get('avg_waa', 0):.1%}.\n\n"
            f"Detection employed MinHash Locality-Sensitive Hashing for scalable pairwise comparison, "
            f"followed by exact Jaccard similarity and Wrong Answer Agreement computation. Community "
            f"detection was performed using the Louvain algorithm on the resulting similarity graph."
        ),
        "center_anomaly": (
            f"Center anomaly detection identified {data.get('anomalous_count', 0)} examination centers "
            f"exhibiting irregular statistical patterns. Eight features were engineered per center, "
            f"including score distributions, answer diversity, and difficulty correlations. "
            f"Anomaly detection was performed using Isolation Forest with Z-score validation."
        ),
        "leak_signature": (
            f"Difficulty curve analysis detected {data.get('leaked_group_size', 0)} students exhibiting "
            f"the characteristic signature of paper leak beneficiaries. Normal students show a negative "
            f"difficulty gradient (gradient = {data.get('normal_gradient', -0.3):.3f}), performing worse "
            f"on harder questions. Flagged students show a flat or positive gradient "
            f"(gradient = {data.get('leaked_gradient', 0):.3f}), indicating pre-knowledge of answers "
            f"regardless of question difficulty."
        ),
    }
    return templates.get(section, f"Analysis data: {json.dumps(data, indent=2)}")
