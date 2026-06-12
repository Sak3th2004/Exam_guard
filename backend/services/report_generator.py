"""PDF Forensic Report Generator.

Generates comprehensive PDF reports with:
  - Cover page with assessment verdict
  - Executive summary with overall stats
  - Per-engine findings with ACTUAL data tables
  - Top 20 flagged students with fraud probabilities
  - Copy ring cluster details
  - Center anomaly evidence
  - XGBoost ensemble feature importance
  - Methodology section
  - Evidence chain
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)

from services.llm_narrator import generate_narrative

logger = logging.getLogger(__name__)

# Professional color scheme
HEADER_BG = colors.HexColor("#1a2332")
HEADER_FG = colors.white
TABLE_HEADER = colors.HexColor("#2c3e50")
TABLE_ALT = colors.HexColor("#f8f9fa")
ACCENT = colors.HexColor("#2980b9")
DANGER = colors.HexColor("#e74c3c")
WARNING = colors.HexColor("#f39c12")
SUCCESS = colors.HexColor("#27ae60")
TEXT_DARK = colors.HexColor("#2c3e50")
TEXT_LIGHT = colors.HexColor("#7f8c8d")
BORDER = colors.HexColor("#bdc3c7")


def _make_table(data: list[list], col_widths: list[int], header_color=TABLE_HEADER) -> Table:
    """Create a professionally styled table."""
    table = Table(data, colWidths=col_widths)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]
    # Alternating row colors
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), TABLE_ALT))
    table.setStyle(TableStyle(style))
    return table


async def generate_report(
    analysis_id: str,
    exam_name: str,
    total_students: int,
    total_centers: int,
    overall_score: float,
    engine_data: dict[str, dict[str, Any]],
    output_path: Path,
) -> Path:
    """Generate a comprehensive PDF forensic report with actual data."""
    logger.info(f"Generating PDF report for analysis {analysis_id}")

    styles = getSampleStyleSheet()
    now = datetime.now()

    # Custom styles
    title_style = ParagraphStyle(
        "Title2", parent=styles["Title"],
        fontSize=28, spaceAfter=6, textColor=HEADER_BG,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=14, spaceAfter=20, textColor=ACCENT,
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=16, spaceBefore=18, spaceAfter=8,
        textColor=HEADER_BG, borderPadding=(0, 0, 2, 0),
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=13, spaceBefore=12, spaceAfter=6, textColor=ACCENT,
    )
    body = ParagraphStyle(
        "Body2", parent=styles["Normal"],
        fontSize=10, spaceAfter=8, leading=14,
    )
    small = ParagraphStyle(
        "Small2", parent=styles["Normal"],
        fontSize=8, textColor=TEXT_LIGHT,
    )
    bold_body = ParagraphStyle(
        "BoldBody", parent=body, fontName="Helvetica-Bold",
    )

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        topMargin=20*mm, bottomMargin=20*mm,
        leftMargin=18*mm, rightMargin=18*mm,
    )
    W = A4[0] - 36*mm  # usable width

    story = []

    # Helper to extract data from EngineOutput or dict
    def _get(obj, key, default=None):
        if obj is None:
            return default
        if hasattr(obj, key):
            return getattr(obj, key, default)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    # ═══════════════════════════════════════════════════
    # COVER PAGE
    # ═══════════════════════════════════════════════════
    story.append(Spacer(1, 60))
    story.append(Paragraph("EXAMGUARD", title_style))
    story.append(Paragraph("AI Forensic Intelligence Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT))
    story.append(Spacer(1, 20))

    # Key info table
    score_text = "COMPROMISED" if overall_score < 50 else "SUSPICIOUS" if overall_score < 75 else "ACCEPTABLE"
    score_col = DANGER if overall_score < 50 else WARNING if overall_score < 75 else SUCCESS

    info_data = [
        ["Field", "Value"],
        ["Examination", exam_name],
        ["Analysis ID", analysis_id[:12] + "..."],
        ["Date", now.strftime("%d %B %Y, %I:%M %p")],
        ["Total Students", f"{total_students:,}"],
        ["Total Centers", f"{total_centers}"],
        ["Integrity Score", f"{overall_score:.1f} / 100"],
        ["Assessment", score_text],
    ]
    info_table = _make_table(info_data, [140, int(W) - 140])
    story.append(info_table)
    story.append(Spacer(1, 30))

    # Compute total flagged per engine
    engine_names_display = {
        "copy_ring": ("E1: Copy Ring Detection", "MinHash LSH + Louvain Community"),
        "stat_impossibility": ("E2: Statistical Impossibility", "Binomial + Bonferroni Correction"),
        "center_anomaly": ("E3: Center Anomaly Detector", "Isolation Forest + Z-Score"),
        "leak_signature": ("E4: Paper Leak Signature", "IRT 2PL + Difficulty Curve"),
        "response_time": ("E5: Response Time Analysis", "KDE + K-Means Clustering"),
        "gnn_copy_ring": ("E6: GNN Fraud Detection", "GraphSAGE (PyTorch Geometric)"),
        "vae_anomaly": ("E7: VAE Anomaly Detection", "Variational Autoencoder (PyTorch)"),
        "question_similarity": ("E8: Question Similarity", "Sentence Transformer NLP"),
        "xgboost_ensemble": ("META: XGBoost Ensemble", "Gradient Boosted Classifier"),
        "benford_law": ("E9: Benford's Law", "Chi-Squared First-Digit Forensics"),
    }

    all_flagged: set[str] = set()
    engine_flagged_counts: dict[str, int] = {}
    for ename, edata in engine_data.items():
        fids = _get(edata, "flagged_student_ids", []) or []
        engine_flagged_counts[ename] = len(fids)
        all_flagged.update(fids)

    total_unique_flagged = len(all_flagged)

    story.append(Paragraph(
        f"<b>Total unique entities flagged:</b> {total_unique_flagged:,} "
        f"({total_unique_flagged/max(total_students,1)*100:.1f}% of {total_students:,} students)",
        body,
    ))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<i>This report was generated by ExamGuard v2.0. "
        "All fraud detection is performed using mathematical and machine learning algorithms. "
        "The LLM component (Mistral 7B) is used exclusively for report narration.</i>",
        small,
    ))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════
    # EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════
    story.append(Paragraph("1. Executive Summary", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Spacer(1, 6))

    exec_text = await generate_narrative("executive_summary", {
        "total_students": total_students,
        "total_centers": total_centers,
        "integrity_score": overall_score,
        "total_flagged": total_unique_flagged,
        "flagged_pct": round(total_unique_flagged / max(total_students, 1) * 100, 1),
        "engines_with_findings": ", ".join(
            name for name, count in engine_flagged_counts.items() if count > 0
        ),
    })
    story.append(Paragraph(exec_text, body))
    story.append(Spacer(1, 10))

    # Engine summary table
    story.append(Paragraph("1.1 Detection Engine Summary", h2))
    summary_rows = [["Engine", "Algorithm", "Flagged", "Status"]]
    for ename, (display, algo) in engine_names_display.items():
        edata = engine_data.get(ename)
        status = _get(edata, "status", "skipped") or "skipped"
        count = engine_flagged_counts.get(ename, 0)
        summary_rows.append([display, algo, str(count), status.upper()])

    summary_tbl = _make_table(summary_rows, [140, 160, 50, 60])
    story.append(summary_tbl)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════
    # PER-ENGINE FINDINGS
    # ═══════════════════════════════════════════════════
    story.append(Paragraph("2. Detection Engine Findings", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))

    # E1: Copy Ring
    e1 = engine_data.get("copy_ring")
    if e1 and engine_flagged_counts.get("copy_ring", 0) > 0:
        story.append(Paragraph("2.1 Copy Ring Detection (E1)", h2))
        e1_rd = _get(e1, "result_data", {}) or {}
        clusters = e1_rd.get("clusters", []) if isinstance(e1_rd, dict) else []

        narrative = await generate_narrative("copy_ring", {
            "clusters_found": len(clusters),
            "flagged_count": engine_flagged_counts.get("copy_ring", 0),
            "largest_cluster_size": max((c.get("size", 0) for c in clusters), default=0),
            "avg_waa": round(sum(c.get("avg_waa", 0) for c in clusters) / max(len(clusters), 1), 3),
        })
        story.append(Paragraph(narrative, body))

        if clusters:
            cluster_rows = [["Cluster", "Size", "Avg Similarity", "Avg WAA", "Center", "Confidence"]]
            for i, c in enumerate(clusters[:10]):
                cluster_rows.append([
                    f"Ring {i+1}",
                    str(c.get("size", 0)),
                    f"{c.get('avg_similarity', 0):.3f}",
                    f"{c.get('avg_waa', 0):.3f}",
                    str(c.get("center_ids", ["?"])[0] if c.get("center_ids") else "?"),
                    f"{c.get('confidence', 0):.1%}",
                ])
            story.append(Spacer(1, 6))
            story.append(_make_table(cluster_rows, [50, 40, 85, 70, 80, 75]))
        story.append(Spacer(1, 10))

    # E2: Statistical Impossibility
    e2 = engine_data.get("stat_impossibility")
    if e2 and engine_flagged_counts.get("stat_impossibility", 0) > 0:
        story.append(Paragraph("2.2 Statistical Impossibility (E2)", h2))
        e2_rd = _get(e2, "result_data", {}) or {}
        pairs = e2_rd.get("pairs", []) if isinstance(e2_rd, dict) else []

        story.append(Paragraph(
            f"Identified <b>{len(pairs)}</b> statistically impossible pairs using "
            f"Binomial probability with Bonferroni correction. These pairs share "
            f"answers at rates that cannot occur by chance.",
            body,
        ))
        if pairs:
            pair_rows = [["Student A", "Student B", "Matching", "Expected", "p-value", "Verdict"]]
            for p in pairs[:15]:
                pair_rows.append([
                    str(p.get("student_a", "")),
                    str(p.get("student_b", "")),
                    str(p.get("matching_total", 0)),
                    str(p.get("expected_matches", 0)),
                    f"{p.get('p_value', 1):.2e}",
                    p.get("human_readable", "Impossible")[:30],
                ])
            story.append(_make_table(pair_rows, [70, 70, 55, 55, 65, 95]))
        story.append(Spacer(1, 10))

    # E3: Center Anomaly
    e3 = engine_data.get("center_anomaly")
    if e3 and engine_flagged_counts.get("center_anomaly", 0) > 0:
        story.append(Paragraph("2.3 Center Anomaly Detection (E3)", h2))
        e3_rd = _get(e3, "result_data", {}) or {}
        centers = e3_rd.get("centers", []) if isinstance(e3_rd, dict) else []

        narrative = await generate_narrative("center_anomaly", {
            "anomalous_count": len(centers),
            "worst_center": centers[0].get("center_id", "?") if centers else "N/A",
        })
        story.append(Paragraph(narrative, body))

        if centers:
            center_rows = [["Center ID", "City", "Students", "Anomaly Score", "Flags"]]
            for c in centers[:10]:
                flags_list = c.get("flags", [])
                center_rows.append([
                    str(c.get("center_id", "")),
                    str(c.get("city", "?")),
                    str(c.get("student_count", 0)),
                    f"{c.get('anomaly_score', 0):.3f}",
                    ", ".join(flags_list[:2]) if flags_list else "N/A",
                ])
            story.append(_make_table(center_rows, [65, 70, 55, 80, 140]))
        story.append(Spacer(1, 10))

    # E4: Leak Signature
    e4 = engine_data.get("leak_signature")
    if e4 and engine_flagged_counts.get("leak_signature", 0) > 0:
        story.append(Paragraph("2.4 Paper Leak Signature (E4)", h2))
        e4_rd = _get(e4, "result_data", {}) or {}
        story.append(Paragraph(
            f"Paper leak detection using IRT 2PL person-fit analysis identified "
            f"<b>{engine_flagged_counts.get('leak_signature', 0)}</b> students with "
            f"suspicious difficulty-gradient inversion patterns consistent with "
            f"prior knowledge of exam questions.",
            body,
        ))

    # E5: Response Time
    e5 = engine_data.get("response_time")
    if e5 and engine_flagged_counts.get("response_time", 0) > 0:
        story.append(Paragraph("2.5 Response Time Analysis (E5)", h2))
        story.append(Paragraph(
            f"KDE-based response time analysis flagged "
            f"<b>{engine_flagged_counts.get('response_time', 0)}</b> students with "
            f"impossibly fast response patterns (speed ratio &lt; 0.2 on hard questions).",
            body,
        ))

    story.append(PageBreak())

    # GPU engines
    story.append(Paragraph("2.6 Deep Learning Engine Results (GPU)", h2))

    gpu_rows = [["Engine", "Method", "Device", "Flagged"]]
    for ename in ["gnn_copy_ring", "vae_anomaly", "question_similarity"]:
        edata = engine_data.get(ename)
        display = engine_names_display.get(ename, (ename, ""))[0]
        algo = engine_names_display.get(ename, ("", ename))[1]
        summary = _get(edata, "summary", {}) or {}
        device = summary.get("device", "cpu") if isinstance(summary, dict) else "cpu"
        gpu_rows.append([display, algo, device.upper(), str(engine_flagged_counts.get(ename, 0))])

    story.append(_make_table(gpu_rows, [130, 150, 50, 55]))
    story.append(Spacer(1, 10))

    # XGBoost Ensemble
    xgb = engine_data.get("xgboost_ensemble")
    if xgb:
        story.append(Paragraph("2.7 XGBoost Meta-Ensemble", h2))
        xgb_rd = _get(xgb, "result_data", {}) or {}
        xgb_summary = _get(xgb, "summary", {}) or {}

        risk_dist = xgb_rd.get("risk_distribution", {}) if isinstance(xgb_rd, dict) else {}
        auc = xgb_rd.get("validation_auc_pr", 0) if isinstance(xgb_rd, dict) else 0
        model_info = xgb_rd.get("model", "XGBoost") if isinstance(xgb_rd, dict) else "XGBoost"

        story.append(Paragraph(
            f"The XGBoost meta-classifier (<b>{model_info}</b>) combined outputs "
            f"from all 8 engines into a calibrated fraud probability per student. "
            f"Validation AUC-PR: <b>{auc:.3f}</b>.",
            body,
        ))

        if risk_dist:
            risk_rows = [["Risk Tier", "Student Count", "Description"]]
            risk_rows.append(["CRITICAL", str(risk_dist.get("CRITICAL", 0)), "Fraud probability > 80%"])
            risk_rows.append(["HIGH", str(risk_dist.get("HIGH", 0)), "Fraud probability 60-80%"])
            risk_rows.append(["MEDIUM", str(risk_dist.get("MEDIUM", 0)), "Fraud probability 30-60%"])
            risk_rows.append(["LOW", str(risk_dist.get("LOW", 0)), "Fraud probability < 30%"])
            story.append(_make_table(risk_rows, [80, 80, 200]))
            story.append(Spacer(1, 8))

        # Feature importance
        fi = xgb_rd.get("feature_importance", []) if isinstance(xgb_rd, dict) else []
        if fi:
            story.append(Paragraph("<b>Feature Importance (Top 8):</b>", body))
            fi_rows = [["Feature", "Importance"]]
            for item in fi[:8]:
                fi_rows.append([str(item.get("feature", "")), f"{item.get('importance', 0):.4f}"])
            story.append(_make_table(fi_rows, [180, 100]))
            story.append(Spacer(1, 8))

        # Top flagged students
        rankings = xgb_rd.get("final_rankings", []) if isinstance(xgb_rd, dict) else []
        if rankings:
            story.append(Paragraph("<b>Top 20 Highest-Risk Students:</b>", body))
            rank_rows = [["#", "Student ID", "Fraud Prob", "Risk Tier", "Center", "Engines Flagged"]]
            for i, r in enumerate(rankings[:20]):
                engines_str = ", ".join(r.get("engines_flagged", [])[:3])
                rank_rows.append([
                    str(i + 1),
                    str(r.get("student_id", "")),
                    f"{r.get('fraud_probability', 0):.3f}",
                    str(r.get("risk_tier", "")),
                    str(r.get("center_id", "?") or "?"),
                    engines_str[:40],
                ])
            story.append(_make_table(rank_rows, [25, 75, 60, 55, 55, 140]))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════
    # METHODOLOGY
    # ═══════════════════════════════════════════════════
    story.append(Paragraph("3. Methodology", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "ExamGuard employs a <b>4-layer hybrid architecture</b> combining classical "
        "statistical methods, deep learning, machine learning ensemble, and natural "
        "language generation.",
        body,
    ))

    layer_data = [
        ["Layer", "Purpose", "Technology"],
        ["Layer 1: Classical\n(CPU)", "Mathematical rigor, explainable,\nlegally defensible proofs",
         "MinHash LSH, Binomial Tests,\nIsolation Forest, IRT 2PL, KDE"],
        ["Layer 2: Deep Learning\n(GPU — RTX 4060)", "Detects unknown fraud patterns\nbeyond rule-based methods",
         "GraphSAGE (PyG), VAE (PyTorch),\nSentence Transformer (HF)"],
        ["Layer 3: Ensemble\n(GPU)", "Optimal weighting, reduces\nfalse positives",
         "XGBoost (gpu_hist)"],
        ["Layer 4: Narrator\n(GPU, Ollama)", "Human-readable forensic\nreport generation",
         "Mistral 7B (local LLM)"],
    ]
    story.append(_make_table(layer_data, [105, 140, 165]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Key Principle:</b> All fraud detection is performed through mathematical "
        "and ML algorithms (Layers 1-3). The LLM (Layer 4) is used exclusively for "
        "report narration — it never makes detection decisions.",
        body,
    ))

    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Paragraph(
        f"<i>Report generated by ExamGuard v2.0 on {now.strftime('%d %B %Y at %H:%M')}. "
        f"Analysis ID: {analysis_id}.</i>",
        small,
    ))

    # Build PDF
    doc.build(story)
    logger.info(f"PDF report saved to {output_path}")
    return output_path
""",
<parameter name="toolAction">Rewriting PDF report
