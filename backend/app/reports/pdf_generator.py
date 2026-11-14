"""Professional PDF Report Generator for FairLens Model Audits."""
import os
from typing import Dict, Any
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
from app.schemas.audit import AuditSummary

def generate_pdf_report(summary: AuditSummary, output_path: str):
    """Generates an executive-grade AI Fairness & Bias Audit report."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#0f172a")    # Slate 900
    c_secondary = colors.HexColor("#3b82f6")  # Blue 500
    c_accent = colors.HexColor("#6366f1")     # Indigo 500
    c_text = colors.HexColor("#334155")       # Slate 700
    c_border = colors.HexColor("#cbd5e1")     # Slate 300
    c_bg_light = colors.HexColor("#f8fafc")   # Slate 50
    c_danger = colors.HexColor("#ef4444")     # Red 500
    c_warning = colors.HexColor("#f59e0b")    # Amber 500

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=c_primary
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=c_text
    )

    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=c_text
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=c_primary
    )

    elements = []

    # Title & Metadata Header
    elements.append(Paragraph("FairLens™ Algorithmic Fairness & Bias Audit Report", title_style))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(f"Audit Reference: <b>{summary.audit_id}</b> | Generated: {summary.created_at}", subtitle_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceBefore=4, spaceAfter=14))

    # Executive Overview
    elements.append(Paragraph("1. Executive Summary", h2_style))
    ov = summary.overall_performance
    overview_text = (
        f"This evaluation audited <b>{escape(summary.dataset_name)}</b> "
        f"({'Predictions-Only Mode' if summary.is_predictions_only else f'Model: {escape(str(summary.model_name or summary.model_id))}'}). "
        f"The evaluation set comprises <b>{ov.total_samples:,} samples</b> across protected attributes: "
        f"<b>{escape(', '.join(summary.sensitive_attributes))}</b>. "
        f"The rules engine recorded <b>{len(summary.findings)} investigation findings</b> "
        "(including model-quality warnings where applicable)."
    )
    elements.append(Paragraph(overview_text, body_style))
    elements.append(Spacer(1, 10))

    # Overall Performance Table
    perf_data = [
        [Paragraph("Metric", table_header_style), Paragraph("Score", table_header_style), 
         Paragraph("Metric", table_header_style), Paragraph("Score", table_header_style)],
        [Paragraph("Accuracy", table_cell_style), Paragraph(f"{ov.accuracy:.1%}", table_cell_style),
         Paragraph("Balanced Accuracy", table_cell_style), Paragraph(f"{ov.balanced_accuracy:.1%}", table_cell_style)],
        [Paragraph("Precision", table_cell_style), Paragraph(f"{ov.precision:.1%}" if ov.precision is not None else "N/A", table_cell_style),
         Paragraph("Recall (TPR)", table_cell_style), Paragraph(f"{ov.recall:.1%}" if ov.recall is not None else "N/A", table_cell_style)],
        [Paragraph("F1-Score", table_cell_style), Paragraph(f"{ov.f1_score:.3f}" if ov.f1_score is not None else "N/A", table_cell_style),
         Paragraph("Selection Rate", table_cell_style), Paragraph(f"{ov.selection_rate:.1%}", table_cell_style)],
    ]
    if ov.roc_auc is not None:
        perf_data.append([
            Paragraph("ROC-AUC", table_cell_style), Paragraph(f"{ov.roc_auc:.3f}", table_cell_style),
            Paragraph("Pos / Neg Count", table_cell_style), Paragraph(f"{ov.positive_samples} / {ov.negative_samples}", table_cell_style)
        ])

    perf_table = Table(perf_data, colWidths=[130, 130, 130, 130])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [c_bg_light, colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
    ]))
    elements.append(perf_table)
    elements.append(Spacer(1, 14))

    # Class-distribution and confusion-matrix context: fairness gaps alone do not measure utility.
    elements.append(Paragraph("Evaluation Outcome & Confusion Matrix", h2_style))
    cm = ov.confusion_matrix
    cm_rows = [
        ["", "Predicted positive", "Predicted negative"],
        ["Actual positive", str(cm.true_positive), str(cm.false_negative)],
        ["Actual negative", str(cm.false_positive), str(cm.true_negative)],
    ]
    cm_table = Table(cm_rows, colWidths=[160, 180, 180])
    cm_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [c_bg_light, colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(cm_table)
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(
        "Precision is undefined when a model makes no positive predictions; "
        "recall is undefined when there are no actual positives. "
        "The F1-score is reported as N/A when its required precision or recall is undefined. "
        "A low disparity score is not proof of adequate predictive performance.", body_style))
    elements.append(Spacer(1, 8))

    # Fairness Disparities Section
    elements.append(Paragraph("2. Protected Attribute Subgroup Disparities", h2_style))
    for attr, res in summary.attribute_fairness.items():
        elements.append(Paragraph(f"<b>Attribute: {escape(attr)}</b> (Reference Group: <i>{escape(res.reference_group)}</i>)", subtitle_style))
        if res.overall_demographic_parity_ratio is None:
            elements.append(Paragraph("DP ratio: N/A (all subgroup selection rates are zero; 0/0 is undefined).", body_style))
        elements.append(Spacer(1, 4))
        
        # Subgroup table
        sub_headers = ["Subgroup", "Sample n", "Selection Rate", "Accuracy", "Recall (TPR)", "DP Diff (vs Ref)", "Eq. Opp Diff"]
        sub_rows = [[Paragraph(h, table_header_style) for h in sub_headers]]

        for grp, m in res.subgroup_metrics.items():
            disp = res.subgroup_disparities.get(grp)
            dp_diff_str = f"{disp.demographic_parity_diff:+.1%}" if disp and disp.demographic_parity_diff is not None else "0.0%"
            eq_opp_str = f"{disp.equal_opportunity_diff:+.1%}" if disp and disp.equal_opportunity_diff is not None else "N/A"
            tpr_str = f"{m.recall_tpr:.1%}" if m.recall_tpr is not None else "N/A"

            sub_rows.append([
                Paragraph(f"<b>{escape(grp)}</b>{' (Ref)' if grp == res.reference_group else ''}", table_cell_style),
                Paragraph(f"{m.sample_size}", table_cell_style),
                Paragraph(f"{m.selection_rate:.1%}", table_cell_style),
                Paragraph(f"{m.accuracy:.1%}", table_cell_style),
                Paragraph(tpr_str, table_cell_style),
                Paragraph(dp_diff_str, table_cell_style),
                Paragraph(eq_opp_str, table_cell_style)
            ])

        sub_table = Table(sub_rows, colWidths=[100, 55, 75, 65, 75, 80, 80])
        sub_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), c_accent),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [c_bg_light, colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ]))
        elements.append(sub_table)
        elements.append(Spacer(1, 8))

    # Start the findings on a fresh page to keep each recommendation readable.
    # With short audits this is still a compact two-page professional report.
    # Findings Section
    elements.append(PageBreak())
    elements.append(Paragraph("3. Ranked Audit Findings & Recommendations", h2_style))
    if not summary.findings:
        elements.append(Paragraph("No configured investigation rules fired. This does not establish fairness, model utility, or legal compliance.", body_style))
    else:
        for idx, f in enumerate(summary.findings[:6], 1):
            badge_color = "#ef4444" if f.severity == "high" else "#f59e0b" if f.severity == "medium" else "#64748b"
            item_text = (
                f"<b>{idx}. [{f.severity.upper()}] {escape(f.title)}</b><br/>"
                f"<b>Observation:</b> {escape(f.explanation)}<br/>"
                f"<b>Evidence:</b> {escape(f.evidence)} ({escape(f.sample_size_info)})<br/>"
                f"<b>Recommendation:</b> {escape(f.suggested_action)}"
            )
            elements.append(Paragraph(item_text, body_style))
            elements.append(Spacer(1, 6))

    # Regulatory & Methodological Limitations
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("4. Methodological Scope & Governance Notice", h2_style))
    disclaimer = (
        "<b>Important Disclaimer:</b> Algorithmic disparity metrics identify statistical variations across measured "
        "subgroups within this specific evaluation dataset. Statistical disparity does not constitute definitive proof of unlawful "
        "discrimination under disparate impact or treatment doctrines. Furthermore, mathematical theorems (Kleinberg et al., 2016) "
        "demonstrate that Demographic Parity, Equal Opportunity, and Predictive Parity cannot generally be satisfied simultaneously "
        "when base rates differ across groups. Stakeholders must evaluate which fairness definitions correspond to ethical, legal, "
        "and operational obligations in their domain."
    )
    elements.append(Paragraph(disclaimer, body_style))

    doc.build(elements)
