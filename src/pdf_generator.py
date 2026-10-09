"""PDF Security Incident Report Generator using ReportLab.

Generates executive-ready, publication-grade cybersecurity incident audit reports
with threat risk scores, MITRE ATT&CK taxonomy, multi-model ensemble matrix,
and 42-feature network packet telemetry.
"""
import io
from datetime import datetime

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_security_incident_report_pdf(row, prob_df, mitre_info, dominant_cat, raw_feat_dict):
    """Generate a PDF report in bytes for Streamlit download button.
    
    Args:
        row (dict/pd.Series): Evaluated hybrid prediction row.
        prob_df (pd.DataFrame): 10-class softmax probabilities.
        mitre_info (dict): MITRE ATT&CK metadata.
        dominant_cat (str): Primary predicted attack category.
        raw_feat_dict (dict): Raw 42 network features dictionary.
        
    Returns:
        bytes: Compiled PDF file buffer.
    """
    if not REPORTLAB_AVAILABLE:
        fallback_txt = f"""===================================================================
XAI ZERO-DAY INTRUSION DETECTION SYSTEM - INCIDENT REPORT
===================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Threat Verdict: {row.get('Hybrid Verdict', 'Normal')}
Calibrated Threat Risk: {row.get('Risk Score (%)', '0.0')}%
Dominant Attack Category: {dominant_cat}
MITRE Technique ID: {mitre_info.get('technique_id', 'N/A')} ({mitre_info.get('tactic', 'N/A')})

MITRE ATT&CK Metadata:
- Technique Name: {mitre_info.get('technique_name', 'N/A')}
- Sub-technique: {mitre_info.get('sub_technique', 'N/A')}
- Severity Rating: {mitre_info.get('severity', 'N/A')}
- Description: {mitre_info.get('description', 'N/A')}

Model Ensemble Breakdown:
- Isolation Forest: {row.get('Isolation Forest', 'N/A')}
- Random Forest: {row.get('Random Forest', 'N/A')}
- XGBoost Classifier: {row.get('XGBoost', 'N/A')}

Containment Command:
{mitre_info.get('containment_command', '# Block command').format(src_ip=raw_feat_dict.get('src_ip', '192.168.1.100'), dst_port=raw_feat_dict.get('dst_port', 80), proto=raw_feat_dict.get('proto', 'tcp'), user_id='root')}
===================================================================
(To generate publication-grade PDF files, run: pip install reportlab)
"""
        return fallback_txt.encode('utf-8')

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    story = []
    styles = getSampleStyleSheet()

    # Custom Color Palette
    DARK_BG = colors.HexColor('#111827')
    TEXT_DARK = colors.HexColor('#1f2937')
    ACCENT_RED = colors.HexColor('#dc2626')
    ACCENT_GREEN = colors.HexColor('#059669')
    ACCENT_BLUE = colors.HexColor('#2563eb')
    TABLE_HEADER_BG = colors.HexColor('#1e293b')
    ALT_ROW_BG = colors.HexColor('#f8fafc')

    # Custom Paragraph Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.white,
        alignment=0
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#94a3b8'),
        alignment=0
    )
    h2_style = ParagraphStyle(
        'H2Style',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=DARK_BG,
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=TEXT_DARK
    )
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#0f172a')
    )

    # Verdict Badge Color Logic
    verdict_str = str(row.get('Hybrid Verdict', 'Normal'))
    if 'Attack' in verdict_str:
        verdict_color = ACCENT_RED
    elif 'Zero-Day' in verdict_str:
        verdict_color = ACCENT_BLUE
    else:
        verdict_color = ACCENT_GREEN

    # 1. Header Title Banner
    header_data = [
        [
            Paragraph("<b>XAI ZERO-DAY INTRUSION DETECTION SYSTEM</b>", title_style),
            Paragraph(f"<b>CONFIDENTIAL REPORT</b><br/>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[4.2 * inch, 3.2 * inch])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), DARK_BG),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    # 2. Executive Summary Box
    risk_score = str(row.get('Risk Score (%)', '0.0'))
    summary_data = [
        [
            Paragraph(f"<b>Final Threat Verdict:</b> <font color='{verdict_color.hexval()}'><b>{verdict_str}</b></font>", body_style),
            Paragraph(f"<b>Calibrated Threat Risk:</b> <b>{risk_score}%</b>", body_style),
        ],
        [
            Paragraph(f"<b>Dominant Attack Category:</b> {dominant_cat}", body_style),
            Paragraph(f"<b>MITRE Technique ID:</b> {mitre_info.get('technique_id', 'N/A')} ({mitre_info.get('tactic', 'N/A')})", body_style),
        ]
    ]
    summary_table = Table(summary_data, colWidths=[3.7 * inch, 3.7 * inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f5f9')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # 3. MITRE ATT&CK Threat Intelligence Profile
    story.append(Paragraph("1. MITRE ATT&CK Framework Threat Intelligence", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=2, spaceAfter=6))

    mitre_table_data = [
        [Paragraph("<b>Attribute</b>", body_style), Paragraph("<b>MITRE ATT&CK Metric</b>", body_style)],
        [Paragraph("Technique Name", body_style), Paragraph(f"<b>{mitre_info.get('technique_name', 'N/A')}</b>", body_style)],
        [Paragraph("Technique ID & Sub-type", body_style), Paragraph(f"{mitre_info.get('technique_id', 'N/A')} ({mitre_info.get('sub_technique', 'N/A')})", body_style)],
        [Paragraph("Tactical Phase", body_style), Paragraph(f"{mitre_info.get('tactic', 'N/A')}", body_style)],
        [Paragraph("Threat Severity Level", body_style), Paragraph(f"<font color='{mitre_info.get('color', '#000000')}'><b>{mitre_info.get('severity', 'N/A')}</b></font>", body_style)],
        [Paragraph("Threat Description", body_style), Paragraph(f"{mitre_info.get('description', 'N/A')}", body_style)],
    ]
    mitre_table = Table(mitre_table_data, colWidths=[2.2 * inch, 5.2 * inch])
    mitre_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ALT_ROW_BG]),
    ]))
    story.append(mitre_table)
    story.append(Spacer(1, 12))

    # 4. Multi-Model Ensemble Classification Matrix
    story.append(Paragraph("2. Multi-Model Ensemble Classification Matrix", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=2, spaceAfter=6))

    ensemble_data = [
        [Paragraph("<b>Model Estimator</b>", body_style), Paragraph("<b>Algorithm Type</b>", body_style), Paragraph("<b>Model Decision</b>", body_style)],
        [Paragraph("Isolation Forest", body_style), Paragraph("Unsupervised Anomaly Detector", body_style), Paragraph(f"{row.get('Isolation Forest', 'N/A')}", body_style)],
        [Paragraph("Random Forest", body_style), Paragraph("Supervised Decision Trees", body_style), Paragraph(f"{row.get('Random Forest', 'N/A')}", body_style)],
        [Paragraph("XGBoost Classifier", body_style), Paragraph("Gradient Boosted Trees", body_style), Paragraph(f"{row.get('XGBoost', 'N/A')}", body_style)],
        [Paragraph("<b>Hybrid System Verdict</b>", body_style), Paragraph("Multi-Estimator Fusion Rule", body_style), Paragraph(f"<b>{verdict_str}</b>", body_style)],
    ]
    ensemble_table = Table(ensemble_data, colWidths=[2.5 * inch, 2.7 * inch, 2.2 * inch])
    ensemble_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ALT_ROW_BG]),
    ]))
    story.append(ensemble_table)
    story.append(Spacer(1, 12))

    # 5. Multi-Class Category Probabilities (Top 5)
    if not prob_df.empty:
        story.append(Paragraph("3. Multi-Class Softmax Probability Breakdown (%)", h2_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=2, spaceAfter=6))
        
        row_prob = prob_df.iloc[0]
        sorted_probs = row_prob.sort_values(ascending=False)
        
        prob_headers = [Paragraph("<b>Attack Category</b>", body_style), Paragraph("<b>Confidence Probability (%)</b>", body_style)]
        prob_rows = [prob_headers]
        for cat_name, prob_val in sorted_probs.items():
            clean_cat = cat_name.replace(' (%)', '')
            prob_rows.append([
                Paragraph(clean_cat, body_style),
                Paragraph(f"<b>{prob_val:.2f}%</b>", body_style)
            ])
            
        prob_table = Table(prob_rows[:11], colWidths=[4.2 * inch, 3.2 * inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ALT_ROW_BG]),
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 12))

    # 6. Active Mitigation Playbook & Firewall Command
    story.append(Paragraph("4. Active Defense Mitigation Playbook & Firewall Rule", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=2, spaceAfter=6))

    cmd_str = mitre_info.get('containment_command', '# Block IP command').format(
        src_ip=raw_feat_dict.get('src_ip', '192.168.1.100'),
        dst_port=raw_feat_dict.get('dst_port', 80),
        proto=raw_feat_dict.get('proto', 'tcp'),
        user_id='root'
    )
    
    playbook_data = [
        [Paragraph("<b>Recommended Action Plan</b>", body_style), Paragraph("Immediately isolate host, deploy perimeter firewall rule, and initiate DPI pcap capture.", body_style)],
        [Paragraph("<b>Executable CLI Mitigation Command</b>", body_style), Paragraph(f"<code>{cmd_str}</code>", code_style)],
    ]
    playbook_table = Table(playbook_data, colWidths=[2.5 * inch, 4.9 * inch])
    playbook_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(playbook_table)
    story.append(Spacer(1, 14))

    # 7. Raw Technical Data Features (Selected Sample)
    story.append(Paragraph("5. Packet Specification Telemetry (Sample Features)", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=2, spaceAfter=6))

    sample_feats = list(raw_feat_dict.items())[:16]
    feat_rows = [[Paragraph("<b>Feature Attribute</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Feature Attribute</b>", body_style), Paragraph("<b>Value</b>", body_style)]]
    
    for i in range(0, len(sample_feats), 2):
        k1, v1 = sample_feats[i]
        k2, v2 = sample_feats[i+1] if (i+1) < len(sample_feats) else ("", "")
        feat_rows.append([
            Paragraph(str(k1), body_style), Paragraph(str(v1), body_style),
            Paragraph(str(k2), body_style), Paragraph(str(v2), body_style)
        ])
        
    feat_table = Table(feat_rows, colWidths=[2.0 * inch, 1.7 * inch, 2.0 * inch, 1.7 * inch])
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ALT_ROW_BG]),
    ]))
    story.append(feat_table)

    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data
