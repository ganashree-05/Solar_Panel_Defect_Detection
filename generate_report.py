from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak, KeepTogether
from reportlab.platypus import BalancedColumns
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF
import datetime, os
from pathlib import Path

OUTPUT = 'SolarScan_Project_Report.pdf'
W, H = A4

# ?? Colour palette ??????????????????????????????????????????????????????????
PURPLE     = colors.HexColor('#7C3AED')
PURPLE_LT  = colors.HexColor('#EDE9FE')
PURPLE_DK  = colors.HexColor('#5B21B6')
PURPLE_MD  = colors.HexColor('#C4B5FD')
DARK       = colors.HexColor('#1E1B4B')
GREY       = colors.HexColor('#6B7280')
GREY_LT    = colors.HexColor('#F3F4F6')
GREY_BD    = colors.HexColor('#E5E7EB')
GREEN      = colors.HexColor('#10B981')
RED        = colors.HexColor('#EF4444')
ORANGE     = colors.HexColor('#F97316')
BLUE       = colors.HexColor('#3B82F6')
CYAN       = colors.HexColor('#06B6D4')
YELLOW     = colors.HexColor('#EAB308')
WHITE      = colors.white
BLACK      = colors.black

# ?? Styles ??????????????????????????????????????????????????????????????????
def make_styles():
    base = getSampleStyleSheet()
    s = {}
    s['cover_title'] = ParagraphStyle('cover_title', fontName='Helvetica-Bold',
        fontSize=32, textColor=WHITE, leading=40, alignment=TA_CENTER)
    s['cover_sub'] = ParagraphStyle('cover_sub', fontName='Helvetica',
        fontSize=14, textColor=PURPLE_MD, leading=20, alignment=TA_CENTER)
    s['cover_meta'] = ParagraphStyle('cover_meta', fontName='Helvetica',
        fontSize=10, textColor=WHITE, leading=14, alignment=TA_CENTER)
    s['h1'] = ParagraphStyle('h1', fontName='Helvetica-Bold',
        fontSize=18, textColor=PURPLE_DK, leading=24, spaceBefore=18, spaceAfter=8)
    s['h2'] = ParagraphStyle('h2', fontName='Helvetica-Bold',
        fontSize=13, textColor=PURPLE, leading=18, spaceBefore=14, spaceAfter=6)
    s['h3'] = ParagraphStyle('h3', fontName='Helvetica-Bold',
        fontSize=11, textColor=DARK, leading=15, spaceBefore=10, spaceAfter=4)
    s['body'] = ParagraphStyle('body', fontName='Helvetica',
        fontSize=10, textColor=DARK, leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
    s['body_sm'] = ParagraphStyle('body_sm', fontName='Helvetica',
        fontSize=9, textColor=GREY, leading=13, spaceAfter=4)
    s['bullet'] = ParagraphStyle('bullet', fontName='Helvetica',
        fontSize=10, textColor=DARK, leading=14, leftIndent=16, spaceAfter=3,
        bulletIndent=4)
    s['code'] = ParagraphStyle('code', fontName='Courier',
        fontSize=8.5, textColor=PURPLE_DK, leading=12, backColor=PURPLE_LT,
        leftIndent=10, rightIndent=10, spaceBefore=4, spaceAfter=4)
    s['caption'] = ParagraphStyle('caption', fontName='Helvetica-Oblique',
        fontSize=8.5, textColor=GREY, leading=12, alignment=TA_CENTER, spaceAfter=8)
    s['table_hdr'] = ParagraphStyle('table_hdr', fontName='Helvetica-Bold',
        fontSize=9, textColor=WHITE, leading=12, alignment=TA_CENTER)
    s['table_cell'] = ParagraphStyle('table_cell', fontName='Helvetica',
        fontSize=9, textColor=DARK, leading=12, alignment=TA_CENTER)
    s['metric_big'] = ParagraphStyle('metric_big', fontName='Helvetica-Bold',
        fontSize=28, textColor=PURPLE, leading=34, alignment=TA_CENTER)
    s['metric_lbl'] = ParagraphStyle('metric_lbl', fontName='Helvetica',
        fontSize=9, textColor=GREY, leading=12, alignment=TA_CENTER)
    s['tag'] = ParagraphStyle('tag', fontName='Helvetica-Bold',
        fontSize=8, textColor=PURPLE, leading=10, alignment=TA_CENTER)
    return s

# ?? Helper builders ?????????????????????????????????????????????????????????
def hr(color=None, thickness=1):
    return HRFlowable(width='100%', thickness=thickness,
                      color=color or PURPLE_MD, spaceAfter=6, spaceBefore=6)

def sp(h=6):
    return Spacer(1, h)

def section_header(text, s):
    return [hr(PURPLE, 2), Paragraph(text, s['h1']), sp(4)]

def metric_box(value, label, color, s):
    data = [[Paragraph(str(value), s['metric_big'])],
            [Paragraph(label, s['metric_lbl'])]]
    t = Table(data, colWidths=[3.8*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), PURPLE_LT),
        ('ROUNDEDCORNERS', [8]),
        ('BOX', (0,0), (-1,-1), 1, color),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    return t

def styled_table(headers, rows, col_widths, s):
    hdr_row = [Paragraph(h, s['table_hdr']) for h in headers]
    data = [hdr_row]
    for row in rows:
        data.append([Paragraph(str(c), s['table_cell']) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PURPLE),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, GREY_LT]),
        ('GRID', (0,0), (-1,-1), 0.5, GREY_BD),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ])
    t.setStyle(style)
    return t

def info_box(title, items, s, bg=None):
    bg = bg or PURPLE_LT
    content = [Paragraph(title, s['h3'])]
    for item in items:
        content.append(Paragraph(f'  {chr(8226)}  {item}', s['bullet']))
    data = [[content]]
    t = Table(data, colWidths=[W - 4*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg),
        ('BOX', (0,0), (-1,-1), 1, PURPLE_MD),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 14),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    return t

# ?? Cover page ??????????????????????????????????????????????????????????????
def build_cover(s):
    story = []
    # Purple header block
    cover_data = [[
        Paragraph('SolarScan', s['cover_title']),
        Paragraph('Solar Panel Defect Detection System', s['cover_sub']),
        Spacer(1, 12),
        Paragraph('Project Technical Report', s['cover_sub']),
    ]]
    cover_tbl = Table([[cover_data[0]]], colWidths=[W - 4*cm])
    cover_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), PURPLE),
        ('TOPPADDING', (0,0), (-1,-1), 40),
        ('BOTTOMPADDING', (0,0), (-1,-1), 40),
        ('LEFTPADDING', (0,0), (-1,-1), 20),
        ('RIGHTPADDING', (0,0), (-1,-1), 20),
        ('ROUNDEDCORNERS', [12]),
    ]))
    story.append(cover_tbl)
    story.append(sp(24))
    # Meta info grid
    today = datetime.date.today().strftime('%B %d, %Y')
    meta = [
        ['Author', 'Shubham Madiwalar'],
        ['Date', today],
        ['Version', '1.0'],
        ['GitHub', 'github.com/Shubhammadiwalar/repoproject1'],
        ['Framework', 'YOLOv8 + ResNet-50 + Flask'],
        ['Language', 'Python 3.14'],
    ]
    meta_data = [[Paragraph(f'<b>{k}</b>', s['body']), Paragraph(v, s['body'])] for k,v in meta]
    mt = Table(meta_data, colWidths=[4*cm, 11*cm])
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), GREY_LT),
        ('GRID', (0,0), (-1,-1), 0.5, GREY_BD),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(mt)
    story.append(sp(24))
    # Abstract box
    abstract = ('SolarScan is an end-to-end AI-powered solar panel defect detection and analysis system. '
                'It combines a YOLOv8m object detection model (CSPDarknet CNN backbone) with a ResNet-50 '
                'classifier to identify six types of solar panel conditions: Bird-drop, Clean, Dusty, '
                'Electrical-damage, Physical-Damage, and Snow-Covered. The system uses GradCAM '
                '(Gradient-weighted Class Activation Mapping) to generate precise bounding box labels '
                'that localise the exact damaged region rather than the whole image. A Flask REST API '
                'backend serves a modern purple-themed dashboard frontend with authentication, real-time '
                'analysis, damage percentage calculation, and actionable repair recommendations.')
    story.append(info_box('Abstract', [abstract], s))
    story.append(PageBreak())
    return story

# ?? Section 1: Introduction ?????????????????????????????????????????????????
def build_intro(s):
    story = section_header('1. Introduction', s)
    story.append(Paragraph('1.1 Problem Statement', s['h2']))
    story.append(Paragraph(
        'Solar panels degrade over time due to environmental factors including bird droppings, '
        'dust accumulation, snow coverage, physical damage from hail or debris, and electrical '
        'faults from arc discharge or overheating. Manual inspection is time-consuming, expensive, '
        'and inconsistent. An automated AI-based system can detect defects faster, more accurately, '
        'and at scale across large solar farms.', s['body']))
    story.append(Paragraph('1.2 Objectives', s['h2']))
    objectives = [
        'Detect and classify 6 types of solar panel conditions using deep learning',
        'Generate precise bounding boxes around the damaged region (not the whole panel)',
        'Quantify damage severity as a percentage (0-100%)',
        'Provide actionable repair recommendations per defect type',
        'Deliver results through a modern web dashboard with authentication',
        'Achieve >90% classification accuracy on the test set',
    ]
    for obj in objectives:
        story.append(Paragraph(f'  {chr(8226)}  {obj}', s['bullet']))
    story.append(sp(8))
    story.append(Paragraph('1.3 Scope', s['h2']))
    story.append(Paragraph(
        'The system processes individual solar panel images (JPG/PNG) and returns detection results '
        'in real-time via a REST API. It is designed for deployment on GPU-equipped machines and '
        'tested on an NVIDIA RTX 3050 6GB Laptop GPU. The dataset contains 1,574 labelled images '
        'across 6 classes split into train/val/test sets.', s['body']))
    story.append(PageBreak())
    return story

# ?? Section 2: Dataset ??????????????????????????????????????????????????????
def build_dataset(s):
    story = section_header('2. Dataset', s)
    story.append(Paragraph('2.1 Dataset Overview', s['h2']))
    story.append(Paragraph(
        'The dataset consists of real-world solar panel images collected across diverse conditions. '
        'Images vary in resolution (158x318 to 3000x2250 pixels), lighting, and camera angle. '
        'The dataset was split into training, validation, and test sets.', s['body']))
    story.append(sp(8))
    # Split table
    split_rows = [
        ['Train', '929', '59.0%', 'Model training'],
        ['Validation', '550', '35.0%', 'Hyperparameter tuning'],
        ['Test', '95', '6.0%', 'Final evaluation'],
        ['Total', '1,574', '100%', 'All splits'],
    ]
    story.append(styled_table(
        ['Split', 'Images', 'Percentage', 'Purpose'],
        split_rows,
        [3*cm, 3*cm, 3.5*cm, 6*cm], s))
    story.append(sp(12))
    story.append(Paragraph('2.2 Class Distribution', s['h2']))
    class_rows = [
        ['0', 'Bird-drop',         '177', '104', '17', '298', 'Moderate (60%)'],
        ['1', 'Clean',             '169', '102', '18', '289', 'None (0%)'],
        ['2', 'Dusty',             '162', '97',  '16', '275', 'Low (35%)'],
        ['3', 'Electrical-damage', '135', '77',  '13', '225', 'Critical (95%)'],
        ['4', 'Physical-Damage',   '132', '78',  '15', '225', 'High (90%)'],
        ['5', 'Snow-Covered',      '154', '92',  '16', '262', 'Moderate (50%)'],
    ]
    story.append(styled_table(
        ['ID', 'Class', 'Train', 'Val', 'Test', 'Total', 'Max Damage'],
        class_rows,
        [1.2*cm, 4.2*cm, 1.8*cm, 1.5*cm, 1.5*cm, 1.8*cm, 3.5*cm], s))
    story.append(sp(12))
    story.append(Paragraph('2.3 Image Characteristics', s['h2']))
    char_rows = [
        ['Format', 'JPG, JPEG, PNG (mixed case extensions)'],
        ['Resolution range', '158x318 to 3000x2250 pixels'],
        ['Colour space', 'RGB (3-channel)'],
        ['YOLO input size', '640 x 640 (resized during training)'],
        ['CNN input size', '224 x 224 (resized during training)'],
        ['Augmentation', 'Mosaic, MixUp, HSV shift, flip, rotation, scale'],
    ]
    ct = Table(char_rows, colWidths=[5*cm, 10*cm])
    ct.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), GREY_LT),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, GREY_BD),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(ct)
    story.append(PageBreak())
    return story

