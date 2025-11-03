"""
PDF Report generation for pre/post test analysis.

This module generates comprehensive PDF reports including class-level statistics,
question-level analysis, visualizations, and individual student pages.
"""

import io
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """Generate comprehensive PDF reports for pre/post test analysis."""
    
    def __init__(self, page_size=letter):
        """
        Initialize PDF generator.
        
        Args:
            page_size: Page size (letter or A4)
        """
        self.page_size = page_size
        self.width = page_size[0]
        self.height = page_size[1]
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Set up custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        # Subheading style
        self.styles.add(ParagraphStyle(
            name='CustomSubheading',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=6
        ))
    
    def generate_full_report(
        self,
        output_path: str,
        df_merged: pd.DataFrame,
        class_stats: Dict[str, Any],
        question_stats: pd.DataFrame,
        faculty_rating: Dict[str, Any],
        class_summary_text: str,
        question_columns: List[str],
        charts: Dict[str, Any],
        class_name: str = "Pre-test / Post-test Analysis",
        include_individual_pages: bool = True
    ):
        """
        Generate full PDF report with all sections.
        
        Args:
            output_path: Output file path
            df_merged: Merged student data
            class_stats: Class-level statistics
            question_stats: Question-level statistics
            faculty_rating: Faculty rating information
            class_summary_text: Human-readable class summary
            question_columns: List of question column names
            charts: Dictionary of chart image paths
            class_name: Name of the class/lecture
            include_individual_pages: Whether to include individual student pages
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.page_size,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        story = []
        
        # Title page
        story.extend(self._create_title_page(class_name, class_stats, df_merged))
        story.append(PageBreak())
        
        # Table of contents
        story.extend(self._create_table_of_contents(include_individual_pages))
        story.append(PageBreak())
        
        # Executive summary
        story.append(Paragraph("Executive Summary", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        for line in class_summary_text.split('\n'):
            if line.strip():
                story.append(Paragraph(line, self.styles['BodyText']))
                story.append(Spacer(1, 0.1*inch))
        story.append(PageBreak())
        
        # Class-level analysis
        story.extend(self._create_class_analysis_section(
            class_stats, charts
        ))
        story.append(PageBreak())
        
        # Question-level analysis
        story.extend(self._create_question_analysis_section(
            question_stats, charts
        ))
        story.append(PageBreak())
        
        # Top performers and needs improvement
        story.extend(self._create_performance_lists(df_merged))
        story.append(PageBreak())
        
        # Individual student pages
        if include_individual_pages:
            from utils import generate_student_analysis_text
            
            story.append(Paragraph("Individual Student Reports", self.styles['CustomTitle']))
            story.append(PageBreak())
            
            for idx, row in df_merged.iterrows():
                story.extend(self._create_student_page(row, question_columns))
                if idx < len(df_merged) - 1:
                    story.append(PageBreak())
        
        # Faculty rating page
        story.append(PageBreak())
        story.extend(self._create_faculty_rating_page(faculty_rating, class_stats))
        
        # Build PDF
        try:
            doc.build(story)
            logger.info(f"PDF report generated: {output_path}")
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            raise
    
    def generate_compact_report(
        self,
        output_path: str,
        df_merged: pd.DataFrame,
        class_stats: Dict[str, Any],
        question_stats: pd.DataFrame,
        faculty_rating: Dict[str, Any],
        class_summary_text: str,
        charts: Dict[str, Any],
        class_name: str = "Pre-test / Post-test Analysis"
    ):
        """
        Generate compact PDF report (class-level only, top/bottom performers).
        
        Args:
            output_path: Output file path
            df_merged: Merged student data
            class_stats: Class-level statistics
            question_stats: Question-level statistics
            faculty_rating: Faculty rating information
            class_summary_text: Human-readable class summary
            charts: Dictionary of chart image paths
            class_name: Name of the class/lecture
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.page_size,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        story = []
        
        # Title page
        story.extend(self._create_title_page(class_name, class_stats, df_merged))
        story.append(PageBreak())
        
        # Executive summary
        story.append(Paragraph("Executive Summary", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        for line in class_summary_text.split('\n'):
            if line.strip():
                story.append(Paragraph(line, self.styles['BodyText']))
                story.append(Spacer(1, 0.1*inch))
        story.append(PageBreak())
        
        # Class-level analysis
        story.extend(self._create_class_analysis_section(class_stats, charts))
        story.append(PageBreak())
        
        # Question-level analysis
        story.extend(self._create_question_analysis_section(question_stats, charts))
        story.append(PageBreak())
        
        # Top/bottom performers
        story.extend(self._create_performance_lists(df_merged))
        story.append(PageBreak())
        
        # Faculty rating
        story.extend(self._create_faculty_rating_page(faculty_rating, class_stats))
        
        # Build PDF
        doc.build(story)
        logger.info(f"Compact PDF report generated: {output_path}")
    
    def _create_title_page(
        self,
        class_name: str,
        class_stats: Dict[str, Any],
        df_merged: pd.DataFrame
    ) -> List:
        """Create title page elements."""
        elements = []
        
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph(class_name, self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.3*inch))
        
        elements.append(Paragraph(
            "Pre-test / Post-test Analysis Report",
            self.styles['Heading2']
        ))
        elements.append(Spacer(1, 0.5*inch))
        
        # Report metadata
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        elements.append(Paragraph(f"Report Generated: {report_date}", self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Summary statistics
        data = [
            ["Total Students Analyzed:", str(class_stats['n_students'])],
            ["Total Questions:", str(class_stats['max_score'])],
            ["Students Discarded:", str(len(df_merged.index))],  # Placeholder
            ["Mean Pre-test Score:", f"{class_stats['mean_pre']:.2f}"],
            ["Mean Post-test Score:", f"{class_stats['mean_post']:.2f}"],
            ["Mean Gain:", f"{class_stats['mean_gain']:.2f}"]
        ]
        
        table = Table(data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(table)
        
        return elements
    
    def _create_table_of_contents(self, include_individual: bool) -> List:
        """Create table of contents."""
        elements = []
        
        elements.append(Paragraph("Table of Contents", self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.3*inch))
        
        toc_items = [
            "1. Executive Summary",
            "2. Class-Level Analysis",
            "3. Question-Level Analysis",
            "4. Top Performers and Areas Needing Support",
        ]
        
        if include_individual:
            toc_items.append("5. Individual Student Reports")
            toc_items.append("6. Faculty Rating and Recommendations")
        else:
            toc_items.append("5. Faculty Rating and Recommendations")
        
        for item in toc_items:
            elements.append(Paragraph(item, self.styles['Normal']))
            elements.append(Spacer(1, 0.15*inch))
        
        return elements
    
    def _create_class_analysis_section(
        self,
        class_stats: Dict[str, Any],
        charts: Dict[str, Any]
    ) -> List:
        """Create class-level analysis section."""
        elements = []
        
        elements.append(Paragraph("Class-Level Analysis", self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Statistics table
        elements.append(Paragraph("Summary Statistics", self.styles['CustomHeading']))
        
        stats_data = [
            ["Metric", "Pre-test", "Post-test", "Change"],
            ["Mean Score", 
             f"{class_stats['mean_pre']:.2f}", 
             f"{class_stats['mean_post']:.2f}",
             f"{class_stats['mean_gain']:+.2f}"],
            ["Standard Deviation",
             f"{class_stats['std_pre']:.2f}",
             f"{class_stats['std_post']:.2f}",
             "—"],
            ["Median",
             f"{class_stats['median_pre']:.1f}",
             f"{class_stats['median_post']:.1f}",
             f"{class_stats['median_post'] - class_stats['median_pre']:+.1f}"],
        ]
        
        stats_table = Table(stats_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(stats_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Statistical significance
        elements.append(Paragraph("Statistical Test Results", self.styles['CustomHeading']))
        
        sig_text = f"Paired t-test p-value: {class_stats['p_value']:.4f}<br/>"
        sig_text += f"Result: {'Statistically significant' if class_stats['is_significant'] else 'Not significant'} (α=0.05)<br/>"
        sig_text += f"Cohen's d (effect size): {class_stats['cohens_d']:.3f}<br/>"
        sig_text += f"Mean Normalized Gain: {class_stats['mean_normalized_gain']:.2%}"
        
        elements.append(Paragraph(sig_text, self.styles['BodyText']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Add charts if available
        if 'class_histogram' in charts and charts['class_histogram']:
            elements.append(Paragraph("Score Distribution", self.styles['CustomHeading']))
            img = Image(charts['class_histogram'], width=6*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
        
        if 'boxplot' in charts and charts['boxplot']:
            elements.append(Paragraph("Score Comparison (Boxplot)", self.styles['CustomHeading']))
            img = Image(charts['boxplot'], width=6*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_question_analysis_section(
        self,
        question_stats: pd.DataFrame,
        charts: Dict[str, Any]
    ) -> List:
        """Create question-level analysis section."""
        elements = []
        
        elements.append(Paragraph("Question-Level Analysis", self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Question statistics table
        table_data = [["Question", "Pre %", "Post %", "Improvement"]]
        
        for _, row in question_stats.iterrows():
            table_data.append([
                row['question'],
                f"{row['pct_correct_pre']:.1f}%",
                f"{row['pct_correct_post']:.1f}%",
                f"{row['improvement']:+.1f}%"
            ])
        
        q_table = Table(table_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        q_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        elements.append(q_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Add question charts
        if 'question_comparison' in charts and charts['question_comparison']:
            elements.append(Paragraph("Question Performance Comparison", self.styles['CustomHeading']))
            img = Image(charts['question_comparison'], width=6*inch, height=4*inch)
            elements.append(img)
        
        return elements
    
    def _create_performance_lists(self, df_merged: pd.DataFrame) -> List:
        """Create top/bottom performers lists."""
        elements = []
        
        elements.append(Paragraph("Performance Highlights", self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Top 10 improvers
        elements.append(Paragraph("Top 10 Improvers", self.styles['CustomHeading']))
        
        top_10 = df_merged.nlargest(10, 'absolute_gain')[
            ['name', 'ticket_no', 'score_pre', 'score_post', 'absolute_gain']
        ]
        
        top_data = [["Rank", "Name", "Ticket", "Pre", "Post", "Gain"]]
        for idx, (_, row) in enumerate(top_10.iterrows(), 1):
            top_data.append([
                str(idx),
                row['name'][:25],  # Truncate long names
                row['ticket_no'],
                str(int(row['score_pre'])),
                str(int(row['score_post'])),
                f"+{int(row['absolute_gain'])}"
            ])
        
        top_table = Table(top_data, colWidths=[0.5*inch, 2.5*inch, 1*inch, 0.7*inch, 0.7*inch, 0.7*inch])
        top_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.green),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(top_table)
        elements.append(Spacer(1, 0.4*inch))
        
        # Bottom 10 (most regression)
        elements.append(Paragraph("Students Needing Support (Regressions)", self.styles['CustomHeading']))
        
        bottom_10 = df_merged.nsmallest(10, 'absolute_gain')[
            ['name', 'ticket_no', 'score_pre', 'score_post', 'absolute_gain']
        ]
        
        bottom_data = [["Rank", "Name", "Ticket", "Pre", "Post", "Change"]]
        for idx, (_, row) in enumerate(bottom_10.iterrows(), 1):
            bottom_data.append([
                str(idx),
                row['name'][:25],
                row['ticket_no'],
                str(int(row['score_pre'])),
                str(int(row['score_post'])),
                f"{int(row['absolute_gain']):+d}"
            ])
        
        bottom_table = Table(bottom_data, colWidths=[0.5*inch, 2.5*inch, 1*inch, 0.7*inch, 0.7*inch, 0.7*inch])
        bottom_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.red),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(bottom_table)
        
        return elements
    
    def _create_student_page(self, row: pd.Series, question_columns: List[str]) -> List:
        """Create individual student page."""
        from utils import generate_student_analysis_text
        
        elements = []
        
        # Student header
        elements.append(Paragraph(
            f"Student Report: {row['name']}", 
            self.styles['CustomHeading']
        ))
        elements.append(Spacer(1, 0.1*inch))
        
        # Generate analysis text
        analysis_text = generate_student_analysis_text(row, question_columns)
        
        # Split into paragraphs and add to elements
        for paragraph in analysis_text.split('\n\n'):
            if paragraph.strip():
                # Replace newlines with <br/> for reportlab
                formatted = paragraph.replace('\n', '<br/>')
                elements.append(Paragraph(formatted, self.styles['BodyText']))
                elements.append(Spacer(1, 0.15*inch))
        
        # Question details table
        q_data = [["Question", "Pre", "Post", "Transition"]]
        
        for q_col in question_columns:
            pre_val = "✓" if row[f"{q_col}_pre"] == 1 else "✗"
            post_val = "✓" if row[f"{q_col}_post"] == 1 else "✗"
            trans = row[f"{q_col}_transition"]
            
            # Simplify transition label
            trans_short = trans.replace('PreRight_PostRight', 'Mastered') \
                              .replace('PreWrong_PostRight', 'Learned') \
                              .replace('PreRight_PostWrong', 'Forgot') \
                              .replace('PreWrong_PostWrong', 'Struggling')
            
            q_data.append([q_col, pre_val, post_val, trans_short])
        
        q_detail_table = Table(q_data, colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 1.8*inch])
        q_detail_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("Detailed Question Breakdown", self.styles['CustomSubheading']))
        elements.append(q_detail_table)
        
        return elements
    
    def _create_faculty_rating_page(
        self,
        faculty_rating: Dict[str, Any],
        class_stats: Dict[str, Any]
    ) -> List:
        """Create faculty rating page."""
        elements = []
        
        elements.append(Paragraph("Faculty Rating", self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Rating score (big and bold)
        score_text = f"<font size=36 color='#1f77b4'><b>{faculty_rating['score']}/100</b></font>"
        elements.append(Paragraph(score_text, self.styles['BodyText']))
        elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Paragraph(
            faculty_rating['interpretation'],
            self.styles['Heading3']
        ))
        elements.append(Spacer(1, 0.3*inch))
        
        # Explanation
        elements.append(Paragraph("Rating Methodology", self.styles['CustomHeading']))
        
        explanation_text = faculty_rating['explanation'].replace('\n', '<br/>')
        elements.append(Paragraph(explanation_text, self.styles['BodyText']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Components breakdown
        elements.append(Paragraph("Rating Components", self.styles['CustomHeading']))
        
        comp_data = [
            ["Component", "Value", "Weight"],
            ["Average Normalized Gain", 
             f"{faculty_rating['components']['normalized_gain']:.2%}",
             "65%"],
            ["Students Improved",
             f"{faculty_rating['components']['pct_improved']:.1f}%",
             "25%"],
            ["Students Regressed",
             f"{faculty_rating['components']['pct_regressed']:.1f}%",
             "10% (inverse)"],
        ]
        
        comp_table = Table(comp_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(comp_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Recommendations
        elements.append(Paragraph("Recommendations", self.styles['CustomHeading']))
        
        if faculty_rating['score'] >= 70:
            rec_text = "Excellent performance. Continue with current teaching methods and consider:"
            recs = [
                "Sharing best practices with colleagues",
                "Documenting successful teaching strategies",
                "Exploring advanced topics for high performers"
            ]
        elif faculty_rating['score'] >= 50:
            rec_text = "Good progress made. Consider the following improvements:"
            recs = [
                "Identify specific topics where students struggled",
                "Incorporate more active learning activities",
                "Provide additional practice materials for weak areas"
            ]
        else:
            rec_text = "Significant improvement needed. Recommended actions:"
            recs = [
                "Review teaching methodology and materials",
                "Increase student engagement and interaction",
                "Provide supplementary resources and support",
                "Consider peer observation and feedback"
            ]
        
        elements.append(Paragraph(rec_text, self.styles['BodyText']))
        elements.append(Spacer(1, 0.1*inch))
        
        for rec in recs:
            elements.append(Paragraph(f"• {rec}", self.styles['BodyText']))
            elements.append(Spacer(1, 0.05*inch))
        
        return elements


def create_matplotlib_chart(
    chart_type: str,
    data: Any,
    **kwargs
) -> io.BytesIO:
    """
    Create a matplotlib chart and return as BytesIO object.
    
    Args:
        chart_type: Type of chart ('histogram', 'boxplot', 'bar', etc.)
        data: Data for the chart
        **kwargs: Additional arguments for customization
        
    Returns:
        BytesIO object containing the chart image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if chart_type == 'histogram':
        # Overlay histogram for pre and post
        ax.hist(data['pre'], bins=kwargs.get('bins', 15), alpha=0.6, label='Pre-test', color='#1f77b4')
        ax.hist(data['post'], bins=kwargs.get('bins', 15), alpha=0.6, label='Post-test', color='#ff7f0e')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title(kwargs.get('title', 'Score Distribution'))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    elif chart_type == 'boxplot':
        ax.boxplot([data['pre'], data['post']], labels=['Pre-test', 'Post-test'])
        ax.set_ylabel('Score')
        ax.set_title(kwargs.get('title', 'Score Comparison'))
        ax.grid(True, alpha=0.3)
        
    elif chart_type == 'bar':
        x = data['x']
        y_pre = data['y_pre']
        y_post = data['y_post']
        
        x_pos = np.arange(len(x))
        width = 0.35
        
        ax.bar(x_pos - width/2, y_pre, width, label='Pre-test', color='#1f77b4')
        ax.bar(x_pos + width/2, y_post, width, label='Post-test', color='#ff7f0e')
        
        ax.set_xlabel(kwargs.get('xlabel', 'Questions'))
        ax.set_ylabel(kwargs.get('ylabel', 'Percent Correct'))
        ax.set_title(kwargs.get('title', 'Question Performance'))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save to BytesIO
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf
