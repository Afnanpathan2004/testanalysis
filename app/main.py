"""
Pre-test / Post-test Analysis Streamlit Application

This is the main entry point for the Streamlit app. It provides an interactive
interface for uploading test files, validating data, computing analytics,
visualizing results, and generating comprehensive PDF reports.
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Import our custom modules
from utils import (
    validate_excel_file,
    validate_matching_structure,
    match_participants,
    compute_scores_and_gains,
    compute_class_statistics,
    compute_question_statistics,
    compute_faculty_rating,
    generate_student_analysis_text,
    generate_class_summary_text,
    QUESTION_PATTERN
)
from report import PDFReportGenerator, create_matplotlib_chart
from generate_samples import generate_blank_template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Pre/Post Test Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .kpi-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #666;
    }
    .success-box {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_validate_file(file_bytes: bytes, file_name: str, file_label: str):
    """Load and validate Excel file with caching."""
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), engine='openpyxl')
        validation_result = validate_excel_file(df, file_label)
        return df, validation_result
    except Exception as e:
        logger.error(f"Error loading {file_label}: {e}")
        return None, None


@st.cache_data
def perform_analysis(
    pre_bytes: bytes,
    post_bytes: bytes,
    q_cols_pre: list,
    q_cols_post: list
):
    """Perform complete analysis with caching for performance."""
    # Load DataFrames
    df_pre = pd.read_excel(io.BytesIO(pre_bytes), engine='openpyxl')
    df_post = pd.read_excel(io.BytesIO(post_bytes), engine='openpyxl')
    
    # Match participants
    df_pre_filtered, df_post_filtered, df_discarded = match_participants(df_pre, df_post)
    
    if len(df_pre_filtered) == 0:
        return None
    
    # Compute scores and gains
    df_merged = compute_scores_and_gains(df_pre_filtered, df_post_filtered, q_cols_pre)
    
    # Compute statistics
    class_stats = compute_class_statistics(df_merged, len(q_cols_pre))
    question_stats = compute_question_statistics(df_merged, q_cols_pre)
    faculty_rating = compute_faculty_rating(class_stats, df_merged)
    
    # Generate summary text
    class_summary = generate_class_summary_text(
        class_stats, faculty_rating, question_stats, df_merged
    )
    
    return {
        'df_merged': df_merged,
        'df_discarded': df_discarded,
        'class_stats': class_stats,
        'question_stats': question_stats,
        'faculty_rating': faculty_rating,
        'class_summary': class_summary,
        'question_columns': q_cols_pre
    }


def create_sample_template_download():
    """Create download button for sample template."""
    df_template = generate_blank_template(10)
    
    # Add sample rows
    sample_data = [
        {"name": "Asha Sharma", "ticket_no": "T001", "q1": 1, "q2": 0, "q3": 0, "q4": 1, "q5": 1, "q6": 0, "q7": 1, "q8": 0, "q9": 1, "q10": 0},
        {"name": "Ravi Kumar", "ticket_no": "T002", "q1": 0, "q2": 0, "q3": 1, "q4": 0, "q5": 1, "q6": 1, "q7": 0, "q8": 1, "q9": 0, "q10": 1},
        {"name": "Meera Patel", "ticket_no": "T003", "q1": 0, "q2": 1, "q3": 0, "q4": 0, "q5": 0, "q6": 0, "q7": 0, "q8": 0, "q9": 0, "q10": 0},
    ]
    
    df_template = pd.DataFrame(sample_data)
    
    # Convert to Excel
    buffer = io.BytesIO()
    df_template.to_excel(buffer, index=False, engine='openpyxl')
    buffer.seek(0)
    
    st.download_button(
        label="📥 Download Sample Template",
        data=buffer,
        file_name="prepost_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download a sample Excel template with the correct format"
    )


def display_kpi_cards(class_stats: Dict, faculty_rating: Dict, n_matched: int, n_discarded: int):
    """Display KPI cards with key metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{n_matched}</div>
            <div class="kpi-label">Students Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        mean_gain = class_stats['mean_gain']
        gain_color = "#28a745" if mean_gain > 0 else "#dc3545"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: {gain_color};">{mean_gain:+.2f}</div>
            <div class="kpi-label">Mean Gain (points)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        pct_improved = class_stats['pct_improved']
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{pct_improved:.1f}%</div>
            <div class="kpi-label">Students Improved</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        rating = faculty_rating['score']
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{rating}/100</div>
            <div class="kpi-label">Faculty Rating</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)


def plot_score_distribution(df_merged: pd.DataFrame, max_score: int):
    """Create interactive histogram of score distribution."""
    fig = go.Figure()
    
    # Pre-test histogram
    fig.add_trace(go.Histogram(
        x=df_merged['score_pre'],
        name='Pre-test',
        opacity=0.7,
        marker_color='#1f77b4',
        nbinsx=max_score + 1
    ))
    
    # Post-test histogram
    fig.add_trace(go.Histogram(
        x=df_merged['score_post'],
        name='Post-test',
        opacity=0.7,
        marker_color='#ff7f0e',
        nbinsx=max_score + 1
    ))
    
    fig.update_layout(
        title='Score Distribution: Pre-test vs Post-test',
        xaxis_title='Score',
        yaxis_title='Number of Students',
        barmode='overlay',
        hovermode='x unified',
        height=500
    )
    
    return fig


def plot_boxplot_comparison(df_merged: pd.DataFrame):
    """Create boxplot comparison of pre and post scores."""
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=df_merged['score_pre'],
        name='Pre-test',
        marker_color='#1f77b4',
        boxmean='sd'
    ))
    
    fig.add_trace(go.Box(
        y=df_merged['score_post'],
        name='Post-test',
        marker_color='#ff7f0e',
        boxmean='sd'
    ))
    
    fig.update_layout(
        title='Score Comparison (Boxplot with Mean ± SD)',
        yaxis_title='Score',
        height=500
    )
    
    return fig


def plot_paired_lines(df_merged: pd.DataFrame):
    """Create spaghetti plot showing individual student trajectories."""
    fig = go.Figure()
    
    # Add line for each student (sample if too many)
    sample_size = min(100, len(df_merged))
    df_sample = df_merged.sample(n=sample_size) if len(df_merged) > 100 else df_merged
    
    for idx, row in df_sample.iterrows():
        fig.add_trace(go.Scatter(
            x=['Pre-test', 'Post-test'],
            y=[row['score_pre'], row['score_post']],
            mode='lines+markers',
            line=dict(width=1, color='lightgray'),
            marker=dict(size=4),
            showlegend=False,
            hovertext=f"{row['name']}<br>Gain: {row['absolute_gain']:+d}",
            hoverinfo='text'
        ))
    
    # Highlight top and bottom performers
    top_student = df_merged.nlargest(1, 'absolute_gain').iloc[0]
    bottom_student = df_merged.nsmallest(1, 'absolute_gain').iloc[0]
    
    fig.add_trace(go.Scatter(
        x=['Pre-test', 'Post-test'],
        y=[top_student['score_pre'], top_student['score_post']],
        mode='lines+markers',
        line=dict(width=3, color='green'),
        marker=dict(size=8),
        name=f'Top: {top_student["name"]}',
        hovertext=f"Gain: {top_student['absolute_gain']:+d}"
    ))
    
    fig.add_trace(go.Scatter(
        x=['Pre-test', 'Post-test'],
        y=[bottom_student['score_pre'], bottom_student['score_post']],
        mode='lines+markers',
        line=dict(width=3, color='red'),
        marker=dict(size=8),
        name=f'Bottom: {bottom_student["name"]}',
        hovertext=f"Gain: {bottom_student['absolute_gain']:+d}"
    ))
    
    fig.update_layout(
        title=f'Individual Student Trajectories ({sample_size} {"sampled" if len(df_merged) > 100 else "students"})',
        yaxis_title='Score',
        height=600,
        hovermode='closest'
    )
    
    return fig


def plot_gain_waterfall(df_merged: pd.DataFrame):
    """Create waterfall chart of absolute gains."""
    df_sorted = df_merged.sort_values('absolute_gain', ascending=False).reset_index(drop=True)
    
    # Limit to top/bottom performers if too many students
    if len(df_sorted) > 50:
        df_display = pd.concat([df_sorted.head(25), df_sorted.tail(25)])
    else:
        df_display = df_sorted
    
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' 
              for x in df_display['absolute_gain']]
    
    fig = go.Figure(go.Bar(
        x=df_display.index,
        y=df_display['absolute_gain'],
        marker_color=colors,
        text=df_display['absolute_gain'],
        textposition='outside',
        hovertext=df_display['name'],
        hoverinfo='text+y'
    ))
    
    fig.update_layout(
        title='Absolute Gains (Sorted Descending)',
        xaxis_title='Student Index (sorted)',
        yaxis_title='Absolute Gain (points)',
        height=500,
        showlegend=False
    )
    
    return fig


def plot_normalized_gain_histogram(df_merged: pd.DataFrame):
    """Create histogram of normalized gains."""
    normalized_gains = df_merged['normalized_gain'].dropna()
    
    fig = go.Figure(go.Histogram(
        x=normalized_gains,
        nbinsx=30,
        marker_color='#2ca02c',
        opacity=0.8
    ))
    
    # Add mean line
    mean_gain = normalized_gains.mean()
    fig.add_vline(
        x=mean_gain,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_gain:.2%}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title='Distribution of Normalized Gains (Hake\'s Gain)',
        xaxis_title='Normalized Gain',
        yaxis_title='Number of Students',
        height=500
    )
    
    return fig


def plot_question_comparison(question_stats: pd.DataFrame):
    """Create bar chart comparing question performance."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=question_stats['question'],
        y=question_stats['pct_correct_pre'],
        name='Pre-test',
        marker_color='#1f77b4'
    ))
    
    fig.add_trace(go.Bar(
        x=question_stats['question'],
        y=question_stats['pct_correct_post'],
        name='Post-test',
        marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        title='Question-Level Performance: Pre vs Post',
        xaxis_title='Question',
        yaxis_title='Percent Correct (%)',
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    
    return fig


def plot_question_improvement(question_stats: pd.DataFrame):
    """Create bar chart of question improvement deltas."""
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' 
              for x in question_stats['improvement']]
    
    fig = go.Figure(go.Bar(
        x=question_stats['question'],
        y=question_stats['improvement'],
        marker_color=colors,
        text=question_stats['improvement'].round(1),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Question Improvement (Post % - Pre %)',
        xaxis_title='Question',
        yaxis_title='Improvement (%)',
        height=500,
        showlegend=False
    )
    
    return fig


def plot_transition_heatmap(question_stats: pd.DataFrame):
    """Create heatmap of transition categories."""
    transition_cols = ['PreWrong_PostRight', 'PreRight_PostRight', 
                       'PreWrong_PostWrong', 'PreRight_PostWrong']
    
    z_data = question_stats[transition_cols].values.T
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=question_stats['question'],
        y=['Wrong→Right<br>(Learned)', 'Right→Right<br>(Mastered)', 
           'Wrong→Wrong<br>(Struggling)', 'Right→Wrong<br>(Forgot)'],
        colorscale='Blues',
        text=z_data,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{y}<br>%{x}: %{z} students<extra></extra>'
    ))
    
    fig.update_layout(
        title='Transition Pattern Heatmap (Count of Students)',
        xaxis_title='Question',
        yaxis_title='Transition Type',
        height=500
    )
    
    return fig


def plot_statistical_summary(class_stats: Dict):
    """Create visualization of statistical test results."""
    fig = go.Figure()
    
    # Mean with confidence interval
    fig.add_trace(go.Bar(
        x=['Pre-test', 'Post-test'],
        y=[class_stats['mean_pre'], class_stats['mean_post']],
        error_y=dict(
            type='data',
            array=[class_stats['std_pre'], class_stats['std_post']],
            visible=True
        ),
        marker_color=['#1f77b4', '#ff7f0e'],
        text=[f"{class_stats['mean_pre']:.2f}", f"{class_stats['mean_post']:.2f}"],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Mean Scores ± SD (p-value: {class_stats['p_value']:.4f})",
        yaxis_title='Mean Score',
        height=500,
        showlegend=False
    )
    
    return fig


def generate_pdf_reports(results: Dict, class_name: str, temp_dir: Path):
    """Generate PDF reports and return file paths."""
    pdf_gen = PDFReportGenerator()
    
    # Create chart images
    charts = {}
    
    # Class-level charts
    try:
        # Histogram
        buf = create_matplotlib_chart(
            'histogram',
            {'pre': results['df_merged']['score_pre'], 
             'post': results['df_merged']['score_post']},
            bins=results['class_stats']['max_score'],
            title='Score Distribution: Pre vs Post'
        )
        hist_path = temp_dir / 'histogram.png'
        with open(hist_path, 'wb') as f:
            f.write(buf.getvalue())
        charts['class_histogram'] = str(hist_path)
        
        # Boxplot
        buf = create_matplotlib_chart(
            'boxplot',
            {'pre': results['df_merged']['score_pre'],
             'post': results['df_merged']['score_post']},
            title='Score Comparison (Boxplot)'
        )
        box_path = temp_dir / 'boxplot.png'
        with open(box_path, 'wb') as f:
            f.write(buf.getvalue())
        charts['boxplot'] = str(box_path)
        
        # Question comparison
        buf = create_matplotlib_chart(
            'bar',
            {
                'x': results['question_stats']['question'].tolist(),
                'y_pre': results['question_stats']['pct_correct_pre'].tolist(),
                'y_post': results['question_stats']['pct_correct_post'].tolist()
            },
            title='Question Performance: Pre vs Post',
            xlabel='Question',
            ylabel='Percent Correct (%)'
        )
        q_path = temp_dir / 'question_comparison.png'
        with open(q_path, 'wb') as f:
            f.write(buf.getvalue())
        charts['question_comparison'] = str(q_path)
        
    except Exception as e:
        logger.error(f"Error creating charts: {e}")
        charts = {}
    
    # Generate full PDF
    full_pdf_path = temp_dir / 'full_report.pdf'
    pdf_gen.generate_full_report(
        output_path=str(full_pdf_path),
        df_merged=results['df_merged'],
        class_stats=results['class_stats'],
        question_stats=results['question_stats'],
        faculty_rating=results['faculty_rating'],
        class_summary_text=results['class_summary'],
        question_columns=results['question_columns'],
        charts=charts,
        class_name=class_name,
        include_individual_pages=st.session_state.get('include_individual', True)
    )
    
    # Generate compact PDF
    compact_pdf_path = temp_dir / 'compact_report.pdf'
    pdf_gen.generate_compact_report(
        output_path=str(compact_pdf_path),
        df_merged=results['df_merged'],
        class_stats=results['class_stats'],
        question_stats=results['question_stats'],
        faculty_rating=results['faculty_rating'],
        class_summary_text=results['class_summary'],
        charts=charts,
        class_name=class_name
    )
    
    return full_pdf_path, compact_pdf_path


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Pre-test / Post-test Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File uploads
        st.subheader("📁 Upload Test Files")
        pre_file = st.file_uploader(
            "Pre-test Excel File",
            type=['xlsx'],
            help="Upload the pre-test Excel file following the required template"
        )
        
        post_file = st.file_uploader(
            "Post-test Excel File",
            type=['xlsx'],
            help="Upload the post-test Excel file following the required template"
        )
        
        st.markdown("---")
        
        # Sample template download
        st.subheader("📥 Template")
        create_sample_template_download()
        
        st.markdown("---")
        
        # Optional settings
        st.subheader("🎯 Analysis Options")
        class_name = st.text_input(
            "Class/Lecture Name",
            value="Pre-test / Post-test Analysis",
            help="Optional: Name of the class or lecture for the report"
        )
        
        include_individual = st.checkbox(
            "Include Individual Student Pages in PDF",
            value=True,
            help="Include detailed analysis for each student in the full PDF report"
        )
        
        st.session_state['include_individual'] = include_individual
        
        st.markdown("---")
        st.caption("📖 Tip: Both files must follow the exact same template format")
    
    # Main content area
    if pre_file is None or post_file is None:
        st.info("👆 Please upload both pre-test and post-test Excel files to begin analysis.")
        
        # Show instructions
        with st.expander("📋 Instructions & File Format Requirements", expanded=True):
            st.markdown("""
            ### Required File Format
            
            Both Excel files must have **identical column structure**:
            
            **Required Headers:**
            - `name` - Student's full name (text)
            - `ticket_no` - Unique student ID (text or number)
            - `q1, q2, q3, ...` - Question columns (values must be 0 or 1)
            
            **Example:**
            
            | name | ticket_no | q1 | q2 | q3 | q4 | q5 |
            |------|-----------|----|----|----|----|-----|
            | Asha Sharma | T001 | 1 | 0 | 0 | 1 | 1 |
            | Ravi Kumar | T002 | 0 | 0 | 1 | 0 | 1 |
            
            **Validation Rules:**
            - Both files must have the same question columns
            - Question values must be 0 (incorrect) or 1 (correct)
            - No duplicate ticket numbers within each file
            - Only students appearing in BOTH files will be analyzed
            
            **Download the sample template** from the sidebar to see the exact format.
            """)
        
        return
    
    # Load and validate files
    st.header("Step 1: Validation")
    
    with st.spinner("Validating files..."):
        # Load files
        pre_bytes = pre_file.read()
        post_bytes = post_file.read()
        
        df_pre, val_pre = load_and_validate_file(pre_bytes, pre_file.name, "Pre-test")
        df_post, val_post = load_and_validate_file(post_bytes, post_file.name, "Post-test")
        
        # Check if loading was successful
        if df_pre is None or df_post is None:
            st.error("❌ Error loading Excel files. Please ensure they are valid .xlsx files.")
            return
        
        # Display validation results
        all_valid = True
        
        # Pre-test validation
        if not val_pre.is_valid:
            all_valid = False
            st.markdown(f'<div class="error-box"><strong>❌ Pre-test Validation Failed</strong><br>', unsafe_allow_html=True)
            for error in val_pre.errors:
                st.markdown(f"• {error}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-box">✅ Pre-test validated successfully ({len(val_pre.question_columns)} questions, {len(df_pre)} students)</div>', unsafe_allow_html=True)
        
        if val_pre.warnings:
            st.markdown('<div class="warning-box"><strong>⚠️ Pre-test Warnings</strong><br>', unsafe_allow_html=True)
            for warning in val_pre.warnings:
                st.markdown(f"• {warning}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Post-test validation
        if not val_post.is_valid:
            all_valid = False
            st.markdown('<div class="error-box"><strong>❌ Post-test Validation Failed</strong><br>', unsafe_allow_html=True)
            for error in val_post.errors:
                st.markdown(f"• {error}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-box">✅ Post-test validated successfully ({len(val_post.question_columns)} questions, {len(df_post)} students)</div>', unsafe_allow_html=True)
        
        if val_post.warnings:
            st.markdown('<div class="warning-box"><strong>⚠️ Post-test Warnings</strong><br>', unsafe_allow_html=True)
            for warning in val_post.warnings:
                st.markdown(f"• {warning}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if not all_valid:
            st.stop()
        
        # Validate matching structure
        structure_errors = validate_matching_structure(
            df_pre, df_post, val_pre.question_columns, val_post.question_columns
        )
        
        if structure_errors:
            st.markdown('<div class="error-box"><strong>❌ File Structure Mismatch</strong><br>', unsafe_allow_html=True)
            for error in structure_errors:
                st.markdown(f"• {error}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.stop()
    
    st.success("✅ All validation checks passed!")
    st.markdown("---")
    
    # Perform analysis
    st.header("Step 2: Analysis")
    
    with st.spinner("Computing analytics..."):
        results = perform_analysis(
            pre_bytes, post_bytes,
            val_pre.question_columns, val_post.question_columns
        )
        
        if results is None:
            st.error("❌ No students found in both pre and post tests. Cannot perform analysis.")
            st.stop()
    
    n_matched = len(results['df_merged'])
    n_discarded = len(results['df_discarded'])
    
    st.success(f"✅ Analysis complete! {n_matched} students matched, {n_discarded} discarded.")
    
    # Show discarded students
    if n_discarded > 0:
        with st.expander(f"⚠️ {n_discarded} Students Discarded (click to view)"):
            st.dataframe(results['df_discarded'], use_container_width=True)
            
            # Download button for discarded list
            csv_discarded = results['df_discarded'].to_csv(index=False)
            st.download_button(
                label="📥 Download Discarded List (CSV)",
                data=csv_discarded,
                file_name="discarded_students.csv",
                mime="text/csv"
            )
    
    st.markdown("---")
    
    # Display KPIs
    st.header("Step 3: Key Performance Indicators")
    display_kpi_cards(
        results['class_stats'],
        results['faculty_rating'],
        n_matched,
        n_discarded
    )
    
    st.markdown("---")
    
    # Visualizations in tabs
    st.header("Step 4: Interactive Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["📈 Class-Level", "📊 Question-Level", "👤 Student-Level"])
    
    with tab1:
        st.subheader("Class-Level Analysis")
        
        # Score distribution
        st.plotly_chart(
            plot_score_distribution(results['df_merged'], results['class_stats']['max_score']),
            use_container_width=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Boxplot
            st.plotly_chart(
                plot_boxplot_comparison(results['df_merged']),
                use_container_width=True
            )
        
        with col2:
            # Statistical summary
            st.plotly_chart(
                plot_statistical_summary(results['class_stats']),
                use_container_width=True
            )
        
        # Paired lines
        st.plotly_chart(
            plot_paired_lines(results['df_merged']),
            use_container_width=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gain waterfall
            st.plotly_chart(
                plot_gain_waterfall(results['df_merged']),
                use_container_width=True
            )
        
        with col2:
            # Normalized gain histogram
            st.plotly_chart(
                plot_normalized_gain_histogram(results['df_merged']),
                use_container_width=True
            )
    
    with tab2:
        st.subheader("Question-Level Analysis")
        
        # Question comparison
        st.plotly_chart(
            plot_question_comparison(results['question_stats']),
            use_container_width=True
        )
        
        # Improvement delta
        st.plotly_chart(
            plot_question_improvement(results['question_stats']),
            use_container_width=True
        )
        
        # Transition heatmap
        st.plotly_chart(
            plot_transition_heatmap(results['question_stats']),
            use_container_width=True
        )
        
        # Question statistics table
        st.subheader("Detailed Question Statistics")
        st.dataframe(
            results['question_stats'].style.background_gradient(
                subset=['improvement'], cmap='RdYlGn', vmin=-50, vmax=50
            ),
            use_container_width=True
        )
    
    with tab3:
        st.subheader("Individual Student Analysis")
        
        # Search/filter
        search_query = st.text_input(
            "🔍 Search by name or ticket number",
            placeholder="Enter student name or ticket number..."
        )
        
        if search_query:
            filtered_df = results['df_merged'][
                results['df_merged']['name'].str.contains(search_query, case=False, na=False) |
                results['df_merged']['ticket_no'].astype(str).str.contains(search_query, case=False, na=False)
            ]
            
            if len(filtered_df) == 0:
                st.warning("No students found matching your search.")
            else:
                st.success(f"Found {len(filtered_df)} student(s)")
                
                # Display results
                for idx, row in filtered_df.iterrows():
                    with st.expander(f"📄 {row['name']} ({row['ticket_no']})"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Pre-test Score", f"{int(row['score_pre'])}/{results['class_stats']['max_score']}")
                        col2.metric("Post-test Score", f"{int(row['score_post'])}/{results['class_stats']['max_score']}")
                        col3.metric("Absolute Gain", f"{int(row['absolute_gain']):+d}")
                        
                        # Analysis text
                        st.markdown("**Detailed Analysis:**")
                        analysis_text = generate_student_analysis_text(row, results['question_columns'])
                        st.text(analysis_text)
        else:
            # Show top/bottom performers
            st.subheader("Top 5 Improvers")
            top_5 = results['df_merged'].nlargest(5, 'absolute_gain')[
                ['name', 'ticket_no', 'score_pre', 'score_post', 'absolute_gain', 'normalized_gain']
            ]
            st.dataframe(top_5, use_container_width=True)
            
            st.subheader("Bottom 5 (Regressions)")
            bottom_5 = results['df_merged'].nsmallest(5, 'absolute_gain')[
                ['name', 'ticket_no', 'score_pre', 'score_post', 'absolute_gain', 'normalized_gain']
            ]
            st.dataframe(bottom_5, use_container_width=True)
    
    st.markdown("---")
    
    # Text analysis
    st.header("Step 5: Summary & Insights")
    
    with st.expander("📝 Class Summary (Human-Readable)", expanded=True):
        st.text(results['class_summary'])
    
    with st.expander("🎯 Faculty Rating Details"):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.metric(
                "Faculty Rating",
                f"{results['faculty_rating']['score']}/100",
                help="Overall effectiveness rating based on normalized gains and student improvement"
            )
        
        with col2:
            st.markdown(f"**{results['faculty_rating']['interpretation']}**")
            st.text(results['faculty_rating']['explanation'])
    
    st.markdown("---")
    
    # Downloads
    st.header("Step 6: Download Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download merged dataset
        merged_csv = results['df_merged'].to_csv(index=False)
        st.download_button(
            label="📥 Download Merged Data (CSV)",
            data=merged_csv,
            file_name="merged_analysis.csv",
            mime="text/csv",
            help="Download complete dataset with all computed metrics"
        )
    
    with col2:
        # Download discarded list
        if n_discarded > 0:
            discarded_csv = results['df_discarded'].to_csv(index=False)
            st.download_button(
                label="📥 Download Discarded List (CSV)",
                data=discarded_csv,
                file_name="discarded_students.csv",
                mime="text/csv"
            )
    
    with col3:
        # Download Excel with all data
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            results['df_merged'].to_excel(writer, sheet_name='Merged Data', index=False)
            results['question_stats'].to_excel(writer, sheet_name='Question Stats', index=False)
            if n_discarded > 0:
                results['df_discarded'].to_excel(writer, sheet_name='Discarded', index=False)
        buffer.seek(0)
        
        st.download_button(
            label="📥 Download Complete Report (Excel)",
            data=buffer,
            file_name="complete_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # PDF generation
    st.subheader("📄 PDF Reports")
    
    with st.spinner("Generating PDF reports... This may take a minute."):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                full_pdf, compact_pdf = generate_pdf_reports(
                    results, class_name, temp_path
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    with open(full_pdf, 'rb') as f:
                        st.download_button(
                            label="📥 Download Full PDF Report",
                            data=f.read(),
                            file_name="full_report.pdf",
                            mime="application/pdf",
                            help="Complete report with individual student pages" if include_individual else "Complete report"
                        )
                
                with col2:
                    with open(compact_pdf, 'rb') as f:
                        st.download_button(
                            label="📥 Download Compact PDF Report",
                            data=f.read(),
                            file_name="compact_report.pdf",
                            mime="application/pdf",
                            help="Summary report with class-level analysis only"
                        )
        except Exception as e:
            st.error(f"Error generating PDF reports: {e}")
            logger.error(f"PDF generation error: {e}", exc_info=True)
    
    st.markdown("---")
    st.success("✅ Analysis complete! All reports are ready for download.")
    
    # Footer
    st.markdown("---")
    st.caption("Pre-test / Post-test Analysis App | Built with Streamlit | © 2025")


if __name__ == "__main__":
    main()
