"""
Utility functions for pre-test/post-test analysis.

This module contains all validation, data processing, and statistical
analysis functions for the pre/post test comparison application.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
REQUIRED_COLUMNS = ["name", "ticket_no"]
QUESTION_PATTERN = re.compile(r"^q\d+$", re.IGNORECASE)
VALID_ANSWER_VALUES = {0, 1}
MASTERY_THRESHOLD = 0.40  # 40% for penalty calculation


@dataclass
class ValidationResult:
    """Result of file validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    question_columns: List[str]
    

@dataclass
class AnalysisResults:
    """Container for all analysis results."""
    df_merged: pd.DataFrame
    df_discarded: pd.DataFrame
    class_stats: Dict[str, Any]
    question_stats: pd.DataFrame
    student_stats: pd.DataFrame
    transition_counts: pd.DataFrame
    faculty_rating: Dict[str, Any]


def validate_excel_file(df: pd.DataFrame, file_label: str = "File") -> ValidationResult:
    """
    Validate that the Excel file matches the required template.
    
    Args:
        df: DataFrame loaded from Excel file
        file_label: Label for error messages (e.g., "Pre-test" or "Post-test")
        
    Returns:
        ValidationResult with validation status and any errors/warnings
    """
    errors = []
    warnings = []
    question_columns = []
    
    # Check if DataFrame is empty
    if df.empty:
        errors.append(f"{file_label}: File is empty")
        return ValidationResult(False, errors, warnings, question_columns)
    
    # Normalize column names (strip whitespace, lowercase for comparison)
    df.columns = df.columns.str.strip()
    column_names_lower = [col.lower() for col in df.columns]
    
    # Check required columns
    for req_col in REQUIRED_COLUMNS:
        if req_col.lower() not in column_names_lower:
            errors.append(f"{file_label}: Missing required column '{req_col}'")
    
    # Find question columns
    for col in df.columns:
        if QUESTION_PATTERN.match(col):
            question_columns.append(col)
    
    if not question_columns:
        errors.append(f"{file_label}: No question columns found (expected q1, q2, q3, ...)")
        return ValidationResult(False, errors, warnings, question_columns)
    
    # Sort question columns numerically
    question_columns = sorted(question_columns, key=lambda x: int(re.findall(r'\d+', x)[0]))
    
    # Check for duplicate ticket numbers
    if 'ticket_no' in df.columns:
        ticket_col = df['ticket_no'].astype(str).str.strip()
        duplicates = ticket_col[ticket_col.duplicated()].unique()
        if len(duplicates) > 0:
            errors.append(f"{file_label}: Duplicate ticket numbers found: {', '.join(duplicates[:5])}")
    
    # Validate question column values
    for q_col in question_columns:
        # Check for non-numeric or invalid values
        invalid_rows = []
        for idx, val in enumerate(df[q_col]):
            try:
                num_val = int(val)
                if num_val not in VALID_ANSWER_VALUES:
                    invalid_rows.append(idx + 2)  # +2 for header and 0-indexing
            except (ValueError, TypeError):
                invalid_rows.append(idx + 2)
        
        if invalid_rows:
            errors.append(
                f"{file_label}: Column '{q_col}' has invalid values (must be 0 or 1) "
                f"in rows: {invalid_rows[:10]}"  # Show first 10
            )
    
    # Check for missing values in required columns
    if 'name' in df.columns:
        missing_mask = df['name'].isna()
        if missing_mask.any():
            missing_count = int(missing_mask.sum())
            warnings.append(f"{file_label}: {missing_count} rows with missing names")
    
    if 'ticket_no' in df.columns:
        missing_mask = df['ticket_no'].isna()
        if missing_mask.any():
            missing_count = int(missing_mask.sum())
            errors.append(f"{file_label}: {missing_count} rows with missing ticket numbers")
    
    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings, question_columns)


def validate_matching_structure(
    df_pre: pd.DataFrame, 
    df_post: pd.DataFrame,
    q_cols_pre: List[str],
    q_cols_post: List[str]
) -> List[str]:
    """
    Validate that pre and post test files have matching structure.
    
    Args:
        df_pre: Pre-test DataFrame
        df_post: Post-test DataFrame
        q_cols_pre: Question columns from pre-test
        q_cols_post: Question columns from post-test
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Check same number of questions
    if len(q_cols_pre) != len(q_cols_post):
        errors.append(
            f"Pre-test has {len(q_cols_pre)} questions, "
            f"but post-test has {len(q_cols_post)} questions. They must match."
        )
    
    # Check same question column names
    if q_cols_pre != q_cols_post:
        missing_in_post = set(q_cols_pre) - set(q_cols_post)
        missing_in_pre = set(q_cols_post) - set(q_cols_pre)
        
        if missing_in_post:
            errors.append(f"Questions in pre-test but not in post-test: {missing_in_post}")
        if missing_in_pre:
            errors.append(f"Questions in post-test but not in pre-test: {missing_in_pre}")
    
    return errors


def match_participants(
    df_pre: pd.DataFrame,
    df_post: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Match participants who appear in both pre and post tests.
    
    Args:
        df_pre: Pre-test DataFrame
        df_post: Post-test DataFrame
        
    Returns:
        Tuple of (df_pre_filtered, df_post_filtered, df_discarded)
    """
    # Normalize ticket numbers
    df_pre = df_pre.copy()
    df_post = df_post.copy()
    df_pre['ticket_no'] = df_pre['ticket_no'].astype(str).str.strip()
    df_post['ticket_no'] = df_post['ticket_no'].astype(str).str.strip()
    
    # Find intersection
    tickets_pre = set(df_pre['ticket_no'])
    tickets_post = set(df_post['ticket_no'])
    tickets_both = tickets_pre & tickets_post
    
    # Create filtered DataFrames
    df_pre_filtered = df_pre[df_pre['ticket_no'].isin(list(tickets_both))].copy()
    df_post_filtered = df_post[df_post['ticket_no'].isin(list(tickets_both))].copy()
    
    # Sort by ticket_no for consistent ordering
    df_pre_filtered = df_pre_filtered.sort_values(by='ticket_no').reset_index(drop=True)
    df_post_filtered = df_post_filtered.sort_values(by='ticket_no').reset_index(drop=True)
    
    # Create discarded DataFrame
    discarded_records = []
    
    # Students only in pre-test
    for ticket in tickets_pre - tickets_both:
        name_series = df_pre[df_pre['ticket_no'] == ticket]['name']
        discarded_records.append({
            'ticket_no': ticket,
            'name': name_series.iloc[0] if len(name_series) > 0 else 'Unknown',
            'reason': 'Missing post-test'
        })
    
    # Students only in post-test
    for ticket in tickets_post - tickets_both:
        name_series = df_post[df_post['ticket_no'] == ticket]['name']
        discarded_records.append({
            'ticket_no': ticket,
            'name': name_series.iloc[0] if len(name_series) > 0 else 'Unknown',
            'reason': 'Missing pre-test'
        })
    
    df_discarded = pd.DataFrame(discarded_records)
    
    logger.info(f"Matched {len(tickets_both)} participants")
    logger.info(f"Discarded {len(discarded_records)} participants")
    
    return df_pre_filtered, df_post_filtered, df_discarded


def compute_scores_and_gains(
    df_pre: pd.DataFrame,
    df_post: pd.DataFrame,
    question_columns: List[str]
) -> pd.DataFrame:
    """
    Compute scores, gains, and normalized gains for matched participants.
    
    Args:
        df_pre: Filtered pre-test DataFrame
        df_post: Filtered post-test DataFrame
        question_columns: List of question column names
        
    Returns:
        Merged DataFrame with computed metrics
    """
    # Ensure both DataFrames are sorted by ticket_no
    df_pre = df_pre.sort_values('ticket_no').reset_index(drop=True)
    df_post = df_post.sort_values('ticket_no').reset_index(drop=True)
    
    max_score = len(question_columns)
    
    # Compute scores
    df_pre['score_pre'] = df_pre[question_columns].astype(int).sum(axis=1)
    df_post['score_post'] = df_post[question_columns].astype(int).sum(axis=1)
    
    # Create merged DataFrame
    df_merged = pd.DataFrame({
        'ticket_no': df_pre['ticket_no'],
        'name': df_pre['name'],
        'score_pre': df_pre['score_pre'],
        'score_post': df_post['score_post']
    })
    
    # Add question-level data
    for q_col in question_columns:
        df_merged[f"{q_col}_pre"] = df_pre[q_col].astype(int).values
        df_merged[f"{q_col}_post"] = df_post[q_col].astype(int).values
    
    # Compute metrics
    df_merged['absolute_gain'] = df_merged['score_post'] - df_merged['score_pre']
    
    # Compute normalized gain (Hake's gain)
    # Formula: (post - pre) / (max - pre) if denominator > 0
    denominator = max_score - df_merged['score_pre']
    df_merged['normalized_gain'] = np.where(
        denominator > 0,
        df_merged['absolute_gain'] / denominator,
        np.nan
    )
    
    # Compute percentages
    df_merged['percent_correct_pre'] = (df_merged['score_pre'] / max_score) * 100
    df_merged['percent_correct_post'] = (df_merged['score_post'] / max_score) * 100
    
    # Compute transition categories for each question
    for q_col in question_columns:
        pre_col = f"{q_col}_pre"
        post_col = f"{q_col}_post"
        
        # Determine transition category
        conditions = [
            (df_merged[pre_col] == 1) & (df_merged[post_col] == 1),
            (df_merged[pre_col] == 1) & (df_merged[post_col] == 0),
            (df_merged[pre_col] == 0) & (df_merged[post_col] == 0),
            (df_merged[pre_col] == 0) & (df_merged[post_col] == 1)
        ]
        choices = [
            'PreRight_PostRight',
            'PreRight_PostWrong',
            'PreWrong_PostWrong',
            'PreWrong_PostRight'
        ]
        df_merged[f"{q_col}_transition"] = np.select(conditions, choices, default='Unknown')
    
    return df_merged


def compute_class_statistics(df_merged: pd.DataFrame, max_score: int) -> Dict[str, Any]:
    """
    Compute class-level statistics.
    
    Args:
        df_merged: Merged DataFrame with scores and gains
        max_score: Maximum possible score
        
    Returns:
        Dictionary with class-level statistics
    """
    stats_dict = {}
    
    # Basic statistics
    stats_dict['n_students'] = len(df_merged)
    stats_dict['max_score'] = max_score
    
    # Pre-test statistics
    stats_dict['mean_pre'] = df_merged['score_pre'].mean()
    stats_dict['std_pre'] = df_merged['score_pre'].std()
    stats_dict['median_pre'] = df_merged['score_pre'].median()
    
    # Post-test statistics
    stats_dict['mean_post'] = df_merged['score_post'].mean()
    stats_dict['std_post'] = df_merged['score_post'].std()
    stats_dict['median_post'] = df_merged['score_post'].median()
    
    # Gain statistics
    stats_dict['mean_gain'] = df_merged['absolute_gain'].mean()
    stats_dict['std_gain'] = df_merged['absolute_gain'].std()
    stats_dict['median_gain'] = df_merged['absolute_gain'].median()
    
    # Normalized gain statistics (ignoring NaN)
    normalized_gains = df_merged['normalized_gain'].dropna()
    stats_dict['mean_normalized_gain'] = normalized_gains.mean() if len(normalized_gains) > 0 else 0
    stats_dict['std_normalized_gain'] = normalized_gains.std() if len(normalized_gains) > 0 else 0
    
    # Student improvement categories
    improved = (df_merged['absolute_gain'] > 0).sum()
    unchanged = (df_merged['absolute_gain'] == 0).sum()
    regressed = (df_merged['absolute_gain'] < 0).sum()
    
    stats_dict['n_improved'] = improved
    stats_dict['n_unchanged'] = unchanged
    stats_dict['n_regressed'] = regressed
    stats_dict['pct_improved'] = (improved / len(df_merged)) * 100
    stats_dict['pct_unchanged'] = (unchanged / len(df_merged)) * 100
    stats_dict['pct_regressed'] = (regressed / len(df_merged)) * 100
    
    # Paired t-test
    try:
        t_stat, p_value = stats.ttest_rel(df_merged['score_post'], df_merged['score_pre'])
        stats_dict['t_statistic'] = float(t_stat)
        stats_dict['p_value'] = float(p_value)
        stats_dict['is_significant'] = bool(float(p_value) < 0.05)
    except Exception as e:
        logger.warning(f"Could not compute t-test: {e}")
        stats_dict['t_statistic'] = np.nan
        stats_dict['p_value'] = np.nan
        stats_dict['is_significant'] = False
    
    # Wilcoxon test (non-parametric alternative)
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
            df_merged['score_post'], 
            df_merged['score_pre'],
            alternative='greater'
        )
        stats_dict['wilcoxon_statistic'] = wilcoxon_stat
        stats_dict['wilcoxon_p_value'] = wilcoxon_p
    except Exception as e:
        logger.warning(f"Could not compute Wilcoxon test: {e}")
        stats_dict['wilcoxon_statistic'] = np.nan
        stats_dict['wilcoxon_p_value'] = np.nan
    
    # Effect size (Cohen's d for paired samples)
    # d = mean_difference / std_difference
    differences = df_merged['score_post'] - df_merged['score_pre']
    cohens_d = differences.mean() / differences.std() if differences.std() > 0 else 0
    stats_dict['cohens_d'] = cohens_d
    
    # Confidence intervals
    ci_95 = stats.t.interval(
        0.95,
        len(df_merged) - 1,
        loc=stats_dict['mean_gain'],
        scale=stats.sem(df_merged['absolute_gain'])
    )
    stats_dict['ci_95_lower'] = ci_95[0]
    stats_dict['ci_95_upper'] = ci_95[1]
    
    return stats_dict


def compute_question_statistics(
    df_merged: pd.DataFrame,
    question_columns: List[str]
) -> pd.DataFrame:
    """
    Compute per-question statistics.
    
    Args:
        df_merged: Merged DataFrame
        question_columns: List of question column names
        
    Returns:
        DataFrame with question-level statistics
    """
    question_stats = []
    
    for q_col in question_columns:
        pre_col = f"{q_col}_pre"
        post_col = f"{q_col}_post"
        trans_col = f"{q_col}_transition"
        
        # Percent correct
        pct_correct_pre = (df_merged[pre_col].sum() / len(df_merged)) * 100
        pct_correct_post = (df_merged[post_col].sum() / len(df_merged)) * 100
        improvement = pct_correct_post - pct_correct_pre
        
        # Transition counts
        transitions = df_merged[trans_col].value_counts()
        
        question_stats.append({
            'question': q_col,
            'pct_correct_pre': pct_correct_pre,
            'pct_correct_post': pct_correct_post,
            'improvement': improvement,
            'PreRight_PostRight': transitions.get('PreRight_PostRight', 0),
            'PreRight_PostWrong': transitions.get('PreRight_PostWrong', 0),
            'PreWrong_PostWrong': transitions.get('PreWrong_PostWrong', 0),
            'PreWrong_PostRight': transitions.get('PreWrong_PostRight', 0),
        })
    
    df_stats = pd.DataFrame(question_stats)
    
    # Item difficulty (average of pre and post)
    df_stats['difficulty'] = (df_stats['pct_correct_pre'] + df_stats['pct_correct_post']) / 2
    
    return df_stats


def compute_faculty_rating(
    class_stats: Dict[str, Any],
    df_merged: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compute faculty rating score (0-100) based on class performance.
    
    Formula:
        avg_normalized_gain = mean(normalized_gain ignoring NaN)
        pct_students_improved = % with absolute_gain > 0
        pct_students_regressed = % with absolute_gain < 0
        
        S = 100 * (0.65 * avg_normalized_gain + 
                   0.25 * pct_students_improved + 
                   0.10 * (1 - pct_students_regressed))
        
        rating = round(S) clipped to [0, 100]
        
        Penalty if mean_post% < 40%
    
    Args:
        class_stats: Dictionary of class-level statistics
        df_merged: Merged DataFrame
        
    Returns:
        Dictionary with rating and explanation
    """
    # Extract components
    avg_normalized_gain = class_stats['mean_normalized_gain']
    pct_improved = class_stats['pct_improved'] / 100  # Convert to 0-1
    pct_regressed = class_stats['pct_regressed'] / 100
    mean_post_pct = (class_stats['mean_post'] / class_stats['max_score'])
    
    # Clamp normalized gain to [0, 1]
    avg_normalized_gain = max(0, min(1, avg_normalized_gain))
    
    # Compute preliminary score
    preliminary_score = 100 * (
        0.65 * avg_normalized_gain +
        0.25 * pct_improved +
        0.10 * (1 - pct_regressed)
    )
    
    # Apply penalty if post-test mastery is low
    penalty = 0
    if mean_post_pct < MASTERY_THRESHOLD:
        penalty = (MASTERY_THRESHOLD - mean_post_pct) * 100 * 0.5  # 0.5 multiplier
        penalty = min(penalty, 15)  # Cap penalty at 15 points
    
    final_score = preliminary_score - penalty
    final_score = max(0, min(100, round(final_score)))
    
    # Create explanation
    explanation_parts = [
        f"Average Normalized Gain: {avg_normalized_gain:.2%} (weight: 65%)",
        f"Students Improved: {pct_improved:.1%} (weight: 25%)",
        f"Students Regressed: {pct_regressed:.1%} (inverse weight: 10%)",
        f"Preliminary Score: {preliminary_score:.1f}/100"
    ]
    
    if penalty > 0:
        explanation_parts.append(
            f"Penalty: -{penalty:.1f} points (post-test mastery {mean_post_pct:.1%} < {MASTERY_THRESHOLD:.0%})"
        )
    
    explanation_parts.append(f"Final Rating: {final_score}/100")
    
    # Interpretation
    if final_score >= 80:
        interpretation = "Excellent - High effectiveness demonstrated"
    elif final_score >= 65:
        interpretation = "Good - Solid learning gains achieved"
    elif final_score >= 50:
        interpretation = "Adequate - Moderate improvement shown"
    elif final_score >= 35:
        interpretation = "Needs Improvement - Limited learning gains"
    else:
        interpretation = "Poor - Significant concerns about effectiveness"
    
    return {
        'score': final_score,
        'preliminary_score': preliminary_score,
        'penalty': penalty,
        'components': {
            'normalized_gain': avg_normalized_gain,
            'pct_improved': pct_improved * 100,
            'pct_regressed': pct_regressed * 100
        },
        'explanation': '\n'.join(explanation_parts),
        'interpretation': interpretation
    }


def generate_student_analysis_text(row: pd.Series, question_columns: List[str]) -> str:
    """
    Generate human-readable analysis text for a single student.
    
    Args:
        row: Row from merged DataFrame for one student
        question_columns: List of question column names
        
    Returns:
        Human-readable analysis string
    """
    name = row['name']
    ticket = row['ticket_no']
    score_pre = row['score_pre']
    score_post = row['score_post']
    gain = row['absolute_gain']
    norm_gain = row['normalized_gain']
    max_score = len(question_columns)
    
    # Opening
    text = f"Student: {name} (Ticket: {ticket})\n\n"
    text += f"Pre-test Score: {score_pre}/{max_score} ({row['percent_correct_pre']:.1f}%)\n"
    text += f"Post-test Score: {score_post}/{max_score} ({row['percent_correct_post']:.1f}%)\n"
    text += f"Absolute Gain: {gain:+d} points\n"
    
    if pd.notna(norm_gain):
        text += f"Normalized Gain: {norm_gain:.1%}\n\n"
    else:
        text += "Normalized Gain: N/A (already at maximum)\n\n"
    
    # Overall assessment
    if gain > 0:
        text += f"{name} showed improvement, gaining {gain} points. "
    elif gain < 0:
        text += f"{name} regressed, losing {abs(gain)} points. "
    else:
        text += f"{name} maintained the same score. "
    
    # Question-by-question summary
    learned = []
    forgot = []
    maintained_correct = []
    maintained_incorrect = []
    
    for q_col in question_columns:
        trans = row[f"{q_col}_transition"]
        if trans == 'PreWrong_PostRight':
            learned.append(q_col)
        elif trans == 'PreRight_PostWrong':
            forgot.append(q_col)
        elif trans == 'PreRight_PostRight':
            maintained_correct.append(q_col)
        elif trans == 'PreWrong_PostWrong':
            maintained_incorrect.append(q_col)
    
    text += "\n\nQuestion-by-question breakdown:\n"
    
    if learned:
        text += f"• Learned (wrong→right): {', '.join(learned)}\n"
    if forgot:
        text += f"• Needs review (right→wrong): {', '.join(forgot)}\n"
    if maintained_correct:
        text += f"• Mastered (right→right): {', '.join(maintained_correct)}\n"
    if maintained_incorrect:
        text += f"• Still struggling (wrong→wrong): {', '.join(maintained_incorrect)}\n"
    
    # Recommendations
    text += "\nRecommendations:\n"
    if forgot:
        text += f"• Review concepts covered in {', '.join(forgot[:3])} - these were regressed.\n"
    if maintained_incorrect:
        text += f"• Focus on {', '.join(maintained_incorrect[:3])} - persistent difficulty.\n"
    if len(learned) > len(question_columns) * 0.5:
        text += "• Strong learning progress - keep up the good work!\n"
    
    return text


def generate_class_summary_text(
    class_stats: Dict[str, Any],
    faculty_rating: Dict[str, Any],
    question_stats: pd.DataFrame,
    df_merged: pd.DataFrame
) -> str:
    """
    Generate human-readable class-level summary.
    
    Args:
        class_stats: Class statistics dictionary
        faculty_rating: Faculty rating dictionary
        question_stats: Question-level statistics DataFrame
        df_merged: Merged student data
        
    Returns:
        Summary text string
    """
    n = class_stats['n_students']
    max_score = class_stats['max_score']
    
    text = f"CLASS SUMMARY\n{'='*50}\n\n"
    text += f"Total Students Analyzed: {n}\n"
    text += f"Total Questions: {max_score}\n\n"
    
    text += f"PRE-TEST PERFORMANCE\n"
    text += f"Mean Score: {class_stats['mean_pre']:.2f}/{max_score} "
    text += f"({class_stats['mean_pre']/max_score*100:.1f}%)\n"
    text += f"Standard Deviation: {class_stats['std_pre']:.2f}\n\n"
    
    text += f"POST-TEST PERFORMANCE\n"
    text += f"Mean Score: {class_stats['mean_post']:.2f}/{max_score} "
    text += f"({class_stats['mean_post']/max_score*100:.1f}%)\n"
    text += f"Standard Deviation: {class_stats['std_post']:.2f}\n\n"
    
    text += f"LEARNING GAINS\n"
    text += f"Mean Absolute Gain: {class_stats['mean_gain']:.2f} points\n"
    text += f"Mean Normalized Gain: {class_stats['mean_normalized_gain']:.2%}\n"
    text += f"95% CI for Gain: [{class_stats['ci_95_lower']:.2f}, {class_stats['ci_95_upper']:.2f}]\n\n"
    
    text += f"STUDENT IMPROVEMENT DISTRIBUTION\n"
    text += f"Improved: {class_stats['n_improved']} ({class_stats['pct_improved']:.1f}%)\n"
    text += f"Unchanged: {class_stats['n_unchanged']} ({class_stats['pct_unchanged']:.1f}%)\n"
    text += f"Regressed: {class_stats['n_regressed']} ({class_stats['pct_regressed']:.1f}%)\n\n"
    
    text += f"STATISTICAL SIGNIFICANCE\n"
    text += f"Paired t-test p-value: {class_stats['p_value']:.4f}\n"
    text += f"Result: {'Statistically significant' if class_stats['is_significant'] else 'Not significant'} "
    text += f"(α=0.05)\n"
    text += f"Cohen's d (effect size): {class_stats['cohens_d']:.3f}\n\n"
    
    # Interpretation
    text += f"INTERPRETATION\n"
    if class_stats['mean_gain'] > 0 and class_stats['is_significant']:
        text += "The intervention/lecture was EFFECTIVE. Students showed statistically significant improvement.\n"
    elif class_stats['mean_gain'] > 0:
        text += "Students showed improvement, but the gains were not statistically significant.\n"
    else:
        text += "WARNING: No improvement detected. The intervention may not have been effective.\n"
    
    text += "\n"
    
    # Question-level insights
    weak_questions = question_stats[question_stats['pct_correct_post'] < 50].sort_values(by='pct_correct_post')
    if len(weak_questions) > 0:
        text += f"AREAS NEEDING ATTENTION\n"
        text += f"The following questions still show <50% mastery after the post-test:\n"
        for _, q in weak_questions.head(5).iterrows():
            text += f"  • {q['question']}: {q['pct_correct_post']:.1f}% correct\n"
        text += "\nRecommendation: Consider re-teaching these concepts.\n\n"
    
    # Top improved questions
    top_improved = question_stats.nlargest(3, 'improvement')
    text += f"MOST IMPROVED TOPICS\n"
    for _, q in top_improved.iterrows():
        text += f"  • {q['question']}: {q['improvement']:+.1f}% improvement\n"
    
    text += f"\n{faculty_rating['interpretation']}\n"
    
    return text
