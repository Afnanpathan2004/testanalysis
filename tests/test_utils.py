"""
Unit tests for utility functions in the pre/post test analysis app.

Tests cover validation, participant matching, score calculations,
and statistical computations.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'app'))

from utils import (
    validate_excel_file,
    validate_matching_structure,
    match_participants,
    compute_scores_and_gains,
    compute_class_statistics,
    compute_question_statistics,
    compute_faculty_rating,
    generate_student_analysis_text,
    QUESTION_PATTERN
)


# Fixtures

@pytest.fixture
def valid_pre_test_df():
    """Create a valid pre-test DataFrame."""
    return pd.DataFrame({
        'name': ['Alice Smith', 'Bob Jones', 'Carol White', 'David Brown'],
        'ticket_no': ['T001', 'T002', 'T003', 'T004'],
        'q1': [1, 0, 1, 0],
        'q2': [0, 1, 0, 1],
        'q3': [1, 1, 0, 0],
        'q4': [0, 0, 1, 1],
        'q5': [1, 0, 1, 0]
    })


@pytest.fixture
def valid_post_test_df():
    """Create a valid post-test DataFrame (with improvements)."""
    return pd.DataFrame({
        'name': ['Alice Smith', 'Bob Jones', 'Carol White', 'David Brown'],
        'ticket_no': ['T001', 'T002', 'T003', 'T004'],
        'q1': [1, 1, 1, 0],
        'q2': [1, 1, 0, 1],
        'q3': [1, 1, 1, 1],
        'q4': [1, 0, 1, 1],
        'q5': [1, 1, 1, 0]
    })


@pytest.fixture
def question_columns():
    """Return list of question columns."""
    return ['q1', 'q2', 'q3', 'q4', 'q5']


# Test validation functions

def test_validate_excel_file_valid(valid_pre_test_df):
    """Test validation of a valid Excel file."""
    result = validate_excel_file(valid_pre_test_df, "Test")
    
    assert result.is_valid
    assert len(result.errors) == 0
    assert len(result.question_columns) == 5
    assert result.question_columns == ['q1', 'q2', 'q3', 'q4', 'q5']


def test_validate_excel_file_missing_name_column():
    """Test validation fails when name column is missing."""
    df = pd.DataFrame({
        'ticket_no': ['T001', 'T002'],
        'q1': [1, 0],
        'q2': [0, 1]
    })
    
    result = validate_excel_file(df, "Test")
    
    assert not result.is_valid
    assert any('name' in error.lower() for error in result.errors)


def test_validate_excel_file_missing_ticket_column():
    """Test validation fails when ticket_no column is missing."""
    df = pd.DataFrame({
        'name': ['Alice', 'Bob'],
        'q1': [1, 0],
        'q2': [0, 1]
    })
    
    result = validate_excel_file(df, "Test")
    
    assert not result.is_valid
    assert any('ticket_no' in error.lower() for error in result.errors)


def test_validate_excel_file_no_questions():
    """Test validation fails when no question columns present."""
    df = pd.DataFrame({
        'name': ['Alice', 'Bob'],
        'ticket_no': ['T001', 'T002'],
        'score': [5, 3]  # Wrong column name
    })
    
    result = validate_excel_file(df, "Test")
    
    assert not result.is_valid
    assert any('question' in error.lower() for error in result.errors)


def test_validate_excel_file_invalid_values():
    """Test validation fails with invalid question values."""
    df = pd.DataFrame({
        'name': ['Alice', 'Bob'],
        'ticket_no': ['T001', 'T002'],
        'q1': [1, 2],  # Invalid value: should be 0 or 1
        'q2': [0, 1]
    })
    
    result = validate_excel_file(df, "Test")
    
    assert not result.is_valid
    assert any('invalid values' in error.lower() for error in result.errors)


def test_validate_excel_file_duplicate_tickets():
    """Test validation fails with duplicate ticket numbers."""
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Carol'],
        'ticket_no': ['T001', 'T001', 'T002'],  # Duplicate T001
        'q1': [1, 0, 1],
        'q2': [0, 1, 0]
    })
    
    result = validate_excel_file(df, "Test")
    
    assert not result.is_valid
    assert any('duplicate' in error.lower() for error in result.errors)


def test_validate_matching_structure_valid(valid_pre_test_df, valid_post_test_df):
    """Test validation of matching file structures."""
    q_cols = ['q1', 'q2', 'q3', 'q4', 'q5']
    
    errors = validate_matching_structure(
        valid_pre_test_df, valid_post_test_df, q_cols, q_cols
    )
    
    assert len(errors) == 0


def test_validate_matching_structure_different_question_count():
    """Test validation fails when question counts differ."""
    df_pre = pd.DataFrame({
        'name': ['Alice'], 'ticket_no': ['T001'],
        'q1': [1], 'q2': [0], 'q3': [1]
    })
    df_post = pd.DataFrame({
        'name': ['Alice'], 'ticket_no': ['T001'],
        'q1': [1], 'q2': [0]
    })
    
    errors = validate_matching_structure(
        df_pre, df_post, ['q1', 'q2', 'q3'], ['q1', 'q2']
    )
    
    assert len(errors) > 0
    assert any('questions' in error.lower() for error in errors)


def test_validate_matching_structure_different_question_names():
    """Test validation fails when question names differ."""
    df_pre = pd.DataFrame({
        'name': ['Alice'], 'ticket_no': ['T001'],
        'q1': [1], 'q2': [0]
    })
    df_post = pd.DataFrame({
        'name': ['Alice'], 'ticket_no': ['T001'],
        'q1': [1], 'q3': [0]  # q3 instead of q2
    })
    
    errors = validate_matching_structure(
        df_pre, df_post, ['q1', 'q2'], ['q1', 'q3']
    )
    
    assert len(errors) > 0


# Test participant matching

def test_match_participants_all_match(valid_pre_test_df, valid_post_test_df):
    """Test participant matching when all students are in both tests."""
    df_pre_filtered, df_post_filtered, df_discarded = match_participants(
        valid_pre_test_df, valid_post_test_df
    )
    
    assert len(df_pre_filtered) == 4
    assert len(df_post_filtered) == 4
    assert len(df_discarded) == 0
    
    # Check that ticket numbers match
    assert list(df_pre_filtered['ticket_no']) == list(df_post_filtered['ticket_no'])


def test_match_participants_partial_match():
    """Test participant matching with some students missing in each test."""
    df_pre = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Carol'],
        'ticket_no': ['T001', 'T002', 'T003'],
        'q1': [1, 0, 1]
    })
    
    df_post = pd.DataFrame({
        'name': ['Alice', 'Carol', 'David'],
        'ticket_no': ['T001', 'T003', 'T004'],
        'q1': [1, 1, 0]
    })
    
    df_pre_filtered, df_post_filtered, df_discarded = match_participants(df_pre, df_post)
    
    # Only Alice and Carol should be matched
    assert len(df_pre_filtered) == 2
    assert len(df_post_filtered) == 2
    assert set(df_pre_filtered['ticket_no']) == {'T001', 'T003'}
    
    # Bob and David should be discarded
    assert len(df_discarded) == 2
    assert set(df_discarded['ticket_no']) == {'T002', 'T004'}


def test_match_participants_no_match():
    """Test participant matching with no common students."""
    df_pre = pd.DataFrame({
        'name': ['Alice', 'Bob'],
        'ticket_no': ['T001', 'T002'],
        'q1': [1, 0]
    })
    
    df_post = pd.DataFrame({
        'name': ['Carol', 'David'],
        'ticket_no': ['T003', 'T004'],
        'q1': [1, 0]
    })
    
    df_pre_filtered, df_post_filtered, df_discarded = match_participants(df_pre, df_post)
    
    assert len(df_pre_filtered) == 0
    assert len(df_post_filtered) == 0
    assert len(df_discarded) == 4


# Test score and gain computations

def test_compute_scores_and_gains(valid_pre_test_df, valid_post_test_df, question_columns):
    """Test score and gain calculations."""
    df_merged = compute_scores_and_gains(
        valid_pre_test_df, valid_post_test_df, question_columns
    )
    
    # Check basic structure
    assert len(df_merged) == 4
    assert 'score_pre' in df_merged.columns
    assert 'score_post' in df_merged.columns
    assert 'absolute_gain' in df_merged.columns
    assert 'normalized_gain' in df_merged.columns
    
    # Verify score calculations for first student
    alice = df_merged[df_merged['ticket_no'] == 'T001'].iloc[0]
    assert alice['score_pre'] == 3  # 1+0+1+0+1
    assert alice['score_post'] == 5  # 1+1+1+1+1
    assert alice['absolute_gain'] == 2
    
    # Verify normalized gain calculation
    # normalized_gain = (post - pre) / (max - pre)
    # For Alice: (5 - 3) / (5 - 3) = 2/2 = 1.0
    assert alice['normalized_gain'] == pytest.approx(1.0)


def test_compute_scores_perfect_score_pre():
    """Test normalized gain when student has perfect pre-test score."""
    df_pre = pd.DataFrame({
        'name': ['Alice'],
        'ticket_no': ['T001'],
        'q1': [1], 'q2': [1], 'q3': [1]
    })
    
    df_post = pd.DataFrame({
        'name': ['Alice'],
        'ticket_no': ['T001'],
        'q1': [1], 'q2': [1], 'q3': [1]
    })
    
    df_merged = compute_scores_and_gains(df_pre, df_post, ['q1', 'q2', 'q3'])
    
    # Normalized gain should be NaN when denominator is 0
    assert pd.isna(df_merged.iloc[0]['normalized_gain'])


def test_transition_categories():
    """Test that transition categories are correctly assigned."""
    df_pre = pd.DataFrame({
        'name': ['Alice'],
        'ticket_no': ['T001'],
        'q1': [1],  # Will stay correct
        'q2': [0],  # Will become correct
        'q3': [1],  # Will become incorrect
        'q4': [0]   # Will stay incorrect
    })
    
    df_post = pd.DataFrame({
        'name': ['Alice'],
        'ticket_no': ['T001'],
        'q1': [1],
        'q2': [1],
        'q3': [0],
        'q4': [0]
    })
    
    df_merged = compute_scores_and_gains(df_pre, df_post, ['q1', 'q2', 'q3', 'q4'])
    
    row = df_merged.iloc[0]
    assert row['q1_transition'] == 'PreRight_PostRight'
    assert row['q2_transition'] == 'PreWrong_PostRight'
    assert row['q3_transition'] == 'PreRight_PostWrong'
    assert row['q4_transition'] == 'PreWrong_PostWrong'


# Test class statistics

def test_compute_class_statistics(valid_pre_test_df, valid_post_test_df, question_columns):
    """Test class-level statistics computation."""
    df_merged = compute_scores_and_gains(
        valid_pre_test_df, valid_post_test_df, question_columns
    )
    
    stats = compute_class_statistics(df_merged, len(question_columns))
    
    # Check required fields
    assert 'n_students' in stats
    assert 'mean_pre' in stats
    assert 'mean_post' in stats
    assert 'mean_gain' in stats
    assert 'std_pre' in stats
    assert 'p_value' in stats
    assert 'cohens_d' in stats
    
    # Verify counts
    assert stats['n_students'] == 4
    assert stats['max_score'] == 5
    
    # Mean gain should be positive (post-test improved)
    assert stats['mean_gain'] > 0


def test_compute_class_statistics_improvement_counts():
    """Test that improvement/regression counts are correct."""
    df_pre = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Carol', 'David'],
        'ticket_no': ['T001', 'T002', 'T003', 'T004'],
        'q1': [1, 0, 1, 1],
        'q2': [0, 1, 0, 1]
    })
    
    df_post = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Carol', 'David'],
        'ticket_no': ['T001', 'T002', 'T003', 'T004'],
        'q1': [1, 1, 0, 1],  # Alice same, Bob improved, Carol regressed, David same
        'q2': [1, 1, 0, 1]
    })
    
    df_merged = compute_scores_and_gains(df_pre, df_post, ['q1', 'q2'])
    stats = compute_class_statistics(df_merged, 2)
    
    assert stats['n_improved'] == 2  # Alice +1, Bob +1
    assert stats['n_unchanged'] == 1  # David
    assert stats['n_regressed'] == 1  # Carol


# Test question statistics

def test_compute_question_statistics(valid_pre_test_df, valid_post_test_df, question_columns):
    """Test question-level statistics computation."""
    df_merged = compute_scores_and_gains(
        valid_pre_test_df, valid_post_test_df, question_columns
    )
    
    q_stats = compute_question_statistics(df_merged, question_columns)
    
    # Check structure
    assert len(q_stats) == 5
    assert 'question' in q_stats.columns
    assert 'pct_correct_pre' in q_stats.columns
    assert 'pct_correct_post' in q_stats.columns
    assert 'improvement' in q_stats.columns
    
    # Check transition columns
    assert 'PreRight_PostRight' in q_stats.columns
    assert 'PreWrong_PostRight' in q_stats.columns
    
    # Verify percentages are in valid range
    assert all(q_stats['pct_correct_pre'] >= 0)
    assert all(q_stats['pct_correct_pre'] <= 100)
    assert all(q_stats['pct_correct_post'] >= 0)
    assert all(q_stats['pct_correct_post'] <= 100)


# Test faculty rating

def test_compute_faculty_rating_good_performance():
    """Test faculty rating with good performance."""
    df = pd.DataFrame({
        'score_pre': [3, 3, 4, 2],
        'score_post': [5, 5, 5, 4],
        'absolute_gain': [2, 2, 1, 2],
        'normalized_gain': [1.0, 1.0, 1.0, 0.67]
    })
    
    class_stats = compute_class_statistics(df, 5)
    rating = compute_faculty_rating(class_stats, df)
    
    # Should have high rating (all students improved)
    assert rating['score'] >= 70
    assert 'score' in rating
    assert 'explanation' in rating
    assert 'interpretation' in rating


def test_compute_faculty_rating_with_penalty():
    """Test faculty rating with mastery penalty."""
    df = pd.DataFrame({
        'score_pre': [1, 1, 1, 1],
        'score_post': [2, 2, 2, 2],  # Only 40% correct - right at threshold
        'absolute_gain': [1, 1, 1, 1],
        'normalized_gain': [0.25, 0.25, 0.25, 0.25]
    })
    
    class_stats = compute_class_statistics(df, 5)
    rating = compute_faculty_rating(class_stats, df)
    
    # Should have some penalty due to low post-test performance
    assert 'penalty' in rating
    # Rating should be moderate
    assert 20 <= rating['score'] <= 80


def test_compute_faculty_rating_regression():
    """Test faculty rating with some regressions."""
    df = pd.DataFrame({
        'score_pre': [4, 4, 3, 3],
        'score_post': [2, 3, 4, 4],  # First regressed, second unchanged, last two improved
        'absolute_gain': [-2, -1, 1, 1],
        'normalized_gain': [-2.0, -1.0, 0.5, 0.5]
    })
    
    class_stats = compute_class_statistics(df, 5)
    rating = compute_faculty_rating(class_stats, df)
    
    # Rating should be lower due to regressions
    assert rating['score'] < 60


# Test text generation

def test_generate_student_analysis_text():
    """Test student analysis text generation."""
    row = pd.Series({
        'name': 'Alice Smith',
        'ticket_no': 'T001',
        'score_pre': 3,
        'score_post': 5,
        'absolute_gain': 2,
        'normalized_gain': 0.67,
        'percent_correct_pre': 60.0,
        'percent_correct_post': 100.0,
        'q1_pre': 1, 'q1_post': 1, 'q1_transition': 'PreRight_PostRight',
        'q2_pre': 0, 'q2_post': 1, 'q2_transition': 'PreWrong_PostRight',
        'q3_pre': 1, 'q3_post': 1, 'q3_transition': 'PreRight_PostRight',
        'q4_pre': 0, 'q4_post': 1, 'q4_transition': 'PreWrong_PostRight',
        'q5_pre': 1, 'q5_post': 1, 'q5_transition': 'PreRight_PostRight'
    })
    
    text = generate_student_analysis_text(row, ['q1', 'q2', 'q3', 'q4', 'q5'])
    
    # Check that key information is present
    assert 'Alice Smith' in text
    assert 'T001' in text
    assert '3/5' in text or '3' in text
    assert '5/5' in text or '5' in text
    assert 'improved' in text.lower() or 'improvement' in text.lower()


# Test question pattern regex

def test_question_pattern_matching():
    """Test that question pattern regex works correctly."""
    assert QUESTION_PATTERN.match('q1')
    assert QUESTION_PATTERN.match('q10')
    assert QUESTION_PATTERN.match('q999')
    assert QUESTION_PATTERN.match('Q1')  # Case insensitive
    
    assert not QUESTION_PATTERN.match('question1')
    assert not QUESTION_PATTERN.match('q')
    assert not QUESTION_PATTERN.match('q1a')
    assert not QUESTION_PATTERN.match('1q')


# Integration test

def test_full_analysis_workflow(valid_pre_test_df, valid_post_test_df, question_columns):
    """Test complete analysis workflow from validation to results."""
    # Step 1: Validate
    val_pre = validate_excel_file(valid_pre_test_df, "Pre")
    val_post = validate_excel_file(valid_post_test_df, "Post")
    
    assert val_pre.is_valid
    assert val_post.is_valid
    
    # Step 2: Check structure
    errors = validate_matching_structure(
        valid_pre_test_df, valid_post_test_df,
        val_pre.question_columns, val_post.question_columns
    )
    assert len(errors) == 0
    
    # Step 3: Match participants
    df_pre_f, df_post_f, df_disc = match_participants(
        valid_pre_test_df, valid_post_test_df
    )
    assert len(df_pre_f) == 4
    assert len(df_disc) == 0
    
    # Step 4: Compute scores
    df_merged = compute_scores_and_gains(df_pre_f, df_post_f, question_columns)
    assert len(df_merged) == 4
    assert 'absolute_gain' in df_merged.columns
    
    # Step 5: Compute statistics
    class_stats = compute_class_statistics(df_merged, len(question_columns))
    assert class_stats['n_students'] == 4
    
    question_stats = compute_question_statistics(df_merged, question_columns)
    assert len(question_stats) == 5
    
    faculty_rating = compute_faculty_rating(class_stats, df_merged)
    assert 0 <= faculty_rating['score'] <= 100
    
    # All steps completed successfully
    assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
