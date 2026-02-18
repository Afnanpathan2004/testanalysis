import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent / 'app'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import streamlit as st
        import plotly.graph_objects as go
        import matplotlib.pyplot as plt
        from reportlab.lib.pagesizes import letter
        import openpyxl
        import scipy.stats
        
        from app.utils import (
            validate_excel_file,
            match_participants,
            compute_scores_and_gains,
            compute_class_statistics,
            compute_question_statistics,
            compute_faculty_rating
        )
        
        from app.report import PDFReportGenerator
        
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_sample_files():
    """Test that sample files exist."""
    print("\nTesting sample files...")
    examples_dir = Path(__file__).parent / 'examples'
    
    pre_file = examples_dir / 'sample_pre.xlsx'
    post_file = examples_dir / 'sample_post.xlsx'
    
    if pre_file.exists():
        print(f"✓ Pre-test sample exists: {pre_file}")
    else:
        print(f"✗ Pre-test sample missing: {pre_file}")
        return False
    
    if post_file.exists():
        print(f"✓ Post-test sample exists: {post_file}")
    else:
        print(f"✗ Post-test sample missing: {post_file}")
        return False
    
    return True


def test_validation():
    """Test validation on sample files."""
    print("\nTesting validation on sample files...")
    try:
        import pandas as pd
        from app.utils import validate_excel_file, validate_matching_structure
        
        examples_dir = Path(__file__).parent / 'examples'
        
        df_pre = pd.read_excel(examples_dir / 'sample_pre.xlsx')
        df_post = pd.read_excel(examples_dir / 'sample_post.xlsx')
        
        val_pre = validate_excel_file(df_pre, "Pre-test")
        val_post = validate_excel_file(df_post, "Post-test")
        
        if val_pre.is_valid:
            print(f"✓ Pre-test validation passed ({len(val_pre.question_columns)} questions)")
        else:
            print(f"✗ Pre-test validation failed: {val_pre.errors}")
            return False
        
        if val_post.is_valid:
            print(f"✓ Post-test validation passed ({len(val_post.question_columns)} questions)")
        else:
            print(f"✗ Post-test validation failed: {val_post.errors}")
            return False
        
        errors = validate_matching_structure(
            df_pre, df_post, val_pre.question_columns, val_post.question_columns
        )
        
        if len(errors) == 0:
            print("✓ File structure matching passed")
        else:
            print(f"✗ Structure mismatch: {errors}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Validation test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analysis():
    """Test analysis on sample files."""
    print("\nTesting analysis pipeline...")
    try:
        import pandas as pd
        from app.utils import (
            match_participants,
            compute_scores_and_gains,
            compute_class_statistics,
            compute_question_statistics,
            compute_faculty_rating
        )
        
        examples_dir = Path(__file__).parent / 'examples'
        
        df_pre = pd.read_excel(examples_dir / 'sample_pre.xlsx')
        df_post = pd.read_excel(examples_dir / 'sample_post.xlsx')
        
        # Get question columns
        q_cols = [col for col in df_pre.columns if col.lower().startswith('q') and col[1:].isdigit()]
        q_cols = sorted(q_cols, key=lambda x: int(x[1:]))
        
        # Match participants
        df_pre_f, df_post_f, df_disc = match_participants(df_pre, df_post)
        print(f"✓ Matched {len(df_pre_f)} participants, discarded {len(df_disc)}")
        
        if len(df_pre_f) == 0:
            print("✗ No matched participants")
            return False
        
        # Compute scores
        df_merged = compute_scores_and_gains(df_pre_f, df_post_f, q_cols)
        print(f"✓ Computed scores and gains for {len(df_merged)} students")
        
        # Compute statistics
        class_stats = compute_class_statistics(df_merged, len(q_cols))
        print(f"✓ Class mean pre: {class_stats['mean_pre']:.2f}, post: {class_stats['mean_post']:.2f}")
        
        question_stats = compute_question_statistics(df_merged, q_cols)
        print(f"✓ Computed statistics for {len(question_stats)} questions")
        
        faculty_rating = compute_faculty_rating(class_stats, df_merged)
        print(f"✓ Faculty rating: {faculty_rating['score']}/100")
        
        return True
        
    except Exception as e:
        print(f"✗ Analysis test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("PREPOST-ANALYSIS VERIFICATION")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Sample Files", test_sample_files()))
    results.append(("Validation", test_validation()))
    results.append(("Analysis", test_analysis()))
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} : {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Application is ready to run!")
        print("\nTo start the application, run:")
        print("  streamlit run app/main.py")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Please check errors above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
