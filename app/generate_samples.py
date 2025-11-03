"""
Sample data generator for pre-test and post-test Excel files.
Creates synthetic data following the exact required template format.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_files():
    """Generate sample pre and post test Excel files with synthetic data."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Sample student data
    students = [
        ("Asha Sharma", "T001"),
        ("Ravi Kumar", "T002"),
        ("Meera Patel", "T003"),
        ("Amit Singh", "T004"),
        ("Priya Verma", "T005"),
        ("Rahul Gupta", "T006"),
        ("Sneha Reddy", "T007"),
        ("Vikram Joshi", "T008"),
        ("Anita Desai", "T009"),
        ("Karan Mehta", "T010"),
        ("Divya Iyer", "T011"),
        ("Arjun Nair", "T012"),
        ("Kavita Pillai", "T013"),
        ("Sanjay Rao", "T014"),
        ("Neha Kapoor", "T015"),
    ]
    
    num_questions = 10
    
    # Generate pre-test data (lower scores)
    pre_data = []
    for name, ticket in students:
        row = {"name": name, "ticket_no": ticket}
        # Pre-test: 20-50% correct on average
        for q in range(1, num_questions + 1):
            row[f"q{q}"] = np.random.choice([0, 1], p=[0.65, 0.35])
        pre_data.append(row)
    
    # Generate post-test data (higher scores, showing improvement)
    post_data = []
    for i, (name, ticket) in enumerate(students):
        # Skip 2 students in post-test to demonstrate discarded participants
        if i in [1, 7]:  # Skip Ravi Kumar and Vikram Joshi
            continue
            
        row = {"name": name, "ticket_no": ticket}
        # Post-test: 50-85% correct on average
        for q in range(1, num_questions + 1):
            row[f"q{q}"] = np.random.choice([0, 1], p=[0.35, 0.65])
        post_data.append(row)
    
    # Add 2 students who only took post-test
    post_data.append({
        "name": "New Student One",
        "ticket_no": "T016",
        **{f"q{q}": np.random.choice([0, 1]) for q in range(1, num_questions + 1)}
    })
    post_data.append({
        "name": "New Student Two",
        "ticket_no": "T017",
        **{f"q{q}": np.random.choice([0, 1]) for q in range(1, num_questions + 1)}
    })
    
    # Create DataFrames
    df_pre = pd.DataFrame(pre_data)
    df_post = pd.DataFrame(post_data)
    
    # Create examples directory
    examples_dir = Path(__file__).parent.parent / "examples"
    examples_dir.mkdir(exist_ok=True)
    
    # Save to Excel
    pre_path = examples_dir / "sample_pre.xlsx"
    post_path = examples_dir / "sample_post.xlsx"
    
    df_pre.to_excel(pre_path, index=False, engine='openpyxl')
    df_post.to_excel(post_path, index=False, engine='openpyxl')
    
    print(f"Generated sample files:")
    print(f"  Pre-test:  {pre_path}")
    print(f"  Post-test: {post_path}")
    print(f"\nPre-test students: {len(df_pre)}")
    print(f"Post-test students: {len(df_post)}")
    print(f"Expected matched: {len(set(df_pre['ticket_no']) & set(df_post['ticket_no']))}")
    

def generate_blank_template(num_questions: int = 10) -> pd.DataFrame:
    """
    Generate a blank template DataFrame with proper headers.
    
    Args:
        num_questions: Number of question columns to include
        
    Returns:
        Empty DataFrame with required headers
    """
    columns = ["name", "ticket_no"] + [f"q{i}" for i in range(1, num_questions + 1)]
    return pd.DataFrame(columns=columns)


if __name__ == "__main__":
    generate_sample_files()
