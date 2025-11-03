# Pre-test / Post-test Analysis App

A production-level Streamlit application for comprehensive pre-test and post-test analysis with automated PDF reporting.

## Features

- **Strict Template Validation**: Ensures both pre and post test files follow exact required format
- **Participant Matching**: Analyzes only students who completed both tests
- **Comprehensive Analytics**:
  - Class-level statistics (mean, std, paired t-test, effect size)
  - Question-level analysis with transition tracking
  - Individual student performance with normalized gains
  - Faculty rating score (0-100) with transparent methodology
- **Interactive Visualizations**: 15+ charts using Plotly and Matplotlib
- **Automated PDF Reports**: Full reports with individual student pages
- **Export Options**: CSV downloads, filtered datasets, discarded participants list

## Requirements

- Python 3.10 or higher
- Dependencies listed in `requirements.txt`

## Quick Start - Local Installation

### 1. Clone or download this repository

### 2. Create virtual environment and install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app/main.py
```

The app will open in your browser at `http://localhost:8501`

## Docker Deployment

### Build the Docker image

```bash
docker build -t prepost-analysis .
```

### Run the container

```bash
docker run -p 8501:8501 prepost-analysis
```

Access the app at `http://localhost:8501`

## Streamlit Cloud Deployment

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your repository
4. Set Python version to 3.10+ in advanced settings

## Input File Format

Both pre-test and post-test files must be `.xlsx` Excel files with **identical column structure**:

### Required Headers (exact, case-insensitive)

```
name, ticket_no, q1, q2, q3, ..., qN
```

- **name**: Participant's full name (string)
- **ticket_no**: Unique identifier (string or numeric)
- **q1...qN**: Question columns (values must be 0 or 1)
  - 0 = incorrect answer
  - 1 = correct answer

### Example

| name | ticket_no | q1 | q2 | q3 | q4 | q5 |
|------|-----------|----|----|----|----|-----|
| Asha Sharma | T001 | 1 | 0 | 0 | 1 | 1 |
| Ravi Kumar | T002 | 0 | 0 | 1 | 0 | 1 |
| Meera Patel | T003 | 0 | 1 | 0 | 0 | 0 |

**Download a sample template directly from the app sidebar.**

## Faculty Rating Formula

The app calculates a Faculty Rating (0-100) using this transparent formula:

```
avg_normalized_gain = mean(normalized_gain) [ignoring NaN]
pct_students_improved = % of students with absolute_gain > 0
pct_students_regressed = % of students with absolute_gain < 0

Preliminary_Score = 100 × (0.65 × avg_normalized_gain + 
                            0.25 × pct_students_improved + 
                            0.10 × (1 - pct_students_regressed))

Faculty_Rating = round(Preliminary_Score) [clipped to 0-100]
```

**Penalty**: If class post-test mean < 40%, a small penalty is applied.

**Components**:
- **65%** weight on normalized learning gain (Hake's gain)
- **25%** weight on proportion who improved
- **10%** weight on minimizing regressions

## Statistical Tests Used

- **Paired t-test**: Tests if mean difference between pre and post is significant
- **Wilcoxon signed-rank test**: Non-parametric alternative (fallback)
- **Cohen's d**: Effect size for paired samples
- **Normalized Gain**: Hake's gain = (post - pre) / (max - pre)

## Project Structure

```
prepost-analysis/
├── app/
│   ├── main.py           # Streamlit entrypoint
│   ├── utils.py          # Analysis & validation functions
│   └── report.py         # PDF generation logic
├── examples/
│   ├── sample_pre.xlsx   # Example pre-test file
│   └── sample_post.xlsx  # Example post-test file
├── tests/
│   └── test_utils.py     # Unit tests
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
└── README.md           # This file
```

## Usage Guide

1. **Upload Files**: Use sidebar to upload pre-test and post-test Excel files
2. **Validate**: App automatically validates format and displays errors if any
3. **View Analytics**: Explore interactive charts across three tabs:
   - Class-level analysis
   - Question-level breakdown
   - Individual student performance
4. **Download Reports**:
   - Full PDF with individual student pages
   - Compact PDF with class-level summary only
   - CSV of discarded participants
   - Merged dataset with all computed metrics

## Testing

Run unit tests with pytest:

```bash
pytest tests/test_utils.py -v
```

Tests cover:
- Header validation
- Participant matching logic
- Score/gain calculations
- Transition category assignment
- Edge cases and error handling

## Performance

- Handles up to 5000 students and 100 questions efficiently
- Uses `@st.cache_data` for heavy computations
- Optimized pandas operations for fast processing

## Support

For issues or questions, please refer to the code documentation or modify constants at the top of each module for customization.

## License

MIT License - Free for educational and commercial use.
