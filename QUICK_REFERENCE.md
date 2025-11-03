# Quick Reference Guide

## 🚀 Quick Start (3 steps)

### Step 1: Install Dependencies
```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Or simply double-click: start.bat
```

### Step 2: Run Application
```bash
streamlit run app/main.py
```

### Step 3: Upload & Analyze
1. Upload pre-test Excel file
2. Upload post-test Excel file
3. View results and download reports

---

## 📋 Excel File Format

### Required Columns (exact names, case-insensitive)

| Column | Type | Description | Values |
|--------|------|-------------|--------|
| `name` | Text | Student name | Any text |
| `ticket_no` | Text/Number | Unique student ID | Must be unique |
| `q1, q2, q3...` | Integer | Question responses | 0 or 1 only |

### Example Template
```
name          | ticket_no | q1 | q2 | q3 | q4 | q5
------------- | --------- | -- | -- | -- | -- | --
Asha Sharma   | T001      | 1  | 0  | 0  | 1  | 1
Ravi Kumar    | T002      | 0  | 0  | 1  | 0  | 1
Meera Patel   | T003      | 0  | 1  | 0  | 0  | 0
```

**Download template from app sidebar** 📥

---

## 📊 Understanding Results

### Faculty Rating (0-100)
- **80-100**: Excellent effectiveness
- **65-79**: Good learning gains
- **50-64**: Adequate improvement
- **35-49**: Needs improvement
- **0-34**: Significant concerns

### Formula Components
```
65% weight: Normalized gain (Hake's formula)
25% weight: % students improved
10% weight: 1 - % students regressed
Penalty:    If post-test mean < 40%
```

### Normalized Gain (Hake's Formula)
```
g = (post - pre) / (max - pre)

Where:
  post = score on post-test
  pre  = score on pre-test
  max  = maximum possible score
```

**Interpretation**:
- `g > 0.7`: High gain
- `0.3 ≤ g ≤ 0.7`: Medium gain
- `g < 0.3`: Low gain

### Transition Categories (per question)

| Category | Pre | Post | Meaning |
|----------|-----|------|---------|
| **Mastered** | ✓ | ✓ | Kept correct answer |
| **Learned** | ✗ | ✓ | Changed wrong to right |
| **Forgot** | ✓ | ✗ | Changed right to wrong |
| **Struggling** | ✗ | ✗ | Kept incorrect answer |

---

## 📥 Download Options

### From App Interface

1. **Full PDF Report**
   - All class-level charts
   - Question analysis
   - Individual student pages (optional)
   - Faculty rating and recommendations
   - ~10-50 MB depending on cohort size

2. **Compact PDF Report**
   - Class summary only
   - Top/bottom performers
   - No individual pages
   - ~2-5 MB

3. **Merged Data CSV**
   - All computed metrics
   - Per-student scores and gains
   - Per-question transitions

4. **Discarded Students CSV**
   - Students missing pre or post test
   - Reason for exclusion

5. **Complete Excel Workbook**
   - Multiple sheets
   - Merged data
   - Question statistics
   - Discarded list

---

## 🔍 Common Issues & Solutions

### Issue: "Missing required column 'name'"
**Solution**: Ensure Excel file has columns exactly named: `name`, `ticket_no`, `q1`, `q2`, etc.

### Issue: "Invalid values in q3"
**Solution**: All question columns must contain only 0 or 1. Check for:
- Empty cells
- Text values
- Numbers other than 0 or 1

### Issue: "Duplicate ticket numbers"
**Solution**: Each ticket_no must appear only once per file. Check for duplicates.

### Issue: "No students found in both tests"
**Solution**: Ensure ticket_no values match exactly between files (case-sensitive).

### Issue: "File structure mismatch"
**Solution**: Both files must have the same question columns (q1-qN) in the same order.

---

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/test_utils.py -v
```

### Run Verification
```bash
python verify.py
```

### Test with Sample Data
```bash
# Sample files are in examples/
# Use sample_pre.xlsx and sample_post.xlsx to test
```

---

## 🐳 Docker Commands

### Build Image
```bash
docker build -t prepost-analysis .
```

### Run Container
```bash
docker run -p 8501:8501 prepost-analysis
```

### Access App
```
http://localhost:8501
```

---

## 📈 Performance Tips

1. **Large Cohorts (>1000 students)**
   - Use compact PDF option
   - Consider disabling individual pages
   - Processing time: ~30-60 seconds

2. **Many Questions (>50)**
   - Some visualizations auto-sample
   - All analysis still computed
   - PDF generation may take 1-2 minutes

3. **Caching**
   - Results are cached automatically
   - Re-running with same files is instant
   - Change files to clear cache

---

## 🎯 Interpretation Guide

### Statistical Significance
- **p-value < 0.05**: Significant improvement
- **p-value ≥ 0.05**: Not significant (could be chance)

### Effect Size (Cohen's d)
- **d < 0.2**: Small effect
- **0.2 ≤ d < 0.8**: Medium effect
- **d ≥ 0.8**: Large effect

### Class Performance
- **Mean gain > 2**: Strong improvement
- **Mean gain 1-2**: Moderate improvement
- **Mean gain 0-1**: Weak improvement
- **Mean gain < 0**: Regression (concern)

### Question Analysis
- **Post % > 70%**: Well-understood topic
- **Post % 50-70%**: Adequately understood
- **Post % < 50%**: Needs re-teaching

---

## 🔧 Customization

### Change Mastery Threshold
Edit `app/utils.py`, line ~20:
```python
MASTERY_THRESHOLD = 0.40  # Change to 0.50 for 50%
```

### Change Faculty Rating Weights
Edit `app/utils.py`, function `compute_faculty_rating`:
```python
preliminary_score = 100 * (
    0.65 * avg_normalized_gain +    # Change weights here
    0.25 * pct_improved +
    0.10 * (1 - pct_regressed)
)
```

### Change PDF Page Size
Edit `app/main.py`, PDF generation section:
```python
from reportlab.lib.pagesizes import A4
pdf_gen = PDFReportGenerator(page_size=A4)  # Instead of letter
```

---

## 📞 Support

### For Issues:
1. Check this guide
2. Review README.md
3. Check validation errors in app
4. Run `python verify.py` to diagnose

### For Questions:
- All formulas explained in app
- Hover over metrics for tooltips
- PDF reports include methodology

---

## ✅ Checklist Before Running

- [ ] Python 3.10+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Excel files prepared with correct format
- [ ] Both files have same question columns
- [ ] No duplicate ticket numbers
- [ ] All question values are 0 or 1

---

## 🎓 Example Workflow

1. **Prepare Files**
   ```
   - Export pre-test responses to Excel
   - Export post-test responses to Excel
   - Ensure format matches template
   ```

2. **Run Application**
   ```bash
   streamlit run app/main.py
   ```

3. **Upload & Validate**
   ```
   - Upload pre-test file
   - Upload post-test file
   - Check validation passes
   ```

4. **Analyze**
   ```
   - View KPI cards
   - Explore visualizations in tabs
   - Read text summaries
   ```

5. **Download**
   ```
   - Full PDF for complete analysis
   - Compact PDF for quick review
   - CSV files for further analysis
   ```

6. **Share Results**
   ```
   - Send PDF to stakeholders
   - Use insights for curriculum planning
   - Identify students needing support
   ```

---

## 🎉 Success!

You're all set! The app is designed to be intuitive and self-explanatory. Just upload your files and explore the results.

**Happy Analyzing! 📊**

---

*Last Updated: 2025-11-03*
