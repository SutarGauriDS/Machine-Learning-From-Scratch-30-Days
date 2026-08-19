#  Day 6 – Machine Learning Workflow & Train-Test Split

## 30 Days of Machine Learning Challenge

Day 6 focuses on understanding the **Machine Learning workflow** and preparing a real-world dataset for model training.

For this day, I used the **Student Performance Factors dataset** containing **6,607 records and 20 columns**.

## 🎯 Topics Covered

* **Data Exploration** — Understanding dataset shape, columns, data types, and statistics.
* **Data Quality** — Checking missing values and duplicate records.
* **Feature Selection** — Selecting relevant numerical variables for ML.
* **Target Selection** — Using `Exam_Score` as the prediction target.
* **Correlation Analysis** — Understanding relationships between features and exam scores.
* **Data Visualization** — Using scatter plots and a correlation heatmap.
* **Train-Test Split** — Dividing data into training and testing sets.
* **Data Leakage** — Understanding why test data should not influence training.

## 📊 Dataset

**Dataset:** Student Performance Factors

```text
Records: 6,607
Columns: 20
Target: Exam_Score
```

### Selected Features

```text
Hours_Studied
Attendance
Previous_Scores
Sleep_Hours
Tutoring_Sessions
```

## 📈 Key Observations

| Feature           | Correlation with Exam Score |
| ----------------- | --------------------------: |
| Attendance        |                        0.58 |
| Hours_Studied     |                        0.45 |
| Previous_Scores   |                        0.18 |
| Tutoring_Sessions |                        0.16 |
| Sleep_Hours       |                       -0.02 |

> Correlation shows association, not causation.

## ✂️ Train-Test Split

Used:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Split Result

```text
Total Records:     6607
Training Records:  5285
Testing Records:   1322
```

Approximately:

**80% → Training | 20% → Testing**

## 🔄 ML Workflow

```text
CSV Dataset
     ↓
Data Exploration
     ↓
Data Quality Check
     ↓
Feature Selection
     ↓
Target Selection
     ↓
Correlation Analysis
     ↓
Train-Test Split
     ↓
Model Training
     ↓
Prediction
     ↓
Evaluation
```
## 📂 Files

| File                     | Description                 |
| ------------------------ | --------------------------- |
| `Day6.ipynb`             | Step-by-step implementation |
| `Day6.py`                | Python implementation       |
| `requirements.txt`       | Required libraries          |
| `references.md`          | Learning resources          |
| `interview_questions.md` | Interview preparation       |

## 💡 Key Learning

**Before training a Machine Learning model, understanding and preparing the data correctly is essential.**

Feature selection, data exploration, correlation analysis, and a proper train-test split help create a reliable ML workflow.

##  Day 6 Completed!

➡️ **Next: Day 7 – Linear Regression** 🚀
