
# 🧠 Day 11 - Naive Bayes Classification

## 📌 30 Days Machine Learning Challenge

For Day 11 of my Machine Learning Challenge, I implemented a **Naive Bayes Classification model** using the **Student Performance Factors** dataset.

The objective is to predict whether a student is likely to **Pass or Fail** based on academic and lifestyle-related factors.

---

## 🎯 Objective

Build a Gaussian Naive Bayes classification model to predict student performance.

### Features Used

- Hours Studied
- Attendance
- Previous Scores
- Sleep Hours
- Tutoring Sessions
- Physical Activity

### Target

- `0` → Fail
- `1` → Pass

---

## 🧠 Algorithm Used

### Gaussian Naive Bayes

Naive Bayes is a supervised machine learning classification algorithm based on **Bayes' Theorem**.

It assumes that the features are conditionally independent given the target class.

Since the selected features are numerical, **GaussianNB** is used.

```python
from sklearn.naive_bayes import GaussianNB
````

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Exploration
   ↓
Missing Value Check
   ↓
Duplicate Removal
   ↓
Create Pass/Fail Target
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Gaussian Naive Bayes
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
   ↓
New Student Prediction
```

---

## 📊 Model Evaluation

The model is evaluated using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score

The project also uses `predict_proba()` to understand the probability of a student belonging to each class.

---

## 🔮 New Student Prediction

The trained model is used to predict the performance of a new student based on:

```text
Hours Studied
Attendance
Previous Scores
Sleep Hours
Tutoring Sessions
Physical Activity
```

The model returns:

```text
PASS
```

or

```text
FAIL
```

along with the probability of each class.

---

## ⚠️ Target Leakage Prevention

The original dataset does not contain a Pass/Fail target.

The target is created using the median `Exam_Score`.

```python
median_score = df["Exam_Score"].median()

df["Pass"] = (
    df["Exam_Score"] >= median_score
).astype(int)
```

`Exam_Score` is **not included in the model features** because it was used to create the target. This prevents target leakage.

---
## 📁 Project Structure

```text
Day11-Naive-Bayes/
│
├── StudentPerformanceFactors.csv
├── Day11_Naive_Bayes_Classification.ipynb
├── Day11_Naive_Bayes_Classification.py
├── README.md
├── requirements.txt
└── REFERENCES.md
```

---

## 🚀 How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Python file:

```bash
python Day11_Naive_Bayes_Classification.py
```

Or open:

```text
Day11_Naive_Bayes_Classification.ipynb
```

in Jupyter Notebook or Google Colab.

---

## 📚 Key Learnings

* Bayes' Theorem
* Conditional Probability
* Naive Bayes Classification
* Gaussian Naive Bayes
* Train/Test Split
* Classification Metrics
* Confusion Matrix
* Prediction Probability
* Target Leakage

---

## ✅ Day 11 Outcome

Successfully implemented a **Gaussian Naive Bayes Classification model** for predicting student performance.

---

## 👩‍💻 30 Days ML Challenge

**Day 11/30 — Completed ✅**

Next: **Day 12 — Random Forest Classification 🌲**
