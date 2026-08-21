
# 🚀 Day 8 – Logistic Regression

## 30 Days of Machine Learning Challenge

Day 8 marks the transition from **Regression to Classification**.

Today I used the **Student Performance Factors** dataset to build a Logistic Regression model that classifies students into **High Performance** and **Low Performance** categories.

---

## 🎯 Objectives

Today I learned:

- Classification
- Logistic Regression
- Creating a categorical target
- Feature selection
- Train-Test Split
- Stratified splitting
- Model training
- Making predictions
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Prediction probabilities

---

## 📊 Dataset

**Dataset:** `StudentPerformanceFactors.csv`

### Original Target

```text
Exam_Score
````

For classification, I created a new target:

```text
Performance
```

### Classification Rule

```text
Exam_Score >= 75  → 1 (High Performance)

Exam_Score < 75   → 0 (Low Performance)
```

---

## 🔍 Selected Features

The model uses the following features:

* `Hours_Studied`
* `Attendance`
* `Previous_Scores`
* `Sleep_Hours`
* `Tutoring_Sessions`

### Target Variable

```text
Performance
```

---

## 🔄 Machine Learning Workflow

```text
Student Performance Dataset
          ↓
     Data Exploration
          ↓
   Create Performance Target
          ↓
     Feature Selection
          ↓
     Train-Test Split
          ↓
   Logistic Regression
          ↓
       Predictions
          ↓
    Model Evaluation
          ↓
   Confusion Matrix
```

---

## 📏 Evaluation Metrics

### Accuracy

Measures the percentage of correct predictions.

### Precision

Measures how many predicted positive cases were actually positive.

### Recall

Measures how many actual positive cases were correctly identified.

### F1 Score

Combines Precision and Recall into a single metric.

### Confusion Matrix

Shows:

* True Positive
* True Negative
* False Positive
* False Negative

---

## 📈 Visualizations

Created:

* Performance Class Distribution
* Confusion Matrix Heatmap

These visualizations help understand the class distribution and model performance.

---

## 🧠 Key Learning

Logistic Regression is a **classification algorithm** used to predict the probability of an observation belonging to a class.

In this project:

```text
0 → Low Performance
1 → High Performance
```

The model can also provide the probability of a student belonging to each class.

---

## 🛠️ Tools Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook

---

## 📂 Files

| File                     | Purpose                        |
| ------------------------ | ------------------------------ |
| `Day8.ipynb`             | Step-by-step implementation    |
| `Day8.py`                | Complete Python implementation |
| `requirements.txt`       | Required libraries             |
| `references.md`          | Learning resources             |
| `interview_questions.md` | Interview preparation          |

---

## 📁 Dataset Structure

```text
Day08/
│
├── data/
│   └── StudentPerformanceFactors.csv
│
├── Day8.ipynb
├── Day8.py
├── README.md
├── requirements.txt
├── references.md
└── interview_questions.md
```

The same `StudentPerformanceFactors.csv` dataset used in previous days is continued here to maintain consistency throughout the 30-day Machine Learning challenge.

---

## 💡 Key Takeaway

**Linear Regression predicts continuous values, while Logistic Regression is used for classification problems.**

Day 8 helped me understand how to convert a continuous target into a classification problem and evaluate a classification model.

---

## 🎉 Day 8 Completed!

➡️ **Next: Day 9 – K-Nearest Neighbors (KNN)** 🚀

