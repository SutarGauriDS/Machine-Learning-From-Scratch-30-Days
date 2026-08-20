# 🚀 Day 7 – Linear Regression

## 30 Days of Machine Learning Challenge

Today I built my **first Machine Learning regression model** using the **Student Performance Factors** dataset.

The goal is to predict a student's **`Exam_Score`** using selected performance-related features.

---

## 🎯 Topics Covered

* Linear Regression
* Multiple Linear Regression
* Model Training
* Predictions
* MAE
* MSE
* RMSE
* R² Score
* Model Coefficients
* Actual vs Predicted Visualization

---

## 📊 Dataset

**Dataset:** `StudentPerformanceFactors.csv`

**Target Variable:**

```text
Exam_Score
```

**Selected Features:**

```text
Hours_Studied
Attendance
Previous_Scores
Sleep_Hours
Tutoring_Sessions
```

---

## 🔄 Machine Learning Workflow

```text
Student Performance Dataset
          ↓
    Feature Selection
          ↓
     Train-Test Split
          ↓
    Linear Regression
          ↓
     Model Training
          ↓
       Predictions
          ↓
   Model Evaluation
          ↓
 Model Interpretation
```

---

## 📏 Evaluation Metrics

### MAE — Mean Absolute Error

Measures the **average absolute difference** between actual and predicted values.

### MSE — Mean Squared Error

Measures the **average squared difference** between actual and predicted values. Larger errors receive more penalty.

### RMSE — Root Mean Squared Error

The square root of MSE. It is expressed in the **same unit as the target variable**.

### R² Score

Measures how much of the variation in the target variable is explained by the model.

---

## 📈 Visualization

Created an **Actual vs Predicted Exam Scores** plot to visually evaluate how closely the model's predictions match the actual scores.

---

## 🧠 Key Learning

Linear Regression can be used to predict **continuous numerical values**.

I also learned that building a Machine Learning model is not only about training the algorithm. **Model evaluation and interpretation are equally important** for understanding how well the model performs on unseen data.

---

## 🛠️ Tools Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* Jupyter Notebook

---

## 📂 Files

| File                     | Purpose                        |
| ------------------------ | ------------------------------ |
| `Day7.ipynb`             | Step-by-step implementation    |
| `Day7.py`                | Complete Python implementation |
| `requirements.txt`       | Required libraries             |
| `references.md`          | Learning resources             |
| `interview_questions.md` | Interview preparation          |

---

## 📁 Dataset Location

The same dataset used in Day 6 is continued in Day 7:

```text
Day07/
│
├── data/
│   └── StudentPerformanceFactors.csv
│
├── Day7.ipynb
├── Day7.py
├── README.md
├── requirements.txt
├── references.md
└── interview_questions.md
```

---

## 🎉 Day 7 Completed!

➡️ **Next: Day 8 – Logistic Regression** 🚀

