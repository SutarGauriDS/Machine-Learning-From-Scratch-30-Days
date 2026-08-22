
# 🚀 Day 9 – K-Nearest Neighbors (KNN)

## 30 Days of Machine Learning Challenge

Day 9 focuses on **K-Nearest Neighbors (KNN)**, a distance-based Machine Learning algorithm used for classification.

Today I used the same **Student Performance Factors** dataset from previous days to predict whether a student belongs to the **High Performance** or **Low Performance** category.

## 🎯 Objectives

Today I learned:

- K-Nearest Neighbors (KNN)
- Distance-based classification
- Feature Scaling
- StandardScaler
- Choosing the value of K
- Model Training
- Making Predictions
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Overfitting and Underfitting

---

## 📊 Dataset

**Dataset:** `StudentPerformanceFactors.csv`

### Target Variable

```text
Performance
````

### Target Creation

```text
Exam_Score >= 75  → 1 (High Performance)

Exam_Score < 75   → 0 (Low Performance)
```

---

## 🔍 Selected Features

The following features were used:

* `Hours_Studied`
* `Attendance`
* `Previous_Scores`
* `Sleep_Hours`
* `Tutoring_Sessions`

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
      Feature Scaling
          ↓
          KNN
          ↓
       Predictions
          ↓
    Model Evaluation
          ↓
     Find Best K Value
```

---

## ⚖️ Feature Scaling

KNN is a **distance-based algorithm**, so feature scaling is important.

I used `StandardScaler` to standardize the numerical features.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data to help prevent **data leakage**.

---

## 🤖 KNN Model

The initial model was created using:

```python
KNeighborsClassifier(n_neighbors=5)
```

KNN predicts the class of a new observation based on its nearest neighbors.

---

## 🔢 Choosing the Best K

I tested different values of K from **1 to 20** and compared their accuracy.

```text
K = 1
K = 2
K = 3
...
K = 20
```

A **K vs Accuracy** graph was created to identify a suitable K value.

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
* K Value vs Accuracy
* KNN Confusion Matrix

---

## 🧠 Key Learning

KNN is a simple but powerful classification algorithm that makes predictions based on the **nearest data points**.

I also learned why **feature scaling is important for distance-based algorithms** and how the choice of K can affect model performance.

### Important Concept

```text
Small K → More sensitive to noise → Possible Overfitting

Large K → More generalized → Possible Underfitting
```

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
| `Day9.ipynb`             | Step-by-step implementation    |
| `Day9.py`                | Complete Python implementation |
| `requirements.txt`       | Required libraries             |
| `references.md`          | Learning resources             |
| `interview_questions.md` | Interview preparation          |

---

## 📁 Dataset Structure

```text
Day09/
│
├── StudentPerformanceFactors.csv
├── Day9.ipynb
├── Day9.py
├── README.md
├── requirements.txt
├── references.md
└── interview_questions.md
```

The same `StudentPerformanceFactors.csv` dataset used in previous days is continued here to maintain consistency throughout the challenge.

---

## 💡 Key Takeaway

**KNN classifies new observations based on their nearest neighbors, and feature scaling is essential because KNN relies on distance calculations.**

---

## 🎉 Day 9 Completed!

➡️ **Next: Day 10 – Decision Tree Classification 🌳🚀**
