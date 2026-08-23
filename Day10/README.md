# 🌳 Day 10 - Decision Tree Classification

## 🎯 Objective

Build a Decision Tree Classification model to predict student performance using:

- Hours Studied
- Attendance
- Previous Scores
- Sleep Hours
- Tutoring Sessions
- Physical Activity

The target variable `Pass` is created from the student's `Exam_Score`.

---

## 📂 Dataset

**Dataset:** Student Performance Factors

The dataset contains information related to students':

- Study habits
- Attendance
- Previous academic performance
- Sleep
- Tutoring
- Physical activity
- Exam performance

The original dataset does not contain a `Pass/Fail` target.

Therefore, the target is created using the median exam score:

```python
median_score = df["Exam_Score"].median()

df["Pass"] = (
    df["Exam_Score"] >= median_score
).astype(int)
````

Where:

* `1` → Pass
* `0` → Fail

Using the median ensures that the classification problem contains both classes.

---

## 🧠 Machine Learning Algorithm

### Decision Tree Classifier

A Decision Tree is a supervised machine learning algorithm used for classification and regression.

The algorithm repeatedly splits the data based on feature values to create groups that are as pure as possible.

### Criterion

This project uses:

```python
criterion="gini"
```

Gini impurity measures how mixed the classes are within a node.

---

## 🔧 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook / Google Colab

---

## 📊 Features Used

| Feature             | Description                    |
| ------------------- | ------------------------------ |
| `Hours_Studied`     | Number of hours spent studying |
| `Attendance`        | Student attendance             |
| `Previous_Scores`   | Previous academic scores       |
| `Sleep_Hours`       | Average sleep duration         |
| `Tutoring_Sessions` | Number of tutoring sessions    |
| `Physical_Activity` | Physical activity level        |

### Target

| Value | Meaning |
| ----- | ------- |
| `0`   | Fail    |
| `1`   | Pass    |

---

## ⚙️ Machine Learning Workflow

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
Decision Tree Classifier
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Feature Importance
   ↓
New Student Prediction
```

---

## 📈 Model Evaluation

The following evaluation metrics are used:

### Accuracy

Measures the percentage of correctly classified students.

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

### Classification Report

Includes:

* Precision
* Recall
* F1-score
* Support

---

## 🌟 Feature Importance

Decision Trees provide feature importance values that show how much each feature contributes to the model's decisions.

The project visualizes feature importance using a bar chart.

```python
model.feature_importances_
```

---

## 🌳 Decision Tree Visualization

The trained Decision Tree is visualized using:

```python
plot_tree()
```

This helps understand how the model makes classification decisions.

---

## 🔮 New Student Prediction

The trained model can also predict the performance of a new student.

Example:

```python
new_student = pd.DataFrame({
    "Hours_Studied": [6],
    "Attendance": [85],
    "Previous_Scores": [70],
    "Sleep_Hours": [7],
    "Tutoring_Sessions": [2],
    "Physical_Activity": [3]
})
```

The model predicts either:

```text
Student is likely to PASS
```

or

```text
Student is likely to FAIL
```

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 2. Open the notebook

```text
Day10_Decision_Tree_Classification.ipynb
```

Run the cells from top to bottom.

### 3. Run the Python file

```bash
python Day10_Decision_Tree_Classification.py
```

If using Google Colab, update the dataset path if necessary:

```python
df = pd.read_csv("/content/StudentPerformanceFactors.csv")
```

---

## 📚 Key Concepts Learned

* Supervised Learning
* Classification
* Decision Trees
* Gini Impurity
* Train/Test Split
* Model Training
* Predictions
* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score
* Feature Importance
* Overfitting
* Tree Depth

---

## ⚠️ Important Note

`Exam_Score` is used to create the `Pass` target, so `Exam_Score` is **not included as an input feature**.

This prevents **target leakage**, where the model would receive information directly related to the answer it is supposed to predict.

---

## 🚀 Day 10 Outcome

Successfully built a **Decision Tree Classification model** for student performance prediction and learned how to evaluate and interpret a classification model.

---

## 👩‍💻 30 Days ML Challenge

**Day 10/30 — Completed ✅**

Next: **Day 11 — Support Vector Machine (SVM)**

