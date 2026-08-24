# 📊 Day 13 - Machine Learning Model Comparison

## 🚀 30 Days Machine Learning Challenge

For Day 13 of my **30 Days Machine Learning Challenge**, I compared three classification algorithms using the **Student Performance Factors** dataset:

- 🌳 Decision Tree
- 🧠 Gaussian Naive Bayes
- 🌲 Random Forest

The goal was to determine which model performs best for predicting whether a student is likely to **Pass or Fail**.

---

## 🎯 Objective

Instead of evaluating a single machine learning model, this project compares multiple models using the same dataset, features, training data, and testing data.

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Training Accuracy
- Testing Accuracy
- Confusion Matrix

---

## 📂 Dataset

### Student Performance Factors

The dataset contains information about different factors affecting student performance.

### Features Used

- `Hours_Studied`
- `Attendance`
- `Previous_Scores`
- `Sleep_Hours`
- `Tutoring_Sessions`
- `Physical_Activity`

### Target Variable

The original dataset does not contain a Pass/Fail target.

A classification target was created using the median `Exam_Score`.

```python
median_score = df["Exam_Score"].median()

df["Pass"] = (
    df["Exam_Score"] >= median_score
).astype(int)
````

Where:

```text
0 → Fail
1 → Pass
```

---

## ⚠️ Target Leakage Prevention

`Exam_Score` is used to create the `Pass` target.

Therefore, `Exam_Score` is **not included as an input feature**.

This prevents target leakage, where information directly related to the target could artificially improve model performance.

---

# 🤖 Models Compared

## 1. Decision Tree

A Decision Tree creates a series of decision rules to classify observations.

```python
DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)
```

---

## 2. Gaussian Naive Bayes

Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.

```python
GaussianNB()
```

Gaussian Naive Bayes is suitable for the numerical features used in this project.

---

## 3. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees to produce a more robust prediction.

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
```

---

# 🔄 Machine Learning Workflow

```text
Student Performance Dataset
          ↓
     Data Loading
          ↓
   Data Exploration
          ↓
   Remove Duplicates
          ↓
   Create Pass/Fail
          ↓
    Feature Selection
          ↓
     Train/Test Split
          ↓
   ┌──────┼─────────┐
   ↓      ↓         ↓
Decision Naive    Random
 Tree    Bayes    Forest
   ↓      ↓         ↓
   └──────┼─────────┘
          ↓
   Model Evaluation
          ↓
    Model Comparison
          ↓
   Select Best Model
```

---

# 📊 Evaluation Metrics

### Accuracy

Measures the percentage of correctly classified observations.

### Precision

Measures how many predicted positive cases were actually positive.

### Recall

Measures how many actual positive cases were correctly identified.

### F1-score

Provides a balance between Precision and Recall.

```text
F1 Score = 2 × (Precision × Recall)
           ---------------------------
           (Precision + Recall)
```

### Confusion Matrix

Used to understand:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

# 📈 Model Comparison

The project creates a comparison table containing:

| Model         | Training Accuracy | Testing Accuracy |  Precision |     Recall |   F1 Score |
| ------------- | ----------------: | ---------------: | ---------: | ---------: | ---------: |
| Decision Tree |        Calculated |       Calculated | Calculated | Calculated | Calculated |
| Naive Bayes   |        Calculated |       Calculated | Calculated | Calculated | Calculated |
| Random Forest |        Calculated |       Calculated | Calculated | Calculated | Calculated |

The actual values are generated when the Python script is executed.

---

# 🏆 Best Model

The best model is selected based on the **highest F1-score**.

```python
best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]
```

This approach avoids selecting a model based only on accuracy.

---

# 🌟 Feature Importance

Feature importance is analyzed for:

* Decision Tree
* Random Forest

This helps identify which student-related factors contribute most to the model's classification decisions.

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook
* Google Colab

---

# 📁 Project Structure

```text
Day13-Model-Comparison/
│
├── StudentPerformanceFactors.csv
├── Day13_Model_Comparison.ipynb
├── Day13_Model_Comparison.py
├── README.md
├── requirements.txt
└── REFERENCES.md
```

---

# ▶️ How to Run

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Run Python File

```bash
python Day13_Model_Comparison.py
```

## 3. Run Jupyter Notebook

Open:

```text
Day13_Model_Comparison.ipynb
```

and run the cells sequentially.

---

# 🧠 Key Learnings

Through this project, I learned:

* Why model comparison is important
* Difference between Decision Tree, Naive Bayes and Random Forest
* Accuracy vs Precision vs Recall vs F1-score
* Training vs Testing performance
* Confusion Matrix interpretation
* Feature Importance
* Model selection
* Target Leakage
* Why the highest accuracy isn't always the best model

---

# 💡 Key Takeaway

> Building a machine learning model is only one part of the process. Comparing multiple models using appropriate evaluation metrics is essential for selecting the most suitable model for a problem.

---

# ✅ Day 13 Outcome

Successfully compared three machine learning classification algorithms on the Student Performance dataset and identified the best-performing model using evaluation metrics.

---

## 🚀 30 Days ML Challenge

**Day 13/30 — Model Comparison Completed ✅**

### Models Compared:

🌳 Decision Tree
🧠 Naive Bayes
🌲 Random Forest
