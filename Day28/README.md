
# 🚀 Day 27 — End-to-End Machine Learning Pipeline

## 🍷 Wine Classification

Part of my **30 Days of Machine Learning Challenge**, focused on building practical, interview-ready machine learning projects.

---

## 🎯 Project Objective

The objective of this project is to build a complete **end-to-end machine learning pipeline** for classifying different types of wine.

Instead of handling preprocessing and model training separately, this project combines them into a single **Scikit-learn Pipeline**.

The project demonstrates how a machine learning workflow can move from **raw data to a trained, evaluated, saved, and reusable model**.

---

## 📊 Dataset

This project uses the **Wine Dataset built into Scikit-learn**.

The dataset contains:

- 178 samples
- 13 numerical features
- 3 wine classes

The features represent different chemical properties of wine.

### Target Classes

```text
0 → class_0
1 → class_1
2 → class_2
````

---

## 🔄 End-to-End ML Workflow

```text
Built-in Wine Dataset
        ↓
Data Understanding
        ↓
Exploratory Analysis
        ↓
Train-Test Split
        ↓
Missing Value Handling
        ↓
Feature Scaling
        ↓
Pipeline Creation
        ↓
Logistic Regression
        ↓
Model Evaluation
        ↓
Confusion Matrix
        ↓
Save Results
        ↓
Save Complete Pipeline
        ↓
Load Pipeline
        ↓
New Prediction
```

---

## 🧠 Machine Learning Model

### Logistic Regression

Logistic Regression is used as the classification algorithm.

Although the model is relatively simple, it provides a strong baseline and allows us to demonstrate the complete ML pipeline workflow.

---

## 🔧 Preprocessing

The preprocessing stage includes:

### 1. Missing Value Imputation

`SimpleImputer` with the median strategy is used to handle missing numerical values.

### 2. Feature Scaling

`StandardScaler` is used to standardize the features before training Logistic Regression.

---

## 🔗 Scikit-learn Pipeline

The preprocessing and model are combined into a single pipeline:

```python
Pipeline([
    ("preprocessing", preprocessing),
    ("model", model)
])
```

This ensures that the same preprocessing steps are automatically applied during both training and prediction.

---

## 📈 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix

For this multiclass problem, weighted Precision, Recall, and F1-score are used.

---

## 💾 Model Persistence

The complete ML pipeline is saved using **Joblib**.

```text
day27_wine_classification_pipeline.pkl
```

The saved pipeline contains both:

* Preprocessing steps
* Trained machine learning model

It can then be loaded and reused for predictions without rebuilding the pipeline.

---

## 📁 Project Structure

```text
Day27_End_to_End_ML/
│
├── day27_end_to_end_ml_pipeline.py
├── Day27_End_to_End_ML_Pipeline_Complete.ipynb
├── Day27_Model_Results.csv
├── day27_wine_classification_pipeline.pkl
├── README.md
├── requirements.txt
└── references.md
```

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Joblib
* Jupyter Notebook
* Google Colab

---

## 🎯 Key Learnings

1. Built an end-to-end machine learning workflow.
2. Used a built-in Scikit-learn dataset.
3. Performed train-test splitting.
4. Implemented missing-value handling.
5. Applied feature scaling.
6. Combined preprocessing and model training using Pipeline.
7. Evaluated a multiclass classification model.
8. Created a confusion matrix and classification report.
9. Saved model results to CSV.
10. Saved and loaded the complete ML pipeline using Joblib.
11. Used the saved pipeline for a new prediction.

---

## 💼 Why This Project Matters

A real ML project does not end after training a model.

An industry workflow usually involves:

**Data → Preprocessing → Model → Evaluation → Serialization → Prediction**

This project demonstrates that complete workflow and provides a foundation for the next step: **deploying a machine learning model as an application or API.**

---

## 🚀 Challenge Progress

**Day 27/30 — End-to-End Machine Learning Pipeline completed! 🎯**

Next → **Day 28: Model Deployment Basics 🚀**
