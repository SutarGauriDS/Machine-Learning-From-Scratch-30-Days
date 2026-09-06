# Day 24 — Model Evaluation

## Classification Model Evaluation using Multiple Metrics

Part of my **30 Days of Machine Learning Challenge**.

### 🎯 Objective

Evaluate a **Random Forest Classifier** using multiple performance metrics instead of relying only on accuracy.

### 🧠 Concepts Covered

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report
* ROC Curve
* ROC-AUC
* Feature Importance

### ⚙️ Workflow

```text
Breast Cancer Dataset
        ↓
Train-Test Split
        ↓
Random Forest Classifier
        ↓
Predictions
        ↓
Evaluation Metrics
        ↓
Confusion Matrix
        ↓
ROC Curve & AUC
        ↓
Feature Importance
```

### 📊 Dataset

**Breast Cancer Wisconsin Dataset** from Scikit-learn.

* 569 samples
* 30 numerical features
* Binary classification

### 📈 Results

The project evaluates the model using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

It also analyzes the **confusion matrix** and identifies the most important features.

### 💡 Key Takeaway

A model should not be evaluated using accuracy alone. **Precision, Recall, F1-Score, and ROC-AUC** provide additional insights into how well a classification model performs.

### 🛠️ Technologies

Python | Pandas | Scikit-learn | Matplotlib | Jupyter

### 📂 Files

* `day24_model_evaluation.py`
* `Day24_Model_Evaluation_Complete.ipynb`
* `README.md`
* `requirements.txt`
* `references.md`

### 🚀 Day 24/30 Completed

**Next:** Day 25 — Ensemble Learning
