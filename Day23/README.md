# Day 23 — Cross Validation

## Model Performance Evaluation using K-Fold Cross Validation

### 🎯 Objective

Use **Cross Validation** to evaluate and compare multiple machine learning classification models more reliably.

### 🧠 Concepts Covered

* Cross Validation
* K-Fold Cross Validation
* Stratified K-Fold
* Model Evaluation
* Mean Accuracy
* Standard Deviation
* Model Stability
* Model Comparison

### ⚙️ Workflow

```text
Breast Cancer Dataset
        ↓
Train-Test Split
        ↓
Define ML Models
        ↓
5-Fold Stratified Cross Validation
        ↓
Calculate Mean & Std Accuracy
        ↓
Compare Models
        ↓
Select Best Model
        ↓
Final Test Evaluation
```

### 🤖 Models Used

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Random Forest

### 📊 Dataset

**Breast Cancer Wisconsin Dataset** from Scikit-learn.

* 569 samples
* 30 numerical features
* Binary classification

### 📈 Results

The project:

* Evaluates each model across 5 folds
* Calculates mean cross-validation accuracy
* Measures performance consistency using standard deviation
* Compares the models
* Evaluates the best model on unseen test data

### 💡 Key Takeaway

Cross-validation provides a more reliable estimate of model performance than relying on a single validation split. **Mean accuracy** shows average performance, while **standard deviation** helps understand model stability.

### 🛠️ Technologies

Python | Pandas | Scikit-learn | Jupyter

### 📂 Files

* `day23_cross_validation.py`
* `Day23_Cross_Validation_Complete.ipynb`
* `README.md`
* `requirements.txt`
* `references.md`

### 🚀 Day 23/30 Completed

**Next:** Day 24 — Model Evaluation
