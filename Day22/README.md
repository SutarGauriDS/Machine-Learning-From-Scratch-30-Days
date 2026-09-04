## Day 22 — Hyperparameter Tuning

## Optimizing a Random Forest Classifier

Part of my **30 Days of Machine Learning Challenge**.

### 🎯 Objective

Use **GridSearchCV** to find the best hyperparameters for a Random Forest Classifier and compare the tuned model with a baseline model.

### 🧠 Concepts Covered

- Hyperparameters
- Hyperparameter Tuning
- GridSearchCV
- 5-Fold Cross-Validation
- Random Forest
- Model Evaluation
- Confusion Matrix
- Feature Importance

### ⚙️ Workflow

```text
Breast Cancer Dataset
        ↓
Train-Test Split
        ↓
Baseline Random Forest
        ↓
Define Hyperparameter Grid
        ↓
GridSearchCV
        ↓
5-Fold Cross-Validation
        ↓
Best Parameters
        ↓
Tuned Random Forest
        ↓
Model Evaluation
        ↓
Compare Results
````

### 📊 Dataset

**Breast Cancer Wisconsin Dataset** from Scikit-learn.

* 569 samples
* 30 numerical features
* Binary classification

### 🔍 Hyperparameters Tuned

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`

### 📈 Results

The project compares:

* Baseline Random Forest accuracy
* Best cross-validation accuracy
* Tuned Random Forest accuracy

It also includes a confusion matrix and feature importance analysis.

### 💡 Key Takeaway

Hyperparameter tuning helps improve model performance by systematically searching for a better combination of model settings instead of relying only on default parameters.

### 🛠️ Technologies

Python | Pandas | Scikit-learn | Matplotlib | Jupyter

### 📂 Files

* `day22_hyperparameter_tuning.py`
* `Day22_Hyperparameter_Tuning_Complete.ipynb`
* `README.md`
* `requirements.txt`
* `references.md`

### 🚀 Day 22/30 Completed

**Next:** Day 23 — Cross Validation

#MachineLearning #HyperparameterTuning #GridSearchCV #RandomForest #Python #DataScience #DataAnalytics #30DaysOfMachineLearning
