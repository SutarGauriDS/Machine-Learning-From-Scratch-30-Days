Day 14 — SVM Hyperparameter Tuning 🔍

A Machine Learning project focused on improving a **Support Vector Machine (SVM)** classifier using **GridSearchCV** and **5-Fold Cross-Validation**.

This project builds on **Day 13**, where Linear SVM and RBF SVM were implemented and compared.

---

## 📌 Project Overview

Hyperparameter tuning is the process of finding the best settings for a machine learning model.

In this project, different SVM hyperparameters are tested systematically using `GridSearchCV` to find the combination that provides the best cross-validation performance.

The project compares a **Baseline SVM** with a **Tuned SVM**.

---

## 🎯 Objectives

- Understand SVM hyperparameters
- Learn hyperparameter tuning
- Use `GridSearchCV`
- Apply 5-Fold Cross-Validation
- Find the best `C`, `kernel`, and `gamma`
- Compare baseline and tuned SVM
- Evaluate model performance
- Visualize the results

---

## 📊 Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) dataset** provided by Scikit-learn.

### Dataset Details

- **Samples:** 569
- **Features:** 30
- **Classes:** 2
- **Problem Type:** Binary Classification

The target classes are:

- Malignant
- Benign

The dataset is loaded directly using:

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
````

---

## 🛠️ Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

## 🔄 Project Workflow

```text
Breast Cancer Dataset
        ↓
Data Preparation
        ↓
Train-Test Split
        ↓
Feature Scaling
        ↓
Baseline SVM
        ↓
Define Hyperparameter Grid
        ↓
GridSearchCV
        ↓
5-Fold Cross-Validation
        ↓
Best Parameters
        ↓
Tuned SVM
        ↓
Model Evaluation
        ↓
Visualization
```

---

# 🧠 Hyperparameter Tuning

Hyperparameters are model settings that are selected before training.

For SVM, this project tunes:

```text
C
Kernel
Gamma
```

Instead of manually selecting values, `GridSearchCV` tests multiple combinations and selects the best-performing configuration.

---

# 🔧 Hyperparameters

## 1. C

`C` controls how strongly the SVM penalizes classification errors.

Values tested:

```python
[0.1, 1, 10, 100]
```

### Lower C

* Allows more classification errors
* Generally produces a wider margin
* Can create a simpler decision boundary

### Higher C

* Penalizes classification errors more strongly
* Tries harder to classify training samples correctly
* Can create a more complex decision boundary

---

## 2. Kernel

The kernel determines how SVM creates its decision boundary.

The project tests:

```python
["linear", "rbf", "poly"]
```

### Linear Kernel

Used when the classes can be separated using a linear decision boundary.

### RBF Kernel

Useful for non-linear relationships between features and classes.

### Polynomial Kernel

Creates a polynomial decision boundary.

---

## 3. Gamma

`gamma` controls the influence of individual training points for kernels such as RBF and polynomial.

Values tested:

```python
["scale", "auto"]
```

A higher effective gamma generally produces a more localized and complex decision boundary, while a lower gamma produces a smoother boundary.

---

# ⚙️ Feature Scaling

SVM is sensitive to feature scales.

Therefore, `StandardScaler` is used before training.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data.

This prevents information from the test set from leaking into the training process.

---

# 🤖 Baseline SVM

Before hyperparameter tuning, a baseline RBF SVM is trained:

```python
SVC(
    kernel="rbf",
    C=1,
    gamma="scale"
)
```

The baseline model provides a reference point for measuring whether hyperparameter tuning improves performance.

---

# 🔎 GridSearchCV

`GridSearchCV` systematically evaluates different combinations of hyperparameters.

The parameter grid used in this project is:

```python
param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__kernel": ["linear", "rbf", "poly"],
    "svm__gamma": ["scale", "auto"]
}
```

GridSearchCV is configured with:

```python
GridSearchCV(
    estimator=baseline_model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
```

---

# 🔁 5-Fold Cross-Validation

The training data is divided into **5 folds**.

The model is trained and validated multiple times using different combinations of training and validation folds.

This helps provide a more reliable estimate of model performance when selecting hyperparameters.

---

# 🏆 Best Parameters

The best hyperparameter combination is obtained using:

```python
grid_search.best_params_
```

The best cross-validation accuracy is obtained using:

```python
grid_search.best_score_
```

The best model is obtained using:

```python
grid_search.best_estimator_
```

---

# 📈 Model Evaluation

The tuned model is evaluated on the unseen test dataset.

The project uses:

### Accuracy

Measures the proportion of correctly classified samples.

### Classification Report

Provides:

* Precision
* Recall
* F1-score
* Support

### Confusion Matrix

Shows the number of correct and incorrect predictions for each class.

---

# 📊 Visualizations

The project generates visualizations for:

### 1. Confusion Matrix

Shows correct and incorrect classifications for each target class.

### 2. Top Hyperparameter Combinations

Displays the highest-performing parameter combinations based on cross-validation accuracy.

### 3. Baseline vs Tuned SVM

Compares the accuracy of the baseline and tuned models.

---

# 📁 Project Structure

```text
Day14-SVM-Hyperparameter-Tuning/
│
├── day14_svm_hyperparameter_tuning.py
│
├── Day14_SVM_Hyperparameter_Tuning.ipynb
│
├── README.md
│
└── requirements.txt
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd Day14-SVM-Hyperparameter-Tuning
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Python File

```bash
python day14_svm_hyperparameter_tuning.py
```

---

# 📓 Run the Jupyter Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Day14_SVM_Hyperparameter_Tuning.ipynb
```

Run the cells sequentially.

---

# 📊 Expected Output

The project displays:

```text
Dataset Shape
Training Samples
Testing Samples

Baseline SVM Accuracy

Best Parameters
Best Cross-Validation Accuracy

Tuned SVM Test Accuracy

Baseline vs Tuned Accuracy

Classification Report

Confusion Matrix

Top Hyperparameter Combinations
```

The exact results may vary if the dataset, random state, or parameter grid is changed.

---

# 💡 Key Learnings

Through this project, I learned:

* How SVM hyperparameters affect model performance
* How `C` controls regularization
* How different kernels affect decision boundaries
* How `gamma` affects non-linear SVM
* Why feature scaling is important for SVM
* How GridSearchCV works
* How 5-Fold Cross-Validation works
* How to select the best hyperparameters
* How to compare baseline and tuned models
* How to evaluate a classification model

---

# 🔬 Day 13 vs Day 14

| Day    | Topic                     | Main Focus                      |
| ------ | ------------------------- | ------------------------------- |
| Day 13 | SVM Classification        | Linear SVM vs RBF SVM           |
| Day 14 | SVM Hyperparameter Tuning | GridSearchCV + Cross-Validation |

### Progression

```text
Day 13
SVM Classification
      ↓
Linear SVM vs RBF SVM
      ↓
Day 14
Hyperparameter Tuning
      ↓
C + Kernel + Gamma
      ↓
GridSearchCV
      ↓
Best SVM
```

---

# 🚀 Future Improvements

Possible improvements include:

* Test the model on a real-world Kaggle dataset
* Explore additional SVM kernels
* Try `RandomizedSearchCV`
* Tune additional hyperparameters
* Compare SVM with Logistic Regression
* Compare SVM with Random Forest
* Evaluate using additional metrics
* Apply the optimized model to a real-world application

---

# 📅 30 Days of Machine Learning

### Day 14/30 — SVM Hyperparameter Tuning

Continuing my journey of learning Machine Learning through hands-on implementation, experimentation, and practical projects.

---

## 👩‍💻 Author

**Gauri**

Learning and building projects in:

* Data Analytics
* Machine Learning
* Python
* Data Science

```
```
