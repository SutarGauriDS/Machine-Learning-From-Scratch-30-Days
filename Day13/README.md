# Day 13 — Support Vector Machine (SVM)

A machine learning classification project implementing **Support Vector Machine (SVM)** using the Breast Cancer dataset provided by Scikit-learn.

This project compares **Linear SVM** and **RBF (Non-Linear) SVM** after applying feature scaling.

---

## 📌 Project Overview

Support Vector Machine is a supervised machine learning algorithm mainly used for classification and regression tasks.

In this project, SVM is used to classify breast cancer cases into two categories based on several medical features.

The project demonstrates:

- Data loading
- Train-test splitting
- Feature scaling
- Linear SVM
- RBF / Non-Linear SVM
- Model comparison
- Accuracy evaluation
- Classification report
- Confusion matrix
- Visualization

---

## 📂 Dataset

The project uses the **Breast Cancer Wisconsin dataset** built into Scikit-learn.

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
````

### Dataset Details

* Samples: **569**
* Features: **30**
* Classes: **2**

The two classes represent:

* Malignant
* Benign

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
Dataset
   ↓
Data Exploration
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Linear SVM
   ↓
RBF SVM
   ↓
Model Comparison
   ↓
Evaluation
   ↓
Confusion Matrix
```

---

## 🧠 Models Used

### 1. Linear SVM

```python
SVC(
    kernel="linear",
    C=1
)
```

The linear kernel attempts to separate the classes using a linear decision boundary.

### 2. RBF SVM

```python
SVC(
    kernel="rbf",
    C=1,
    gamma="scale"
)
```

The RBF kernel can handle non-linear relationships between features and target classes.

---

## 📊 Feature Scaling

SVM is sensitive to the scale of features.

Therefore, `StandardScaler` is used:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data to avoid data leakage.

---

## 📈 Evaluation

The models are evaluated using:

### Accuracy

Measures the percentage of correctly classified samples.

### Classification Report

Includes:

* Precision
* Recall
* F1-score

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

## 📊 Visualizations

The project generates:

1. Target class distribution
2. Linear SVM vs RBF SVM accuracy comparison
3. Confusion matrix

---

## 📁 Project Structure

```text
Day13-SVM/
│
├── day13_svm_classification.py
├── Day13_SVM_Classification.ipynb
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project:

```bash
cd Day13-SVM
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Python File

```bash
python day13_svm_classification.py
```

---

## 📓 Run Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
Day13_SVM_Classification.ipynb
```

---

## 📦 Requirements

Create a `requirements.txt` file:

```text
pandas
matplotlib
scikit-learn
jupyter
```

Install:

```bash
pip install -r requirements.txt
```

---

## 🎯 Key Learnings

Through this project, I practiced:

* Understanding SVM classification
* Difference between linear and non-linear kernels
* Importance of feature scaling
* Train-test splitting
* Model evaluation
* Confusion matrix interpretation
* Comparing machine learning models

---

## 🚀 Future Improvements

The next step is to explore **SVM hyperparameter tuning** using:

```text
GridSearchCV
```

Parameters such as:

* `C`
* `kernel`
* `gamma`

can be optimized to find a better-performing SVM model.

---

## 📚 30 Days ML Challenge

**Day 13/30 — Support Vector Machine (SVM)**

Continuing my journey of learning and implementing Machine Learning concepts through hands-on projects.

---

## 👩‍💻 Author

**Gauri**
