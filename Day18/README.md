# Day 18 — PCA (Principal Component Analysis) 🧠

A Machine Learning project demonstrating **Principal Component Analysis (PCA)** for dimensionality reduction using the **Breast Cancer Wisconsin (Diagnostic) dataset**.

The project compares **Logistic Regression before and after PCA** to understand how dimensionality reduction affects model performance.

---

## 📌 Project Overview

PCA is a dimensionality reduction technique that transforms a large number of correlated features into a smaller number of **principal components** while retaining important information from the original dataset.

In this project:

```text
Original Features
       ↓
Feature Scaling
       ↓
PCA
       ↓
Explained Variance Analysis
       ↓
Select Components
       ↓
Reduced Dataset
       ↓
Logistic Regression
       ↓
Model Evaluation
````

---

## 🎯 Objectives

* Understand Dimensionality Reduction
* Understand Principal Component Analysis
* Apply Feature Scaling before PCA
* Calculate Explained Variance
* Analyze Cumulative Explained Variance
* Select components retaining at least 95% variance
* Reduce the number of features
* Visualize high-dimensional data in 2D
* Compare model performance before and after PCA

---

## 📊 Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) dataset** provided by Scikit-learn.

### Dataset Details

| Property          |                 Value |
| ----------------- | --------------------: |
| Samples           |                   569 |
| Original Features |                    30 |
| Problem Type      | Binary Classification |

The dataset contains numerical measurements related to breast cancer cell characteristics.

### Target

The dataset contains two target classes:

* Malignant
* Benign

---

# 🧠 What is PCA?

**Principal Component Analysis (PCA)** is a dimensionality reduction technique.

Instead of using all original features, PCA creates new features called **Principal Components**.

The first principal component captures the maximum possible variance, while subsequent components capture additional variance while remaining orthogonal to previous components.

### Simple Representation

```text
30 Original Features
        ↓
       PCA
        ↓
Fewer Principal Components
        ↓
Retain Important Information
```

---

# 📐 Why Scaling Before PCA?

PCA is sensitive to feature magnitudes.

If one feature has a much larger numerical range than another, it can have a disproportionate influence on PCA.

Therefore, the project first applies:

```python
StandardScaler()
```

The scaler is fitted only on the training data to avoid data leakage.

---

# 📊 Explained Variance

**Explained Variance Ratio** tells us how much of the total variance is captured by each principal component.

For example:

```text
PC1 → captures some percentage of variance
PC2 → captures additional variance
PC3 → captures additional variance
...
```

The cumulative explained variance helps determine how many components should be retained.

---

# 🎯 Selecting Components

The project selects the smallest number of components that retains at least **95% of the variance**.

```python
n_components_95 = (
    cumulative_variance >= 0.95
).argmax() + 1
```

This allows dimensionality to be reduced while retaining most of the information represented by the original features.

---

# 🤖 Machine Learning Model

The project uses **Logistic Regression** for binary classification.

Two models are compared:

### Model 1 — Without PCA

Uses all 30 standardized features.

```text
30 Features
    ↓
Logistic Regression
    ↓
Prediction
```

### Model 2 — With PCA

Uses the reduced principal-component representation.

```text
30 Features
    ↓
PCA
    ↓
Reduced Components
    ↓
Logistic Regression
    ↓
Prediction
```

---

# 🔬 Model Comparison

The project compares:

| Model                             | Input                         |
| --------------------------------- | ----------------------------- |
| Logistic Regression — Without PCA | All standardized features     |
| Logistic Regression — With PCA    | Selected principal components |

The accuracy difference is calculated to understand the effect of dimensionality reduction.

---

# 🗺️ 2D PCA Visualization

PCA is also applied with:

```python
PCA(n_components=2)
```

This reduces the data to two dimensions.

The two principal components can then be plotted to visually examine the separation between the target classes.

```text
High-Dimensional Data
        ↓
       PCA
        ↓
     PC1 + PC2
        ↓
   2D Visualization
```

---

# 📈 Model Evaluation

The PCA-based Logistic Regression model is evaluated using:

### Accuracy

Measures the percentage of correctly classified samples.

### Classification Report

Includes:

* Precision
* Recall
* F1-score
* Support

### Confusion Matrix

Shows the classification results for the two target classes.

---

# 📊 Visualizations

The notebook contains:

### 1. Cumulative Explained Variance

Shows how much variance is retained as more principal components are added.

### 2. Model Accuracy Comparison

Compares Logistic Regression before and after PCA.

### 3. 2D PCA Visualization

Projects the high-dimensional dataset into two principal components.

### 4. Confusion Matrix

Visualizes classification results from the PCA model.

---

# 📉 Feature Reduction

The project calculates:

* Original number of features
* Number of selected PCA components
* Percentage of feature reduction
* Total variance retained

This demonstrates how PCA can reduce dimensionality while preserving important information.

---

# 🔄 Complete Project Workflow

```text
Breast Cancer Dataset
          ↓
Data Exploration
          ↓
Train-Test Split
          ↓
StandardScaler
          ↓
Baseline Logistic Regression
          ↓
Apply PCA
          ↓
Explained Variance
          ↓
Cumulative Variance
          ↓
Select Components for 95% Variance
          ↓
Reduced Dataset
          ↓
Logistic Regression
          ↓
Model Comparison
          ↓
2D PCA Visualization
          ↓
Confusion Matrix
          ↓
Classification Report
```

---

# ⚠️ Data Leakage Prevention

The scaler and PCA are fitted only using the training data.

### Training data

```python
X_train_scaled = scaler.fit_transform(X_train)
```

### Test data

```python
X_test_scaled = scaler.transform(X_test)
```

Similarly, PCA is fitted on the training data and then applied to the test data.

This prevents information from the test set from influencing model training.

---

# 🛠️ Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

# 📁 Project Structure

```text
Day18-PCA/
│
├── day18_pca.py
├── Day18_PCA_Complete.ipynb
├── README.md
└── requirements.txt
```

---

# 📦 Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Or install them directly:

```bash
pip install pandas matplotlib scikit-learn jupyter
```

---

# ▶️ Run Python File

```bash
python day18_pca.py
```

---

# 📓 Run Jupyter Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Day18_PCA_Complete.ipynb
```

Run the cells sequentially.

---

# 💡 Key Learnings

Through this project, I learned:

* What dimensionality reduction means
* What PCA is
* How PCA creates principal components
* Why feature scaling is important before PCA
* What explained variance means
* How cumulative explained variance helps select components
* How to reduce the dimensionality of a dataset
* How to visualize high-dimensional data in 2D
* How to compare model performance before and after PCA
* How to avoid data leakage

---

# 🔑 Key Findings

* PCA can represent high-dimensional data using fewer components.
* Standardization should be considered before PCA when features have different scales.
* Explained variance helps determine how many components should be retained.
* The project retains at least 95% of the variance.
* PCA makes it possible to visualize high-dimensional data in two dimensions.
* Model performance can be compared before and after dimensionality reduction.

---

# 🏁 Conclusion

PCA is an important dimensionality reduction technique in Machine Learning.

In this project, the Breast Cancer Wisconsin dataset was standardized and PCA was applied to reduce the number of dimensions while retaining at least 95% of the variance.

Logistic Regression was then trained using the reduced representation and compared with a baseline model using all standardized features.

This project demonstrates how PCA can simplify high-dimensional datasets while preserving a large portion of the information contained in the original features.

---

# 📚 References

1. **Scikit-learn — PCA**
   [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

2. **Scikit-learn — StandardScaler**
   [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

3. **Scikit-learn — Logistic Regression**
   [https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

4. **Scikit-learn — Breast Cancer Dataset**
   [https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

5. **Scikit-learn — Decomposing Signals using PCA**
   [https://scikit-learn.org/stable/modules/decomposition.html](https://scikit-learn.org/stable/modules/decomposition.html)

---

# 📅 30 Days of Machine Learning

## Day 18/30 — PCA 🧠

Continuing my Machine Learning journey through hands-on implementation and practical experimentation.

```text
Learn
  ↓
Code
  ↓
Experiment
  ↓
Reduce Dimensions
  ↓
Evaluate
```

---

## 👩‍💻 Author

**Gauri**
