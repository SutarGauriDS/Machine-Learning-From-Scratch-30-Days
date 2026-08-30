# Day 17 — Feature Scaling 📏

A Machine Learning project demonstrating the importance of **Feature Scaling** using the Breast Cancer Wisconsin dataset and a **K-Nearest Neighbors (KNN)** classifier.

The project compares KNN performance using:

- Original features without scaling
- `StandardScaler`
- `MinMaxScaler`

---

## 📌 Project Overview

Feature Scaling is a preprocessing technique used to bring numerical features to comparable scales.

This is especially important for distance-based algorithms such as **KNN**, because features with larger numerical ranges can have a greater influence on distance calculations.

In this project, the same KNN model is trained using different scaling approaches and their performance is compared.

---

## 🎯 Objectives

- Understand Feature Scaling
- Understand why scaling is important for KNN
- Learn Standardization
- Learn Min-Max Normalization
- Use `StandardScaler`
- Use `MinMaxScaler`
- Compare KNN with and without scaling
- Visualize feature distributions
- Evaluate model performance

---

## 📊 Dataset

This project uses the **Breast Cancer Wisconsin (Diagnostic) dataset** provided by Scikit-learn.

### Dataset Details

- **Samples:** 569
- **Features:** 30
- **Problem Type:** Binary Classification

The dataset contains numerical features extracted from breast cancer cell images.

The target contains two classes:

- Malignant
- Benign

---

## 🧠 Why Feature Scaling?

Different features can have very different numerical ranges.

For example:

```text
Feature A → 0.01 to 1
Feature B → 10 to 1000
````

For distance-based algorithms, the larger-scale feature can disproportionately influence the distance.

Feature Scaling solves this by putting features on comparable scales.

---

# 📐 Scaling Techniques

## 1. Standardization — StandardScaler

Standardization transforms features so they generally have:

```text
Mean ≈ 0
Standard Deviation ≈ 1
```

Implementation:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 2. Min-Max Normalization — MinMaxScaler

Min-Max scaling transforms values to a specified range, commonly:

```text
0 to 1
```

Implementation:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

# 🤖 Machine Learning Algorithm

## K-Nearest Neighbors (KNN)

KNN is a distance-based classification algorithm.

The model predicts the class of a new observation based on its nearest training observations.

In this project:

```python
KNeighborsClassifier(
    n_neighbors=5
)
```

is used.

The same `K=5` is used for all three experiments so that the effect of scaling can be compared fairly.

---

# 🔄 Project Workflow

```text
Breast Cancer Dataset
        ↓
Data Exploration
        ↓
Train-Test Split
        ↓
KNN Without Scaling
        ↓
StandardScaler
        ↓
KNN with StandardScaler
        ↓
MinMaxScaler
        ↓
KNN with MinMaxScaler
        ↓
Compare Accuracy
        ↓
Confusion Matrix
        ↓
Classification Report
        ↓
Visualizations
```

---

# 🧪 Experiments

Three versions of the KNN model are evaluated.

### Experiment 1 — Without Scaling

```text
Original Data
      ↓
KNN
      ↓
Accuracy
```

### Experiment 2 — StandardScaler

```text
Original Data
      ↓
StandardScaler
      ↓
KNN
      ↓
Accuracy
```

### Experiment 3 — MinMaxScaler

```text
Original Data
      ↓
MinMaxScaler
      ↓
KNN
      ↓
Accuracy
```

---

# 📊 Model Comparison

The project compares:

| Method          | Model |
| --------------- | ----- |
| Without Scaling | KNN   |
| StandardScaler  | KNN   |
| MinMaxScaler    | KNN   |

The notebook generates a bar chart showing the accuracy of all three approaches.

---

# 📦 Feature Distribution

The project also visualizes selected features before and after standardization.

Features used for visualization include:

```text
mean radius
mean texture
mean area
mean smoothness
```

Box plots are used to demonstrate the difference in feature scales.

---

# 📈 Model Evaluation

The standardized KNN model is evaluated using:

### Accuracy

Measures the percentage of correctly classified samples.

### Classification Report

Provides:

* Precision
* Recall
* F1-score
* Support

### Confusion Matrix

Shows:

* True positives
* True negatives
* False positives
* False negatives

---

# ⚠️ Avoiding Data Leakage

The scaler is fitted only on the training data:

```python
scaler.fit_transform(X_train)
```

The same fitted scaler is then applied to the test data:

```python
scaler.transform(X_test)
```

We should **not** fit the scaler separately on the test set because that would allow information from the test data to influence preprocessing.

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
Day17-Feature-Scaling/
│
├── day17_feature_scaling.py
├── Day17_Feature_Scaling_Complete.ipynb
├── README.md
└── requirements.txt
```

---

# 📦 Requirements

Create a `requirements.txt` file:

```text
pandas
matplotlib
scikit-learn
jupyter
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install pandas matplotlib scikit-learn jupyter
```

---

# ▶️ Run the Python File

```bash
python day17_feature_scaling.py
```

---

# 📓 Run the Jupyter Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Day17_Feature_Scaling_Complete.ipynb
```

Run the cells sequentially.

---

# 💡 Key Learnings

Through this project, I learned:

* What Feature Scaling means
* Why scaling is important for KNN
* How StandardScaler works
* How MinMaxScaler works
* Difference between standardization and normalization
* How to scale training and testing data correctly
* How scaling affects distance-based algorithms
* How to compare model performance
* How to visualize feature distributions
* How to evaluate classification models

---

# 🔑 Key Findings

* KNN is sensitive to feature scale because it relies on distances.
* Features with larger ranges can influence distance calculations more strongly.
* StandardScaler converts features to a standardized scale.
* MinMaxScaler converts features to a fixed range.
* The effect of scaling can be observed by comparing KNN accuracy before and after scaling.
* The best scaling method should be selected based on the characteristics of the dataset and model performance.

---

# 🏁 Conclusion

Feature Scaling is an important preprocessing step for many machine learning algorithms.

In this project, KNN was trained using the original data, StandardScaler, and MinMaxScaler.

The results demonstrate how preprocessing choices can affect a distance-based machine learning model.

The project also reinforces the importance of fitting preprocessing transformations only on training data to avoid data leakage.

---

## 👩‍💻 Author

**Gauri**

### Focus Areas

* Data Analytics
* Machine Learning
* Python
* Data Science

```
```
