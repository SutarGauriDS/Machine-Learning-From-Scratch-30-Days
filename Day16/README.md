# Day 16 — Feature Selection 🎯

A Machine Learning project focused on **Feature Selection** using the Titanic dataset.

In this project, I use **SelectKBest** with the **ANOVA F-test** to identify the most relevant features and compare a Logistic Regression model using all features with a model using only the selected features.

---

## 📌 Project Overview

Feature Selection is the process of selecting the most useful features from a dataset while removing less informative features.

The goal is to reduce unnecessary features while maintaining or improving model performance.

### Project Workflow

```text
Titanic Dataset
      ↓
Data Exploration
      ↓
Feature Engineering
      ↓
Train-Test Split
      ↓
Data Preprocessing
      ↓
Baseline Model
      ↓
SelectKBest + ANOVA F-test
      ↓
Select Top 10 Features
      ↓
Train Selected-Feature Model
      ↓
Compare Models
      ↓
Evaluate Results
````

---

## 🎯 Objectives

* Understand Feature Selection
* Identify important features
* Learn how `SelectKBest` works
* Understand the ANOVA F-test
* Select the top features
* Compare all features vs selected features
* Evaluate model performance
* Visualize feature importance scores

---

## 📊 Dataset

The project uses the **Titanic dataset**.

The target variable is:

```text
survived
```

Where:

```text
0 → Not Survived
1 → Survived
```

The project uses original Titanic features along with engineered features.

---

## 🔧 Feature Engineering

Before feature selection, the following features are created:

| Feature         | Description                                     |
| --------------- | ----------------------------------------------- |
| `FamilySize`    | Total family members travelling together        |
| `IsAlone`       | Indicates whether the passenger travelled alone |
| `FarePerPerson` | Fare divided by family size                     |
| `Title`         | Category derived from passenger information     |
| `AgeGroup`      | Age converted into meaningful groups            |

Example:

```python
df["FamilySize"] = df["sibsp"] + df["parch"] + 1
```

---

## 🔎 Feature Selection

### SelectKBest

`SelectKBest` selects the top `K` features according to a scoring function.

In this project:

```python
selector = SelectKBest(
    score_func=f_classif,
    k=10
)
```

The model selects the **top 10 processed features**.

---

## 📐 ANOVA F-test

The ANOVA F-test is used to calculate a score for each feature.

```python
from sklearn.feature_selection import f_classif
```

A higher F-score indicates a stronger statistical relationship between the feature and the classification target for this selection method.

---

## 🧹 Data Preprocessing

### Numerical Features

Numerical features are:

1. Imputed using the median
2. Standardized using `StandardScaler`

```python
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

### Categorical Features

Categorical features are:

1. Imputed using the most frequent value
2. Encoded using `OneHotEncoder`

```python
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
```

---

## 🤖 Machine Learning Model

The project uses **Logistic Regression** because the target variable is binary.

```python
LogisticRegression(
    max_iter=1000
)
```

Two models are compared:

### Model 1 — All Features

Uses all processed features.

### Model 2 — Selected Features

Uses only the top 10 features selected by `SelectKBest`.

---

## 📈 Model Evaluation

The following metrics are used:

### Accuracy

Measures the percentage of correctly classified samples.

### Classification Report

Provides:

* Precision
* Recall
* F1-score
* Support

### Confusion Matrix

Shows correct and incorrect classifications for each class.

---

## 📊 Visualizations

The notebook contains:

### 1. Survival Rate by Family Size

Shows the relationship between family size and survival rate.

### 2. Top 10 Features

Displays the highest ANOVA F-scores.

### 3. Confusion Matrix

Shows the classification performance of the selected-feature model.

### 4. Model Comparison

Compares the accuracy of:

```text
All Features
      vs
Selected Features
```

---

## 🔬 All Features vs Selected Features

```text
All Processed Features
          ↓
     SelectKBest
          ↓
     Top 10 Features
          ↓
   Logistic Regression
          ↓
     Model Evaluation
```

The notebook compares the accuracy of both approaches to understand the effect of Feature Selection.

---
## 💡 Key Learnings

Through this project, I learned:

* What Feature Selection means
* Why irrelevant features can be removed
* How `SelectKBest` works
* How `f_classif` is used for classification
* How ANOVA F-scores can rank features
* How to select the top K features
* How to compare models before and after feature selection
* How Feature Selection can affect model performance
* How to visualize selected features

---

## 🔑 Key Findings

* Feature Selection reduces the number of input features.
* `SelectKBest` can identify the strongest features according to a chosen statistical score.
* ANOVA F-test provides a score for ranking features.
* The selected-feature model can be compared directly with the baseline model.
* Feature Selection can make a machine learning workflow more focused and interpretable.

---

## 🏁 Conclusion

Feature Selection is an important step in a Machine Learning workflow.

In this project, I used **SelectKBest with the ANOVA F-test** to select the top 10 processed features from the Titanic dataset.

A Logistic Regression model using the selected features was then compared with a model using all processed features.

This project helped me understand how selecting relevant features can simplify a machine learning model while allowing us to evaluate whether model performance is maintained.

---

## 📅 30 Days of Machine Learning

### Day 16/30 — Feature Selection 🎯

Continuing my Machine Learning journey through hands-on projects, experimentation, and practical implementation.

```text
Learn
 ↓
Code
 ↓
Experiment
 ↓
Evaluate
 ↓
Improve
```
