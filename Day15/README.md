# Day 15 — Feature Engineering 🚀

A Machine Learning project demonstrating **Feature Engineering** using the Titanic dataset.

Feature Engineering is the process of creating meaningful and useful features from existing raw data to help a machine learning model identify patterns more effectively.

---

## 📌 Project Overview

In this project, the Titanic dataset is used to create new features from existing passenger information.

The newly engineered features are then used to train a **Logistic Regression** classification model.

### Main steps:

- Load the Titanic dataset
- Explore the data
- Check missing values
- Create new features
- Analyze engineered features
- Prepare numerical and categorical features
- Handle missing values
- Scale numerical features
- Encode categorical features
- Train Logistic Regression
- Evaluate model performance
- Visualize results

---

## 🎯 Objectives

The main objectives of this project are:

- Understand the importance of Feature Engineering
- Create meaningful features from existing columns
- Handle missing values
- Transform categorical variables
- Scale numerical variables
- Build a machine learning pipeline
- Evaluate the effect of engineered features on a classification model

---

## 📊 Dataset

This project uses the **Titanic dataset**.

The dataset contains information about passengers who travelled on the Titanic.

### Target Variable

```text
survived
````

Where:

* `0` = Did not survive
* `1` = Survived

### Important Original Features

Some of the original columns include:

* `pclass`
* `sex`
* `age`
* `sibsp`
* `parch`
* `fare`
* `embarked`
* `who`

---

# 🔧 Feature Engineering

The project creates five new features.

## 1. FamilySize

Calculates the total number of family members travelling together, including the passenger.

```python
df["FamilySize"] = df["sibsp"] + df["parch"] + 1
```

---

## 2. IsAlone

Identifies whether a passenger was travelling alone.

```python
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
```

Values:

```text
0 → Travelling with family
1 → Travelling alone
```

---

## 3. FarePerPerson

Calculates the fare paid per family member.

```python
df["FarePerPerson"] = df["fare"] / df["FamilySize"]
```

This provides additional information beyond the original total fare.

---

## 4. Title

Creates a simplified title/category from the `who` column.

```python
df["Title"] = df["who"].astype(str).str.title()
```

---

## 5. AgeGroup

Converts continuous age values into meaningful groups.

```python
df["AgeGroup"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=[
        "Child",
        "Teen",
        "Young Adult",
        "Adult",
        "Senior"
    ]
)
```

---

# 🔄 Project Workflow

```text
Titanic Dataset
       ↓
Data Exploration
       ↓
Missing Value Analysis
       ↓
Feature Engineering
       ↓
Create New Features
       ↓
Train-Test Split
       ↓
Data Preprocessing
       ↓
Feature Scaling
       ↓
One-Hot Encoding
       ↓
Logistic Regression
       ↓
Prediction
       ↓
Model Evaluation
       ↓
Visualization
```

---

# 🧹 Data Preprocessing

The project uses separate preprocessing pipelines for numerical and categorical features.

### Numerical Features

Missing values are handled using the median and then the features are standardized using `StandardScaler`.

```python
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

### Categorical Features

Missing values are replaced using the most frequent value and categorical features are converted into numerical representations using `OneHotEncoder`.

```python
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
```

---

# 🤖 Machine Learning Model

The project uses **Logistic Regression** for binary classification.

```python
LogisticRegression(
    max_iter=1000
)
```

The complete preprocessing and model are combined using a Scikit-learn `Pipeline`.

```text
Preprocessing
      ↓
Logistic Regression
      ↓
Prediction
```

---

# 📈 Model Evaluation

The model is evaluated using:

### Accuracy

Measures the percentage of correctly classified observations.

### Classification Report

Provides:

* Precision
* Recall
* F1-score
* Support

### Confusion Matrix

Shows:

* Correct predictions
* Incorrect predictions
* True positives
* True negatives
* False positives
* False negatives

---

# 📊 Visualizations

The project includes visualizations for:

### 1. Survival Rate by Family Size

Shows how survival rate varies across different family sizes.

### 2. Confusion Matrix

Visualizes the classification results of the Logistic Regression model.

---
# 📁 Project Structure

```text
Day15-Feature-Engineering/
│
├── day15_feature_engineering.py
│
├── Day15_Feature_Engineering.ipynb
│
├── README.md
│
└── requirements.txt
```

---

# ▶️ Run the Python File

```bash
python day15_feature_engineering.py
```

---

# 📓 Run the Jupyter Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Day15_Feature_Engineering.ipynb
```

Run the cells sequentially.

---

# 💡 Key Learnings

Through this project, I learned:

* What Feature Engineering means
* How to create new features from existing data
* How `FamilySize` can be derived from family-related columns
* How to identify passengers travelling alone
* How to calculate fare per person
* How to create meaningful age groups
* How to handle missing values
* How to encode categorical variables
* How to scale numerical features
* How to build a preprocessing pipeline
* How to train a classification model
* How to evaluate model performance

---

# 🔬 Engineered Features Summary

| Feature         | Description                                 |
| --------------- | ------------------------------------------- |
| `FamilySize`    | Total family size                           |
| `IsAlone`       | Indicates whether passenger travelled alone |
| `FarePerPerson` | Fare divided by family size                 |
| `Title`         | Category derived from passenger information |
| `AgeGroup`      | Age converted into meaningful groups        |

---

# 📅 Day 15 of 30 Days of Machine Learning

**Day 15 — Feature Engineering**

Today's focus was on transforming raw data into meaningful features that can provide additional information to a machine learning model.

```text
Raw Data
   ↓
Feature Engineering
   ↓
Better Representation
   ↓
Machine Learning Model
   ↓
Prediction
```

---

# 🚀 Future Improvements

Possible improvements include:

* Compare the model with and without engineered features
* Try different feature selection techniques
* Experiment with other classification algorithms
* Perform hyperparameter tuning
* Use additional feature engineering techniques
* Test the approach on another real-world dataset

---
# 👩‍💻 Author

**Gauri**
