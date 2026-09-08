# 🚀 Day 26 — XGBoost

## Credit Card Default Risk Prediction

Part of my **30 Days of Machine Learning Challenge**, focused on building practical, interview-ready machine learning projects.

---

## 🎯 Project Objective

The objective of this project is to predict whether a credit-card client is likely to **default on their next payment** using machine learning.

The main focus of this project is **XGBoost**, a powerful gradient boosting algorithm.

Instead of using XGBoost alone, I compared its performance with:

- Logistic Regression
- Random Forest
- XGBoost

This helps understand whether a more advanced boosting algorithm provides better predictive performance for the problem.

---

## 📊 Dataset

**UCI Default of Credit Card Clients Dataset**

The dataset contains information about credit-card clients, including:

- Credit limit
- Age
- Education
- Marital status
- Repayment history
- Bill amounts
- Previous payment amounts

### Target

`DEFAULT`

- `0` → No default
- `1` → Default

---

## 🔄 Project Workflow

```text
UCI Credit Card Dataset
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Target Analysis
        ↓
Feature Engineering
        ↓
Class Imbalance Handling
        ↓
Train-Test Split
        ↓
Logistic Regression
        ↓
Random Forest
        ↓
XGBoost
        ↓
Model Comparison
        ↓
Model Evaluation
        ↓
Feature Importance
        ↓
Business Insights
````

---

## 🧠 Machine Learning Models

### 1. Logistic Regression

Used as a baseline classification model.

### 2. Random Forest

Used as a bagging-based ensemble model for comparison.

### 3. XGBoost ⭐

The main algorithm of this project.

XGBoost uses gradient boosting to build trees sequentially, where each new tree attempts to improve the errors made by previous trees.

---

## 🔧 Feature Engineering

Several additional features were created from the original financial variables:

* `AVG_BILL_AMT`
* `AVG_PAY_AMT`
* `TOTAL_PAY_AMT`
* `TOTAL_BILL_AMT`
* `PAYMENT_BILL_RATIO`
* `MAX_PAYMENT_DELAY`
* `DELAYED_MONTHS`

These features provide additional representations of a customer's billing and repayment behavior.

---

## ⚖️ Class Imbalance

The dataset contains more non-default cases than default cases.

To address this imbalance, the XGBoost model uses:

```python
scale_pos_weight
```

This gives greater importance to the minority class during training.

---

## 📈 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Classification Report
* ROC Curve

ROC-AUC is particularly useful for evaluating how effectively the models distinguish between default and non-default customers.

---

## 🔍 Feature Importance

XGBoost feature importance is used to identify the features that contribute most to the model's predictions.

This helps connect the machine learning model with the underlying credit-risk problem.

---

## 💼 Business Application

Credit-default prediction can support financial institutions in identifying customers who may have a higher probability of default.

Potential applications include:

* Credit risk analysis
* Customer risk segmentation
* Portfolio monitoring
* Early risk identification
* Supporting financial decision-making

In a real-world financial system, model predictions should be combined with appropriate human review, business rules, and responsible model governance.

---

## 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**
* **XGBoost**
* **Google Colab**
* **Jupyter Notebook**

---

## 📁 Project Structure

```text
Day26_XGBoost/
│
├── day26_xgboost_credit_risk.py
├── Day26_XGBoost_Credit_Risk_Complete.ipynb
├── default of credit card clients.xls
├── Day26_XGBoost_Model_Results.csv
├── Day26_XGBoost_Feature_Importance.csv
├── README.md
├── requirements.txt
└── references.md
```

---

## 🎯 Key Learnings

1. Understood the working principle of **XGBoost**.
2. Learned how boosting differs from bagging.
3. Implemented XGBoost for a real-world classification problem.
4. Learned how `scale_pos_weight` can help with class imbalance.
5. Compared XGBoost with Logistic Regression and Random Forest.
6. Evaluated models using multiple classification metrics.
7. Used feature importance to interpret model behavior.
8. Connected machine learning predictions with a practical **credit-risk use case**.

---

## 🚀 Challenge Progress

**Day 26/30 — XGBoost completed! ⚡**

Next → **Day 27: End-to-End ML Pipeline**
