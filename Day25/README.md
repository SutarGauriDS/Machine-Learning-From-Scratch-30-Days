#  Day 25 — Ensemble Learning

## Bank Customer Subscription Prediction

Part of my **30 Days of Machine Learning Challenge**, focused on building practical, interview-ready machine learning projects.

### 🎯 Objective

Build a machine learning classification system that predicts whether a bank customer will subscribe to a **term deposit** using multiple ensemble learning techniques.

The project focuses on comparing individual models with ensemble approaches and understanding how combining models can improve predictive performance.

---

## 📊 Dataset

**UCI Bank Marketing Dataset**

The dataset contains **45,211 customer records** with **17 input features**, including customer demographics, account information, and marketing campaign details.

**Target variable:** `y`

* `yes` → Customer subscribed
* `no` → Customer did not subscribe

The dataset contains both **numerical and categorical features**, making it suitable for a realistic end-to-end ML workflow.

---

## 🔄 Project Workflow

```text
Bank Marketing Dataset
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Target Encoding
        ↓
Feature Engineering
        ↓
Train-Test Split
        ↓
Preprocessing Pipeline
        ↓
Individual ML Models
        ↓
Ensemble Learning
        ↓
Model Evaluation
        ↓
Cross-Validation
        ↓
Feature Importance
        ↓
Business Insights
```

---

## 🧠 Machine Learning Models

### 1. Logistic Regression

Used as a baseline classification model.

### 2. Random Forest

Uses **Bagging** to combine multiple decision trees and reduce variance.

### 3. Gradient Boosting

Uses **Boosting** to sequentially improve predictions by learning from previous errors.

### 4. Soft Voting Ensemble

Combines the probability predictions of:

* Logistic Regression
* Random Forest
* Gradient Boosting

The final prediction is based on the combined probabilities of the individual models.

---

## ⚙️ Techniques Used

* Data Cleaning
* Duplicate Removal
* Target Encoding
* Feature Engineering
* One-Hot Encoding
* Feature Scaling
* ColumnTransformer
* Scikit-learn Pipelines
* Train-Test Split
* Stratified Cross-Validation
* Bagging
* Boosting
* Voting Ensemble
* Hyperparameter Configuration
* Classification Metrics
* ROC-AUC Analysis
* Feature Importance
* Business Interpretation

---

## 📈 Model Evaluation

The models are evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1 Score**
* **ROC-AUC**
* **Confusion Matrix**
* **Classification Report**
* **ROC Curve**

ROC-AUC is particularly useful for evaluating the model's ability to distinguish between customers who subscribe and those who do not.

---

## 🔍 Feature Engineering

Additional features were created to improve the representation of customer information:

* `age_group`
* `balance_category`
* `campaign_intensity`

These features transform raw numerical values into meaningful business categories.

---

## 💼 Business Application

The trained models can help a bank identify customers who are more likely to subscribe to a term deposit.

This can support:

* Targeted marketing campaigns
* Customer prioritization
* Better campaign allocation
* Reduced unnecessary outreach
* Data-driven marketing decisions

---

## 📁 Project Files

```text
Day25_Ensemble_Learning_Complete.ipynb
Day25_Ensemble_Model_Results.csv
Day25_Cross_Validation_Results.csv
Day25_RandomForest_Feature_Importance.csv
README.md
requirements.txt
references.md
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Google Colab
* Jupyter Notebook

---

## 🎯 Key Learnings

1. **Ensemble learning** combines multiple models to produce stronger predictions.
2. **Bagging** mainly helps reduce model variance.
3. **Boosting** focuses on correcting errors made by previous models.
4. **Voting ensembles** can combine different models to leverage their strengths.
5. **Cross-validation** provides a more reliable estimate of model performance.
6. Using multiple evaluation metrics gives a better understanding of classification performance.
7. **Feature importance** helps connect machine learning results with business decisions.

---

## 🚀 Challenge Progress

**Day 25/30 — Ensemble Learning Completed!**

Next: **Day 26 — XGBoost** 🔥
