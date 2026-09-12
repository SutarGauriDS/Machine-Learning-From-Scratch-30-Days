# Day 30 — Customer Churn Prediction

## Final Project | 30 Days of Machine Learning

A complete end-to-end Machine Learning project that predicts whether a telecom customer is likely to **churn**.

The project brings together the major concepts covered throughout the 30 Days of Machine Learning challenge, including data preprocessing, feature engineering, classification, ensemble learning, model evaluation, cross-validation, and model persistence.

---

## 🎯 Project Objective

Customer churn is an important business problem for telecom companies.

The objective of this project is to build a Machine Learning system that predicts:

> **Will a customer leave the telecom service?**

The target variable is:

- `0` → No Churn
- `1` → Churn

The trained model can help businesses identify high-risk customers and support targeted customer-retention strategies.

---

## 📊 Dataset

This project uses the **Telco Customer Churn dataset**.

The dataset contains information about telecom customers, including:

- Customer demographics
- Account information
- Contract details
- Internet services
- Payment methods
- Monthly charges
- Total charges
- Customer tenure
- Churn status

### Target Variable

`Churn`

| Value | Meaning |
|---|---|
| `Yes` | Customer churned |
| `No` | Customer did not churn |

For Machine Learning, the target is converted to:

| Value | Meaning |
|---|---|
| `1` | Churn |
| `0` | No Churn |

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Train-Test Split
   ↓
Preprocessing Pipeline
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Cross-Validation
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Feature Importance
   ↓
Model Persistence
   ↓
Customer Churn Prediction
````

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Converted `TotalCharges` into a numeric format.
2. Replaced blank values with missing values.
3. Removed duplicate records.
4. Removed the `customerID` identifier.
5. Converted the `Churn` target into binary values.
6. Used median imputation for numerical features.
7. Used most-frequent imputation for categorical features.
8. Applied `StandardScaler` to numerical features.
9. Applied One-Hot Encoding to categorical features.

A Scikit-learn `Pipeline` and `ColumnTransformer` were used to keep preprocessing consistent between training and prediction.

---

## 🛠️ Feature Engineering

Three additional features were created.

### 1. Tenure Group

Customers were grouped based on their tenure:

* `0-12`
* `13-24`
* `25-48`
* `49-72` months

### 2. Monthly Charge Category

Monthly charges were divided into:

* Low
* Medium
* High

### 3. Service Count

The number of selected subscribed services was calculated for each customer.

This provides an additional representation of customer engagement with the company's services.

---

## 🤖 Machine Learning Models

Four classification algorithms were compared.

### 1. Logistic Regression

Used as a strong and interpretable baseline classification model.

### 2. Random Forest

An ensemble learning algorithm that combines multiple decision trees.

Advantages:

* Handles non-linear relationships
* Works well with mixed feature patterns
* Provides feature importance

### 3. Gradient Boosting

Builds models sequentially, with each new model attempting to improve previous errors.

### 4. XGBoost

A powerful gradient boosting implementation designed for efficient and high-performance Machine Learning.

---

## 📈 Model Evaluation

The models are evaluated using:

### Accuracy

Measures the overall percentage of correct predictions.

### Precision

Measures how many predicted churn customers actually churned.

### Recall

Measures how many actual churn customers were correctly identified.

### F1 Score

Combines precision and recall into a single metric.

### ROC-AUC

Measures the model's ability to distinguish between churn and non-churn customers across classification thresholds.

For a churn problem, **recall and ROC-AUC are particularly useful alongside accuracy**, because missing a customer who is likely to churn can have business consequences.

---

## 🔁 Cross-Validation

A **5-fold Stratified Cross-Validation** approach is used.

```text
Training Data
     ↓
 ┌───┬───┬───┬───┬───┐
 │ F1│ F2│ F3│ F4│ F5│
 └───┴───┴───┴───┴───┘
     ↓
5 validation rounds
     ↓
Mean ROC-AUC
     +
Standard Deviation
```

Stratification helps preserve the proportion of churn and non-churn customers across folds.

---

## 📊 Visualizations

The project generates visualizations including:

* Customer churn distribution
* Customer tenure distribution
* Model performance comparison
* Confusion matrix
* ROC curve comparison
* Top feature importance

---

## 🔍 Feature Importance

Feature importance is extracted from the selected model when supported.

For tree-based models such as Random Forest and XGBoost, `feature_importances_` is used.

For Logistic Regression, the absolute value of model coefficients is used as an importance measure.

The resulting data is saved to:

```text
Day30_Feature_Importance.csv
```

---

## 💾 Model Persistence

The final selected pipeline is saved using Joblib:

```text
churn_model.pkl
```

The saved object contains both:

* Preprocessing steps
* Machine Learning model

This makes it possible to load the complete trained pipeline later and directly make predictions on new customer data.

---

## 📁 Project Structure

```text
Day30_Customer_Churn_Prediction/
│
├── day30_customer_churn.py
├── Day30_Customer_Churn.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── churn_model.pkl
├── Day30_Model_Results.csv
├── Day30_Feature_Importance.csv
│
├── README.md
├── requirements.txt
├── references.md
└── interview_questions.md
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the project directory

```bash
cd Day30_Customer_Churn_Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Add the dataset

Place:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

inside the project folder.

### 7. Run the Python project

```bash
python day30_customer_churn.py
```

### 8. Run the Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
Day30_Customer_Churn.ipynb
```

---

## 📦 Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**
* **XGBoost**
* **Joblib**
* **Jupyter Notebook**

---

## 💡 Key Learnings

This final project helped consolidate the complete Machine Learning workflow:

1. Data cleaning is an essential part of real-world ML.
2. Feature engineering can improve the representation of business problems.
3. Pipelines make preprocessing reproducible.
4. Multiple models should be compared rather than relying on one algorithm.
5. Cross-validation provides a more reliable estimate of model performance.
6. Accuracy alone may not be sufficient for an imbalanced classification problem.
7. ROC-AUC, precision, recall and F1-score provide additional perspectives.
8. Feature importance helps connect model predictions with business understanding.
9. Saving the complete pipeline makes the model reusable.
10. A successful ML project should connect technical predictions with a real business objective.

---

## 💼 Business Use Case

A telecom company could use a churn prediction model to:

* Identify customers with a high probability of leaving.
* Prioritize retention campaigns.
* Analyze characteristics associated with churn.
* Improve customer engagement strategies.
* Allocate retention resources more efficiently.

The model should support business decisions rather than automatically determine customer treatment.

---

## 🚀 Possible Future Improvements

Future versions could include:

* Hyperparameter tuning
* Advanced feature engineering
* SHAP-based model explainability
* Probability threshold optimization
* Cost-sensitive evaluation
* Model monitoring
* FastAPI deployment
* Cloud deployment
* Automated retraining
* A customer churn prediction dashboard

---

## 🏆 Final Project

**Day 30 completes the 30 Days of Machine Learning challenge.**

The project demonstrates an end-to-end workflow from:

**Raw Data → Cleaning → Feature Engineering → ML Models → Evaluation → Model Selection → Saved Model → Prediction**
