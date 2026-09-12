# 📚 References — Day 30: Customer Churn Prediction

## 1. Dataset

### Telco Customer Churn Dataset
IBM Sample Data — Telco Customer Churn dataset.

Used for customer churn prediction and classification.

Dataset contains customer demographic, account, service, billing, and churn information.

---

## 2. Python Libraries

### NumPy
Numerical computing and array operations.

https://numpy.org/doc/

### Pandas
Data loading, cleaning, transformation, and analysis.

https://pandas.pydata.org/docs/

### Matplotlib
Data visualization and model-performance plots.

https://matplotlib.org/stable/

---

## 3. Scikit-learn

### Scikit-learn
Machine Learning algorithms, preprocessing, pipelines, model evaluation, and cross-validation.

https://scikit-learn.org/stable/

### train_test_split
Used to divide the dataset into training and testing subsets.

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

### StratifiedKFold
Used for stratified cross-validation while maintaining class proportions.

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

### Pipeline
Used to combine preprocessing and the Machine Learning model into one reusable workflow.

https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

### ColumnTransformer
Used to apply different preprocessing operations to numerical and categorical features.

https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html

### StandardScaler
Used to standardize numerical features.

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

### SimpleImputer
Used to replace missing values using strategies such as median and most frequent value.

https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

### OneHotEncoder
Used to convert categorical variables into numerical representations.

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

---

## 4. Machine Learning Models

### Logistic Regression
Used as a baseline binary classification model.

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

### Random Forest Classifier
An ensemble learning algorithm based on multiple decision trees.

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

### Gradient Boosting Classifier
A boosting-based classification algorithm that builds models sequentially.

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

### XGBoost
Gradient boosting framework used for efficient and high-performance Machine Learning.

https://xgboost.readthedocs.io/

---

## 5. Model Evaluation

### Accuracy
Measures the proportion of correctly classified observations.

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

### Precision
Measures the proportion of predicted positive observations that are actually positive.

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

### Recall
Measures the proportion of actual positive observations correctly identified.

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html

### F1 Score
Harmonic mean of precision and recall.

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

### ROC-AUC
Used to evaluate the ability of a binary classifier to distinguish between classes.

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

### Classification Report
Provides precision, recall, F1-score, and support.

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

### Confusion Matrix
Used to examine correct and incorrect classification outcomes.

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

### ROC Curve
Plots the relationship between true-positive rate and false-positive rate.

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html

---

## 6. Model Persistence

### Joblib
Used to save and reload the trained Machine Learning pipeline.

https://joblib.readthedocs.io/

### Scikit-learn Model Persistence
Guidance for saving and loading trained models.

https://scikit-learn.org/stable/model_persistence.html

---

## 7. Notebook Environment

### Jupyter
Interactive environment used to develop and document the Machine Learning project.

https://jupyter.org/documentation

-
