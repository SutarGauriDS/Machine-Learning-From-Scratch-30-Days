## Day 30 — Customer Churn Prediction

---

## 1. What is customer churn?

Customer churn occurs when a customer stops using a company's product or service.

In this project, the model predicts whether a telecom customer is likely to leave the service.

---

## 2. What is the objective of your project?

The objective is to build a Machine Learning classification system that predicts whether a customer will churn.

The prediction can help telecom companies identify high-risk customers and support retention strategies.

---

## 3. What type of Machine Learning problem is this?

It is a **supervised binary classification problem**.

The target has two possible outcomes:

- `0` → No Churn
- `1` → Churn

---

## 4. Which dataset did you use?

I used the **Telco Customer Churn dataset**.

It contains customer information such as:

- Demographics
- Tenure
- Contract
- Internet services
- Payment method
- Monthly charges
- Total charges
- Churn status

---

## 5. Why did you remove `customerID`?

`customerID` is an identifier rather than a meaningful predictive feature.

Keeping it could introduce noise without providing useful information for predicting churn.

---

## 6. Why did you convert `TotalCharges` to numeric?

`TotalCharges` may contain blank or non-numeric values.

I converted it using:

```python
pd.to_numeric(df["TotalCharges"], errors="coerce")
````

Invalid values become `NaN`, which can then be handled by the preprocessing pipeline.

---

## 7. What feature engineering did you perform?

I created:

1. `tenure_group`
2. `monthly_charge_category`
3. `service_count`

These features provide additional representations of customer tenure, spending level, and service usage.

---

## 8. Why did you create a `service_count` feature?

Instead of treating every service independently, `service_count` provides an overall measure of how many selected services a customer uses.

This can help represent customer engagement.

---

## 9. Why did you use a train-test split?

The dataset was divided into training and testing sets.

The training set is used to learn patterns, while the test set evaluates how the trained model performs on unseen data.

I used an **80/20 split**.

---

## 10. Why did you use `stratify=y`?

Churn datasets can have an imbalance between churn and non-churn customers.

`stratify=y` helps maintain approximately the same class distribution in both training and testing sets.

---

## 11. Why did you use a preprocessing pipeline?

A pipeline combines preprocessing and model training into one reproducible workflow.

It also ensures that the same preprocessing logic is applied during prediction.

---

## 12. Why did you use `ColumnTransformer`?

The dataset contains both numerical and categorical features.

`ColumnTransformer` allows different preprocessing operations to be applied to each type.

For example:

* Numerical → imputation + scaling
* Categorical → imputation + one-hot encoding

---

## 13. Why is scaling required?

Scaling puts numerical features on comparable ranges.

It is particularly useful for algorithms such as Logistic Regression.

Tree-based models generally do not require feature scaling.

---

## 14. Why did you use One-Hot Encoding?

Machine Learning algorithms require numerical input.

One-Hot Encoding converts categorical values into numerical indicator columns.

For example:

```text
Contract
  |
  ├── Month-to-month
  ├── One year
  └── Two year
```

can be represented using binary columns.

---

## 15. Which models did you compare?

I compared four classification algorithms:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. XGBoost

---

## 16. Why did you use Logistic Regression?

Logistic Regression provides a strong and interpretable baseline for binary classification.

It also produces probability estimates.

---

## 17. Why did you use Random Forest?

Random Forest combines multiple decision trees.

It can capture non-linear relationships and interactions between features and also provides feature importance.

---

## 18. What is Gradient Boosting?

Gradient Boosting builds models sequentially.

Each new model attempts to improve the errors made by previous models.

---

## 19. What is XGBoost?

XGBoost is an optimized gradient boosting framework.

It is widely used for structured or tabular data and can model complex non-linear relationships.

---

## 20. Why compare multiple models?

Different algorithms can learn different patterns.

Rather than assuming one algorithm is best, I compared their performance using multiple evaluation metrics and cross-validation.

---

## 21. Which evaluation metrics did you use?

I used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

---

## 22. Why isn't accuracy alone enough?

A churn dataset can have class imbalance.

A model could achieve good accuracy while still failing to identify many customers who actually churn.

Therefore, precision, recall, F1-score, and ROC-AUC provide additional information.

---

## 23. What is precision?

Precision answers:

> Of the customers predicted as churners, how many actually churned?

Formula:

```text
Precision = TP / (TP + FP)
```

---

## 24. What is recall?

Recall answers:

> Of all customers who actually churned, how many did the model correctly identify?

Formula:

```text
Recall = TP / (TP + FN)
```

---

## 25. Why is recall important for churn prediction?

If a real churner is classified as a non-churner, the company may miss an opportunity to retain that customer.

Therefore, recall can be an important metric in a churn use case.

---

## 26. What is F1-score?

F1-score is the harmonic mean of precision and recall.

```text
F1 = 2 × (Precision × Recall)
     ---------------------------
       (Precision + Recall)
```

It is useful when both precision and recall matter.

---

## 27. What is ROC-AUC?

ROC-AUC measures how effectively a classifier separates the two classes across different probability thresholds.

A higher ROC-AUC generally indicates better discrimination.

---

## 28. What is a confusion matrix?

A confusion matrix shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

For this project:

```text
                 Predicted
              No Churn   Churn

Actual
No Churn         TN        FP
Churn            FN        TP
```

---

## 29. What is cross-validation?

Cross-validation evaluates a model across multiple train-validation splits.

I used **5-fold Stratified Cross-Validation**.

This provides a more reliable estimate of model performance than relying on only one split.

---

## 30. Why use StratifiedKFold?

StratifiedKFold maintains approximately the same class distribution in every fold.

This is especially useful for classification problems where classes may be imbalanced.

---

## 31. How did you select the final model?

I compared the models using holdout-test metrics and selected the best-performing model based on **ROC-AUC**.

I also examined cross-validation performance to understand model stability.

---

## 32. What is feature importance?

Feature importance indicates which input features contribute strongly to predictions.

For tree-based models, I used `feature_importances_`.

For Logistic Regression, absolute coefficient magnitude was used as an importance measure.

---

## 33. What is model persistence?

Model persistence means saving a trained model so it can be loaded and reused later without retraining it.

I used **Joblib**.

```python
joblib.dump(best_pipeline, "churn_model.pkl")
```

---

## 34. Why save the complete pipeline instead of only the model?

The pipeline contains both:

* Preprocessing
* Machine Learning model

Therefore, when a new customer is provided, the same preprocessing steps can automatically be applied before prediction.

---

## 35. How would you deploy this project?

The saved pipeline could be exposed through an API using a framework such as FastAPI.

A frontend or business application could then send customer information to the API and receive:

* Churn prediction
* Churn probability

---

## 36. How would a company use this model?

A company could use predicted churn probabilities to:

* Identify high-risk customers
* Prioritize retention campaigns
* Analyze churn patterns
* Improve customer engagement
* Allocate retention resources

---

## 37. What would you do if recall was too low?

I would investigate:

* Class imbalance
* Classification threshold
* Feature engineering
* Model hyperparameters
* Sampling strategies
* Alternative algorithms

I would not simply optimize accuracy.

---

## 38. How could you improve this project?

Possible improvements include:

* Hyperparameter tuning
* More feature engineering
* SHAP explainability
* Threshold optimization
* Cost-sensitive learning
* Model monitoring
* FastAPI deployment
* Cloud deployment
* Automated retraining

---

## 39. What is data leakage?

Data leakage occurs when information that would not legitimately be available during prediction is used while training the model.

It can produce unrealistically high evaluation results.

Using a preprocessing pipeline helps reduce leakage by fitting transformations only on training data during model fitting.

---

## 40. What was your biggest learning from this project?

My biggest learning was that building a Machine Learning model is not just about training an algorithm.

A practical ML solution requires:

**Data Cleaning → Feature Engineering → Preprocessing → Model Training → Evaluation → Validation → Persistence → Business Interpretation**

---

# 🎯 Explain your project.

> I built an end-to-end Customer Churn Prediction system using the Telco Customer Churn dataset. I cleaned the data, performed feature engineering, and created a preprocessing pipeline for numerical and categorical features. I compared Logistic Regression, Random Forest, Gradient Boosting, and XGBoost using metrics such as precision, recall, F1-score, and ROC-AUC. I also used 5-fold stratified cross-validation, analyzed feature importance, and saved the complete best-performing pipeline using Joblib so it can be reused for future predictions.
