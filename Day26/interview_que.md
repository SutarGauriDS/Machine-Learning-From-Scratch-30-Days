# 🎯 Day 26 — XGBoost Interview Questions & Answers

## 1. What is XGBoost?

XGBoost stands for **Extreme Gradient Boosting**.

It is an optimized implementation of the gradient boosting algorithm used for classification and regression problems.

XGBoost builds decision trees sequentially, where each new tree tries to reduce the errors made by the previous trees.

---

## 2. Why did you use XGBoost in this project?

I used XGBoost because the project is a **credit-card default prediction problem** involving both numerical and categorical information.

XGBoost is powerful for tabular data and can capture complex non-linear relationships between features.

I also compared it with Logistic Regression and Random Forest to determine which approach performed better.

---

## 3. What is Gradient Boosting?

Gradient Boosting is an ensemble technique that builds models sequentially.

Each new model focuses on correcting the errors of the previous models.

The predictions from all models are combined to produce the final prediction.

---

## 4. What is the difference between Bagging and Boosting?

### Bagging

- Models are generally trained independently.
- Reduces variance.
- Random Forest is an example.

### Boosting

- Models are trained sequentially.
- Each model tries to improve the previous model.
- Can reduce bias and improve predictive performance.
- XGBoost and Gradient Boosting are examples.

---

## 5. Why did you compare XGBoost with Logistic Regression and Random Forest?

I used Logistic Regression as a **baseline model**, Random Forest as a **bagging-based ensemble model**, and XGBoost as the main **boosting-based model**.

Comparing different approaches helps determine whether the additional complexity of XGBoost provides better predictive performance.

---

## 6. What dataset did you use?

I used the **UCI Default of Credit Card Clients Dataset**.

It contains information about credit-card clients such as:

- Credit limit
- Age
- Education
- Marital status
- Repayment history
- Bill amounts
- Payment amounts

The target variable represents whether the client defaulted on the next payment.

---

## 7. What is the target variable?

The target variable is:

```text
DEFAULT
````

Where:

```text
0 → No default
1 → Default
```

This makes the problem a **binary classification problem**.

---

## 8. What feature engineering did you perform?

I created several aggregated features from the billing and payment history.

Examples include:

* `AVG_BILL_AMT`
* `AVG_PAY_AMT`
* `TOTAL_PAY_AMT`
* `TOTAL_BILL_AMT`
* `PAYMENT_BILL_RATIO`
* `MAX_PAYMENT_DELAY`
* `DELAYED_MONTHS`

These features summarize payment behavior and billing patterns.

---

## 9. Why is feature engineering important?

Feature engineering can help models identify meaningful patterns more easily.

For example, instead of making the model separately interpret six payment amounts, an aggregated feature such as `AVG_PAY_AMT` can provide a useful summary of the customer's payment behavior.

---

## 10. What is class imbalance?

Class imbalance occurs when one target class contains significantly more observations than another.

In credit-default prediction, the number of customers who do not default can be much larger than the number who default.

This can make a model biased toward the majority class.

---

## 11. How did you handle class imbalance?

For XGBoost, I calculated:

```python
scale_pos_weight = negative_count / positive_count
```

and passed this value to:

```python
XGBClassifier(
    scale_pos_weight=scale_pos_weight
)
```

This gives more importance to the minority class during training.

---

## 12. What is `scale_pos_weight` in XGBoost?

`scale_pos_weight` is a parameter used to handle class imbalance.

It increases the importance of positive-class examples during training.

A common starting point is:

```text
Number of negative samples / Number of positive samples
```

---

## 13. What is `learning_rate` in XGBoost?

`learning_rate` controls how much each new tree contributes to the final model.

A smaller learning rate generally requires more trees but can provide more controlled learning.

In this project, I used:

```python
learning_rate=0.05
```

---

## 14. What is `n_estimators`?

`n_estimators` represents the number of boosting trees used by the model.

In this project:

```python
n_estimators=300
```

means XGBoost builds 300 boosting trees.

---

## 15. What is `max_depth`?

`max_depth` controls the maximum depth of each decision tree.

A larger depth allows the model to learn more complex patterns but can increase the risk of overfitting.

I used:

```python
max_depth=5
```

---

## 16. What is `subsample`?

`subsample` specifies the fraction of training samples used to build each boosting tree.

I used:

```python
subsample=0.8
```

This means each tree uses approximately 80% of the training samples.

It can help reduce overfitting.

---

## 17. What is `colsample_bytree`?

`colsample_bytree` controls the fraction of features randomly selected for each tree.

I used:

```python
colsample_bytree=0.8
```

This means approximately 80% of the features are considered for each tree.

---

## 18. What are `reg_alpha` and `reg_lambda`?

They are regularization parameters.

### `reg_alpha`

Controls **L1 regularization**.

### `reg_lambda`

Controls **L2 regularization**.

Regularization helps control model complexity and reduce overfitting.

---

## 19. What is overfitting?

Overfitting occurs when a model learns the training data too closely, including noise and random patterns.

As a result, it performs very well on training data but poorly on unseen data.

Ways to reduce overfitting include:

* Limiting tree depth
* Using regularization
* Reducing learning rate
* Using subsampling
* Cross-validation
* Early stopping

---

## 20. Why did you use a train-test split?

I used an **80/20 stratified train-test split**.

The training set is used to train the models, while the test set is kept separate for evaluating performance on unseen data.

Stratification helps preserve the class distribution in both sets.

---

## 21. Why did you use ROC-AUC?

ROC-AUC is useful for evaluating binary classification models, especially when class distribution is imbalanced.

It measures how well the model separates positive and negative classes across different classification thresholds.

A higher ROC-AUC generally indicates better ranking/separation ability.

---

## 22. What is the difference between accuracy and recall?

### Accuracy

Measures the proportion of all predictions that are correct.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Recall

Measures how many actual positive cases were correctly identified.

```text
Recall = TP / (TP + FN)
```

For credit-default prediction, recall can be important because missing a customer who is actually likely to default can be costly.

---

## 23. Why is precision also important in credit risk?

Precision tells us how many customers predicted as high-risk actually belong to the positive class.

If precision is low, many customers may be incorrectly flagged as risky.

Therefore, a financial institution may need to balance precision and recall depending on the business cost of different errors.

---

## 24. What is F1-score?

F1-score is the harmonic mean of precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

It is useful when we want to consider both false positives and false negatives.

---

## 25. What is a confusion matrix?

A confusion matrix summarizes classification predictions using four categories:

* True Positive
* True Negative
* False Positive
* False Negative

It helps understand what types of mistakes the model is making.

---

## 26. What is the difference between Random Forest and XGBoost?

### Random Forest

* Uses bagging.
* Builds many trees independently.
* Combines their predictions.
* Primarily helps reduce variance.

### XGBoost

* Uses boosting.
* Builds trees sequentially.
* Each tree tries to improve previous errors.
* Uses regularization and optimization techniques.

---

## 27. Why might XGBoost perform better than Logistic Regression?

Logistic Regression primarily models a linear relationship between features and the target.

XGBoost can capture:

* Non-linear relationships
* Feature interactions
* Complex decision boundaries

Therefore, XGBoost can be more effective when the underlying relationships are complex.

---

## 28. What is feature importance?

Feature importance indicates how useful different features were for making predictions.

In this project, I used XGBoost feature importance to understand which variables contributed most to the model.

However, feature importance should not automatically be interpreted as causal relationships.

---

## 29. What is data leakage?

Data leakage occurs when information that would not be available at prediction time is unintentionally used during model training.

This can result in unrealistically high evaluation scores.

For a credit-risk model, it is important to ensure that features represent information available before the prediction decision.

---

## 30. How would you improve this project?

Possible improvements include:

1. Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
2. Cross-validation.
3. Threshold optimization based on business costs.
4. SHAP-based model explainability.
5. More detailed error analysis.
6. Model calibration.
7. Monitoring model performance after deployment.
8. Testing the model on newer data.

---

## 31. What would you do if the model has high accuracy but poor recall?

I would not rely on accuracy alone.

I would investigate:

* Class imbalance
* Confusion matrix
* Precision
* Recall
* F1-score
* ROC-AUC
* Prediction threshold

For a credit-risk problem, I might adjust the classification threshold if identifying more potential defaults is more important to the business.

---

## 32. What is early stopping in XGBoost?

Early stopping stops training when the model stops improving on a validation dataset for a specified number of rounds.

It can help prevent overfitting and avoid unnecessary training.

It was not used in the current implementation, but it would be a useful improvement.

---

## 33. Why is XGBoost popular for tabular data?

XGBoost is popular because it can:

* Capture non-linear relationships
* Handle feature interactions
* Provide strong predictive performance
* Support regularization
* Work effectively with structured/tabular datasets
* Provide feature importance

---

## 34. What are the limitations of your project?

Some limitations include:

* The model is trained on historical data.
* Model performance may change on new populations.
* Feature importance does not prove causation.
* Credit decisions involve fairness and regulatory considerations.
* A production system would require monitoring and validation.
* The current project focuses on prediction rather than a complete production credit decision system.

---

## 35. Explain your project in 30 seconds.

> "I built a credit-card default prediction system using the UCI Default of Credit Card Clients dataset. I performed data cleaning and feature engineering, then compared Logistic Regression, Random Forest, and XGBoost models. I handled class imbalance using `scale_pos_weight` and evaluated the models using precision, recall, F1-score, and ROC-AUC. Finally, I analyzed XGBoost feature importance to understand which customer and repayment-related features were most useful for predicting default risk."

---

# ⭐ Quick Revision

### XGBoost

→ Gradient Boosting algorithm

### `n_estimators`

→ Number of trees

### `learning_rate`

→ Contribution of each tree

### `max_depth`

→ Maximum tree depth

### `subsample`

→ Fraction of rows used per tree

### `colsample_bytree`

→ Fraction of features used per tree

### `reg_alpha`

→ L1 regularization

### `reg_lambda`

→ L2 regularization

### `scale_pos_weight`

→ Helps handle class imbalance

### ROC-AUC

→ Measures class separation across thresholds

### Precision

→ Correct positive predictions among predicted positives

### Recall

→ Correctly identified actual positives

### F1

→ Balance between precision and recall

### Random Forest

→ Bagging

### XGBoost

→ Boosting

### Data Leakage

→ Using information that should not be available during prediction

```
```
