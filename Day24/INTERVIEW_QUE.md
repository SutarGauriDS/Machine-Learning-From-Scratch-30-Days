# 🎯 Day 24 — Model Evaluation: Interview Questions & Answers

### 1. What is model evaluation?

**Model evaluation** is the process of measuring how well a machine learning model performs on unseen data using appropriate evaluation metrics.

---

### 2. Why isn't accuracy always enough?

Accuracy only tells us the percentage of overall predictions that are correct.

For imbalanced datasets, a model can have high accuracy while performing poorly on the minority class. Therefore, metrics like **Precision, Recall, F1-Score, and ROC-AUC** are also important.

---

### 3. What is a confusion matrix?

A **confusion matrix** summarizes classification predictions into four categories:

|                     | Predicted Negative | Predicted Positive |
| ------------------- | -----------------: | -----------------: |
| **Actual Negative** |                 TN |                 FP |
| **Actual Positive** |                 FN |                 TP |

* **TN:** True Negative
* **TP:** True Positive
* **FP:** False Positive
* **FN:** False Negative

---

### 4. What is Accuracy?

Accuracy measures the proportion of all predictions that are correct.

**Formula:**

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

---

### 5. What is Precision?

Precision tells us:

> **Of all the instances predicted as positive, how many were actually positive?**

**Formula:**

```text
Precision = TP / (TP + FP)
```

Precision is important when **false positives are costly**.

---

### 6. What is Recall?

Recall tells us:

> **Of all actual positive instances, how many did the model correctly identify?**

**Formula:**

```text
Recall = TP / (TP + FN)
```

Recall is especially important when **missing a positive case is costly**.

---

### 7. What is F1-Score?

F1-Score is the **harmonic mean of Precision and Recall**.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

It is useful when we want a balance between precision and recall.

---

### 8. Precision vs Recall — what's the difference?

**Precision:** "When my model says positive, how often is it correct?"

**Recall:** "Of all actual positives, how many did my model find?"

---

### 9. What is ROC-AUC?

**ROC-AUC** measures how well a classification model distinguishes between two classes across different classification thresholds.

* ROC = Receiver Operating Characteristic
* AUC = Area Under the Curve

A higher AUC generally indicates better class discrimination.

---

### 10. What is an ROC Curve?

An **ROC curve** plots:

* **True Positive Rate (TPR)** on the Y-axis
* **False Positive Rate (FPR)** on the X-axis

It shows how the model's performance changes as the classification threshold changes.

---

### 11. What is the difference between ROC-AUC and accuracy?

**Accuracy** evaluates predictions at a particular classification threshold.

**ROC-AUC** evaluates the model's ability to distinguish between classes across different thresholds.

---

### 12. What is a classification report?

`classification_report()` provides important classification metrics such as:

* Precision
* Recall
* F1-Score
* Support

Example:

```python
print(classification_report(y_test, y_pred))
```

---

### 13. Why did you use Random Forest?

I used **Random Forest** because it is a powerful ensemble classification algorithm that can capture nonlinear relationships and also provides feature importance.

---

### 14. What is Feature Importance?

Feature importance indicates how much each feature contributes to the predictions made by the model.

In Random Forest, feature importance can be accessed using:

```python
model.feature_importances_
```

---

### 15. What is `predict_proba()`?

`predict_proba()` returns the **probability estimates for each class**.

I used it for ROC-AUC and ROC curve calculations:

```python
y_probability = model.predict_proba(X_test)[:, 1]
```

---

### 16. Why do you need probabilities for ROC-AUC?

ROC curves evaluate model performance across different classification thresholds. Therefore, probability or decision scores are used rather than only the final class predictions.

---

### 17. What does `zero_division=0` do?

It prevents warnings or errors when a metric encounters a situation where its denominator becomes zero.

```python
precision_score(
    y_test,
    y_pred,
    zero_division=0
)
```

---

### 18. What is the difference between False Positive and False Negative?

**False Positive:** Model predicts positive, but the actual class is negative.

**False Negative:** Model predicts negative, but the actual class is positive.

---

### 19. Which is more important: Precision or Recall?

It depends on the application.

* If **false positives** are more costly → prioritize Precision.
* If **false negatives** are more costly → prioritize Recall.
* If both matter → F1-Score can be useful.

---

### 20. Explain your Day 24 project in an interview.

> **In Day 24 of my Machine Learning challenge, I focused on classification model evaluation. I used the Breast Cancer Wisconsin dataset and trained a Random Forest Classifier. Instead of evaluating the model only using accuracy, I calculated Precision, Recall, F1-Score and ROC-AUC. I also created a confusion matrix and ROC curve to understand the model's errors and class discrimination ability. Finally, I analyzed Random Forest feature importance to identify the features contributing most to the predictions.**
