
# 🎯 Day 8 – Logistic Regression
# Interview Questions & Answers

## 1. What is Logistic Regression?

Logistic Regression is a supervised Machine Learning algorithm
used mainly for classification problems.

It predicts the probability of an observation belonging to a class.

---

## 2. Is Logistic Regression a regression or classification algorithm?

Despite its name, Logistic Regression is primarily used for
classification problems.

For example:

```text
0 → Low Performance
1 → High Performance
````

---

## 3. What is the difference between Linear Regression and Logistic Regression?

### Linear Regression

Used to predict continuous numerical values.

Example:

```text
Exam Score = 82.5
```

### Logistic Regression

Used to predict classes or class probabilities.

Example:

```text
High Performance
Low Performance
```

---

## 4. What is the Sigmoid Function?

The sigmoid function converts a numerical value into a probability
between 0 and 1.

Its formula is:

```text
σ(z) = 1 / (1 + e⁻ᶻ)
```

---

## 5. What is a classification threshold?

A threshold is used to convert predicted probability into a class.

A common threshold is:

```text
Probability >= 0.5 → Class 1

Probability < 0.5 → Class 0
```

---

## 6. What is Accuracy?

Accuracy is the proportion of correct predictions out of all predictions.

```text
Accuracy =
Correct Predictions / Total Predictions
```

---

## 7. What is Precision?

Precision answers:

> Of all observations predicted as positive, how many were actually positive?

```text
Precision =
TP / (TP + FP)
```

---

## 8. What is Recall?

Recall answers:

> Of all actual positive observations, how many did the model correctly identify?

```text
Recall =
TP / (TP + FN)
```

---

## 9. What is F1 Score?

F1 Score is the harmonic mean of Precision and Recall.

```text
F1 =
2 × (Precision × Recall)
------------------------
Precision + Recall
```

---

## 10. What is a Confusion Matrix?

A confusion matrix summarizes classification predictions using:

* True Positive (TP)
* True Negative (TN)
* False Positive (FP)
* False Negative (FN)

---

## 11. What is `predict_proba()`?

`predict_proba()` returns the probability of an observation
belonging to each class.

```python
model.predict_proba(X_test)
```

---

## 12. Why did we use `stratify=y`?

`stratify=y` helps maintain a similar class distribution
in both the training and testing datasets.

This is particularly useful for classification problems.

---

## 13. What is class imbalance?

Class imbalance occurs when one class contains significantly
more observations than another class.

For example:

```text
High Performance → 90%
Low Performance  → 10%
```

This can make accuracy misleading.

---

## 14. Why are Precision and Recall important?

Accuracy alone may not provide enough information, especially
when classes are imbalanced.

Precision and Recall help us understand different types
of classification errors.

---

## 15. What is overfitting in classification?

Overfitting occurs when a model learns the training data too
closely and performs poorly on unseen data.

---

# 🎯 Project-Based Interview Question

## Explain your Day 8 project.

### Answer:

"I used the Student Performance Factors dataset to build a binary classification model using Logistic Regression. Since the original target, Exam_Score, is continuous, I created a new Performance target where students scoring 75 or above were classified as High Performance and the remaining students as Low Performance. I selected features such as Hours_Studied, Attendance, Previous_Scores, Sleep_Hours, and Tutoring_Sessions. I then performed a stratified train-test split, trained a Logistic Regression model, and evaluated it using Accuracy, Precision, Recall, F1 Score, and a Confusion Matrix."
