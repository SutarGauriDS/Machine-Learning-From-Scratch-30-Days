# 🎯 Day 9 – K-Nearest Neighbors (KNN)
# Interview Questions & Answers

## 1. What is K-Nearest Neighbors?

K-Nearest Neighbors (KNN) is a supervised Machine Learning algorithm
used for classification and regression.

For classification, it predicts the class of a new data point based
on the classes of its nearest neighbors.

---

## 2. How does KNN work?

KNN follows these basic steps:

1. Choose a value of K.
2. Calculate the distance between the new data point and training points.
3. Find the K nearest data points.
4. Check their classes.
5. Assign the majority class to the new data point.

---

## 3. What does K represent in KNN?

K represents the number of nearest neighbors considered when making
a prediction.

Example:

```text
K = 5
````

means the model considers the 5 nearest observations.

---

## 4. What happens when K is too small?

A very small K can make the model sensitive to noise and individual
data points.

This can result in:

**Overfitting**

Example:

```text
K = 1
```

---

## 5. What happens when K is too large?

A very large K considers too many neighboring points and can make
the model overly generalized.

This can result in:

**Underfitting**

---

## 6. Why is feature scaling important in KNN?

KNN uses distance calculations.

If features have very different scales, a feature with a larger
numerical range can dominate the distance calculation.

Therefore, feature scaling is important.

---

## 7. What is StandardScaler?

`StandardScaler` standardizes numerical features so that they have
approximately:

* Mean = 0
* Standard deviation = 1

Example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 8. Why do we use `fit_transform()` on training data?

The scaler learns the mean and standard deviation from the training
data.

```python
scaler.fit_transform(X_train)
```

For test data, we only transform:

```python
scaler.transform(X_test)
```

This prevents information from the test data from influencing the
training process.

---

## 9. What distance metric is commonly used in KNN?

Euclidean distance is commonly used.

The formula is:

```text
d = √[(x₂-x₁)² + (y₂-y₁)²]
```

Scikit-Learn's KNN classifier uses Euclidean distance by default
when using the default Minkowski metric with `p=2`.

---

## 10. Is KNN a supervised or unsupervised algorithm?

KNN is a:

**Supervised Machine Learning algorithm**

because it learns from labeled training data.

---

## 11. Is KNN used for classification or regression?

It can be used for both:

### Classification

Predicts a class.

Example:

```text
High Performance
Low Performance
```

### Regression

Predicts a numerical value.

Example:

```text
Exam Score = 82.5
```

---

## 12. What is lazy learning?

KNN is often called a **lazy learning algorithm** because it does
not build a traditional model during training.

Instead, most of the computation happens when making predictions.

---

## 13. What is the difference between KNN and Logistic Regression?

### KNN

* Distance-based
* Non-parametric
* Requires feature scaling
* Prediction can be computationally expensive with large datasets

### Logistic Regression

* Linear classification model
* Parametric
* Feature scaling may help but is not inherently required
* Generally faster for prediction

---

## 14. What is overfitting in KNN?

Overfitting occurs when the model becomes too sensitive to the
training data.

In KNN, very small values of K can increase the risk of overfitting.

---

## 15. What is underfitting in KNN?

Underfitting occurs when the model is too simple to capture useful
patterns in the data.

Very large values of K can increase the risk of underfitting.

---

## 16. Why did we test different K values?

There is no universally best K value.

Different K values can produce different model performance.

Therefore, we tested values from:

```text
K = 1 to 20
```

and compared their accuracy.

---

## 17. What is a confusion matrix?

A confusion matrix summarizes classification predictions using:

* True Positive (TP)
* True Negative (TN)
* False Positive (FP)
* False Negative (FN)

---

## 18. What is accuracy?

Accuracy measures the proportion of correct predictions.

```text
Accuracy =
(TP + TN) / (TP + TN + FP + FN)
```

---

## 19. What is precision?

Precision answers:

> Of all observations predicted as positive, how many were actually positive?

```text
Precision = TP / (TP + FP)
```

---

## 20. What is recall?

Recall answers:

> Of all actual positive observations, how many were correctly identified?

```text
Recall = TP / (TP + FN)
```

---

## 21. What is F1 Score?

F1 Score is the harmonic mean of Precision and Recall.

```text
F1 Score =
2 × (Precision × Recall)
-------------------------
Precision + Recall
```
# 🎯 Project-Based Interview Question

## Explain your Day 9 project.

### Answer:

"I used the Student Performance Factors dataset to build a K-Nearest
Neighbors classification model. I created a binary Performance target
where students with an Exam_Score of 75 or above were classified as
High Performance and the remaining students as Low Performance.

I selected Hours_Studied, Attendance, Previous_Scores, Sleep_Hours,
and Tutoring_Sessions as features. Since KNN is a distance-based
algorithm, I standardized the features using StandardScaler.

I then trained the KNN model, evaluated it using Accuracy, Precision,
Recall, F1 Score, and a Confusion Matrix, and tested different K values
from 1 to 20 to identify a suitable value."

---
