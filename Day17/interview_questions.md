##  Feature Scaling: Interview Questions

### 1. What is Feature Scaling?

Feature Scaling is the process of transforming numerical features so that they are on comparable scales.

---

### 2. Why is Feature Scaling important?

It is especially important for algorithms that depend on **distance or magnitude**, such as:

* KNN
* K-Means
* SVM
* PCA
* Neural Networks

---

### 3. Why did you use KNN in this project?

KNN is a **distance-based algorithm**, so differences in feature scales can significantly affect the distance calculation. This makes KNN a good algorithm for demonstrating the effect of Feature Scaling.

---

### 4. What happens if we don't scale features?

A feature with a much larger numerical range can dominate the distance calculation, causing smaller-scale features to have less influence.

---

### 5. What is Standardization?

Standardization transforms a feature so that it generally has:

```text
Mean = 0
Standard Deviation = 1
```

It is performed using:

```python
StandardScaler()
```

---

### 6. What is Normalization?

Normalization commonly transforms values into a fixed range, usually:

```text
0 to 1
```

In this project, I used:

```python
MinMaxScaler()
```

---

### 7. StandardScaler vs MinMaxScaler?

| StandardScaler                   | MinMaxScaler                            |
| -------------------------------- | --------------------------------------- |
| Mean ≈ 0                         | Usually range 0–1                       |
| Std ≈ 1                          | Fixed range                             |
| Uses mean and standard deviation | Uses minimum and maximum                |
| Can produce negative values      | Usually produces values between 0 and 1 |

---

### 8. What is the formula for Standardization?

The standardization formula is:

$$
z = \frac{x-\mu}{\sigma}
$$

Where:

* `x` = original value
* `μ` = mean
* `σ` = standard deviation

---

### 9. What is the formula for Min-Max Scaling?

$$
x' = \frac{x-x_{min}}{x_{max}-x_{min}}
$$

This maps values to the range 0–1 when the default range is used.

---

### 10. Why do you use `fit_transform()` on training data?

```python
X_train_scaled = scaler.fit_transform(X_train)
```

`fit()` learns the scaling parameters from the training data, and `transform()` applies them.

---

### 11. Why only `transform()` on test data?

```python
X_test_scaled = scaler.transform(X_test)
```

The test data must remain unseen during the fitting process.

This helps prevent **data leakage**.

---

### 12. What is Data Leakage?

Data leakage occurs when information from outside the training data, especially the test set, unintentionally influences model training.

For scaling, this can happen if we do:

```python
scaler.fit_transform(X_test)
```

separately on the test set.

---

### 13. Is Feature Scaling required for every ML algorithm?

No.

Scaling is generally important for distance- or magnitude-sensitive algorithms.

Tree-based algorithms such as:

* Decision Tree
* Random Forest

generally don't require feature scaling because their splits are based on feature thresholds rather than distances.

---

### 14. What did you compare in your project?

I compared three KNN approaches:

```text
1. KNN without scaling
2. KNN + StandardScaler
3. KNN + MinMaxScaler
```

I compared their test accuracy to understand the effect of scaling.

---

### 15. What dataset did you use?

I used the **Breast Cancer Wisconsin (Diagnostic) dataset** available through Scikit-learn.

It is a binary classification dataset with **569 samples and 30 numerical features**.

---

### "Explain your Day 17 project."

**Answer:**

> "In Day 17, I worked on Feature Scaling using the Breast Cancer Wisconsin dataset. I used KNN because it is a distance-based algorithm and is sensitive to feature scales. First, I trained KNN without scaling as a baseline. Then I applied StandardScaler and MinMaxScaler to the training data and evaluated KNN again. Finally, I compared the accuracies, visualized the feature distributions, and evaluated the standardized model using a confusion matrix and classification report. I also made sure to fit the scalers only on the training data to avoid data leakage."

**Raw Data → Train-Test Split → No Scaling → StandardScaler → MinMaxScaler → KNN → Compare → Evaluate**
