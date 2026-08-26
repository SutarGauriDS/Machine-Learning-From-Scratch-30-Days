### 1. What is SVM?

**Support Vector Machine (SVM)** is a supervised machine learning algorithm used mainly for classification and also for regression.

It finds a decision boundary, called a **hyperplane**, that separates different classes while trying to maximize the margin between them.

---

### 2. What is a hyperplane?

A hyperplane is the decision boundary used by SVM to separate classes.

For 2D data, it is a line:

```text
Class A       |       Class B
● ● ●         |       ○ ○ ○
● ● ●         |       ○ ○ ○
              |
          Hyperplane
```

For higher-dimensional data, it becomes a mathematical hyperplane.

---

### 3. What are support vectors?

**Support vectors are the data points closest to the decision boundary.**

They are important because they determine the position of the optimal hyperplane.

```text
● ●      ● | ●      ○ ○
           ↑
     Support Vectors
```

---

### 4. What is the margin in SVM?

The **margin** is the distance between the decision boundary and the closest data points from each class.

SVM tries to find the hyperplane with the **maximum possible margin**.

---

### 5. What is `C` in SVM?

`C` is the **regularization parameter**.

It controls the trade-off between:

* having a wider margin
* correctly classifying training samples

#### Small C

Allows more classification errors but generally creates a wider margin.

#### Large C

Penalizes errors more strongly and tries harder to classify training samples correctly.

---

### 6. What is a kernel in SVM?

A kernel allows SVM to handle **non-linear relationships** by transforming the data into a higher-dimensional feature space.

Common kernels:

```text
linear
poly
rbf
sigmoid
```

---

### 7. What is the difference between Linear and RBF SVM?

**Linear SVM:**

```python
SVC(kernel="linear")
```

Used when classes can be separated reasonably well with a linear boundary.

**RBF SVM:**

```python
SVC(kernel="rbf")
```

Useful when the relationship between classes is non-linear.

In your Day 13 project, you compared exactly these two approaches.

---

### 8. What is `gamma`?

`gamma` is mainly important for kernels such as **RBF**.

It controls how far the influence of an individual training point reaches.

* Higher gamma → more localized influence → more complex boundary
* Lower gamma → broader influence → smoother boundary

---

### 9. Why is feature scaling important for SVM?

SVM relies on distances and margins, so features with very different scales can disproportionately affect the model.

That's why your project uses:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

An important interview point:

> We fit the scaler only on training data and use `transform()` on test data to avoid data leakage.

---

### 10. Why use `fit_transform()` on training data but only `transform()` on test data?

Because the scaler should learn the mean and standard deviation **only from the training data**.

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

If we fit the scaler on the test data, information from the test set leaks into the training process.

---

### 11. What happens if `C` is very high?

A high `C` strongly penalizes misclassification.

The model may try to classify almost every training point correctly, potentially creating a more complex decision boundary and increasing the risk of overfitting.

---

### 12. What happens if `C` is very low?

A low `C` allows more training errors in exchange for a larger margin.

This can produce a simpler model, but if it's too low, the model may underfit.

---

### 13. What is the kernel trick?

The **kernel trick** allows SVM to work with non-linear data without explicitly calculating the coordinates of the data in the higher-dimensional space.

This is one reason RBF SVM can create non-linear decision boundaries.

---

### 14. Is SVM sensitive to outliers?

**Yes.**

Outliers can influence the position of the decision boundary and margin.

Feature scaling and appropriate regularization can help, but preprocessing and outlier handling are still important.

---

### 15. What are the advantages of SVM?

* Works well with high-dimensional data
* Effective for classification
* Can model non-linear relationships using kernels
* Works well when the number of features is relatively large
* Uses support vectors rather than treating every training point equally

---

### 16. What are the disadvantages?

* Can be computationally expensive on very large datasets
* Sensitive to feature scaling
* Choosing `C`, `gamma`, and kernel can require tuning
* Less interpretable than simple models such as Logistic Regression
* Performance can be affected by outliers

---
### "Why did you use StandardScaler before SVM?"

A strong answer:

> "I used StandardScaler because SVM is sensitive to feature magnitude. The dataset contains features with different scales, and unscaled features could influence the distance and margin calculations disproportionately. I fitted the scaler only on the training data and then transformed both training and test data to prevent data leakage."

---

### "Explain your Day 13 SVM project."

You can answer:

> "I implemented an SVM classification project using the Breast Cancer dataset from Scikit-learn. I first divided the data into training and testing sets using stratified sampling. Since SVM is sensitive to feature scale, I applied StandardScaler. I then trained two SVM models: Linear SVM and RBF SVM. I compared their accuracy and evaluated the better model using a classification report and confusion matrix. I also visualized the accuracy comparison between the two models."

### Remember these 6 terms for the interview:

**Hyperplane → Margin → Support Vectors → Kernel → C → Gamma**

If you understand these six properly, you'll be able to handle most **basic-to-intermediate SVM interview questions**.
