
### 1. What is PCA?

**PCA (Principal Component Analysis)** is a dimensionality reduction technique that transforms a dataset with many features into a smaller set of new features called **principal components**, while retaining as much variance as possible.

---

### 2. Why do we use PCA?

PCA can help to:

* Reduce the number of features
* Reduce computational complexity
* Remove redundancy from correlated features
* Visualize high-dimensional data
* Reduce noise in some situations
* Make models easier to work with

---

### 3. What is a Principal Component?

A principal component is a new feature created as a **linear combination of the original features**.

The first principal component captures the maximum variance, the second captures the next highest variance, and so on.

---

### 4. What is Explained Variance?

**Explained Variance Ratio** tells us how much of the total information/variance in the dataset is captured by each principal component.

For example:

```text
PC1 → 45%
PC2 → 20%
PC3 → 10%
```

Together, the first three components explain **75%** of the variance.

---

### 5. What is Cumulative Explained Variance?

It is the total variance captured by the first `n` principal components.

We use it to decide how many components should be retained.

In your project, we selected components that retain **at least 95% variance**.

---

### 6. Why did you apply StandardScaler before PCA?

PCA is sensitive to feature magnitudes.

If features have very different scales, features with larger numerical values can have a greater influence on PCA.

Therefore, I standardized the features before applying PCA.

---

### 7. Why did you use `fit_transform()` on training data?

```python
X_train_scaled = scaler.fit_transform(X_train)
```

`fit()` learns the scaling parameters from the training data, while `transform()` applies the transformation.

---

### 8. Why use only `transform()` on test data?

```python
X_test_scaled = scaler.transform(X_test)
```

The test set should remain unseen during preprocessing.

Using `fit_transform()` on the test set could introduce **data leakage**.

---

### 9. How did you decide the number of PCA components?

I calculated the **cumulative explained variance** and selected the smallest number of components that retained at least **95% of the variance**.

```python
n_components_95 = (
    cumulative_variance >= 0.95
).argmax() + 1
```

---

### 10. Can PCA be used without scaling?

Technically, yes, but it can produce misleading results when features have very different scales.

Therefore, scaling is generally recommended before PCA when feature magnitudes differ significantly.

---

### 11. What is the difference between Feature Selection and PCA?

| Feature Selection                    | PCA                                         |
| ------------------------------------ | ------------------------------------------- |
| Selects existing features            | Creates new features                        |
| Original feature meaning is retained | New components may be harder to interpret   |
| Removes less useful features         | Combines information from multiple features |
| Example: SelectKBest                 | Example: PCA                                |

---

### 12. Is PCA supervised or unsupervised?

**PCA is an unsupervised technique.**

It does not use the target variable when finding the principal components.

---

### 13. Does PCA always improve model accuracy?

**No.**

PCA reduces dimensionality and may remove some information. Therefore, model performance can improve, remain similar, or decrease.

The benefit depends on the dataset and model.

---

### 14. Why did you use Logistic Regression after PCA?

Logistic Regression is a classification algorithm suitable for the binary target in the Breast Cancer dataset.

I used it to compare:

```text
Logistic Regression + All Features
              VS
Logistic Regression + PCA Features
```

---

### 15. How did you visualize high-dimensional data?

I used:

```python
PCA(n_components=2)
```

This reduced the dataset to two principal components, allowing the data to be plotted on a 2D graph.

---

# Explain Your Day 18 Project

### Interview Answer:

> "In Day 18, I worked on Principal Component Analysis using the Breast Cancer Wisconsin dataset. The dataset initially contained 30 features. First, I split the data into training and testing sets and standardized the features using StandardScaler. I then trained a Logistic Regression model as a baseline. After that, I applied PCA and analyzed the explained variance and cumulative explained variance. I selected the minimum number of components required to retain at least 95% of the variance. I trained another Logistic Regression model using the reduced PCA features and compared its performance with the baseline model. I also created a 2D PCA visualization and evaluated the final model using a confusion matrix and classification report."

---

## 🔥 Rapid-Fire Questions

**Q: PCA stands for?**  

➡️ Principal Component Analysis.

**Q: PCA is used for?** 

➡️ Dimensionality reduction.

**Q: Is PCA supervised?** 

➡️ No, it is unsupervised.

**Q: What does PC1 represent?** 

➡️ The direction capturing the maximum variance.

**Q: Why scale before PCA?** 

➡️ PCA is sensitive to feature magnitude.

**Q: What does explained variance tell us?** 

➡️ How much variance is captured by each component.

**Q: Why use cumulative variance?** 

➡️ To decide how many components to retain.

**Q: What percentage did you retain?**

➡️ At least **95%**.

**Q: Can PCA reduce 30 features to 2?**
➡️ Yes, for visualization, but that may not retain 95% of the variance.

**Q: Does PCA guarantee better accuracy?**
➡️ No.
