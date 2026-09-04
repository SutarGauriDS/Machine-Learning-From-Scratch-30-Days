# 🎯 Day 22 — Hyperparameter Tuning: 
### 1. What is hyperparameter tuning?

**Hyperparameter tuning** is the process of finding the best combination of hyperparameters to improve a machine learning model's performance.

---

### 2. What is a hyperparameter?

A **hyperparameter** is a setting defined before training the model.

Examples in Random Forest:

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`

---

### 3. What is the difference between parameters and hyperparameters?

| Parameters                  | Hyperparameters      |
| --------------------------- | -------------------- |
| Learned during training     | Set before training  |
| Example: tree split values  | Example: `max_depth` |
| Model learns them from data | We choose them       |

---

### 4. What is GridSearchCV?

**GridSearchCV** systematically tests different combinations of hyperparameters and uses cross-validation to find the best combination.

Example:

```python
GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="accuracy"
)
```

---

### 5. Why did you use GridSearchCV in your project?

I used **GridSearchCV** to systematically search through different Random Forest hyperparameter combinations instead of manually choosing values.

It helped me identify the combination that produced the best cross-validation performance.

---

### 6. What does `cv=5` mean?

`cv=5` means **5-fold cross-validation**.

The training data is divided into 5 parts. The model trains on 4 parts and validates on 1 part, repeating this process 5 times.

The results are then averaged.

---

### 7. What is `best_params_`?

`best_params_` returns the **hyperparameter combination that achieved the best cross-validation score**.

```python
grid_search.best_params_
```

---

### 8. What is `best_score_`?

`best_score_` gives the **best average cross-validation score** obtained during GridSearchCV.

```python
grid_search.best_score_
```

---

### 9. What is `best_estimator_`?

It returns the model trained/configured using the **best hyperparameters found by GridSearchCV**.

```python
grid_search.best_estimator_
```

---

### 10. Which model did you tune in your project?

I tuned a **Random Forest Classifier** using GridSearchCV on the **Breast Cancer Wisconsin dataset**.

---

### 11. Which hyperparameters did you tune?

I tuned:

```python
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
```

---

### 12. What does `n_estimators` mean?

`n_estimators` represents the **number of decision trees** in the Random Forest.

Generally, increasing it can improve stability, but it also increases computational cost.

---

### 13. What does `max_depth` do?

`max_depth` controls the **maximum depth of each decision tree**.

A very large depth can make trees more complex and potentially overfit.

---

### 14. What is `min_samples_split`?

It specifies the **minimum number of samples required to split an internal node**.

Increasing it can make the tree less complex.

---

### 15. What is `min_samples_leaf`?

It specifies the **minimum number of samples that must be present in a leaf node**.

Higher values can help reduce overfitting.

---

### 16. What is `max_features`?

`max_features` determines how many features are considered when looking for the best split in each tree.

In your project, you tested:

```python
"sqrt"
"log2"
```

---

### 17. Why did you create a baseline model?

The baseline provides a **reference point**.

I first trained a default Random Forest and then compared its performance with the tuned model to determine whether hyperparameter tuning actually improved the model.

---

### 18. Does Random Forest require feature scaling?

**No.**

Random Forest is tree-based, so feature scaling such as StandardScaler is generally not required.

---

### 19. What is the difference between Grid Search and Random Search?

**Grid Search:** Tests every specified combination.

**Random Search:** Tests randomly selected combinations from the specified search space.

Random Search can be more efficient when there are many hyperparameters.

---

### 20. Can hyperparameter tuning cause overfitting?

Yes, if tuning is done improperly.

Using **cross-validation** helps reduce the risk of selecting hyperparameters that work well only on one particular validation split.

---

### 21. Why shouldn't you tune hyperparameters using the test set?

The test set should represent **unseen data**.

If we repeatedly use it during tuning, information from the test set influences model selection, making the final evaluation less reliable.

---

### 22. What dataset did you use?

I used Scikit-learn's **Breast Cancer Wisconsin dataset**.

It contains:

* **569 samples**
* **30 numerical features**
* **2 target classes**

---

### 23. Why did you use `stratify=y` in train-test split?

```python
train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

`stratify=y` helps maintain a similar **class distribution** in both training and testing datasets.

---

### 24. How would you explain your Day 22 project in an interview?

> **In Day 22 of my ML challenge, I worked on hyperparameter tuning using a Random Forest Classifier. I first created a baseline model and then used GridSearchCV with 5-fold cross-validation to test different combinations of parameters such as n_estimators, max_depth, min_samples_split, min_samples_leaf, and max_features. I compared the tuned model with the baseline and also analyzed the confusion matrix and feature importance. This helped me understand how systematic hyperparameter optimization can improve model performance.**
