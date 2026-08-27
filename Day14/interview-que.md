## 🎯 Day 14 — SVM Hyperparameter Tuning Interview Questions
### 1. What is hyperparameter tuning?

Hyperparameter tuning is the process of finding the best values of model settings such as `C`, `kernel`, and `gamma` to improve model performance.

### 2. What is GridSearchCV?

`GridSearchCV` systematically tests different combinations of hyperparameters using cross-validation and selects the combination with the best validation score.

### 3. Why did you use GridSearchCV?

> "Instead of manually selecting SVM parameters, I used GridSearchCV to systematically evaluate multiple combinations of C, kernel, and gamma and identify the best-performing configuration."

### 4. What is cross-validation?

Cross-validation divides the training data into multiple parts called folds and repeatedly trains and validates the model on different combinations of those folds.

### 5. Why did you use `cv=5`?

`cv=5` performs **5-fold cross-validation**, giving a more reliable estimate of model performance than relying on a single validation split.

### 6. Which hyperparameters did you tune?

In this project:

```python
C
kernel
gamma
```

### 7. What does `C` do?

`C` controls the penalty for classification errors.

* **Small C:** wider margin, more tolerance for errors
* **Large C:** stronger penalty for errors, potentially more complex boundary

### 8. What does `gamma` do?

For kernels such as RBF, `gamma` controls how strongly individual training points influence the decision boundary.

* Higher gamma → more complex/local boundary
* Lower gamma → smoother boundary

### 9. Why did you test different kernels?

Different kernels allow SVM to model different types of relationships.

```python
["linear", "rbf", "poly"]
```

* **Linear:** linear relationships
* **RBF:** non-linear relationships
* **Polynomial:** polynomial relationships

### 10. What is the difference between `best_score_` and test accuracy?

**`best_score_`** is the best average cross-validation score obtained during GridSearchCV.

**Test accuracy** is the performance of the selected best model on the completely unseen test set.

### 11. Why shouldn't you tune using the test set?

Because the test set should remain unseen until final evaluation.

If we repeatedly use the test set to select parameters, it can lead to **data leakage** and an overly optimistic estimate of performance.

### 12. Why did you use a Pipeline?

Your project uses:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])
```

A pipeline ensures that scaling is correctly performed within each cross-validation training fold, helping prevent data leakage.

### 13. Why is scaling important for SVM?

SVM is sensitive to feature magnitudes. Scaling puts features on comparable scales so that features with larger numerical values don't disproportionately affect the model.

### 14. What is `n_jobs=-1`?

It tells GridSearchCV to use all available CPU cores to perform the parameter search in parallel, when supported.

### 15. What is overfitting in SVM?

Overfitting occurs when the model learns the training data too closely and performs poorly on unseen data.

Very aggressive parameter choices, such as a very large `C` or inappropriate `gamma`, can contribute to overly complex decision boundaries.

--
### "Explain your Day 14 project."

You can answer:

> "In Day 14, I extended my SVM classification project by applying hyperparameter tuning. I first created a baseline RBF SVM and recorded its test accuracy. Then I used GridSearchCV with 5-fold cross-validation to test different combinations of C, kernel, and gamma. After finding the best parameters based on cross-validation performance, I evaluated the selected model on the unseen test set. Finally, I compared the baseline and tuned models and evaluated the tuned model using accuracy, classification report, and confusion matrix."

### 🔥 Remember this flow for interviews:

**Baseline → Parameter Grid → GridSearchCV → 5-Fold CV → Best Parameters → Best Model → Test Set → Evaluation**

