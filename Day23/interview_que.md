#  Day 23 — Cross Validation: Interview Questions & Answers

### 1. What is Cross Validation?

**Cross Validation** is a technique used to evaluate how well a machine learning model performs on unseen data.

Instead of relying on a single validation split, the data is divided into multiple folds and the model is evaluated multiple times.

---

### 2. What is K-Fold Cross Validation?

In **K-Fold Cross Validation**, the dataset is divided into **K equal parts (folds)**.

For example, with `K=5`:

* Train on 4 folds
* Validate on 1 fold
* Repeat 5 times
* Calculate the average score

---

### 3. Why did you use 5-Fold Cross Validation?

I used **5 folds** because it provides a good balance between reliable evaluation and computational cost.

It allows every training sample to participate in validation while keeping the process reasonably efficient.

---

### 4. What is Stratified K-Fold?

**Stratified K-Fold** is mainly used for classification problems.

It maintains approximately the same **class distribution in each fold** as in the complete dataset.

---

### 5. Why did you use StratifiedKFold in your project?

The project uses a **binary classification dataset**, so maintaining the proportion of both classes across folds is important.

Therefore, I used `StratifiedKFold`.

---

### 6. What is `cross_val_score()`?

`cross_val_score()` evaluates a model using cross-validation and returns the score for each fold.

```python
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)
```

---

### 7. What does `cv=5` mean?

It means the model is evaluated using **5 cross-validation folds**.

Each fold gets an opportunity to act as the validation set.

---

### 8. What is Mean CV Accuracy?

It is the **average accuracy across all folds**.

```python
mean_score = scores.mean()
```

A higher mean score generally indicates better average performance.

---

### 9. What does Standard Deviation tell you?

Standard deviation tells us how much the model's performance **varies across different folds**.

* Lower standard deviation → more consistent performance
* Higher standard deviation → greater variation

---

### 10. Why is Cross Validation better than a single train-validation split?

A single split can give a performance estimate that depends heavily on which observations happen to be in the validation set.

Cross-validation evaluates the model across multiple splits, giving a **more robust estimate of performance**.

---

### 11. Which models did you compare?

I compared:

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Random Forest**

---

### 12. Why did you use a Pipeline?

For Logistic Regression and KNN, I used a pipeline containing `StandardScaler` and the model.

This ensures that scaling is performed properly **within each cross-validation training fold**, helping prevent data leakage.

---

### 13. Why doesn't Random Forest require scaling?

Random Forest is based on **decision trees**, which split data based on feature thresholds. Therefore, differences in feature scale generally do not affect the tree splitting process.

---

### 14. What is data leakage?

**Data leakage** occurs when information from outside the training data improperly influences model training.

For example, fitting a scaler on the entire dataset before cross-validation can allow validation-fold information to influence the scaling.

---

### 15. Why did you keep the test set separate?

The test set represents **unseen data**.

I used the training data for cross-validation and kept the test set untouched until the final evaluation.

---

### 16. What is the difference between Cross Validation and Train-Test Split?

| Train-Test Split     | Cross Validation               |
| -------------------- | ------------------------------ |
| Usually one split    | Multiple splits/folds          |
| Faster               | More computationally expensive |
| Less robust estimate | More reliable estimate         |
| Simple evaluation    | Better for model comparison    |

---

### 17. What is `random_state=42`?

`random_state=42` makes the random operations **reproducible**.

Running the same code again with the same settings will produce the same split/fold arrangement.

---

### 18. What dataset did you use?

I used the **Breast Cancer Wisconsin dataset** from Scikit-learn.

It contains:

* **569 samples**
* **30 numerical features**
* **2 target classes**

---

### 19. How did you select the best model?

I compared the models using their **mean cross-validation accuracy**.

The model with the highest mean CV accuracy was selected and then evaluated on the unseen test set.

I also considered standard deviation to understand performance stability.

---

### 20. Explain your Day 23 project in an interview.

> **In Day 23 of my Machine Learning challenge, I worked on Cross Validation for model performance evaluation. I used the Breast Cancer Wisconsin dataset and compared Logistic Regression, KNN, and Random Forest. I used Stratified 5-Fold Cross Validation to preserve class distribution across folds. For each model, I calculated the mean accuracy and standard deviation to evaluate both performance and stability. I selected the best-performing model based on cross-validation results and finally evaluated it on an unseen test set.**
