# 🎯 Day 13 — Interview Questions & Answers

### 1. Why do we compare multiple machine learning models?

**Answer:**
We compare multiple models to determine which algorithm performs best on our dataset. We shouldn't select a model based only on accuracy; we should also consider precision, recall, F1-score, training time, interpretability, and overfitting.

---

### 2. Which models did you compare in your project?

**Answer:**
I compared three classification algorithms:

* Decision Tree
* Gaussian Naive Bayes
* Random Forest

All three models were trained and tested using the same Student Performance dataset and the same train-test split.

---

### 3. Why did you use the same train-test split?

**Answer:**
Using the same training and testing data makes the comparison fair. Otherwise, differences in the data could affect the results rather than differences in the models themselves.

---

### 4. What is accuracy?

**Answer:**
Accuracy is the proportion of correctly predicted observations out of all observations.

[
Accuracy = \frac{Correct\ Predictions}{Total\ Predictions}
]

---

### 5. Is accuracy always a good metric?

**Answer:**
No. Accuracy can be misleading when the classes are imbalanced. That's why I also considered precision, recall, and F1-score.

---

### 6. What is precision?

**Answer:**
Precision tells us how many of the observations predicted as positive were actually positive.

[
Precision = \frac{TP}{TP+FP}
]

For my project, it tells me how reliable the model's **Pass** predictions are.

---

### 7. What is recall?

**Answer:**
Recall tells us how many of the actual positive cases were correctly identified.

[
Recall = \frac{TP}{TP+FN}
]

In this project, it measures how well the model identifies students classified as Pass.

---

### 8. What is F1-score?

**Answer:**
F1-score is the harmonic mean of precision and recall. It provides a balance between the two.

[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
]

---

### 9. What is a confusion matrix?

**Answer:**
A confusion matrix summarizes classification predictions into:

* True Positive
* True Negative
* False Positive
* False Negative

It helps us understand where the model is making mistakes.

---

### 10. Why did you use F1-score to select the best model?

**Answer:**
I used F1-score because it considers both precision and recall. It gives a more balanced evaluation than accuracy alone, especially when we want to consider both types of classification errors.

---

### 11. What is the difference between Decision Tree and Random Forest?

**Answer:**
A Decision Tree uses a single tree to make predictions, whereas Random Forest combines multiple Decision Trees. Random Forest generally provides better generalization and is less prone to overfitting than a single unrestricted Decision Tree.

---

### 12. What is the difference between Naive Bayes and Decision Tree?

**Answer:**
Naive Bayes is a probabilistic algorithm based on Bayes' theorem and assumes conditional independence between features. Decision Tree is rule-based and creates splits based on feature values.

---

### 13. Why can Random Forest perform better than a single Decision Tree?

**Answer:**
Random Forest combines predictions from multiple trees trained using different samples and feature subsets. This reduces the effect of individual trees and generally improves robustness and generalization.

---

### 14. What is overfitting?

**Answer:**
Overfitting occurs when a model performs very well on training data but performs poorly on unseen testing data because it has learned the training data too closely.

---

### 15. How did you identify overfitting?

**Answer:**
I compared training accuracy with testing accuracy. If training accuracy is significantly higher than testing accuracy, it can indicate overfitting.

---

### 16. What is underfitting?

**Answer:**
Underfitting occurs when a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and testing data.

---

### 17. What is target leakage in your project?

**Answer:**
I created the `Pass` target using `Exam_Score`. Therefore, I excluded `Exam_Score` from the input features. If I included it, the model could directly use information that was used to create the target, causing target leakage.

---

### 18. Why did you use `random_state=42`?

**Answer:**
`random_state=42` makes the random operations reproducible. This means I can run the code again and obtain the same train-test split and comparable results.

---

### 19. Why did you use `stratify=y`?

**Answer:**
`stratify=y` maintains approximately the same proportion of Pass and Fail classes in both the training and testing datasets.

---

### 20. How would you improve your model comparison?

**Answer:**
I would use **cross-validation** instead of relying on a single train-test split. I could also perform hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV` and compare the models across multiple folds.

---
