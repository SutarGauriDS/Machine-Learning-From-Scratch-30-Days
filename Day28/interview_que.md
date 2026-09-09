### 1. What is an End-to-End Machine Learning Pipeline?

An end-to-end ML pipeline connects all major stages of a machine learning workflow:

**Data → Preprocessing → Training → Evaluation → Saving → Loading → Prediction**

---

### 2. Why did you use a Pipeline?

I used Scikit-learn’s `Pipeline` to combine preprocessing and model training into a single workflow.

This ensures that the same preprocessing steps are applied consistently during training and prediction.

---

### 3. Which dataset did you use?

I used the **Wine Recognition Dataset** built into Scikit-learn.

It contains:

* **178 samples**
* **13 numerical features**
* **3 target classes**

---

### 4. What is the objective of the project?

The objective is to classify wine samples into **three different classes** based on their chemical properties.

It is a **multiclass classification problem**.

---

### 5. Which algorithm did you use?

I used **Logistic Regression** as the classification model.

It was selected to demonstrate the complete ML pipeline from preprocessing to model persistence and prediction.

---

### 6. Why did you use StandardScaler?

`StandardScaler` standardizes numerical features so they have approximately:

**Mean = 0**
**Standard Deviation = 1**

This is useful for Logistic Regression because the model can be affected by differences in feature scales.

---

### 7. Why did you use SimpleImputer?

`SimpleImputer` is used to handle missing values.

In this project, I used:

```python
SimpleImputer(strategy="median")
```

Missing numerical values are replaced with the median calculated from the training data.

---

### 8. Why did you use train-test split?

I used an **80/20 train-test split**.

The training set is used to train the model, while the test set is kept separate for evaluating performance on unseen data.

---

### 9. Why did you use `stratify=y`?

`stratify=y` helps maintain a similar distribution of target classes in both the training and testing datasets.

---

### 10. What is data leakage?

Data leakage happens when information that should not be available during training is unintentionally used by the model.

For example, fitting a scaler using the complete dataset before splitting it can cause leakage.

---

### 11. How does Pipeline help prevent data leakage?

The preprocessing steps are included inside the Pipeline.

When the Pipeline is trained, preprocessing is fitted using the training data and then applied to the test data.

This helps maintain a proper ML workflow.

---

### 12. What metrics did you use?

I used:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Classification Report**
* **Confusion Matrix**

For this multiclass problem, weighted Precision, Recall, and F1-score were used.

---

### 13. What is Accuracy?

Accuracy measures the percentage of predictions that are correct.

**Accuracy = Correct Predictions / Total Predictions**

---

### 14. What is Precision?

Precision measures how many samples predicted as a particular class actually belong to that class.

**Precision = TP / (TP + FP)**

---

### 15. What is Recall?

Recall measures how many actual samples belonging to a class were correctly identified.

**Recall = TP / (TP + FN)**

---

### 16. What is F1-score?

F1-score is the harmonic mean of Precision and Recall.

**F1 = 2 × (Precision × Recall) / (Precision + Recall)**

It provides a balance between precision and recall.

---

### 17. What is a Confusion Matrix?

A confusion matrix compares the **actual classes** with the **predicted classes**.

It helps identify which classes were correctly classified and which classes were confused with each other.

---

### 18. What is multiclass classification?

Multiclass classification is a classification problem where there are more than two possible target classes.

In this project, there are **three wine classes**.

---

### 19. Why did you use weighted averaging?

Weighted averaging calculates the metric for each class and weights each class according to the number of samples it contains.

This provides an overall metric while considering differences in class sizes.

---

### 20. What is model persistence?

Model persistence means **saving a trained machine learning model so that it can be reused later without retraining it**.

---

### 21. Why did you use Joblib?

I used **Joblib** to save and load the trained ML Pipeline.

The main functions are:

```python
joblib.dump()
```

for saving and:

```python
joblib.load()
```

for loading.

---

### 22. What did you save in this project?

I saved the **complete ML Pipeline**, including:

**Preprocessing + Logistic Regression Model**

The saved file is:

```text
day27_wine_classification_pipeline.pkl
```

---

### 23. Why save the complete Pipeline instead of only the model?

Saving the complete Pipeline ensures that preprocessing and prediction remain together.

This avoids having to manually reproduce preprocessing steps before making predictions.

---

### 24. How would you deploy this model?

A possible deployment architecture is:

**User/Application → API → Saved ML Pipeline → Prediction → Response**

The Pipeline could be integrated into an API using **FastAPI** or **Flask**.

---

### 25. How would you improve this project?

I could improve it by:

1. Comparing multiple ML algorithms.
2. Adding cross-validation.
3. Performing hyperparameter tuning.
4. Adding data validation.
5. Creating a FastAPI prediction API.
6. Building a simple frontend.
7. Dockerizing the application.
8. Adding model monitoring.

---

### 26. What is the difference between a model and a pipeline?

A **model** is responsible for making predictions.

A **pipeline** combines multiple processing steps with the model.

For example:

**Preprocessing → Logistic Regression**

Together, these form the ML Pipeline.

---

### 27. What is the main advantage of an end-to-end pipeline?

An end-to-end pipeline provides:

* **Consistency**
* **Reproducibility**
* **Automation**
* **Maintainability**
* **Easier deployment**
* **Reduced preprocessing errors**

---

### 28. Explain your Day 27 project in 30 seconds.

> I built an end-to-end machine learning pipeline using the Scikit-learn Wine dataset. I performed an 80/20 stratified train-test split, handled missing values using SimpleImputer, standardized features using StandardScaler, and trained a Logistic Regression classifier. I evaluated the model using accuracy, precision, recall, F1-score, and a confusion matrix. Finally, I saved the complete preprocessing and model pipeline using Joblib, loaded it again, and used it to make predictions.

---

## Quick Revision

| Concept                 | Remember                      |
| ----------------------- | ----------------------------- |
| **Pipeline**            | Preprocessing + Model         |
| **SimpleImputer**       | Handles missing values        |
| **StandardScaler**      | Standardizes features         |
| **Logistic Regression** | Classification algorithm      |
| **Train-Test Split**    | Training vs evaluation data   |
| **Stratify**            | Maintains class distribution  |
| **Accuracy**            | Overall correct predictions   |
| **Precision**           | Correct predicted positives   |
| **Recall**              | Correct actual positives      |
| **F1-score**            | Balance of Precision & Recall |
| **Confusion Matrix**    | Actual vs Predicted           |
| **Joblib**              | Save & load ML objects        |
| **Model Persistence**   | Reuse trained model           |
| **End-to-End ML**       | Data → Prediction             |
