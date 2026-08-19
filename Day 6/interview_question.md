#  Day 6 – Machine Learning Workflow Interview Questions

## 1. What is the Machine Learning workflow?

A typical workflow is:

**Data Collection → Exploration → Preprocessing → Feature Selection → Train-Test Split → Model Training → Prediction → Evaluation**

---

## 2. What are features?

Features are the input variables used by a Machine Learning model to make predictions.

Example:

`Study_Hours` is a feature.

---

## 3. What is a target variable?

The target is the output that the model is trying to predict.

Example:

`Marks` is the target.

---

## 4. Why do we split data into training and testing sets?

Training data is used to learn patterns, while testing data is used to evaluate how well the model performs on unseen data.

---

## 5. What is `train_test_split()`?

It is a Scikit-Learn function used to divide data into training and testing subsets.

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 6. What does `test_size=0.2` mean?

It means approximately **20% of the data is reserved for testing** and the remaining 80% is used for training.

---

## 7. What is `random_state`?

It controls the randomness of the split. Using the same value produces the same split, making experiments reproducible.

---

## 8. Why shouldn't we train and test on the same data?

Because the model has already seen the training data. Testing on the same data can give an overly optimistic estimate of performance.

---

## 9. What is data leakage?

Data leakage occurs when information that should not be available during model training influences the training process.

It can result in unrealistically high model performance.

---

## 10. What is overfitting?

Overfitting occurs when a model learns the training data too closely, including noise, and performs poorly on new or unseen data.

---

## ⭐ Important Questions

Focus especially on:

* Features vs Target
* Training vs Testing data
* Why train-test split is required
* `test_size`
* `random_state`
* Data leakage
* Overfitting
* Complete ML workflow
