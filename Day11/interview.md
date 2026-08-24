# 🎯 Day 12 – Decision Tree Interview Questions

## 1. What is a Decision Tree?

A Decision Tree is a supervised Machine Learning algorithm used for
classification and regression.

It makes predictions by repeatedly splitting data based on feature
conditions.

---

## 2. How does a Decision Tree work?

A Decision Tree starts with a root node and divides the dataset into
smaller groups using the best feature split.

The process continues until a stopping condition is reached.

Structure:

Root Node
   ↓
Decision / Internal Nodes
   ↓
Leaf Nodes
   ↓
Prediction

---

## 3. What is the root node?

The root node is the first node of a Decision Tree.

It contains the entire training dataset and performs the first split.

---

## 4. What is a leaf node?

A leaf node is the final node of a Decision Tree.

It contains the final prediction.

---

## 5. What is Gini Impurity?

Gini Impurity measures how mixed the classes are within a node.

A lower Gini value indicates a purer node.

Formula:

Gini = 1 - Σ(pᵢ²)

---

## 6. What is Entropy?

Entropy measures the uncertainty or impurity in a dataset.

Formula:

Entropy = -Σ pᵢ log₂(pᵢ)

Lower entropy means the node is more pure.

---

## 7. What is Information Gain?

Information Gain measures how much uncertainty is reduced after
splitting a node.

A Decision Tree generally selects the split that provides the highest
information gain.

---

## 8. What is Gini vs Entropy?

| Gini | Entropy |
|---|---|
| Measures impurity | Measures uncertainty |
| Usually faster | Slightly more computationally expensive |
| Common default in Scikit-Learn | Based on information theory |

---

## 9. What is overfitting in Decision Trees?

A Decision Tree can become too complex and learn noise from the
training data.

This causes high training performance but poor performance on unseen
data.

---

## 10. How can you prevent Decision Tree overfitting?

Common techniques include:

- Limit `max_depth`
- Increase `min_samples_split`
- Increase `min_samples_leaf`
- Use pruning
- Evaluate using cross-validation

Example:

```python
DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)
````

---

## 11. What is `max_depth`?

`max_depth` controls the maximum depth of a Decision Tree.

A smaller depth generally produces a simpler model.

---

## 12. Can Decision Trees handle categorical data?

Yes, Decision Trees can conceptually work with categorical features,
but Scikit-Learn's standard DecisionTreeClassifier generally requires
features to be represented numerically.

Categorical variables can be encoded before training.

---

## 13. What is feature importance?

Feature importance indicates how useful each feature was in making
decisions in the tree.

In Scikit-Learn:

```python
model.feature_importances_
```

---

## 14. What are the advantages of Decision Trees?

* Easy to understand
* Easy to visualize
* Requires relatively little preprocessing
* Can model non-linear relationships
* Useful for classification and regression
* Provides feature importance

---

## 15. What are the disadvantages?

* Can easily overfit
* Small changes in data can change the tree
* Very deep trees can become difficult to interpret
* Individual trees may have lower generalization performance than
  ensemble methods

---

## 16. Is feature scaling required for Decision Trees?

Generally, no.

Decision Trees split data using feature thresholds, so standardization
is usually not required.

---

## 17. What is a Random Forest?

Random Forest is an ensemble Machine Learning algorithm that combines
multiple Decision Trees to improve generalization.

---

## 18. Decision Tree vs Random Forest

| Decision Tree       | Random Forest                          |
| ------------------- | -------------------------------------- |
| Single tree         | Multiple trees                         |
| Easier to visualize | More complex                           |
| Can overfit easily  | Usually reduces overfitting            |
| Faster to train     | Usually more computationally expensive |

---

## 19. Which metrics can be used to evaluate a Decision Tree classifier?

Common classification metrics include:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 20. What did you use as the target in Day 12?

For this project, the continuous `Exam_Score` was converted into a
classification target:

```text
Exam_Score >= 75 → High Performance
Exam_Score < 75  → Low Performance
```

The Decision Tree then predicts the student's performance category.

---

## ⭐ Quick Interview Revision

Remember:

```text
Decision Tree
      ↓
Root Node
      ↓
Best Feature Split
      ↓
Gini / Entropy
      ↓
Internal Nodes
      ↓
Leaf Node
      ↓
Prediction
```

Key parameters:

```text
max_depth
min_samples_split
min_samples_leaf
criterion
```

Key metrics:

```text
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
```

```
```
