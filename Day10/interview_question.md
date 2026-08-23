## 🌳 Decision Tree — Interview Questions & Answers

**1. What is a Decision Tree?**
A Decision Tree is a supervised machine learning algorithm used for **classification and regression**. It makes predictions by splitting data based on feature values.

**2. Why did you use Decision Tree for your Student Performance project?**
I used Decision Tree because it is easy to understand and interpret, can handle nonlinear relationships, and does not require feature scaling.

**3. What is the difference between classification and regression?**
Classification predicts categories such as **Pass/Fail**, while regression predicts continuous numerical values such as an **exam score**.

**4. What is Gini Impurity?**
Gini Impurity measures how impure or mixed the classes are in a node. A lower Gini value indicates a purer node.

**5. What is Entropy?**
Entropy measures the uncertainty or impurity in a dataset. Lower entropy means the data is more homogeneous.

**6. What is Information Gain?**
Information Gain measures how much uncertainty is reduced after splitting the data. The tree generally chooses the split that provides the greatest information gain.

**7. What is the root node?**
The root node is the **first/top node** of a Decision Tree where the initial split occurs.

**8. What is a leaf node?**
A leaf node is the final node of a Decision Tree where the prediction is made.

**9. What is overfitting in a Decision Tree?**
Overfitting occurs when the tree becomes too complex and learns the training data too closely, resulting in poor performance on unseen data.

**10. How can you prevent overfitting?**
I can control parameters such as:

```python
max_depth
min_samples_split
min_samples_leaf
```

For example:

```python
DecisionTreeClassifier(max_depth=5)
```

**11. Does Decision Tree require feature scaling?**  
No. Decision Trees generally don't require normalization or standardization because they make decisions using feature thresholds rather than distances.

**12. What is `max_depth`?**  
`max_depth` controls the maximum depth of the Decision Tree. A smaller value can help reduce overfitting.

**13. What is feature importance?**  
Feature importance tells us how much each feature contributes to the model's decision-making process.

**14. What is target leakage?**  
Target leakage occurs when information directly related to the target is included as an input feature.

In our project, we shouldn't include `Exam_Score` in `X` because we created `Pass` from `Exam_Score`.
 
**15. Why did you create the `Pass` column using the median?**  
The original dataset didn't have a Pass/Fail target. Using the median creates two classes based on the dataset's own score distribution.

---

