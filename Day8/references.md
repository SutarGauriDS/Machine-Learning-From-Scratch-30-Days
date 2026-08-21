# 📚 Day 8 – References

## Official Documentation

- Scikit-Learn Documentation
- Logistic Regression Documentation
- Scikit-Learn Classification Metrics
- Pandas Documentation
- Matplotlib Documentation
- Seaborn Documentation

## 🔍 Topics Referenced

- Classification
- Logistic Regression
- Train-Test Split
- Stratified Sampling
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report
- Prediction Probability
- Model Coefficients

## 📊 Dataset

**Student Performance Factors**

The dataset is used to practice classification by creating a
`Performance` target from the original `Exam_Score` variable.

### Target Creation

```python
df["Performance"] = (
    df["Exam_Score"] >= 75
).astype(int)

```
Where:

1 → High Performance

0 → Low Performance

💡 Learning Note

The concepts were practiced using a real-world student performance
dataset to understand how Logistic Regression can be applied to
a binary classification problem.
