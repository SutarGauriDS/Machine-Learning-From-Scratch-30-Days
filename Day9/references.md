# 📚 Day 9 – References

## Official Documentation

- Scikit-Learn Documentation
- KNeighborsClassifier Documentation
- StandardScaler Documentation
- Train-Test Split Documentation
- Scikit-Learn Classification Metrics
- Pandas Documentation
- Matplotlib Documentation
- Seaborn Documentation

## 🔍 Topics Referenced

- K-Nearest Neighbors (KNN)
- KNN Classification
- Distance-Based Algorithms
- Feature Scaling
- StandardScaler
- Train-Test Split
- Stratified Sampling
- Choosing the Value of K
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Overfitting
- Underfitting

## 📊 Dataset

**Student Performance Factors**

The same dataset used in previous days was used to
practice KNN classification.

### Target

`Performance`

### Target Definition

```text
Exam_Score >= 75 → 1 (High Performance)
Exam_Score < 75  → 0 (Low Performance)

Selected Features
Hours_Studied
Attendance
Previous_Scores
Sleep_Hours
Tutoring_Sessions
```
💡 Learning Note

KNN is a distance-based algorithm, so feature scaling is
important. StandardScaler was used to bring the numerical
features to a comparable scale.

Different K values were tested to understand how the choice
of neighbors affects model performance.
