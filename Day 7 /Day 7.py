"""
Day 7 - Linear Regression
30 Days of Machine Learning Challenge
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("StudentPerformanceFactors.csv")

print("First 5 Records:")
print(df.head())


# ==========================================
# 2. Explore Dataset
# ==========================================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# ==========================================
# 3. Select Features and Target
# ==========================================

features = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Sleep_Hours",
    "Tutoring_Sessions"
]

X = df[features]

y = df["Exam_Score"]


# ==========================================
# 4. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# ==========================================
# 5. Create Model
# ==========================================

model = LinearRegression()


# ==========================================
# 6. Train Model
# ==========================================

model.fit(X_train, y_train)

print("\nModel Training Completed!")


# ==========================================
# 7. Predictions
# ==========================================

y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])


# ==========================================
# 8. Actual vs Predicted
# ==========================================

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(comparison.head(10))


# ==========================================
# 9. Evaluation Metrics
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\nModel Evaluation")
print("-------------------------")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)


# ==========================================
# 10. Actual vs Predicted Plot
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")

plt.title(
    "Actual vs Predicted Exam Scores"
)

plt.tight_layout()
plt.show()


# ==========================================
# 11. Model Coefficients
# ==========================================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nModel Coefficients:")
print(coefficients)


# ==========================================
# 12. Intercept
# ==========================================

print("\nIntercept:")
print(model.intercept_)


# ==========================================
# 13. Predict New Student
# ==========================================

new_student = pd.DataFrame({
    "Hours_Studied": [6],
    "Attendance": [90],
    "Previous_Scores": [75],
    "Sleep_Hours": [7],
    "Tutoring_Sessions": [2]
})

prediction = model.predict(new_student)

print("\nNew Student Prediction:")
print(prediction[0])


