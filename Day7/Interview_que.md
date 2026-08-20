# 🎯 Day 7 – Linear Regression Interview Questions

## 1. What is Linear Regression?

Linear Regression is a supervised Machine Learning algorithm
used to predict a continuous numerical target.

---

## 2. What is Multiple Linear Regression?

Multiple Linear Regression uses multiple independent variables
to predict one continuous target.

Example:

Hours_Studied
+ Attendance
+ Previous_Scores
+ Sleep_Hours
+ Tutoring_Sessions

→ Exam_Score

---

## 3. What is the equation of Linear Regression?

For multiple features:

y = b0 + b1X1 + b2X2 + ... + bnXn

Where:

y = predicted output
b0 = intercept
b1...bn = coefficients
X1...Xn = features

---

## 4. What is MAE?

Mean Absolute Error measures the average absolute difference
between actual and predicted values.

Lower MAE is generally better.

---

## 5. What is MSE?

Mean Squared Error calculates the average squared difference
between actual and predicted values.

Large errors are penalized more heavily.

---

## 6. What is RMSE?

Root Mean Squared Error is the square root of MSE.

It is expressed in the same unit as the target variable.

---

## 7. What is R² Score?

R² measures how much of the variation in the target is
explained by the model.

Higher values generally indicate better fit,
but the metric should always be interpreted in context.

---

## 8. MAE vs MSE vs RMSE?

### MAE
Easy to interpret and less sensitive to large errors.

### MSE
Penalizes large errors more strongly.

### RMSE
Same unit as the target and more sensitive to large errors.

---

## 9. What are coefficients?

Coefficients represent the estimated change in the predicted
target for a one-unit increase in a feature, while other
features are held constant.

---

## 10. What is the intercept?

The intercept is the predicted value of the target when all
input features are zero.

---

## ⭐ Important Interview Questions

Focus on:

- Linear Regression
- Simple vs Multiple Linear Regression
- MAE vs MSE vs RMSE
- R² Score
- Coefficients
- Intercept
- Overfitting
- Underfitting
- Regression vs Classification
