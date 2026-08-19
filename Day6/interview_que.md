# 🎯 Day 6 – Machine Learning Workflow
# Interview Questions & Answers

## 1. What is a Machine Learning workflow?

A Machine Learning workflow is a sequence of steps used to build an ML model.

Typical workflow:

Data Collection
→ Data Exploration
→ Data Preprocessing
→ Feature Selection
→ Train-Test Split
→ Model Training
→ Prediction
→ Evaluation

---

## 2. What are features in Machine Learning?

Features are the input variables used by a model to make predictions.

In this project:

- Hours_Studied
- Attendance
- Previous_Scores
- Sleep_Hours
- Tutoring_Sessions

are selected features.

---

## 3. What is a target variable?

The target is the output that the Machine Learning model tries to predict.

In this project:

`Exam_Score`

is the target variable.

---

## 4. What is Train-Test Split?

Train-Test Split divides a dataset into two parts:

- Training data → used to train the model
- Testing data → used to evaluate the model on unseen data


## 5. Why do we split data?

If we train and test a model on the same data, we cannot properly determine how well it performs on new data.

The test set helps measure the model's ability to generalize.

---

## 6. What does `test_size=0.2` mean?

It means 20% of the dataset is reserved for testing and approximately 80% is used for training.

For this dataset:

* Total records = 6,607
* Training records = 5,285
* Testing records = 1,322

---

## 7. What is `random_state`?

`random_state` controls the random splitting process.

Using the same value gives the same train-test split every time.

```python
random_state=42
```

The number `42` is not special. It is simply a commonly used value.

---

## 8. What is data leakage?

Data leakage occurs when information from outside the training data accidentally influences the model during training.

This can make model performance look unrealistically good.

### Example:

Using information from the test set while preprocessing the training data.

---

## 9. What is overfitting?

Overfitting occurs when a model learns the training data too closely, including noise, and performs poorly on unseen data.

A model should generalize well to new data.

---

## 10. What is correlation?

Correlation measures the strength and direction of a relationship between two numerical variables.

Its value generally ranges from:

`-1 to +1`

* `+1` → Strong positive relationship
* `0` → Little or no linear relationship
* `-1` → Strong negative relationship

---

## 11. What was the strongest selected feature correlation in this project?

Among the selected features, **Attendance** showed the strongest positive correlation with `Exam_Score`, around **0.58**.

However, correlation does not prove that one variable causes another.

---

## 12. Why did you use a correlation heatmap?

A correlation heatmap makes relationships between numerical variables easier to identify visually.

It can help during:

* Exploratory Data Analysis
* Feature selection
* Understanding relationships
* Detecting highly correlated features

---

## 13. Why is feature selection important?

Feature selection helps identify useful input variables and can:

* Reduce unnecessary data
* Simplify the model
* Improve interpretability
* Reduce computational requirements

---

## 14. What is the difference between X and y?

```python
X = df[features]
y = df["Exam_Score"]
```

`X` contains the **input features**.

`y` contains the **target/output**.


## 15. Why should preprocessing be handled carefully before model training?

Because preprocessing can accidentally introduce information from the test set into the training process.

For example, when calculating statistics such as mean or standard deviation, the transformation should generally be fitted using training data and then applied to the test data.

# 🎯 Project-Based Interview Question

## Explain your Day 6 project.
I worked with a Student Performance Factors dataset containing 6,607 records and 20 columns. I first explored the dataset using Pandas, checked its structure,
missing values, duplicates, and statistical summary. I selected relevant numerical features such as Hours_Studied, Attendance, Previous_Scores, Sleep_Hours, 
and Tutoring_Sessions, with Exam_Score as the target. I then performed correlation analysis and created visualizations to understand relationships between the
features and target. Finally, I used Scikit-Learn's train_test_split to divide the data into 80% training and 20% testing data, preparing the dataset for model 
training.
