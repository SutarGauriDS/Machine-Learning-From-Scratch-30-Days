##  Day 15 — Feature Engineering Interview Questions

### 1. What is Feature Engineering?

Feature Engineering is the process of creating, transforming, or modifying features from raw data to provide useful information to a machine learning model.

### 2. Why is Feature Engineering important?

Good features can help a model identify important patterns, improve performance, and represent the underlying problem more effectively.

### 3. What features did you create in your Titanic project?

I created:

* `FamilySize`
* `IsAlone`
* `FarePerPerson`
* `Title`
* `AgeGroup`

### 4. How did you calculate `FamilySize`?

```python
df["FamilySize"] = df["sibsp"] + df["parch"] + 1
```

It represents the passenger plus their siblings/spouse and parents/children travelling with them.

### 5. What is `IsAlone`?

It is a binary feature indicating whether a passenger was travelling alone.

```python
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
```

`1` means alone and `0` means travelling with family.

### 6. Why did you create `FarePerPerson`?

Instead of using only total fare, I calculated the fare relative to family size:

```python
df["FarePerPerson"] = df["fare"] / df["FamilySize"]
```

This provides additional information about the fare paid per person.

### 7. Why did you create `AgeGroup`?

Age is a continuous variable, so I converted it into meaningful groups such as **Child, Teen, Young Adult, Adult, and Senior**.

### 8. What is the difference between Feature Engineering and Feature Selection?

**Feature Engineering:** Creates or transforms features.

**Feature Selection:** Chooses the most useful features from the available features.

### 9. How did you handle missing values?

For numerical features, I used **median imputation**.

For categorical features, I used **most-frequent-value imputation**.

```python
SimpleImputer(strategy="median")
```

and

```python
SimpleImputer(strategy="most_frequent")
```

### 10. Why did you use OneHotEncoder?

Categorical variables such as `sex`, `embarked`, and `AgeGroup` cannot be directly used by Logistic Regression, so I converted them into numerical representations using `OneHotEncoder`.

### 11. Why did you use StandardScaler?

I used `StandardScaler` to standardize numerical features so they are on comparable scales.

### 12. Why did you use a Pipeline?

A Pipeline combines preprocessing and the machine learning model into a single workflow and helps prevent data leakage during training.

### 13. What is data leakage?

Data leakage occurs when information that should not be available during model training is unintentionally used by the model.

### 14. Why did you split the data before preprocessing?

The model should learn preprocessing parameters only from the training data. The test data must remain unseen until final evaluation.

### 15. Why did you use Logistic Regression?

The Titanic `survived` variable is a **binary classification target**, making Logistic Regression an appropriate baseline classification algorithm.

---
### "Explain your Day 15 project."

**Answer:**

> "In Day 15, I worked on Feature Engineering using the Titanic dataset. I created meaningful features such as FamilySize, IsAlone, FarePerPerson, Title, and AgeGroup from the existing data. I then handled missing values, encoded categorical features using OneHotEncoder, and standardized numerical features using StandardScaler. I combined the preprocessing steps with Logistic Regression using a Pipeline and evaluated the model using accuracy, classification report, and confusion matrix."


**Raw Data → Create Features → Handle Missing Values → Encode → Scale → Model → Evaluate**
