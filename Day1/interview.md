# 🎯 Day 2 Interview Questions
**1. What is Data Preprocessing?**  
Data preprocessing is the process of cleaning, transforming, and preparing raw data before using it for analysis or Machine Learning.

**2. Why is data preprocessing important?**  
Because real-world data can contain missing values, duplicates, incorrect data types, and categorical values. Cleaning the data improves the quality of analysis and model performance.

**3. What are common steps in data preprocessing?**

* Handling missing values
* Removing duplicates
* Correcting data types
* Encoding categorical variables
* Feature selection
* Feature scaling
* Splitting data into training and testing sets

**4. What are missing values?**  
Missing values are data points where no value is available for a particular observation.

```python 
df.isnull().sum()
```

**5. How can you handle missing values?**  

* Remove rows/columns
* Replace with mean
* Replace with median
* Replace with mode
* Use an appropriate estimated value

**6. When would you use mean vs median to fill missing values?**  
Mean is commonly used when the data doesn't have significant outliers. Median is often safer when the data is skewed or contains outliers.

**7. How do you find duplicate records in Pandas?**  

```python
df.duplicated().sum()
```
 
**8. How do you remove duplicate records?**  

```python
df = df.drop_duplicates()
```

**9. What is categorical data?**  
Data that represents categories or labels rather than numerical measurements.

Example:

```text
City
Pune
Mumbai
Nashik
```

**10. Why do we encode categorical data?**  
Many Machine Learning algorithms require numerical input, so categorical values need to be converted into numerical representations.

---

### ⭐ Important Interview Questions  

**11. What is One-Hot Encoding?**   
One-hot encoding converts categories into separate binary columns.

Example:

```text
City
Pune
Mumbai
```

can become:

```text
City_Mumbai  City_Pune
     1           0
     0           1
```

In Pandas:

```python
pd.get_dummies(df, columns=["City"])
```

**12. What are features and target variables?**  

* **Features (`X`)** → Input variables used to make a prediction.
* **Target (`y`)** → Output variable that we want to predict.

Example:

```python
X = df.drop("Marks", axis=1)
y = df["Marks"]
```

**13. What is train-test split?**  
It divides the dataset into training data for learning and testing data for evaluating the model.

```python
train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

**14. Why do we use `random_state`?**  
It ensures that the same random split is produced every time the code runs, making results reproducible.

**15. What is data leakage?**    
Data leakage occurs when information from outside the training data, especially information from the test set or future information, unintentionally influences model training.

**16. What is the difference between data cleaning and data preprocessing?**  
Data cleaning focuses mainly on fixing problems such as missing values, duplicates, and incorrect data. Data preprocessing is broader and can include cleaning, encoding, scaling, feature preparation, and splitting the data.

### 💻 Practical Interview Questions
  
**17. How would you find all missing values in a DataFrame?**  

```python
df.isnull().sum()
```

**18. How would you check the data types?**  

```python
df.dtypes
```

**19. How would you get basic information about a DataFrame?**  

```python
df.info()
```
 
**20. How would you get statistical information?**  

```python
df.describe()
```

**21. How would you remove a column?**  

```python
df.drop("Column_Name", axis=1)
```

**22. How would you convert a categorical column into numerical columns?**  

```python
pd.get_dummies(df, columns=["City"], dtype=int)
```

### 🏆 Top 5 to Remember 

For interviews, make sure you can confidently explain:

1. **What is data preprocessing?**
2. **How do you handle missing values?**
3. **What is One-Hot Encoding?**
4. **What are features and target?**
5. **Why do we use train-test split?**
