# 🎯 Day 4 – Pandas Interview Questions & Answers

### 1. What is Pandas?

**Answer:** Pandas is a Python library used for **data manipulation, cleaning, analysis, and preprocessing**. It provides useful data structures such as Series and DataFrame.

---

### 2. What is a DataFrame?

**Answer:** A DataFrame is a **two-dimensional labeled data structure** consisting of rows and columns, similar to a table in a database or Excel.

```python
df = pd.DataFrame(data)
```

---

### 3. What is a Series?

**Answer:** A Series is a **one-dimensional labeled data structure** in Pandas.

```python
marks = pd.Series([70, 80, 90])
```

---

### 4. Difference between Series and DataFrame?

| Series                     | DataFrame                 |
| -------------------------- | ------------------------- |
| One-dimensional            | Two-dimensional           |
| Single column of data      | Multiple rows and columns |
| Similar to a single column | Similar to a table        |

---

### 5. How do you create a DataFrame?

```python
import pandas as pd

data = {
    "Name": ["A", "B", "C"],
    "Marks": [80, 90, 85]
}

df = pd.DataFrame(data)
```

---

### 6. How do you view the first 5 rows?

```python
df.head()
```

**Answer:** `head()` displays the first 5 rows by default.

---

### 7. How do you view the last 5 rows?

```python
df.tail()
```

---

### 8. How do you check the number of rows and columns?

```python
df.shape
```

**Answer:** It returns a tuple:

```text
(rows, columns)
```

---

### 9. How do you check column names?

```python
df.columns
```

---

### 10. What does `df.info()` do?

**Answer:** `info()` provides information about:

* Number of rows
* Column names
* Non-null values
* Data types
* Memory usage

```python
df.info()
```

---

### 11. What does `df.describe()` do?

**Answer:** It provides statistical summaries of numerical columns, such as:

* Count
* Mean
* Standard deviation
* Minimum
* Maximum
* Quartiles

```python
df.describe()
```

---

### 12. How do you select a single column?

```python
df["Marks"]
```

---

### 13. How do you select multiple columns?

```python
df[["Name", "Marks"]]
```

---

### 14. What is the difference between `loc` and `iloc`?

**Answer:**

* `loc` → selects data using **labels**
* `iloc` → selects data using **integer positions**

```python
df.loc[0, "Name"]
```

```python
df.iloc[0, 1]
```

---

### 15. How do you filter data?

For example, students scoring more than 80:

```python
df[df["Marks"] > 80]
```

---

### 16. How do you sort a DataFrame?

```python
df.sort_values("Marks")
```

For descending order:

```python
df.sort_values("Marks", ascending=False)
```

---

### 17. How do you check missing values?

```python
df.isnull().sum()
```

---

### 18. How do you handle missing values?

One common method is filling missing values with the mean:

```python
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())
```

---

### 19. What is `groupby()`?

**Answer:** `groupby()` groups data based on one or more columns and allows us to perform calculations on each group.

Example:

```python
df.groupby("City")["Marks"].mean()
```

This calculates the **average marks for each city**.

---

### 20. How do you create a new column?

```python
df["Passed"] = df["Marks"] >= 40
```

---

## ⭐ Important Practical Questions

### 21. How would you find the highest marks?

```python
df["Marks"].max()
```

### 22. How would you find the average marks?

```python
df["Marks"].mean()
```

### 23. How would you count unique cities?

```python
df["City"].nunique()
```

### 24. How would you find unique values?

```python
df["City"].unique()
```

### 25. How would you count each category?

```python
df["City"].value_counts()
```
