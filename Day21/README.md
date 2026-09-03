#  Day 21 — DBSCAN Clustering

## Customer Segmentation using DBSCAN

Part of my **30 Days of Machine Learning Challenge**.

### 🎯 Objective

Use **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** to group customers based on:

- Annual Income
- Spending Score

Unlike K-Means, DBSCAN can detect clusters based on density and identify noise/outlier points.

### 🧠 Concepts Covered

- Unsupervised Learning
- DBSCAN
- Density-Based Clustering
- `eps`
- `min_samples`
- Noise / Outliers
- Feature Scaling
- Customer Segmentation

### ⚙️ Workflow

```text
Dataset
   ↓
Feature Selection
   ↓
StandardScaler
   ↓
DBSCAN
   ↓
Identify Clusters & Noise
   ↓
Visualization
   ↓
Cluster Analysis
````

### 📊 Dataset

**Mall Customers Dataset**

Features used:

* `Annual Income (k$)`
* `Spending Score (1-100)`

### 🛠️ Technologies

* Python
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

### 📈 Results

* Applied DBSCAN clustering
* Identified customer groups based on density
* Detected potential noise/outlier points
* Analyzed cluster sizes and characteristics
* Exported the clustered dataset

### 💡 Key Takeaway

DBSCAN is useful when we want to discover clusters **without specifying the number of clusters beforehand** and when detecting **noise or outliers** is important.

### 📂 Files

* `day21_dbscan.py`
* `Day21_DBSCAN_Clustering_Complete.ipynb`
* `Mall_Customers.csv`
* `Mall_Customers_DBSCAN_Clustered.csv`
* `requirements.txt`
* `references.md`

### 🚀 Day 21/30 Completed

**Next:** Day 22 — Hyperparameter Tuning

#MachineLearning #DBSCAN #Clustering #CustomerSegmentation #Python #DataScience #DataAnalytics #30DaysOfMachineLearning

```
```
