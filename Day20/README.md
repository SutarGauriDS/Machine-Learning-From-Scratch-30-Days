#  Day 20 — Hierarchical Clustering

## Customer Segmentation using Agglomerative Clustering

Part of my **30 Days of Machine Learning Challenge**.

### 🎯 Objective

Use Hierarchical Clustering to group customers based on:

- Annual Income
- Spending Score

### 🧠 Concepts Covered

- Unsupervised Learning
- Hierarchical Clustering
- Agglomerative Clustering
- Feature Scaling
- Ward Linkage
- Dendrogram
- Customer Segmentation

### ⚙️ Workflow

```text
Dataset
   ↓
Feature Selection
   ↓
StandardScaler
   ↓
Ward Linkage
   ↓
Dendrogram
   ↓
Agglomerative Clustering
   ↓
Customer Segmentation
````

### 🛠️ Technologies

* Python
* Pandas
* Matplotlib
* Scikit-learn
* SciPy
* Jupyter Notebook

### 📊 Dataset

**Mall Customers Dataset**

Features used:

* `Annual Income (k$)`
* `Spending Score (1-100)`

### 📈 Results

* Created a hierarchical clustering dendrogram
* Segmented customers into **5 clusters**
* Analyzed cluster sizes and average customer characteristics
* Exported the clustered dataset

### 💡 Key Takeaway

Hierarchical Clustering helps discover **hidden customer groups without predefined labels**, making it useful for customer segmentation and behavioral analysis.

### 📂 Files

* `day20_hierarchical_clustering.py`
* `Day20_Hierarchical_Clustering_Complete.ipynb`
* `Mall_Customers.csv`
* `Mall_Customers_Hierarchical_Clustered.csv`
* `requirements.txt`
* `references.md`

### 🚀 Day 20/30 Completed

**Next:** Day 21 — DBSCAN Clustering

#MachineLearning #HierarchicalClustering #Clustering #Python #DataAnalytics #DataScience #30DaysOfMachineLearning
