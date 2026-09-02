### 1. What is Hierarchical Clustering?

Hierarchical Clustering is an **unsupervised machine learning algorithm** that groups similar data points into a hierarchy of clusters.

It is commonly visualized using a **dendrogram**.

---

### 2. What is Agglomerative Clustering?

Agglomerative Clustering is a **bottom-up approach**.

Initially, every data point is treated as an individual cluster. Similar clusters are then repeatedly merged until the required number of clusters is obtained.

```text
Individual Points
       ↓
Merge Similar Points
       ↓
Merge Clusters
       ↓
Final Clusters
```

---

### 3. What is a Dendrogram?

A dendrogram is a **tree-like diagram** that shows how clusters are progressively merged.

It helps us decide the appropriate number of clusters by observing the **distance between merges**.

---

### 4. What is Linkage?

Linkage determines **how the distance between two clusters is calculated**.

Common linkage methods are:

* Single
* Complete
* Average
* Ward

In our project, we used **Ward linkage**.

---

### 5. Why did you use Ward linkage?

Ward linkage tries to minimize the **increase in within-cluster variance** when clusters are merged.

It generally produces relatively compact clusters.

---

### 6. Why did you use StandardScaler?

Our features have different scales.

For example:

* Annual Income → values around tens/hundreds
* Spending Score → values from 1–100

Scaling puts the features on a comparable scale so that distance calculations are not dominated by one feature.

---

### 7. Why is Hierarchical Clustering called unsupervised learning?

Because there is **no predefined target variable or class label**.

The algorithm discovers groups based on similarities within the data.

---

### 8. How did you choose 5 clusters?

I first examined the **dendrogram** to understand the hierarchical structure and possible separation between clusters.

For this project, I used **5 clusters**, maintaining consistency with the previous K-Means customer segmentation project.

---

### 9. What is the difference between K-Means and Hierarchical Clustering?

| K-Means                                | Hierarchical                          |
| -------------------------------------- | ------------------------------------- |
| Centroid-based                         | Hierarchy-based                       |
| Requires K beforehand                  | Dendrogram helps inspect clusters     |
| Uses iterative optimization            | Builds a hierarchy                    |
| Produces final clusters                | Shows merging structure               |
| Generally efficient for large datasets | Can be more computationally expensive |

---

### 10. What is the difference between Agglomerative and Divisive clustering?

**Agglomerative:**
Starts with individual data points and **merges** them.

**Divisive:**
Starts with one large cluster and **splits** it.

So:

```text
Agglomerative → Bottom-Up
Divisive      → Top-Down
```

---

### 11. What features did you use in your project?

I used:

```text
Annual Income (k$)
Spending Score (1-100)
```

These features were selected because they are useful for understanding customer spending behavior.

---

### 12. What is the real-world application of this project?

This approach can be used for:

* Customer segmentation
* Marketing campaigns
* Personalized offers
* Market analysis
* Recommendation systems
* Customer behavior analysis

---
Explain your Day 20 project:

> I worked on customer segmentation using Agglomerative Hierarchical Clustering. I selected Annual Income and Spending Score as features, standardized them using StandardScaler, and used Ward linkage to create a dendrogram. Based on the clustering structure, I used 5 clusters and analyzed the characteristics of each customer segment. This helped me understand how unsupervised learning can identify customer groups without predefined labels.
