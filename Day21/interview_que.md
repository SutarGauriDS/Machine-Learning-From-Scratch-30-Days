##  Day 21 — DBSCAN Interview Questions

### 1. What is DBSCAN?

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** is an unsupervised clustering algorithm that groups data points based on their density.

A major advantage is that it can identify **noise/outliers** and does not require the number of clusters to be specified beforehand.

---

### 2. What are the main parameters of DBSCAN?

There are two important parameters:

* **`eps`** — maximum distance to consider another point as a neighbor.
* **`min_samples`** — minimum number of points required to form a dense region.

In our project, we started with:

```python
DBSCAN(
    eps=0.35,
    min_samples=5
)
```

---

### 3. What is `eps`?

`eps` defines the **neighborhood radius** around each data point.

If another point lies within this distance, it can be considered a neighbor.

Too small → many points may become noise.

Too large → different clusters may get merged.

---

### 4. What is `min_samples`?

`min_samples` specifies the minimum number of points required within the `eps` neighborhood for a point to be considered a **core point**.

---

### 5. What are core, border, and noise points?

DBSCAN classifies points into three categories:

**Core Point:** Has enough neighboring points within `eps`.

**Border Point:** Doesn't have enough neighbors itself but is close to a core point.

**Noise Point:** Doesn't belong to any cluster.

In Scikit-learn, noise points are represented by:

```python
-1
```

---

### 6. Why did you use StandardScaler?

DBSCAN relies on **distance between points**.

Our features have different ranges, so scaling prevents one feature from disproportionately affecting the distance calculation.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### 7. Does DBSCAN require the number of clusters beforehand?

**No.**

This is one of the main differences from K-Means.

DBSCAN determines clusters based on the density of the data.

---

### 8. What is the biggest advantage of DBSCAN?

The important advantages are:

* No need to specify the number of clusters
* Can detect noise/outliers
* Can identify clusters with non-spherical shapes
* Useful for density-based datasets

---

### 9. What are the limitations of DBSCAN?

Some limitations are:

* Sensitive to `eps` and `min_samples`
* Can struggle when clusters have very different densities
* Distance calculations become challenging with high-dimensional data

---

### 10. DBSCAN vs K-Means?

| K-Means                          | DBSCAN                              |
| -------------------------------- | ----------------------------------- |
| Requires number of clusters      | Does not require number of clusters |
| Centroid-based                   | Density-based                       |
| Sensitive to outliers            | Can identify noise                  |
| Works best with compact clusters | Can find irregular-shaped clusters  |
| Uses centroids                   | Uses density                        |

---

### 11. Why did you use DBSCAN for the Mall Customers dataset?

> “I used DBSCAN to explore customer segmentation from a different perspective. Instead of specifying the number of clusters like K-Means, DBSCAN groups customers based on density and can also identify potential noise points.”

---

### 12. Explain your Day 21 project.

> “In Day 21, I implemented DBSCAN for customer segmentation using the Mall Customers dataset. I selected Annual Income and Spending Score, standardized the features using StandardScaler, and applied DBSCAN using `eps` and `min_samples`. I then analyzed the generated clusters and identified noise points. Finally, I visualized the customer segments and exported the clustered dataset.”

---

