## 1. What is K-Means Clustering?

K-Means is an **unsupervised machine learning algorithm** that divides data into `K` groups or clusters based on similarity.

---

## 2. Why is K-Means called unsupervised learning?

Because K-Means does not require a **target/output label** during training.

It discovers patterns or groups directly from the input data.

---

## 3. How does K-Means work?

The basic process is:

```text
Choose K
   ↓
Initialize Centroids
   ↓
Assign Points to Nearest Centroid
   ↓
Recalculate Centroids
   ↓
Repeat
   ↓
Final Clusters
```

---

## 4. What is a centroid?

A **centroid** is the center point of a cluster.

K-Means continuously updates the centroid based on the observations assigned to that cluster.

---

## 5. What is K in K-Means?

`K` represents the **number of clusters** we want the algorithm to create.

For example:

```python
KMeans(n_clusters=5)
```

creates five clusters.

---

## 6. How did you choose K in your project?

I used the **Elbow Method**.

I calculated the inertia for different values of K and plotted them. The point where the reduction in inertia starts becoming less significant is considered the elbow.

For this project, I used **K = 5**.

---

## 7. What is the Elbow Method?

The Elbow Method is a technique for selecting a suitable number of clusters.

We plot:

```text
Number of Clusters
        vs
Inertia
```

The point where the curve forms an **elbow** can indicate a reasonable K.

---

## 8. What is inertia?

Inertia, also called **Within-Cluster Sum of Squares (WCSS)**, measures how close observations are to their assigned cluster centroids.

Lower inertia generally indicates more compact clusters.

However, inertia usually decreases as K increases, so we don't simply choose the largest K.

---

## 9. Why did you scale the data before K-Means?

K-Means relies on **distance calculations**.

If features have different scales, a feature with larger numerical values can have a greater influence on the distance.

Therefore, I used:

```python
StandardScaler()
```

before applying K-Means.

---

## 10. What distance does K-Means commonly use?

K-Means commonly uses **Euclidean distance** to determine the nearest centroid.

---

## 11. Why did you use the Mall Customers dataset?

It is a simple and practical dataset for demonstrating **customer segmentation**.

I used:

```text
Annual Income (k$)
Spending Score (1-100)
```

as the clustering features.

---

## 12. What is customer segmentation?

Customer segmentation means dividing customers into groups based on similar characteristics or behavior.

For example:

```text
High Income + High Spending
        ↓
Potential High-Value Segment
```

---

## 13. Why don't we use CustomerID for clustering?

`CustomerID` is simply an identifier.

It does not represent a meaningful customer characteristic, so including it could introduce irrelevant information into the clustering process.

---

## 14. What is the difference between K-Means and classification?

| K-Means                        | Classification          |
| ------------------------------ | ----------------------- |
| Unsupervised                   | Supervised              |
| No target required             | Target required         |
| Finds clusters                 | Predicts classes        |
| Example: Customer segmentation | Example: Spam detection |

---

## 15. What is the difference between K-Means and KNN?

Despite the similar names, they are completely different.

| K-Means              | KNN                                 |
| -------------------- | ----------------------------------- |
| Clustering algorithm | Classification/regression algorithm |
| Unsupervised         | Supervised                          |
| Creates clusters     | Makes predictions                   |
| Uses centroids       | Uses nearest observations           |

---

## 16. Does K-Means always find the globally optimal clusters?

No.

K-Means can converge to a **local optimum**, and the result can depend on the initial centroid positions.

That's one reason we use multiple initializations.

In our code:

```python
KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
```

---

## 17. What is `n_init=10`?

`n_init=10` means K-Means is run with **10 different centroid initializations**, and the best result is selected based on the objective/inertia.

---

## 18. What is `random_state=42`?

It makes the initialization reproducible so that we can obtain consistent results when running the code again.

---

## 19. Can K-Means work with categorical data?

Standard K-Means is designed primarily for **numerical data** because it relies on numerical distance calculations.

Categorical data generally requires appropriate encoding or a different clustering approach.

---

## 20. What are the limitations of K-Means?

Important limitations include:

* You need to choose `K`.
* It is sensitive to feature scaling.
* It can be affected by outliers.
* Results can depend on initialization.
* It works best when clusters are reasonably compact and separated.
* It is not naturally suited to arbitrary-shaped clusters.

---

Explain Your Day 19 Project



> In Day 19, I worked on customer segmentation using K-Means Clustering. I used the Mall Customers dataset and selected Annual Income and Spending Score as the clustering features. Since K-Means is a distance-based algorithm, I standardized the features using StandardScaler. I then used the Elbow Method to analyze different values of K using inertia and selected K = 5 for the main experiment. After applying K-Means, I assigned each customer a cluster label and analyzed the cluster sizes and centroids. I also visualized the customer segments and used PCA as an additional 2D visualization. Finally, I exported the clustered dataset for further analysis.

---

# 🔥 Rapid-Fire Questions

**1. K-Means is supervised or unsupervised?**


➡️ Unsupervised.

**2. What does K represent?**


➡️ Number of clusters.

**3. What is a centroid?**


➡️ The center of a cluster.

**4. What is inertia?**  


➡️ Sum of squared distances of observations from their assigned centroids.

**5. Which method did you use to select K?**


➡️ Elbow Method.

**6. What K did you use?**


➡️ K = 5.

**7. Why scale before K-Means?**


➡️ Because K-Means is distance-based.

**8. Which features did you use?**


➡️ Annual Income and Spending Score.

**9. Why not CustomerID?**  


➡️ It is an identifier, not a meaningful clustering feature.

**10. What type of learning is K-Means?**


➡️ Unsupervised learning.

**11. What is customer segmentation?**
 
➡️ Grouping customers with similar characteristics or behavior.

**12. What does `n_init=10` do?**


➡️ Tries multiple centroid initializations and selects the best result.

---

