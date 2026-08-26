# 📚 References

## 1. Scikit-learn — Support Vector Machines

The main reference for understanding SVM, including classification with support vectors and different kernels.

[Scikit-learn Support Vector Machines Documentation](https://scikit-learn.org/stable/modules/svm.html?utm_source=chatgpt.com)

---

## 2. Scikit-learn — SVC

The project uses `SVC` for both Linear and RBF classification.

The important parameters used in this project are:

* `kernel="linear"` — linear decision boundary
* `kernel="rbf"` — radial basis function kernel for non-linear classification
* `C=1` — regularization parameter
* `gamma="scale"` — RBF kernel parameter

[Scikit-learn SVC Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?utm_source=chatgpt.com)

---

## 3. Scikit-learn — Breast Cancer Dataset

The project uses `load_breast_cancer()`.

According to the official documentation, the dataset contains **569 samples, 30 features, and 2 target classes**. The class distribution is 212 malignant and 357 benign samples. ([Scikit-learn][1])

[Breast Cancer Dataset — Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html?utm_source=chatgpt.com)

The dataset is a copy of the **UCI Breast Cancer Wisconsin (Diagnostic)** dataset. ([Scikit-learn][1])

---

## 4. Scikit-learn — StandardScaler

`StandardScaler` is used before SVM because the features have different scales.

It standardizes each feature by removing its mean and scaling according to its standard deviation. This is particularly relevant for SVM because feature scale can affect the model's objective and kernel calculations. ([Scikit-learn][2])

[StandardScaler Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html?utm_source=chatgpt.com)

---

## 5. Scikit-learn — Model Evaluation

This project uses:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

for evaluating the classification model.

[Scikit-learn Model Evaluation Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html?utm_source=chatgpt.com)

---

## 6. Scikit-learn — SVM Examples

Scikit-learn also provides examples covering different SVM kernels, decision boundaries, RBF SVM, and support vectors. These are useful for understanding the concepts demonstrated in this project. ([Scikit-learn][3])

[Scikit-learn SVM Examples](https://scikit-learn.org/stable/auto_examples/index.html?utm_source=chatgpt.com)

---

## 📖 Reference Note

This project is an educational implementation created as part of my **30 Days Machine Learning Challenge — Day 13**.

The implementation uses the official Scikit-learn APIs and documentation as the primary technical reference. The dataset itself originates from the UCI Breast Cancer Wisconsin (Diagnostic) dataset and is provided through Scikit-learn's dataset loader. ([Scikit-learn][1])

