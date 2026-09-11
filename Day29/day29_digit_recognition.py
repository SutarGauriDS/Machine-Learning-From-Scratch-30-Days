# ============================================================
# DAY 29/30 — HANDWRITTEN DIGIT RECOGNITION
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("DAY 29 — HANDWRITTEN DIGIT RECOGNITION")
print("=" * 60)

print("\nLoading Digits dataset...")

digits = load_digits()

X = pd.DataFrame(
    digits.data,
    columns=[f"pixel_{i}" for i in range(digits.data.shape[1])]
)

y = pd.Series(digits.target, name="digit")

print("\nDataset loaded successfully!")

print(f"Number of samples : {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes : {len(np.unique(y))}")

print("\nClasses:")
print(sorted(y.unique()))


# ============================================================
# 2. DATA UNDERSTANDING
# ============================================================

print("\n" + "=" * 60)
print("DATA UNDERSTANDING")
print("=" * 60)

print("\nFirst 5 rows:")
print(X.head())

print("\nTarget distribution:")
print(y.value_counts().sort_index())

print("\nMissing values:")
print(X.isnull().sum().sum())

print("\nDuplicate rows:")
print(X.duplicated().sum())


# ============================================================
# 3. VISUALIZE SAMPLE DIGITS
# ============================================================

print("\nDisplaying sample handwritten digits...")

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for ax, image, label in zip(
    axes.ravel(),
    digits.images[:10],
    digits.target[:10]
):
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Digit: {label}")
    ax.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")


# ============================================================
# 5. CREATE MODELS
# ============================================================

print("\n" + "=" * 60)
print("CREATING MODELS")
print("=" * 60)

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                random_state=42
            )
        )
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            )
        )
    ])
}


# ============================================================
# 6. TRAIN AND EVALUATE MODELS
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

results = []

trained_models = {}

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[model_name] = model

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


# ============================================================
# 7. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)

print("\n")
print(results_df.to_string(index=False))

results_df.to_csv(
    "Day29_Model_Results.csv",
    index=False
)

print("\nResults saved to:")
print("Day29_Model_Results.csv")


# ============================================================
# 8. CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("CROSS-VALIDATION")
print("=" * 60)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for model_name, model in models.items():

    print(f"\nRunning CV for {model_name}...")

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    cv_results.append({
        "Model": model_name,
        "CV Mean Accuracy": scores.mean(),
        "CV Std": scores.std()
    })

    print(f"Mean Accuracy: {scores.mean():.4f}")
    print(f"Std          : {scores.std():.4f}")


cv_df = pd.DataFrame(cv_results)

print("\nCross-validation results:")
print(cv_df.to_string(index=False))


# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"\nBest model: {best_model_name}")

best_predictions = best_model.predict(X_test)


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_test,
    best_predictions
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=digits.target_names
)

disp.plot()

plt.title(
    f"Confusion Matrix — {best_model_name}"
)

plt.tight_layout()
plt.show()


# ============================================================
# 12. SAVE BEST MODEL
# ============================================================

MODEL_FILE = "digit_model.pkl"

joblib.dump(
    best_model,
    MODEL_FILE
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(f"\nSaved model: {MODEL_FILE}")


# ============================================================
# 13. LOAD SAVED MODEL
# ============================================================

print("\nLoading saved model...")

loaded_model = joblib.load(
    MODEL_FILE
)

print("Model loaded successfully!")


# ============================================================
# 14. TEST NEW SAMPLE
# ============================================================

sample_index = 0

sample = X_test.iloc[[sample_index]]

actual_digit = y_test.iloc[sample_index]

prediction = loaded_model.predict(sample)[0]

print("\n" + "=" * 60)
print("SAMPLE PREDICTION")
print("=" * 60)

print(f"\nActual digit    : {actual_digit}")
print(f"Predicted digit : {prediction}")


# ============================================================
# 15. PREDICTION PROBABILITIES
# ============================================================

if hasattr(loaded_model, "predict_proba"):

    probabilities = loaded_model.predict_proba(sample)[0]

    probability_df = pd.DataFrame({
        "Digit": range(10),
        "Probability": probabilities
    })

    probability_df = probability_df.sort_values(
        by="Probability",
        ascending=False
    )

    print("\nPrediction probabilities:")

    print(
        probability_df.to_string(
            index=False
        )
    )


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("DAY 29 PROJECT COMPLETED SUCCESSFULLY! 🚀")
print("=" * 60)

print("\nFiles created:")
print("1. Day29_Model_Results.csv")
print("2. digit_model.pkl")

print("\nBest Model:")
print(best_model_name)

print("\nNext step:")
print("Create the Day 29 Jupyter Notebook.")