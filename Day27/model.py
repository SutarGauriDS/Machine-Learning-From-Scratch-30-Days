import pandas as pd
import joblib

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading Wine dataset...")

wine = load_wine()

X = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

y = pd.Series(
    wine.target,
    name="target"
)

print("Dataset loaded successfully!")
print("Shape:", X.shape)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# CREATE PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "model",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model trained successfully!")


# ============================================================
# EVALUATE
# ============================================================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nModel Accuracy: {accuracy:.4f}")


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH = "wine_model.pkl"

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\nModel saved successfully!")
print(f"File: {MODEL_PATH}")