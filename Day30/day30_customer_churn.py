# ============================================================
# DAY 30/30 — CUSTOMER CHURN PREDICTION
# FINAL MACHINE LEARNING PROJECT
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

MODEL_FILE = "churn_model.pkl"

RESULTS_FILE = "Day30_Model_Results.csv"

FEATURE_IMPORTANCE_FILE = "Day30_Feature_Importance.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("DAY 30/30 — CUSTOMER CHURN PREDICTION")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 2. DATA UNDERSTANDING
# ============================================================

print("\n" + "=" * 70)
print("DATA UNDERSTANDING")
print("=" * 70)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 3. DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Replace empty strings with NaN
df = df.replace(r"^\s*$", np.nan, regex=True)

# Remove duplicates
df = df.drop_duplicates()

# Remove customer ID because it is not useful for prediction
df = df.drop(columns=["customerID"])

print("\nData cleaning completed.")

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================================
# 4. TARGET ENCODING
# ============================================================

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

print("\nTarget distribution:")
print(df["Churn"].value_counts())

print("\nChurn percentage:")

print(
    (df["Churn"].value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Churn distribution

plt.figure(figsize=(7, 5))

df["Churn"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.xticks(
    ticks=[0, 1],
    labels=["No Churn", "Churn"],
    rotation=0
)

plt.tight_layout()
plt.show()


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

# Tenure groups

df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=[
        "0-1 Year",
        "1-2 Years",
        "2-4 Years",
        "4+ Years"
    ]
)

# Monthly charge category

df["monthly_charge_category"] = pd.cut(
    df["MonthlyCharges"],
    bins=[-np.inf, 40, 70, 100, np.inf],
    labels=[
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)

# Total service count

service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

existing_service_columns = [
    col for col in service_columns
    if col in df.columns
]

df["service_count"] = (
    df[existing_service_columns]
    .apply(
        lambda row: sum(
            str(value).lower() in ["yes", "yes"]
            for value in row
        ),
        axis=1
    )
)

print("\nNew features created:")

print(
    "- tenure_group"
)

print(
    "- monthly_charge_category"
)

print(
    "- service_count"
)


# ============================================================
# 7. SPLIT FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["Churn"])

y = df["Churn"]


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")


# ============================================================
# 9. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 10. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ============================================================
# 11. CREATE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 12. TRAIN MODELS
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

results = {}

trained_models = {}


for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }

    trained_models[
        model_name
    ] = pipeline

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )


# ============================================================
# 13. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(
    results
).T

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)

print("\n")

print(
    results_df.round(4).to_string()
)

results_df.to_csv(
    RESULTS_FILE
)

print(
    f"\nResults saved as: {RESULTS_FILE}"
)


# ============================================================
# 14. CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []


for model_name, model in models.items():

    print(
        f"\nCross-validating {model_name}..."
    )

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    cv_results.append({
        "Model": model_name,
        "Mean ROC-AUC": scores.mean(),
        "Std ROC-AUC": scores.std()
    })

    print(
        f"Mean ROC-AUC: {scores.mean():.4f}"
    )

    print(
        f"Std ROC-AUC : {scores.std():.4f}"
    )


cv_df = pd.DataFrame(
    cv_results
).sort_values(
    by="Mean ROC-AUC",
    ascending=False
)

print("\nCross-validation results:")

print(
    cv_df.round(4).to_string(index=False)
)


# ============================================================
# 15. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.index[0]

best_model = trained_models[
    best_model_name
]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"\nBest model based on ROC-AUC: {best_model_name}"
)


# ============================================================
# 16. BEST MODEL EVALUATION
# ============================================================

best_predictions = best_model.predict(
    X_test
)

best_probabilities = best_model.predict_proba(
    X_test
)[:, 1]


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    best_predictions
)

print("\nConfusion Matrix:")

print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Churn",
        "Churn"
    ]
)

disp.plot()

plt.title(
    f"Confusion Matrix — {best_model_name}"
)

plt.tight_layout()

plt.show()


# ============================================================
# 18. ROC CURVE
# ============================================================

plt.figure(figsize=(8, 6))

RocCurveDisplay.from_predictions(
    y_test,
    best_probabilities
)

plt.title(
    f"ROC Curve — {best_model_name}"
)

plt.tight_layout()

plt.show()


# ============================================================
# 19. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

model_step = best_model.named_steps["model"]

if hasattr(
    model_step,
    "feature_importances_"
):

    feature_importances = (
        model_step.feature_importances_
    )

    feature_names = (
        best_model
        .named_steps[
            "preprocessor"
        ]
        .get_feature_names_out()
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importances
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False
    )

    print(
        f"\nFeature importance saved as: "
        f"{FEATURE_IMPORTANCE_FILE}"
    )

else:

    print(
        "\nFeature importance is not directly available "
        "for this model."
    )


# ============================================================
# 20. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    MODEL_FILE
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    f"\nSaved model: {MODEL_FILE}"
)


# ============================================================
# 21. LOAD SAVED MODEL
# ============================================================

loaded_model = joblib.load(
    MODEL_FILE
)

print(
    "\nSaved model loaded successfully!"
)


# ============================================================
# 22. SAMPLE CUSTOMER PREDICTION
# ============================================================

sample_customer = X_test.iloc[
    [0]
]

actual_value = y_test.iloc[
    0
]

prediction = loaded_model.predict(
    sample_customer
)[0]

probability = loaded_model.predict_proba(
    sample_customer
)[0, 1]


print("\n" + "=" * 70)
print("SAMPLE CUSTOMER PREDICTION")
print("=" * 70)

print(
    f"\nActual Churn    : {actual_value}"
)

print(
    f"Predicted Churn : {prediction}"
)

print(
    f"Churn Probability: {probability:.2%}"
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DAY 30/30 PROJECT COMPLETED SUCCESSFULLY! 🎉")
print("=" * 70)

print(
    f"\nBest Model: {best_model_name}"
)

print(
    f"Test ROC-AUC: "
    f"{results_df.loc[best_model_name, 'ROC-AUC']:.4f}"
)

print(
    f"Test Accuracy: "
    f"{results_df.loc[best_model_name, 'Accuracy']:.4f}"
)

print("\nFiles created:")

print(
    f"1. {RESULTS_FILE}"
)

print(
    f"2. {MODEL_FILE}"
)

print(
    f"3. {FEATURE_IMPORTANCE_FILE}"
)

print(
    "\n🎯 30 Days of Machine Learning completed!"
)