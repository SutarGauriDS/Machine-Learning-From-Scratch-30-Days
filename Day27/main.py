from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("wine_model.pkl")

print("✅ Model loaded successfully")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Wine Classification API",
    description="Wine classification using Machine Learning",
    version="1.0"
)


# ============================================================
# INPUT DATA
# ============================================================

class WineInput(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Wine Classification API is running!",
        "status": "success"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(data: WineInput):

    print("\n📥 Prediction request received")

    # Create DataFrame
    input_df = pd.DataFrame([{
        "alcohol": data.alcohol,
        "malic_acid": data.malic_acid,
        "ash": data.ash,
        "alcalinity_of_ash": data.alcalinity_of_ash,
        "magnesium": data.magnesium,
        "total_phenols": data.total_phenols,
        "flavanoids": data.flavanoids,
        "nonflavanoid_phenols": data.nonflavanoid_phenols,
        "proanthocyanins": data.proanthocyanins,
        "color_intensity": data.color_intensity,
        "hue": data.hue,
        "od280/od315_of_diluted_wines":
            data.od280_od315_of_diluted_wines,
        "proline": data.proline
    }])

    print("\nInput received:")
    print(input_df)

    # Prediction
    prediction = model.predict(input_df)[0]

    # Probability
    probabilities = model.predict_proba(input_df)[0]

    print("\nPrediction:", prediction)
    print("Probabilities:", probabilities)

    return {
        "status": "success",
        "predicted_class": f"class_{int(prediction)}",
        "predicted_class_number": int(prediction),
        "probabilities": {
            "class_0": round(float(probabilities[0]), 4),
            "class_1": round(float(probabilities[1]), 4),
            "class_2": round(float(probabilities[2]), 4)
        }
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    print("\n==========================================")
    print("🍷 WINE CLASSIFICATION API")
    print("==========================================")
    print("Swagger: http://127.0.0.1:8000/docs")
    print("Health : http://127.0.0.1:8000/health")
    print("Predict: http://127.0.0.1:8000/predict")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )