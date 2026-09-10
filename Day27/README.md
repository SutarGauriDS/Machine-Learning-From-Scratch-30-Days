# 🚀 Day 28/30 — Machine Learning Model Deployment with FastAPI

## 📌 Project Overview

This project demonstrates how to deploy a trained Machine Learning model as a **REST API using FastAPI**.

As part of my **30 Days of Machine Learning Challenge**, I took the Wine Classification model developed in Day 27 and made it accessible through API endpoints.

The API accepts wine-related features as input and returns the **predicted wine class along with prediction probabilities**.

---

## 🎯 Project Objective

The main objective is to understand the basic workflow of **Machine Learning Model Deployment**.

Instead of running the ML model manually inside a Python script, the trained model is exposed through an API so that other applications can send data and receive predictions.

### Workflow

```text
Wine Features
      ↓
FastAPI Request
      ↓
Data Validation
      ↓
Trained ML Pipeline
      ↓
Prediction
      ↓
Prediction Probability
      ↓
JSON Response
````

---

## 🧠 Machine Learning Model

The model used in this project is the **Wine Classification model from Day 27**.

### Dataset

The project uses Scikit-learn's built-in **Wine Recognition Dataset**.

* Samples: 178
* Features: 13
* Classes: 3
* Problem Type: Multiclass Classification

### Model Pipeline

```text
Input Data
    ↓
SimpleImputer
    ↓
StandardScaler
    ↓
Logistic Regression
    ↓
Prediction
```

The complete preprocessing and model are saved together as a `.pkl` file.

---

## ⚙️ Technologies Used

* 🐍 Python
* ⚡ FastAPI
* 🚀 Uvicorn
* 🤖 Scikit-learn
* 🐼 Pandas
* 🔢 NumPy
* 📦 Joblib
* 📖 Swagger UI

---

## 📂 Project Structure

```text
Day28_Model_Deployment/
│
├── create_model.py
├── main.py
├── wine_model.pkl
├── requirements.txt
├── README.md
├── references.md
└── interview_questions.md
```

### File Description

| File                     | Purpose                       |
| ------------------------ | ----------------------------- |
| `create_model.py`        | Trains and saves the ML model |
| `main.py`                | FastAPI application           |
| `wine_model.pkl`         | Saved trained ML pipeline     |
| `requirements.txt`       | Required Python libraries     |
| `README.md`              | Project documentation         |
| `references.md`          | Learning references           |
| `interview_questions.md` | Interview preparation         |

---

# 🛠️ Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Day28_Model_Deployment
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Step 1 — Create the model

If `wine_model.pkl` does not already exist, run:

```bash
python create_model.py
```

This trains the Wine Classification model and creates:

```text
wine_model.pkl
```

---

## Step 2 — Start the FastAPI server

Run:

```bash
python main.py
```

The server will start at:

```text
http://127.0.0.1:8000
```

---

# 📖 Swagger API Documentation

FastAPI automatically provides interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

You should see:

```text
GET  /
GET  /health
POST /predict
```

Swagger UI allows the API to be tested directly from the browser.

---

# 🔗 API Endpoints

## 1. GET `/`

Checks whether the API is running.

### Response

```json
{
  "message": "Wine Classification API is running!"
}
```

---

## 2. GET `/health`

Checks the API and model status.

### Response

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## 3. POST `/predict`

Accepts wine features and returns a prediction.

### Sample Input

```json
{
  "alcohol": 13.2,
  "malic_acid": 1.78,
  "ash": 2.14,
  "alcalinity_of_ash": 11.2,
  "magnesium": 100,
  "total_phenols": 2.65,
  "flavanoids": 2.76,
  "nonflavanoid_phenols": 0.26,
  "proanthocyanins": 1.28,
  "color_intensity": 4.38,
  "hue": 1.05,
  "od280_od315_of_diluted_wines": 3.4,
  "proline": 1050
}
```

### Sample Response

```json
{
  "status": "success",
  "predicted_class": "class_0",
  "predicted_class_number": 0,
  "probabilities": {
    "class_0": 0.98,
    "class_1": 0.02,
    "class_2": 0.0
  }
}
```

> Note: The exact probability values may vary depending on the trained model.

---

# 📊 Example Output

The API successfully receives the input features and returns:

```text
HTTP 200 OK

Prediction:
Class 0

Probabilities:
Class 0 → 98%
Class 1 → 2%
Class 2 → 0%
```

This demonstrates that the trained Machine Learning model can be accessed through an API.

---

# 🔍 Important Implementation Detail

The original Scikit-learn Wine dataset contains the feature:

```text
od280/od315_of_diluted_wines
```

Because `/` is inconvenient in a JSON/Pydantic field name, the API accepts:

```text
od280_od315_of_diluted_wines
```

Inside the prediction function, it is mapped back to the exact feature name expected by the trained Scikit-learn pipeline.

This ensures that the feature names match those used during model training.

---

# 💾 Model Persistence

The trained pipeline is saved using **Joblib**:

```python
joblib.dump(model, "wine_model.pkl")
```

The FastAPI application loads the saved model:

```python
model = joblib.load("wine_model.pkl")
```

This means the model does not need to be retrained every time the API starts.

---

# 🧪 Testing

The API can be tested using:

* Swagger UI
* Browser
* Postman
* Python `requests`
* Any application capable of sending HTTP requests

Swagger UI is used in this project for convenient testing.

---

# 🔐 Data Validation

FastAPI uses **Pydantic** models to validate incoming request data.

For example:

```python
class WineInput(BaseModel):
    alcohol: float
    malic_acid: float
    magnesium: float
```

This helps ensure that API inputs follow the expected structure and data types.

---

# 🚀 Why Model Deployment Matters

Training a Machine Learning model is only one part of an ML project.

A real-world ML workflow often looks like:

```text
Data
 ↓
Preprocessing
 ↓
Model Training
 ↓
Model Evaluation
 ↓
Model Saving
 ↓
API Development
 ↓
Deployment
 ↓
Application Integration
```

FastAPI provides a lightweight way to expose the trained model so that other applications can use its predictions.

---

# 💡 Key Learnings

Through this project, I learned:

1. **How to expose an ML model through a REST API**
2. **How FastAPI handles API endpoints**
3. **How Pydantic validates request data**
4. **How to load a saved Scikit-learn model**
5. **How to return predictions as JSON**
6. **How to test APIs using Swagger UI**
7. **How ML models can be integrated into applications**
8. **Why model deployment is an important part of the ML lifecycle**

---

# 🔮 Possible Future Improvements

This project can be extended by adding:

* Docker containerization
* Cloud deployment
* Authentication
* Logging and monitoring
* Database integration
* Batch prediction endpoint
* Frontend interface
* Automated CI/CD
* Model versioning
* Production server configuration

---

# 🏆 Day 28 Achievement

### **Machine Learning Model Deployment using FastAPI**

I successfully converted a trained Machine Learning model into an API that accepts real-time input and returns predictions.

