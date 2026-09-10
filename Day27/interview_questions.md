## 1. What is Machine Learning Model Deployment?

**Answer:**

Machine Learning model deployment is the process of making a trained ML model available for real-world use.

Instead of keeping the model inside a development environment, we expose it through an application, API, or service so that users or other applications can send data and receive predictions.

---

## 2. Why did you use FastAPI in this project?

**Answer:**

I used **FastAPI** to create a REST API around my trained Machine Learning model.

It allows an application to send wine features through an HTTP request and receive the model's prediction as a JSON response.

---

## 3. What is FastAPI?

**Answer:**

FastAPI is a modern Python web framework used for building APIs.

It provides:

- Fast API development
- Automatic request validation
- Type hints
- Automatic API documentation
- Swagger UI
- Easy integration with Machine Learning models

---

## 4. What is Uvicorn?

**Answer:**

**Uvicorn** is an ASGI web server used to run FastAPI applications.

In this project, Uvicorn starts the FastAPI application locally at:

```text
http://127.0.0.1:8000
````

---

## 5. What is an API?

**Answer:**

API stands for **Application Programming Interface**.

It allows two software applications to communicate with each other.

In this project:

```text
Client
   ↓
FastAPI
   ↓
ML Model
   ↓
Prediction
   ↓
JSON Response
```

---

## 6. What is a REST API?

**Answer:**

A REST API is an API that follows REST principles and uses HTTP methods to communicate between applications.

Common HTTP methods include:

* GET
* POST
* PUT
* DELETE

In my project, I mainly used **GET** and **POST**.

---

## 7. What does the `/predict` endpoint do?

**Answer:**

The `/predict` endpoint receives the 13 wine features from the client, passes them to the trained Machine Learning pipeline, and returns:

* Predicted class
* Predicted class number
* Probability for each class

---

## 8. Why is `/predict` a POST endpoint?

**Answer:**

I used POST because the client needs to send input data to the server.

The wine features are included in the request body as JSON.

Example:

```json
{
  "alcohol": 13.2,
  "malic_acid": 1.78,
  "magnesium": 100
}
```

---

## 9. What is Pydantic used for?

**Answer:**

Pydantic is used for **data validation**.

I created a Pydantic model that defines the expected wine features and their data types.

For example:

```python
class WineInput(BaseModel):
    alcohol: float
    malic_acid: float
    magnesium: float
```

If the client sends invalid data types, FastAPI can detect the validation error.

---

## 10. What is Swagger UI?

**Answer:**

Swagger UI is an interactive interface automatically provided by FastAPI for testing API endpoints.

In this project, I used:

```text
http://127.0.0.1:8000/docs
```

to test:

```text
GET /
GET /health
POST /predict
```

without needing a separate API testing tool.

---

## 11. What does the `/health` endpoint do?

**Answer:**

The `/health` endpoint checks whether the API is running and whether the ML model has been successfully loaded.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## 12. Why did you save the model as a `.pkl` file?

**Answer:**

I saved the trained Machine Learning pipeline so that it could be reused without retraining every time the API starts.

I used **Joblib** for model persistence.

```python
joblib.dump(model, "wine_model.pkl")
```

The model can later be loaded using:

```python
model = joblib.load("wine_model.pkl")
```

---

## 13. What is Joblib?

**Answer:**

Joblib is a Python library commonly used for efficiently saving and loading Python objects, including Scikit-learn models and pipelines.

---

## 14. Why did you save the entire pipeline instead of only the model?

**Answer:**

Saving the entire pipeline ensures that the same preprocessing steps used during training are also applied during prediction.

My pipeline contains:

```text
SimpleImputer
      ↓
StandardScaler
      ↓
Logistic Regression
```

This reduces the risk of applying different preprocessing during deployment.

---

## 15. What is the ML pipeline used in this project?

**Answer:**

The pipeline consists of:

```text
Input Features
      ↓
SimpleImputer
      ↓
StandardScaler
      ↓
Logistic Regression
      ↓
Prediction
```

The preprocessing and model are stored together.

---

## 16. What is the Wine dataset?

**Answer:**

The Wine Recognition dataset is a classification dataset available through Scikit-learn.

It contains:

* **178 samples**
* **13 features**
* **3 target classes**

The objective is to classify wine samples into one of three classes.

---

## 17. What model did you use?

**Answer:**

I used **Logistic Regression** as the classification model.

It is suitable for classification problems and can also provide class probabilities using `predict_proba()`.

---

## 18. What does `predict()` do?

**Answer:**

`predict()` returns the predicted class.

Example:

```python
prediction = model.predict(data)
```

The result could be:

```text
0
```

meaning the model predicted Class 0.

---

## 19. What does `predict_proba()` do?

**Answer:**

`predict_proba()` returns the probability associated with each class.

For example:

```text
Class 0 → 0.98
Class 1 → 0.02
Class 2 → 0.00
```

The probabilities indicate how strongly the model favors each class.

---

## 20. What does HTTP 200 OK mean?

**Answer:**

HTTP status code **200 OK** means that the request was successfully processed by the server.

For example, when `/predict` successfully returns a prediction, the API responds with HTTP 200.

---

## 21. What error did you encounter while developing the API?

**Answer:**

I encountered a feature-name mismatch between the feature used during model training and the feature sent by the API.

The trained model expected:

```text
od280/od315_of_diluted_wines
```

while the API initially provided:

```text
od280_od315_of_diluted_wines
```

I solved this by keeping the API-friendly field name for the request and mapping it internally to the exact feature name expected by the trained Scikit-learn pipeline.

---

## 22. Why are feature names important in Scikit-learn?

**Answer:**

When a model is trained using a Pandas DataFrame, Scikit-learn can retain the feature names.

During prediction, the input feature names should match the names and structure used during training.

Otherwise, the model can raise a feature-name mismatch error.

---

## 23. What is the difference between training and deployment?

**Answer:**

**Training** is the process of learning patterns from historical data.

**Deployment** is the process of making the trained model available for predictions on new data.

```text
Training:
Data → Model → Trained Model

Deployment:
New Data → API → Trained Model → Prediction
```

---

## 24. Is this project production-ready?

**Answer:**

Not completely.

This project demonstrates the **basic model deployment workflow** locally.

For production, I would add:

* Authentication
* HTTPS
* Docker
* Cloud deployment
* Logging
* Monitoring
* Error handling
* Model versioning
* CI/CD
* Load testing

---

## 25. How would you deploy this API to the cloud?

**Answer:**

I could containerize the application using **Docker** and deploy it to a cloud platform such as AWS, Azure, or another container hosting service.

A typical workflow would be:

```text
ML Model
   ↓
FastAPI
   ↓
Docker
   ↓
Cloud Platform
   ↓
Public API
```

---

#  Scenario-Based Interview Questions

## 26. What happens when a user sends data to `/predict`?

**Answer:**

The process is:

1. The client sends a POST request.
2. FastAPI receives the JSON data.
3. Pydantic validates the input.
4. The data is converted into the expected DataFrame structure.
5. The saved ML pipeline processes the input.
6. The model generates a prediction.
7. `predict_proba()` generates class probabilities.
8. FastAPI returns the result as JSON.

---

## 27. How would you improve this project?

**Answer:**

I would improve it by:

1. Adding Docker containerization.
2. Deploying the API to the cloud.
3. Adding authentication.
4. Adding structured logging.
5. Adding monitoring.
6. Implementing model versioning.
7. Creating a frontend application.
8. Adding automated testing and CI/CD.

---

## 28. How would you handle invalid input?

**Answer:**

I would use Pydantic validation to check the input data types and required fields.

I would also add exception handling for unexpected errors and return appropriate HTTP status codes and meaningful error messages.

---

## 29. How would you monitor a deployed ML model?

**Answer:**

I would monitor:

* API response time
* Error rate
* Request volume
* Prediction distribution
* Data drift
* Model performance
* System resource usage

Monitoring helps identify both application and model-related problems.

---

## 30. Explain this project in 30 seconds.

**Answer:**

> **"For Day 28 of my Machine Learning challenge, I deployed a Wine Classification model using FastAPI. I saved the complete Scikit-learn preprocessing and Logistic Regression pipeline using Joblib and created REST API endpoints for health checking and real-time prediction. The `/predict` endpoint accepts 13 wine features and returns the predicted class and class probabilities. I tested the API using FastAPI's Swagger UI and learned how a trained ML model can be integrated into an application through an API."**

---

# 🧠 Quick Revision

```text
FastAPI       → Build the API
Uvicorn       → Run the API server
Pydantic      → Validate input
Joblib        → Save/load model
Scikit-learn  → ML pipeline and model
Swagger UI    → Test API
POST /predict → Get prediction
GET /health   → Check API/model
HTTP 200      → Successful request
```

