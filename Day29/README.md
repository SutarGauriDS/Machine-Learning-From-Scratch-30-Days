
# 🚀 Day 29/30 — Handwritten Digit Recognition

This project implements a **Handwritten Digit Recognition system** using Machine Learning.

The objective is to classify handwritten digits from **0 to 9** using the built-in **Scikit-learn Digits dataset**.

Multiple classification algorithms are trained and compared, and the best-performing model is selected based on test accuracy and cross-validation performance.

---

# 🎯 Project Objective

The main goal of this project is to build a complete Machine Learning classification workflow for recognizing handwritten digits.

The project covers:

- Data loading
- Data understanding
- Data visualization
- Train-test splitting
- Multiple Machine Learning models
- Model evaluation
- Cross-validation
- Model comparison
- Confusion matrix analysis
- Model persistence
- Sample prediction
- Prediction probabilities

---

# 📊 Dataset

The project uses the **Digits dataset** provided by Scikit-learn.

### Dataset Information

| Property | Value |
|---|---:|
| Total Samples | 1,797 |
| Features | 64 |
| Classes | 10 |
| Target Classes | 0–9 |
| Image Size | 8 × 8 pixels |

Each handwritten digit is represented as an **8 × 8 grayscale image**.

The 64 pixel values are converted into numerical features for Machine Learning.

```text
8 × 8 Image
     ↓
64 Pixel Features
     ↓
Machine Learning Model
     ↓
Predicted Digit (0–9)
````

---

# 🧠 Machine Learning Models

Four classification algorithms were implemented and compared.

## 1. Logistic Regression

Used as a strong linear classification baseline.

## 2. K-Nearest Neighbors (KNN)

Classifies a sample based on the nearest training examples.

## 3. Random Forest

Uses multiple decision trees to make predictions.

## 4. Support Vector Machine (SVM)

Finds a decision boundary that separates different digit classes.

For Logistic Regression, KNN, and SVM, **StandardScaler** was used because these algorithms can benefit from feature scaling.

---

# 🔄 Project Workflow

```text
Load Digits Dataset
        ↓
Data Understanding
        ↓
Visualize Handwritten Digits
        ↓
Train-Test Split
        ↓
Create ML Models
        ↓
Train Models
        ↓
Evaluate Models
        ↓
Model Comparison
        ↓
5-Fold Cross-Validation
        ↓
Select Best Model
        ↓
Classification Report
        ↓
Confusion Matrix
        ↓
Save Best Model
        ↓
Load Saved Model
        ↓
Make Prediction
```

---

# 🔍 Data Understanding

The dataset was inspected for:

* Number of samples
* Number of features
* Target classes
* Missing values
* Duplicate records
* Target distribution

### Results

```text
Samples  : 1797
Features : 64
Classes  : 10
Missing values: 0
Duplicate rows: 0
```

The dataset contains a relatively balanced number of examples for each digit.

---

# 🖼️ Digit Visualization

The project visualizes sample handwritten digits using Matplotlib.

Each image represents an **8 × 8 pixel grayscale image**.

This helps understand how the raw image information is represented before it is given to the Machine Learning algorithms.

---

# ✂️ Train-Test Split

The dataset was divided into training and testing sets.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Split

```text
Training Samples: 1437
Testing Samples : 360
```

`stratify=y` was used to maintain a similar class distribution in the training and testing datasets.

---

# 📈 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

Weighted averaging was used for the multiclass precision, recall, and F1-score calculations.

---

# 🏆 Model Comparison

The actual test-set results from the completed project were:

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| **SVM**             | **97.50%** | **97.59%** | **97.50%** | **97.49%** |
| Logistic Regression |     97.22% |     97.24% |     97.22% |     97.22% |
| KNN                 |     96.39% |     96.48% |     96.39% |     96.36% |
| Random Forest       |     96.39% |     96.44% |     96.39% |     96.36% |

### 🥇 Best Test Model

**SVM — 97.50% accuracy**

---

# 🔄 Cross-Validation

A **5-Fold Stratified Cross-Validation** approach was used to evaluate model stability.

### Results

| Model               | Mean CV Accuracy |     CV Std |
| ------------------- | ---------------: | ---------: |
| **SVM**             |       **98.12%** | **0.0061** |
| Random Forest       |           97.63% |     0.0097 |
| KNN                 |           97.56% |     0.0099 |
| Logistic Regression |           96.80% |     0.0074 |

### 🥇 Best Cross-Validation Model

**SVM — approximately 98.12% mean accuracy**

The SVM model performed best on both the test set and cross-validation.

---

# 🔲 Confusion Matrix

A confusion matrix was generated for the best-performing **SVM model**.

It shows:

* Actual digit classes on the Y-axis
* Predicted digit classes on the X-axis
* Correct predictions along the diagonal
* Misclassifications outside the diagonal

The confusion matrix shows that the majority of handwritten digits were classified correctly.

---

# 💾 Model Persistence

The best-performing SVM model was saved using **Joblib**.

```python
joblib.dump(best_model, "digit_model.pkl")
```

The saved model can later be loaded without retraining:

```python
loaded_model = joblib.load("digit_model.pkl")
```

This makes the trained model reusable for future predictions.

---

# 🔢 Sample Prediction

After loading the saved model, a sample from the test dataset was passed to the model.

The system compares:

```text
Actual Digit
     ↓
Machine Learning Model
     ↓
Predicted Digit
```

The model can also return prediction probabilities using:

```python
loaded_model.predict_proba(sample)
```

This provides the probability associated with each digit class.

---

# 📁 Project Structure

```text
Day29_Handwritten_Digit_Recognition/
│
├── day29_digit_recognition.py
├── Day29_Handwritten_Digit_Recognition_Complete.ipynb
├── digit_model.pkl
├── Day29_Model_Results.csv
├── README.md
├── requirements.txt
├── references.md
└── interview_questions.md
```

---

# 📄 File Description

| File                                                 | Description                     |
| ---------------------------------------------------- | ------------------------------- |
| `day29_digit_recognition.py`                         | Complete Python implementation  |
| `Day29_Handwritten_Digit_Recognition_Complete.ipynb` | Jupyter Notebook                |
| `digit_model.pkl`                                    | Saved best-performing SVM model |
| `Day29_Model_Results.csv`                            | Model comparison results        |
| `README.md`                                          | Project documentation           |
| `requirements.txt`                                   | Required Python packages        |
| `references.md`                                      | Learning references             |
| `interview_questions.md`                             | Interview preparation           |

---

# 🛠️ Technologies Used

* 🐍 Python
* 🐼 Pandas
* 🔢 NumPy
* 📊 Matplotlib
* 🤖 Scikit-learn
* 📦 Joblib
* 📓 Jupyter Notebook

---

# ▶️ How to Run

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

## 2. Open the project folder

```bash
cd Day29_Handwritten_Digit_Recognition
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Python project

```bash
python day29_digit_recognition.py
```

The program will:

1. Load the dataset
2. Display sample digits
3. Train four ML models
4. Compare their performance
5. Perform cross-validation
6. Select the best model
7. Display the confusion matrix
8. Save the best model
9. Load the saved model
10. Perform a sample prediction

---

# 💡 Key Learnings

Through this project, I learned:

1. **How image data can be represented as numerical features**
2. **How to work with Scikit-learn's built-in Digits dataset**
3. **How to compare multiple classification algorithms**
4. **Why feature scaling is important for certain ML algorithms**
5. **How to evaluate multiclass classification models**
6. **How to use Stratified K-Fold Cross-Validation**
7. **How to interpret a confusion matrix**
8. **How to select a model based on multiple evaluation methods**
9. **How to save and reload trained Machine Learning models**
10. **How to generate prediction probabilities**

---

# 🚀 Future Improvements

This project can be extended by:

* Building a handwritten digit drawing interface
* Allowing users to upload digit images
* Using CNN for image-based deep learning
* Creating a web application
* Deploying the model using FastAPI
* Adding Docker
* Deploying the application to the cloud

---

# 🏆 Final Result

The project successfully compared four Machine Learning algorithms.

### Best Model:

**Support Vector Machine (SVM)**

### Test Accuracy:

**97.50%**

### Cross-Validation Accuracy:

**98.12%**

The model successfully recognizes handwritten digits from **0 to 9** with high classification accuracy.

---

# 📌 Conclusion

This project demonstrates a complete Machine Learning workflow from **raw image data to model evaluation and prediction**.

It also demonstrates an important practical ML skill: **comparing different algorithms and selecting a reliable model rather than depending on a single algorithm.**
