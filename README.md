# 🥗 NutriSense AI – Intelligent Health Risk Analyzer

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1.2-black?logo=flask)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit--learn-orange)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-95.30%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-20%2C758_Records-blue)

**NutriSense AI** is a machine-learning-powered web application that analyzes an individual's body measurements, dietary habits, and lifestyle information to predict a dataset-derived health risk category.

The application uses a trained Random Forest classification model to categorize inputs into three groups: Low Risk, Moderate Risk, and High Risk. It also provides BMI calculations, prediction probabilities, and personalized general lifestyle suggestions through an interactive web interface.

---

## 🚀 Project Overview

NutriSense AI aims to demonstrate how machine learning can be integrated into a web application to analyze lifestyle and dietary information.

Users can enter their personal measurements and lifestyle details to receive:

- A predicted risk category.
- Model confidence and probability scores.
- BMI calculation and category.
- General lifestyle suggestions.
- Prediction history and statistical summaries.

---

## ✨ Key Features

- 🧠 **Machine Learning Prediction:** Classifies inputs into Low, Moderate, or High Risk groups.
- 📊 **Prediction Confidence:** Displays the model's confidence and probability distribution.
- ⚖️ **BMI Calculator:** Automatically calculates BMI using height and weight.
- 💡 **Personalized Suggestions:** Generates general lifestyle suggestions based on submitted information.
- 📋 **Prediction History:** Maintains a history of predictions during the current server session.
- 📈 **Statistics Dashboard:** Displays prediction statistics and risk distribution.
- 🌐 **REST API:** Provides endpoints for predictions, BMI, history, statistics, and application health.
- 📱 **Responsive Web Interface:** Interactive frontend built using HTML, CSS, and JavaScript.

---

## 🧠 Machine Learning Model

Two machine learning algorithms were evaluated during model development.

| Algorithm | Average 5-Fold CV Accuracy |
|---|---:|
| Logistic Regression | 93.24% |
| **Random Forest** | **94.47%** |

Random Forest was selected because it achieved the higher average cross-validation accuracy.

### Model Performance

| Evaluation Metric | Result |
|---|---:|
| Selected Algorithm | Random Forest |
| Cross-Validation Accuracy | 94.47% |
| Hold-Out Test Accuracy | **95.30%** |
| Training Data | 80% |
| Testing Data | 20% |
| Cross-Validation | 5-Fold |

### Classification Report

| Risk Group | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Low Risk | 0.96 | 0.95 | 0.95 |
| Moderate Risk | 0.89 | 0.92 | 0.90 |
| High Risk | 0.98 | 0.97 | 0.98 |
| **Overall Accuracy** | | | **0.95** |

The reported performance is based on the held-out test dataset and does not represent clinical validation or guaranteed real-world accuracy.

---

## 🧬 Dataset Information

The project uses an obesity-level dataset containing:

- **20,758 records**
- **16 input features**
- **7 original target categories**
- **3 grouped prediction categories**

### Input Features

| Category | Features |
|---|---|
| Body Measurements | Age, Gender, Height, Weight |
| Family History | Family history of overweight |
| Dietary Habits | High-calorie food consumption, vegetable consumption, number of main meals, eating between meals |
| Lifestyle | Smoking, water consumption, calorie monitoring, physical activity, technology usage |
| Other Habits | Alcohol consumption, transportation method |

### Prediction Categories

The original seven dataset labels are grouped into three output categories.

| Prediction Category | Original Dataset Labels |
|---|---|
| Low Risk | Insufficient Weight, Normal Weight |
| Moderate Risk | Overweight Level I, Overweight Level II |
| High Risk | Obesity Type I, Obesity Type II, Obesity Type III |

The original dataset contains the label `0rmal_Weight`, which is retained as written in the source data.

---

## ⚙️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Web Framework | Flask |
| Data Processing | Pandas, NumPy |
| Frontend | HTML5, CSS3, JavaScript |
| Model Serialization | Pickle |

---

## 📁 Project Structure

```text
NutriSense-AI/
│
├── app.py
├── train_model.py
├── check_dataset.py
│
├── obesity_level.csv
├── model.pkl
├── features.pkl
│
├── requirements.txt
├── .gitignore
├── README.md
│
└── templates/
    └── index.html
```

---

## ⚡ Installation and Setup

### 1. Clone the repository

Replace the repository URL below with your actual GitHub repository URL after creating it.

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd NutriSense-AI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the machine learning model

```bash
python train_model.py
```

This generates two important files:

- `model.pkl` – Stores the trained machine learning pipeline.
- `features.pkl` – Stores the input feature names required by the model.

### 4. Run the Flask application

```bash
python app.py
```

### 5. Open the application

Open your browser and visit:

```text
http://127.0.0.1:5000
```

The NutriSense AI dashboard will be displayed.

---

## 🔬 How the Application Works

1. **User Input:** The user enters body measurements, dietary habits, and lifestyle information.
2. **BMI Calculation:** BMI is calculated automatically using height and weight.
3. **Data Preparation:** The submitted information is converted into the required input format.
4. **Preprocessing:** The machine learning pipeline applies numerical scaling and categorical encoding.
5. **Model Prediction:** The Random Forest classifier predicts one of the three grouped risk categories.
6. **Probability Calculation:** The application obtains probability scores for all three categories.
7. **Recommendations:** General lifestyle suggestions are generated based on the submitted information.
8. **Result Display:** The frontend displays the prediction, probabilities, BMI, and suggestions.

---

## 📡 REST API Endpoints

| HTTP Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Loads the main web application |
| POST | `/predict` | Predicts the risk category |
| POST | `/bmi` | Calculates BMI |
| GET | `/history` | Retrieves prediction history |
| GET | `/stats` | Retrieves prediction statistics |
| GET | `/health` | Checks application health |

### Example Prediction Request

**Endpoint:** `POST /predict`

**Content-Type:** `application/json`

```json
{
  "Gender": "Male",
  "Age": 25,
  "Height": 1.75,
  "Weight": 82,
  "family_history_with_overweight": 1,
  "FAVC": 1,
  "FCVC": 2,
  "NCP": 3,
  "CAEC": "Sometimes",
  "SMOKE": 0,
  "CH2O": 2,
  "SCC": 0,
  "FAF": 1,
  "TUE": 1,
  "CALC": "Sometimes",
  "MTRANS": "Public_Transportation"
}

### Example Prediction Response

```json
{
  "risk": "Moderate Risk",
  "emoji": "🟡",
  "confidence": 80.04,
  "bmi": 26.8,
  "bmi_category": "Overweight",
  "probabilities": {
    "low": 10.76,
    "moderate": 80.04,
    "high": 9.2
  },
  "recommendations": [
    "Pay attention to frequent consumption of high-calorie foods.",
    "Family history is one factor to consider alongside lifestyle habits.",
    "Consider reviewing your eating patterns and physical activity."
  ]
}
```

*Note: These percentages represent the model's output for this input, not medical probabilities or a clinical diagnosis.*

```

**Note:** Height must be provided in meters when using the prediction API. The website automatically converts height entered in centimeters into meters.

---

## 📊 Model Evaluation and Validation

The model development process includes:

- Dataset loading and inspection.
- Target category grouping.
- Numerical and categorical feature preprocessing.
- Stratified train-test splitting.
- Five-fold cross-validation.
- Comparison of Logistic Regression and Random Forest.
- Model selection based on cross-validation accuracy.
- Hold-out test evaluation.
- Classification report generation.

---

## ⚠️ Limitations and Disclaimer

- This project is an educational machine learning demonstration, not a medical diagnostic system.
- The three prediction groups are derived from obesity-level dataset labels and are not clinically validated medical risk categories.
- Model confidence represents the classifier's output, not a medical probability.
- The model's performance on the supplied dataset does not guarantee equivalent performance on real-world populations.
- Adult BMI reference ranges may not be appropriate for individuals under 18.
- Prediction history and statistics are stored in application memory and reset when the Flask server restarts.

Users should consult qualified healthcare professionals for medical advice or health assessments.

---

## 🔮 Future Improvements

- Deploy the application to a cloud hosting platform.
- Improve model evaluation using additional datasets.
- Add more comprehensive data visualizations.
- Introduce user authentication and persistent prediction history.
- Improve recommendation generation.
- Explore multilingual support.

---

## 👨‍💻 Author

**Mohith Jhajhrya**

**Role:** Sole Developer

- GitHub: [mohithdon](https://github.com/mohithdon)
- Email: mohithjhajhrya@gmail.com

Designed and developed NutriSense AI, including its web interface, machine learning workflow, Flask backend, model integration, and prediction functionality.

---

**NutriSense AI – Turning Lifestyle Data into Meaningful Insights.** 🥗