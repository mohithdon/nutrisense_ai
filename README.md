# 🥗 NutriSense AI – Intelligent Health Risk Analyzer

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1.2-black?logo=flask)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Random_Forest-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7.2-F7931E?logo=scikitlearn)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-95.30%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-20%2C758_Records-blue)

NutriSense AI is a machine-learning-powered health risk analyzer that uses lifestyle and physical attributes to predict dataset-derived weight-related risk categories.

The application combines a trained Random Forest classification model with a Flask backend and an interactive web interface. Users can enter personal lifestyle information and receive a predicted risk category, model confidence, BMI information, and general lifestyle recommendations.

> **Disclaimer:** NutriSense AI is an educational machine learning project. Its predictions are based on patterns in the training dataset and are not medically validated. It is not intended to diagnose, prevent, or treat any medical condition.

---

## 📌 Project Overview

NutriSense AI analyzes lifestyle-related information such as age, height, weight, eating habits, water consumption, physical activity, and transportation methods.

The trained machine learning model classifies the input into one of three dataset-derived categories:

- 🟢 **Low Risk**
- 🟡 **Moderate Risk**
- 🔴 **High Risk**

The project demonstrates how machine learning can be integrated into a full-stack web application using Python, Flask, and HTML/CSS/JavaScript.

### 🎯 Objectives

- Build a machine learning model for weight-related category prediction.
- Analyze lifestyle and physical attributes from a structured dataset.
- Compare Logistic Regression and Random Forest models.
- Integrate the selected model into a Flask web application.
- Provide an interactive interface for predictions and results.
- Display model confidence, BMI information, and general recommendations.

---

## ✨ Key Features

- **Machine Learning Prediction:** Uses a trained Random Forest classification model.
- **Lifestyle Analysis:** Evaluates 16 input features related to physical attributes and lifestyle habits.
- **Three Risk Categories:** Displays Low, Moderate, and High Risk predictions.
- **BMI Calculator:** Calculates Body Mass Index using height and weight.
- **Model Confidence:** Displays the model's predicted class confidence.
- **Probability Distribution:** Shows the model's output probabilities for all three categories.
- **Personalized Recommendations:** Provides general suggestions based on submitted inputs.
- **Prediction History:** Displays previous predictions during the current server session.
- **Statistics Dashboard:** Shows prediction statistics.
- **Responsive Interface:** Supports desktop and mobile screen sizes.
- **REST API:** Provides Flask endpoints for prediction and related operations.

---

## 🤖 Machine Learning Approach

The project compares two classification algorithms:

1. Logistic Regression
2. Random Forest Classifier

Both models are trained and evaluated using the same dataset and preprocessing pipeline.

### Data Preprocessing

The preprocessing pipeline includes:

- Removing the unnecessary `id` column.
- Mapping the original seven dataset labels into three broader categories.
- Separating numerical and categorical features.
- Applying StandardScaler to numerical features.
- Applying OneHotEncoder to categorical features.
- Splitting the dataset into training and testing sets using stratified sampling.

The preprocessing steps are included inside the saved Scikit-learn pipeline.

### Model Selection

Five-fold cross-validation is used to compare the models. The model with the highest average cross-validation accuracy is selected.

**Random Forest achieved the highest average cross-validation accuracy and was selected as the final model.**

---

## 📊 Model Performance

### Cross-Validation Results

| Algorithm | Average Cross-Validation Accuracy |
|---|---:|
| Logistic Regression | 93.24% |
| Random Forest | 94.47% |

### Final Model Evaluation

The selected Random Forest model was evaluated on a separate hold-out test set.

| Metric | Result |
|---|---:|
| Test Accuracy | 95.30% |
| Training/Test Split | 80% / 20% |
| Cross-Validation | 5-Fold |
| Selected Algorithm | Random Forest |
| Input Features | 16 |
| Output Categories | 3 |

### Classification Report

| Category | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Low Risk | 0.96 | 0.95 | 0.95 |
| Moderate Risk | 0.89 | 0.92 | 0.90 |
| High Risk | 0.98 | 0.97 | 0.98 |
| **Overall Accuracy** | | | **0.95** |

*Note: The 95.30% accuracy is the result obtained on the project's hold-out test split. It does not represent clinical accuracy or guarantee performance on real-world users.*

---

## 📂 Dataset Information

The project uses the `obesity_level.csv` dataset.

| Dataset Detail | Information |
|---|---|
| Total Records | 20,758 |
| Original Columns | 18 |
| Input Features Used | 16 |
| Target Column | `0be1dad` |
| Missing Values | None |
| Original Target Classes | 7 |
| Final Output Categories | 3 |

### Input Features

The model uses the following 16 features:

| Feature | Description |
|---|---|
| Gender | Gender |
| Age | Age in years |
| Height | Height in meters |
| Weight | Weight in kilograms |
| family_history_with_overweight | Family history of overweight |
| FAVC | Frequent consumption of high-calorie food |
| FCVC | Frequency of vegetable consumption |
| NCP | Number of main meals |
| CAEC | Consumption of food between meals |
| SMOKE | Smoking habit |
| CH2O | Daily water consumption |
| SCC | Monitoring of calorie consumption |
| FAF | Physical activity frequency |
| TUE | Time spent using technological devices |
| CALC | Alcohol consumption frequency |
| MTRANS | Main mode of transportation |

### Target Category Mapping

The original dataset contains seven target labels. These are grouped into three categories for this project.

| Original Dataset Labels | Project Category |
|---|---|
| Insufficient_Weight, `0rmal_Weight` | Low Risk |
| Overweight_Level_I, Overweight_Level_II | Moderate Risk |
| Obesity_Type_I, Obesity_Type_II, Obesity_Type_III | High Risk |

*Note: The dataset contains the label `0rmal_Weight` with a zero at the beginning. This original spelling is retained during preprocessing.*

The three categories are project-defined groupings of the dataset labels. They are not clinically validated medical risk classifications.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Flask | Backend web framework |
| Pandas | Dataset processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and preprocessing |
| Random Forest | Final classification algorithm |
| HTML5 | Web page structure |
| CSS3 | Styling and responsive design |
| JavaScript | Frontend interactions and API communication |
| Pickle | Saving and loading trained model artifacts |

---

## 📁 Project Structure

```text
nutrisense_ai/
│
├── app.py                 # Flask application and prediction API
├── train_model.py         # Model training and evaluation
├── check_dataset.py       # Dataset inspection
├── model.pkl              # Trained machine learning pipeline
├── features.pkl           # Model input feature information
├── obesity_level.csv      # Dataset
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignored files configuration
├── README.md              # Project documentation
│
└── templates/
    └── index.html         # Frontend web application
```

---

## ⚙️ Installation and Setup

Follow these steps to run NutriSense AI locally.

### 1. Clone the Repository

```bash
git clone https://github.com/mohithdon/nutrisense_ai.git
```

### 2. Navigate to the Project Directory

```bash
cd nutrisense_ai
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python app.py
```

### 7. Open the Application

Open your browser and visit:

```text
http://localhost:5000
```

The NutriSense AI web application should now be running locally.

---

## 🔄 Application Workflow

```text
User Enters Lifestyle Information
              |
              v
      Frontend Validation
              |
              v
       Flask Backend
              |
              v
       Input Processing
              |
              v
    Saved ML Pipeline (model.pkl)
              |
              v
    Random Forest Prediction
              |
              v
      Prediction Results
              |
              v
   Risk Category and Confidence
              |
              v
       Results Displayed
```

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
```

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

*Note: This is an example based on a prediction produced during local testing. The exact prediction values depend on the input.*

**Important:** Height must be provided in meters when using the prediction API. The website automatically converts height entered in centimeters into meters.

---

## 📈 Model Evaluation and Validation

The project uses the following evaluation methods:

- Train-test split with stratified sampling.
- Five-fold cross-validation.
- Accuracy comparison between Logistic Regression and Random Forest.
- Precision, recall, and F1-score analysis.
- Evaluation of the selected model on a separate test set.

The final trained preprocessing and prediction pipeline is saved in `model.pkl`. The feature information is saved in `features.pkl`.

---

## 🔒 Important Notes and Limitations

- This project is developed for educational and demonstration purposes.
- Predictions are based on patterns learned from the provided dataset.
- The three risk categories are not medically validated classifications.
- Model confidence represents the model's output score, not medical certainty.
- BMI is a general screening measure and is not a complete assessment of health.
- Adult BMI categories may not be appropriate for individuals under 18.
- Prediction history and statistics are maintained in memory and may reset when the Flask server restarts.
- Real-world performance may differ from the reported test results.

**NutriSense AI should not be used as a substitute for professional medical advice, diagnosis, or treatment.**

---

## 🚀 Future Improvements

- Deploy the application to a public cloud platform.
- Add more diverse and representative training data.
- Explore additional machine learning algorithms.
- Improve model interpretability using feature importance and explainability techniques.
- Add persistent storage for prediction history.
- Improve input validation and application security.
- Conduct further evaluation using independent datasets.

---

## 👨‍💻 Author

**Mohith Jhajhrya**

- **Role:** Sole Developer
- **GitHub:** [mohithdon](https://github.com/mohithdon)
- **Email:** mohithjhajhrya@gmail.com

---

## ⭐ Support

If you find this project useful or interesting, consider giving the repository a star ⭐ on GitHub.

**GitHub Repository:** [NutriSense AI](https://github.com/mohithdon/nutrisense_ai)

---

*Built with Python, Flask, and Machine Learning.*
