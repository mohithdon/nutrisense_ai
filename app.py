
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os
from datetime import datetime
from collections import Counter


# -----------------------------------------
# 1. FLASK APPLICATION
# -----------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)


# -----------------------------------------
# 2. LOAD TRAINED MODEL
# -----------------------------------------

def load_pkl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with open(path, "rb") as file:
        return pickle.load(file)


model = load_pkl(
    os.path.join(BASE_DIR, "model.pkl")
)

FEATURES = load_pkl(
    os.path.join(BASE_DIR, "features.pkl")
)

_history = []


# -----------------------------------------
# 3. RISK INFORMATION
# -----------------------------------------

RISK_MAP = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk"
}

RISK_EMOJI = {
    0: "💚",
    1: "🟡",
    2: "🔴"
}


# -----------------------------------------
# 4. SAFE INPUT FUNCTIONS
# -----------------------------------------

def safe_int(value):
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        raise ValueError("Invalid integer input.")


def safe_float(value):
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        raise ValueError("Invalid numerical input.")


# -----------------------------------------
# 5. VALIDATE INPUT DATA
# -----------------------------------------

def prepare_input(data):

    required_fields = FEATURES

    for field in required_fields:
        if field not in data:
            raise ValueError(
                f"Missing required field: {field}"
            )

    gender = str(data["Gender"]).strip()

    if gender not in ["Male", "Female"]:
        raise ValueError("Invalid gender.")

    caec = str(data["CAEC"]).strip()

    if caec not in [
        "Sometimes",
        "Frequently",
        "Always",
        "0"
    ]:
        raise ValueError("Invalid CAEC value.")

    calc = str(data["CALC"]).strip()

    if calc not in [
        "Sometimes",
        "Frequently",
        "0"
    ]:
        raise ValueError("Invalid CALC value.")

    transport = str(data["MTRANS"]).strip()

    if transport not in [
        "Public_Transportation",
        "Automobile",
        "Walking",
        "Motorbike",
        "Bike"
    ]:
        raise ValueError("Invalid transportation value.")

    prepared = {
        "Gender": gender,
        "Age": safe_float(data["Age"]),
        "Height": safe_float(data["Height"]),
        "Weight": safe_float(data["Weight"]),
        "family_history_with_overweight": safe_int(
            data["family_history_with_overweight"]
        ),
        "FAVC": safe_int(data["FAVC"]),
        "FCVC": safe_float(data["FCVC"]),
        "NCP": safe_float(data["NCP"]),
        "CAEC": caec,
        "SMOKE": safe_int(data["SMOKE"]),
        "CH2O": safe_float(data["CH2O"]),
        "SCC": safe_int(data["SCC"]),
        "FAF": safe_float(data["FAF"]),
        "TUE": safe_float(data["TUE"]),
        "CALC": calc,
        "MTRANS": transport
    }

    if not 14 <= prepared["Age"] <= 100:
        raise ValueError("Age must be between 14 and 100.")

    if not 1.0 <= prepared["Height"] <= 2.5:
        raise ValueError("Height must be in metres.")

    if not 20 <= prepared["Weight"] <= 300:
        raise ValueError("Weight must be between 20 and 300 kg.")

    if prepared["family_history_with_overweight"] not in [0, 1]:
        raise ValueError("Invalid family history value.")

    if prepared["FAVC"] not in [0, 1]:
        raise ValueError("Invalid FAVC value.")

    if not 1 <= prepared["FCVC"] <= 3:
        raise ValueError("FCVC must be between 1 and 3.")

    if not 1 <= prepared["NCP"] <= 4:
        raise ValueError("NCP must be between 1 and 4.")

    if prepared["SMOKE"] not in [0, 1]:
        raise ValueError("Invalid smoking value.")

    if not 1 <= prepared["CH2O"] <= 3:
        raise ValueError("CH2O must be between 1 and 3.")

    if prepared["SCC"] not in [0, 1]:
        raise ValueError("Invalid SCC value.")

    if not 0 <= prepared["FAF"] <= 3:
        raise ValueError("FAF must be between 0 and 3.")

    if not 0 <= prepared["TUE"] <= 2:
        raise ValueError("TUE must be between 0 and 2.")

    return prepared


# -----------------------------------------
# 6. BMI CALCULATION
# -----------------------------------------

def calculate_bmi(weight, height):

    bmi = weight / (height ** 2)

    bmi = round(bmi, 1)

    if bmi < 18.5:
        category = "Underweight"

    elif bmi < 25:
        category = "Normal"

    elif bmi < 30:
        category = "Overweight"

    else:
        category = "Obese"

    return bmi, category


# -----------------------------------------
# 7. RECOMMENDATIONS
# -----------------------------------------

def build_recommendations(data, risk):

    tips = []

    if data["FCVC"] < 2:
        tips.append(
            "Consider including vegetables more regularly in your meals."
        )

    if data["CH2O"] < 2:
        tips.append(
            "Review your daily water intake and maintain adequate hydration."
        )

    if data["FAF"] < 1:
        tips.append(
            "Consider gradually increasing your physical activity."
        )

    if data["FAVC"] == 1:
        tips.append(
            "Pay attention to frequent consumption of high-calorie foods."
        )

    if data["family_history_with_overweight"] == 1:
        tips.append(
            "Family history is one factor to consider alongside lifestyle habits."
        )

    if data["SMOKE"] == 1:
        tips.append(
            "Avoid smoking and consider seeking support to quit."
        )

    if data["TUE"] >= 2:
        tips.append(
            "Consider balancing technology use with movement and physical activity."
        )

    if risk == 0:
        tips.append(
            "Continue maintaining balanced eating and activity habits."
        )

    elif risk == 1:
        tips.append(
            "Consider reviewing your eating patterns and physical activity."
        )

    elif risk == 2:
        tips.append(
            "Consider discussing your health and nutrition with a qualified healthcare professional."
        )

    return tips[:5]


# -----------------------------------------
# 8. HOME PAGE
# -----------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -----------------------------------------
# 9. BMI CALCULATOR
# -----------------------------------------

@app.route("/bmi", methods=["POST"])
def calc_bmi():

    try:

        data = request.get_json()

        if not data:
            raise ValueError("No input data received.")

        weight = safe_float(data.get("weight"))
        height = safe_float(data.get("height"))

        if weight <= 0 or height <= 0:
            raise ValueError("Weight and height must be positive.")

        bmi, category = calculate_bmi(
            weight,
            height / 100
        )

        return jsonify({
            "bmi": bmi,
            "category": category
        })

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 400


# -----------------------------------------
# 10. PREDICTION
# -----------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:
            raise ValueError("No JSON data received.")

        # Prepare all 16 genuine dataset features
        prepared = prepare_input(data)

        # Create DataFrame in the exact training order
        input_df = pd.DataFrame(
            [prepared],
            columns=FEATURES
        )

        # Model performs preprocessing internally
        prediction = int(
            model.predict(input_df)[0]
        )

        probabilities = model.predict_proba(input_df)[0]

        confidence = round(
            float(max(probabilities)) * 100,
            2
        )

        bmi, bmi_category = calculate_bmi(
            prepared["Weight"],
            prepared["Height"]
        )

        tips = build_recommendations(
            prepared,
            prediction
        )

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        response = {

            "risk": RISK_MAP[prediction],

            "emoji": RISK_EMOJI[prediction],

            "confidence": confidence,

            "bmi": bmi,

            "bmi_category": bmi_category,

            "probabilities": {
                "low": round(
                    float(probabilities[0]) * 100, 2
                ),

                "moderate": round(
                    float(probabilities[1]) * 100, 2
                ),

                "high": round(
                    float(probabilities[2]) * 100, 2
                )
            },

            "recommendations": tips,

            "timestamp": timestamp
        }

        _history.append({

            "timestamp": timestamp,

            "risk": RISK_MAP[prediction],

            "confidence": confidence,

            "bmi": bmi,

            "age": prepared["Age"]

        })

        return jsonify(response)

    except Exception as error:

        return jsonify({

            "error": "Prediction failed",

            "details": str(error)

        }), 400


# -----------------------------------------
# 11. PREDICTION HISTORY
# -----------------------------------------

@app.route("/history", methods=["GET"])
def history():

    return jsonify({

        "count": len(_history),

        "history": _history[-20:][::-1]

    })


# -----------------------------------------
# 12. STATISTICS
# -----------------------------------------

@app.route("/stats", methods=["GET"])
def stats():

    if not _history:

        return jsonify({
            "message": "No predictions yet."
        })

    counts = Counter(
        item["risk"] for item in _history
    )

    avg_confidence = round(
        sum(item["confidence"] for item in _history)
        / len(_history),
        2
    )

    avg_bmi = round(
        sum(item["bmi"] for item in _history)
        / len(_history),
        1
    )

    return jsonify({

        "total_predictions": len(_history),

        "risk_distribution": dict(counts),

        "average_confidence": avg_confidence,

        "average_bmi": avg_bmi

    })


# -----------------------------------------
# 13. HEALTH CHECK
# -----------------------------------------

@app.route("/health", methods=["GET"])
def health_check():

    return jsonify({

        "status": "healthy",

        "model_loaded": True,

        "model_type": type(model).__name__,

        "feature_count": len(FEATURES),

        "total_analyzed": len(_history),

        "timestamp": datetime.now().isoformat()

    })


# -----------------------------------------
# 14. RUN APPLICATION
# -----------------------------------------

if __name__ == "__main__":

    app.run(
    debug=True,
    host="127.0.0.1",
    port=5000,
    use_reloader=False
)
