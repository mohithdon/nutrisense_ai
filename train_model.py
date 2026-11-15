
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------------------
# 1. LOAD DATASET
# -----------------------------------------

df = pd.read_csv("obesity_level.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# -----------------------------------------
# 2. REMOVE UNNECESSARY COLUMN
# -----------------------------------------

df = df.drop(columns=["id"])


# -----------------------------------------
# 3. CONVERT 7 CATEGORIES INTO 3 RISK LEVELS
# -----------------------------------------

risk_map = {
    "Insufficient_Weight": 0,
    "0rmal_Weight": 0,
    "Overweight_Level_I": 1,
    "Overweight_Level_II": 1,
    "Obesity_Type_I": 2,
    "Obesity_Type_II": 2,
    "Obesity_Type_III": 2
}

df["risk"] = df["0be1dad"].map(risk_map)

if df["risk"].isnull().any():
    raise ValueError("Some target labels could not be mapped!")

df["risk"] = df["risk"].astype(int)

# Separate input features and target
X = df.drop(columns=["0be1dad", "risk"])
y = df["risk"]


# -----------------------------------------
# 4. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# -----------------------------------------

categorical_features = [
    "Gender",
    "CAEC",
    "CALC",
    "MTRANS"
]

numerical_features = [
    column for column in X.columns
    if column not in categorical_features
]

print("\nNumerical features:", numerical_features)
print("Categorical features:", categorical_features)


# -----------------------------------------
# 5. PREPROCESS THE DATA
# -----------------------------------------

numerical_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# -----------------------------------------
# 6. SPLIT TRAINING AND TESTING DATA
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# -----------------------------------------
# 7. DEFINE MODELS
# -----------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        class_weight="balanced",
        n_jobs=1
    )
}


# -----------------------------------------
# 8. COMPARE MODELS USING CROSS-VALIDATION
# -----------------------------------------

best_model = None
best_score = 0
best_name = ""

print("\nMODEL COMPARISON")
print("-" * 40)

for name, classifier in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
        n_jobs=1
    )

    average_score = scores.mean()

    print(name)
    print("Fold scores:", scores)
    print("Average CV accuracy:", round(average_score * 100, 2), "%")
    print("-" * 40)

    if average_score > best_score:
        best_score = average_score
        best_model = pipeline
        best_name = name


# -----------------------------------------
# 9. TRAIN THE BEST MODEL
# -----------------------------------------

print("\nBest model:", best_name)

best_model.fit(X_train, y_train)


# -----------------------------------------
# 10. TEST THE MODEL
# -----------------------------------------

predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nFINAL TEST RESULTS")
print("-" * 40)

print("Selected model:", best_name)
print("Cross-validation accuracy:", round(best_score * 100, 2), "%")
print("Test accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=[
            "Low Risk",
            "Moderate Risk",
            "High Risk"
        ]
    )
)


# -----------------------------------------
# 11. SAVE TRAINED MODEL
# -----------------------------------------

with open("model.pkl", "wb") as file:
    pickle.dump(best_model, file)

with open("features.pkl", "wb") as file:
    pickle.dump(list(X.columns), file)

print("\nModel saved successfully!")
print("Created: model.pkl")
print("Created: features.pkl")
print("\nTraining completed!")
