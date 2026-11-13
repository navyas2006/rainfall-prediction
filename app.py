"""
app.py
------
Flask web application for the "Rainfall Prediction Using Machine Learning"
project.

Run with:   python app.py
Then open:  http://127.0.0.1:5000
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import preprocess as pp

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "rainfall_model.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

# ---------------------------------------------------------------------------
# Load model + preprocessing objects ONCE at startup
# ---------------------------------------------------------------------------
model = None
encoders = None
scaler = None
metadata = None
LOAD_ERROR = None

try:
    if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH)
            and os.path.exists(SCALER_PATH) and os.path.exists(METADATA_PATH)):
        raise FileNotFoundError(
            "Model files not found. Please run: python scripts/train_model.py"
        )

    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

except Exception as e:
    LOAD_ERROR = str(e)
    print(f"WARNING: Could not load model artifacts -> {LOAD_ERROR}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    accuracy = None
    if metadata:
        accuracy = round(metadata.get("accuracy", 0) * 100, 2)
    return render_template("index.html", accuracy=accuracy, load_error=LOAD_ERROR)


@app.route("/predict", methods=["GET"])
def predict_form():
    if LOAD_ERROR:
        return render_template(
            "predict.html",
            load_error=LOAD_ERROR,
            categorical_options={},
            numerical_ranges={},
        )

    return render_template(
        "predict.html",
        load_error=None,
        categorical_options=metadata["categorical_options"],
        numerical_ranges=metadata["numerical_ranges"],
    )


@app.route("/predict", methods=["POST"])
def predict_submit():
    if LOAD_ERROR or model is None:
        return render_template(
            "result.html",
            error="The trained model could not be loaded. Please run "
                  "'python scripts/train_model.py' first, then restart the app.",
        )

    try:
        # ---------------------------------------------------------
        # 1. Read + validate form data
        # ---------------------------------------------------------
        form_values = {}
        errors = []

        for col in pp.CATEGORICAL_COLUMNS:
            value = request.form.get(col, "").strip()
            if value == "":
                errors.append(f"'{col}' is required.")
            form_values[col] = value

        for col in pp.NUMERICAL_COLUMNS:
            raw_value = request.form.get(col, "").strip()
            if raw_value == "":
                errors.append(f"'{col}' is required.")
                continue
            try:
                form_values[col] = float(raw_value)
            except ValueError:
                errors.append(f"'{col}' must be a valid number.")

        if errors:
            return render_template(
                "predict.html",
                load_error=None,
                categorical_options=metadata["categorical_options"],
                numerical_ranges=metadata["numerical_ranges"],
                form_errors=errors,
                previous_values=request.form,
            )

        # ---------------------------------------------------------
        # 2. Build a single-row DataFrame with the SAME columns and
        #    SAME order used during training, then apply the SAME
        #    cleaning / encoding / scaling pipeline.
        # ---------------------------------------------------------
        input_df = pd.DataFrame([form_values])
        input_df = pp.clean_dataset(input_df)  # fills any stray missing values
        X_input = pp.build_feature_matrix(input_df, encoders, fit=False)
        X_input[pp.NUMERICAL_COLUMNS] = scaler.transform(X_input[pp.NUMERICAL_COLUMNS])

        # ---------------------------------------------------------
        # 3. Predict
        # ---------------------------------------------------------
        prediction = int(model.predict(X_input)[0])

        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            confidence = round(float(proba[prediction]) * 100, 2)

        result_text = "Rainfall Expected" if prediction == 1 else "No Rainfall Expected"

        return render_template(
            "result.html",
            error=None,
            prediction=result_text,
            is_rain=(prediction == 1),
            confidence=confidence,
            inputs=form_values,
        )

    except Exception as e:
        return render_template(
            "result.html",
            error=f"Something went wrong while making the prediction: {e}",
        )


if __name__ == "__main__":
    app.run(debug=True)
