from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from fuzzy_knn_health_risk import (
    FuzzyKNNClassifier,
    FuzzyKNNConfig,
    default_feature_columns,
    disease_risk_rule_based_label,
    preprocess_features,
)
from main import SavedModel, load_model, save_model

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.json"
TRAIN_DATA = BASE_DIR / "sample_patient_records.csv"


def build_model_if_missing() -> SavedModel:
    if MODEL_PATH.exists():
        return load_model(MODEL_PATH)

    df = pd.read_csv(TRAIN_DATA)
    target_col = "severity_level"
    if target_col not in df.columns:
        df[target_col] = disease_risk_rule_based_label(df)

    features = default_feature_columns(df, target_col)
    # Use all available data for a tiny demo model.
    x_train, _, encoded_cols, scaling_info = preprocess_features(df, df, features)
    y_train = df[target_col].to_numpy()

    cfg = FuzzyKNNConfig(k=9, m=2.0)
    model = FuzzyKNNClassifier(cfg).fit(x_train, y_train)
    payload = SavedModel(
        k=cfg.k,
        m=cfg.m,
        classes=[str(c) for c in model.classes_],
        feature_columns_raw=features,
        feature_columns_encoded=encoded_cols,
        scaling_info={k: [v[0], v[1]] for k, v in scaling_info.items()},
        X_train=x_train.tolist(),
        y_train=[str(y) for y in y_train.tolist()],
    )
    save_model(MODEL_PATH, payload)
    return payload


def encode_single_patient(patient: Dict[str, str], saved: SavedModel) -> np.ndarray:
    row = {k: patient.get(k, "") for k in saved.feature_columns_raw}
    df = pd.DataFrame([row])

    numeric_fields = [
        "age",
        "systolic_bp",
        "diastolic_bp",
        "heart_rate",
        "respiratory_rate",
        "temperature",
        "spo2",
        "bmi",
        "has_diabetes",
        "has_hypertension",
        "has_heart_disease",
        "prior_hospitalizations",
        "smoker",
    ]
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce").fillna(0)

    df = pd.get_dummies(df, columns=[c for c in df.columns if df[c].dtype == "object"])
    df = df.reindex(columns=saved.feature_columns_encoded, fill_value=0)

    for col in df.columns:
        col_min, col_max = saved.scaling_info[col]
        denom = col_max - col_min
        if denom == 0:
            df[col] = 0.0
        else:
            df[col] = (df[col] - col_min) / denom
    return df.to_numpy(dtype=float)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    saved = build_model_if_missing()
    x = encode_single_patient(request.form.to_dict(), saved)
    model = FuzzyKNNClassifier(FuzzyKNNConfig(k=saved.k, m=saved.m))
    model.fit(np.asarray(saved.X_train, dtype=float), np.asarray(saved.y_train))

    memberships = model.predict_membership(x)[0]
    pred = model.predict(x)[0]
    details = []
    for i, label in enumerate(saved.classes):
        details.append((label, round(float(memberships[i]), 4)))
    details.sort(key=lambda item: item[1], reverse=True)

    return render_template("result.html", severity=pred, memberships=details)


if __name__ == "__main__":
    build_model_if_missing()
    app.run(debug=True)
