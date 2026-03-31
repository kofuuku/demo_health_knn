# Healthcare Risk Assessment (Fuzzy KNN)

This project predicts disease severity levels (`low`, `moderate`, `high`, `critical`) using a fuzzy KNN classifier on patient vitals and historical medical records.

## Features

- Fuzzy KNN implementation from scratch (distance-weighted fuzzy memberships)
- Multiclass disease severity prediction
- Handles mixed numeric/categorical patient attributes
- CLI workflow for model training/evaluation and inference
- Sample datasets for quick testing

## Input Data

Expected columns in CSV (example):

- `age`, `gender`
- `systolic_bp`, `diastolic_bp`
- `heart_rate`, `respiratory_rate`, `temperature`, `spo2`
- `bmi`
- `has_diabetes`, `has_hypertension`, `has_heart_disease`
- `prior_hospitalizations`, `smoker`
- Optional target: `severity_level`

If `severity_level` is not present, it is auto-generated using a transparent rule-based risk scoring function for demo/training bootstrapping.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train and Evaluate

```bash
python3 main.py train \
  --data sample_patient_records.csv \
  --target severity_level \
  --k 9 \
  --m 2.0 \
  --model-out model.json
```

This command prints:

- Accuracy
- Per-class precision/recall/F1 report
- Fuzzy class memberships for sample test predictions

## Predict on New Patients

```bash
python3 main.py predict \
  --model model.json \
  --patient sample_new_patients.csv
```

Output includes predicted severity + class membership scores for each patient.

## Demo Website (Simple)

Run:

```bash
pip install -r requirements.txt
python3 app.py
```

Then open `http://127.0.0.1:5000` in your browser and enter patient details.

## Notes for Real Clinical Use

- Replace demo labels with clinician-validated severity outcomes.
- Validate with cross-validation and external cohorts.
- Add calibration, confidence thresholds, and model monitoring.
- Ensure regulatory, privacy, and safety requirements before deployment.
