from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from fuzzy_knn_health_risk import (
    FuzzyKNNClassifier,
    FuzzyKNNConfig,
    default_feature_columns,
    disease_risk_rule_based_label,
    preprocess_features,
)


@dataclass
class SavedModel:
    k: int
    m: float
    classes: List[str]
    feature_columns_raw: List[str]
    feature_columns_encoded: List[str]
    scaling_info: Dict[str, List[float]]
    X_train: List[List[float]]
    y_train: List[str]


def train_test_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    split = int(len(df) * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(
        drop=True
    )


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    rows = []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = int(np.sum(y_true == cls))
        rows.append((cls, precision, recall, f1, support))

    header = f"{'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"
    lines = [header, "-" * len(header)]
    for cls, p, r, f1, s in rows:
        lines.append(f"{cls:<12}{p:>10.3f}{r:>10.3f}{f1:>10.3f}{s:>10d}")
    return "\n".join(lines)


def ensure_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    if target_col not in out.columns:
        out[target_col] = disease_risk_rule_based_label(out)
    return out


def save_model(path: Path, payload: SavedModel) -> None:
    path.write_text(json.dumps(asdict(payload), indent=2))


def load_model(path: Path) -> SavedModel:
    obj = json.loads(path.read_text())
    return SavedModel(**obj)


def encode_and_scale_for_inference(
    df: pd.DataFrame, saved: SavedModel
) -> np.ndarray:
    x = df[saved.feature_columns_raw].copy()
    x = pd.get_dummies(x, columns=[c for c in x.columns if x[c].dtype == "object"])
    x = x.reindex(columns=saved.feature_columns_encoded, fill_value=0)

    for col in x.columns:
        col_min, col_max = saved.scaling_info[col]
        denom = col_max - col_min
        if denom == 0:
            x[col] = 0.0
        else:
            x[col] = (x[col] - col_min) / denom
    return x.to_numpy(dtype=float)


def run_train(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    out_model = Path(args.model_out)

    df = pd.read_csv(data_path)
    df = ensure_target(df, args.target)
    features = default_feature_columns(df, args.target)

    train_df, test_df = train_test_split(df, test_size=args.test_size, seed=args.seed)
    X_train, X_test, encoded_cols, scaling_info = preprocess_features(
        train_df, test_df, features
    )
    y_train = train_df[args.target].to_numpy()
    y_test = test_df[args.target].to_numpy()

    model = FuzzyKNNClassifier(FuzzyKNNConfig(k=args.k, m=args.m))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_membership = model.predict_membership(X_test)

    print("\n=== Evaluation ===")
    print(f"Accuracy: {accuracy(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Show a few prediction probabilities/memberships for interpretability.
    print("\n=== Sample fuzzy memberships (first 5) ===")
    class_names = list(model.classes_)
    for i in range(min(5, len(y_pred))):
        probs = {class_names[j]: float(y_membership[i, j]) for j in range(len(class_names))}
        print(f"true={y_test[i]:<10} pred={y_pred[i]:<10} memberships={probs}")

    scaling_serializable = {k: [v[0], v[1]] for k, v in scaling_info.items()}
    payload = SavedModel(
        k=args.k,
        m=args.m,
        classes=[str(c) for c in model.classes_],
        feature_columns_raw=features,
        feature_columns_encoded=encoded_cols,
        scaling_info=scaling_serializable,
        X_train=X_train.tolist(),
        y_train=[str(y) for y in y_train.tolist()],
    )
    save_model(out_model, payload)
    print(f"\nModel saved to: {out_model}")


def run_predict(args: argparse.Namespace) -> None:
    model_file = Path(args.model)
    patient_file = Path(args.patient)

    saved = load_model(model_file)
    patient_df = pd.read_csv(patient_file)
    X = encode_and_scale_for_inference(patient_df, saved)

    model = FuzzyKNNClassifier(FuzzyKNNConfig(k=saved.k, m=saved.m))
    model.fit(np.asarray(saved.X_train, dtype=float), np.asarray(saved.y_train))
    pred = model.predict(X)
    memb = model.predict_membership(X)

    print("\n=== Predictions ===")
    for i in range(len(patient_df)):
        probs = {saved.classes[j]: float(memb[i, j]) for j in range(len(saved.classes))}
        print(f"patient_index={i} severity={pred[i]} memberships={probs}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Healthcare risk assessment with fuzzy KNN."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_cmd = sub.add_parser("train", help="Train/evaluate model from CSV.")
    train_cmd.add_argument("--data", required=True, help="Path to training CSV.")
    train_cmd.add_argument(
        "--target",
        default="severity_level",
        help="Target column. Auto-generated if missing.",
    )
    train_cmd.add_argument("--k", type=int, default=9, help="Number of neighbors.")
    train_cmd.add_argument("--m", type=float, default=2.0, help="Fuzzifier (>1).")
    train_cmd.add_argument("--test-size", type=float, default=0.2, help="Test ratio.")
    train_cmd.add_argument("--seed", type=int, default=42, help="Random seed.")
    train_cmd.add_argument(
        "--model-out", default="model.json", help="Path to save model artifact."
    )
    train_cmd.set_defaults(func=run_train)

    pred_cmd = sub.add_parser("predict", help="Predict severity from patient CSV.")
    pred_cmd.add_argument("--model", required=True, help="Path to saved model JSON.")
    pred_cmd.add_argument("--patient", required=True, help="Path to patient CSV.")
    pred_cmd.set_defaults(func=run_predict)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
