from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FuzzyKNNConfig:
    k: int = 9
    m: float = 2.0
    epsilon: float = 1e-8


class FuzzyKNNClassifier:
    """
    Fuzzy KNN classifier for multiclass disease severity prediction.

    Each training sample has a fuzzy membership vector over classes.
    For simplicity, this implementation starts with crisp memberships
    (1.0 for true class, 0.0 otherwise) and computes fuzzy class
    memberships for query points based on distance-weighted neighbors.
    """

    def __init__(self, config: FuzzyKNNConfig | None = None):
        self.config = config or FuzzyKNNConfig()
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.class_to_index: Dict[str, int] = {}
        self.train_memberships: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FuzzyKNNClassifier":
        if len(X) == 0:
            raise ValueError("Training data is empty.")
        if len(X) != len(y):
            raise ValueError("X and y lengths do not match.")
        if self.config.k <= 0:
            raise ValueError("k must be a positive integer.")
        if self.config.m <= 1.0:
            raise ValueError("m must be > 1.0 for fuzzy weighting.")

        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        self.class_to_index = {label: idx for idx, label in enumerate(self.classes_)}

        # Crisp memberships as initialization.
        n_samples = len(self.y_train)
        n_classes = len(self.classes_)
        memberships = np.zeros((n_samples, n_classes), dtype=float)
        for i, label in enumerate(self.y_train):
            memberships[i, self.class_to_index[label]] = 1.0
        self.train_memberships = memberships
        return self

    def _check_is_fitted(self) -> None:
        if (
            self.X_train is None
            or self.y_train is None
            or self.classes_ is None
            or self.train_memberships is None
        ):
            raise RuntimeError("Model is not fitted. Call fit() first.")

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        assert self.X_train is not None
        diff = X[:, None, :] - self.X_train[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))

    def predict_membership(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        assert self.train_memberships is not None
        distances = self._pairwise_distances(np.asarray(X, dtype=float))

        n_queries = distances.shape[0]
        n_classes = self.train_memberships.shape[1]
        out = np.zeros((n_queries, n_classes), dtype=float)

        k_eff = min(self.config.k, distances.shape[1])
        exponent = 2.0 / (self.config.m - 1.0)

        for i in range(n_queries):
            d = distances[i]
            neighbor_idx = np.argpartition(d, k_eff - 1)[:k_eff]
            neighbor_dist = d[neighbor_idx]

            # Strongly favor exact matches while keeping numeric stability.
            weights = 1.0 / np.power(neighbor_dist + self.config.epsilon, exponent)
            weights_sum = np.sum(weights)
            if weights_sum <= 0:
                weights = np.ones_like(weights) / len(weights)
            else:
                weights = weights / weights_sum

            out[i] = np.sum(
                self.train_memberships[neighbor_idx] * weights[:, None],
                axis=0,
            )

        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        memberships = self.predict_membership(X)
        assert self.classes_ is not None
        best_idx = np.argmax(memberships, axis=1)
        return self.classes_[best_idx]


def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def default_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    return [c for c in df.columns if c != target_col]


def disease_risk_rule_based_label(df: pd.DataFrame) -> pd.Series:
    """
    Optional helper to synthesize a severity label when training data
    does not yet contain one. This is useful for bootstrapping demos.
    """

    risk_score = (
        (df.get("age", 0) > 65).astype(int) * 2
        + (df.get("systolic_bp", 0) > 150).astype(int) * 2
        + (df.get("heart_rate", 0) > 110).astype(int) * 2
        + (df.get("spo2", 100) < 92).astype(int) * 3
        + (df.get("temperature", 98.6) > 101.0).astype(int) * 1
        + (df.get("respiratory_rate", 16) > 24).astype(int) * 2
        + (df.get("has_diabetes", 0)).astype(int) * 1
        + (df.get("has_hypertension", 0)).astype(int) * 1
        + (df.get("prior_hospitalizations", 0) >= 2).astype(int) * 2
    )

    bins = [-1, 2, 5, 8, 100]
    labels = ["low", "moderate", "high", "critical"]
    return pd.cut(risk_score, bins=bins, labels=labels).astype(str)


def preprocess_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Tuple[float, float]]]:
    """
    Simple preprocessing:
    - one-hot encode categorical columns
    - min-max scale all features based on train statistics
    """
    x_train = train_df[feature_cols].copy()
    x_test = test_df[feature_cols].copy()

    all_data = pd.concat([x_train, x_test], axis=0, ignore_index=True)
    cat_cols = [c for c in all_data.columns if all_data[c].dtype == "object"]
    all_data = pd.get_dummies(all_data, columns=cat_cols, drop_first=False)

    x_train_enc = all_data.iloc[: len(x_train)].copy()
    x_test_enc = all_data.iloc[len(x_train) :].copy()

    scaling_info: Dict[str, Tuple[float, float]] = {}
    for col in x_train_enc.columns:
        col_min = float(x_train_enc[col].min())
        col_max = float(x_train_enc[col].max())
        scaling_info[col] = (col_min, col_max)
        denom = col_max - col_min
        if denom == 0:
            x_train_enc[col] = 0.0
            x_test_enc[col] = 0.0
        else:
            x_train_enc[col] = (x_train_enc[col] - col_min) / denom
            x_test_enc[col] = (x_test_enc[col] - col_min) / denom

    return (
        x_train_enc.to_numpy(dtype=float),
        x_test_enc.to_numpy(dtype=float),
        list(x_train_enc.columns),
        scaling_info,
    )
