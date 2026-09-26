"""Data loading, training and evaluation for the heart-condition classifier.

Kept free of any GUI code so it can be tested and reused from scripts.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common.protocol import FEATURES

LABELS = ["Normal", "Abnormal"]
TARGET = "target"

ALGORITHMS = {
    "Decision Tree": lambda seed: DecisionTreeClassifier(random_state=seed),
    "Random Forest": lambda seed: RandomForestClassifier(n_estimators=200, random_state=seed),
    "KNN": lambda seed: KNeighborsClassifier(n_neighbors=10),
}


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray = field(repr=False)

    def as_dict(self):
        return {"Accuracy": self.accuracy, "Precision": self.precision,
                "Recall": self.recall, "F1 Score": self.f1}


def load_dataset(path, drop_duplicates: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing columns: {', '.join(missing)}")
    df = df[FEATURES + [TARGET]].fillna(0)
    if drop_duplicates:
        # The public heart.csv repeats most rows; keeping them lets the same
        # patient land in both train and test sets and inflates accuracy.
        df = df.drop_duplicates().reset_index(drop=True)
    return df


class HeartModel:
    """Holds the train/test split, the scaler and every trained classifier."""

    def __init__(self, df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
        self.seed = seed
        X = df[FEATURES].to_numpy(dtype=float)
        y = df[TARGET].to_numpy(dtype=int)
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y)
        # Fit the scaler on training data only to avoid test-set leakage.
        self.scaler = StandardScaler().fit(X_train)
        self.X_train = self.scaler.transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.models = {}
        self.metrics = {}

    def train(self, name: str) -> Metrics:
        clf = ALGORITHMS[name](self.seed).fit(self.X_train, self.y_train)
        pred = clf.predict(self.X_test)
        m = Metrics(
            accuracy=accuracy_score(self.y_test, pred) * 100,
            precision=precision_score(self.y_test, pred, average="macro", zero_division=0) * 100,
            recall=recall_score(self.y_test, pred, average="macro", zero_division=0) * 100,
            f1=f1_score(self.y_test, pred, average="macro", zero_division=0) * 100,
            confusion=confusion_matrix(self.y_test, pred, labels=[0, 1]),
        )
        self.models[name] = clf
        self.metrics[name] = m
        return m

    def best_model_name(self):
        if not self.metrics:
            return None
        return max(self.metrics, key=lambda n: (self.metrics[n].f1, self.metrics[n].accuracy))

    def predict(self, features, model_name=None):
        """Return (label, probability_of_abnormal or None, model_name)."""
        name = model_name or self.best_model_name()
        if name is None:
            raise RuntimeError("no model has been trained yet")
        if len(features) != len(FEATURES):
            raise ValueError(f"expected {len(FEATURES)} values, got {len(features)}")
        x = self.scaler.transform(np.asarray([features], dtype=float))
        clf = self.models[name]
        pred = int(clf.predict(x)[0])
        proba = float(clf.predict_proba(x)[0][1]) if hasattr(clf, "predict_proba") else None
        return LABELS[pred], proba, name
