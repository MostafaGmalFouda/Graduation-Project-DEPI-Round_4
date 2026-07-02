import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from Phase_5_ML.ModelFactory import ModelFactory


class ModelTrainer:
    """
    Prepares X/y from a clean DataFrame, detects (or accepts) the task
    type, and fits a single model. Kept deliberately separate from
    MLPipeline so Developer Mode can call each step individually if needed.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ── Target column guessing (used when User Mode has no explicit target) ──
    TARGET_NAME_HINTS = (
        "target", "label", "class", "y", "outcome", "result", "survived",
        "price", "sales", "score", "rating", "churn", "diagnosis",
    )

    def guess_target_column(self) -> str:
        cols = list(self.df.columns)

        # 1) Name-based hint (case-insensitive exact or substring match)
        for col in cols:
            if col.lower() in self.TARGET_NAME_HINTS:
                return col
        for col in cols:
            lc = col.lower()
            if any(hint in lc for hint in self.TARGET_NAME_HINTS):
                return col

        # 2) Heuristic: prefer a low-cardinality column (looks categorical /
        #    classification-friendly) that isn't an obvious ID column.
        best_col, best_score = None, None
        n = len(self.df)
        for col in cols:
            nunique = self.df[col].nunique(dropna=True)
            if nunique <= 1 or nunique >= n * 0.9:
                continue  # constant or ID-like — not a usable target
            # score: prefer 2-20 unique values (classification sweet spot)
            score = abs(nunique - 5)
            if best_score is None or score < best_score:
                best_col, best_score = col, score

        # 3) Fallback: last column
        return best_col or cols[-1]

    def detect_task_type(self, target_column: str) -> str:
        series = self.df[target_column]
        if series.dtype.kind in "iufc":
            nunique = series.nunique(dropna=True)
            # Numeric but few distinct values → treat as classification
            # (e.g. 0/1 flags, encoded categories with small cardinality).
            if nunique <= max(10, int(len(series) * 0.02)):
                return "classification"
            return "regression"
        return "classification"

    def prepare_xy(self, target_column: str, feature_columns: list = None):
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        y = self.df[target_column]
        if feature_columns:
            feature_columns = [c for c in feature_columns if c in self.df.columns and c != target_column]
        else:
            feature_columns = [c for c in self.df.columns if c != target_column]

        X = self.df[feature_columns].copy()

        # Keep it robust: any leftover non-numeric feature columns get
        # dropped (the Phase 1 pipeline already encodes categoricals, but
        # this guards against columns added after that step).
        non_numeric = [c for c in X.columns if X[c].dtype.kind not in "iufcb"]
        if non_numeric:
            X = X.drop(columns=non_numeric)

        X = X.fillna(X.median(numeric_only=True))
        y = y.fillna(y.mode().iloc[0] if not y.mode().empty else 0)

        return X, y, list(X.columns)

    def split(self, X, y, test_size: float = 0.2, stratify_classification: bool = True, task_type: str = "regression"):
        stratify = y if (task_type == "classification" and stratify_classification and y.nunique() > 1
                          and y.value_counts().min() > 1) else None
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)

    def train_one(self, task_type: str, model_name: str, X_train, y_train, scale: bool = False, **params):
        scaler = None
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

        model = ModelFactory.build(task_type, model_name, **params)
        model.fit(X_train, y_train)
        return model, scaler

    def cross_validate(self, task_type: str, model_name: str, X, y, cv: int = 5, **params):
        model = ModelFactory.build(task_type, model_name, **params)
        scoring = "accuracy" if task_type == "classification" else "r2"
        try:
            scores = cross_val_score(model, X, y, cv=min(cv, max(2, y.value_counts().min()) if task_type == "classification" else cv), scoring=scoring)
            return float(np.mean(scores)), float(np.std(scores))
        except Exception:
            return None, None

    @staticmethod
    def save_model(model, scaler, path: str, feature_columns: list, task_type: str, model_name: str, target_column: str, labels=None):
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "task_type": task_type,
            "model_name": model_name,
            "target_column": target_column,
            "labels": labels,
        }, path)

    @staticmethod
    def load_model(path: str):
        return joblib.load(path)
