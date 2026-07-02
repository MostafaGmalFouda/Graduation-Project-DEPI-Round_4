import pandas as pd
from Phase_5_ML.ModelTrainer import ModelTrainer


class Predictor:
    """Wraps a saved {model, scaler, feature_columns, ...} bundle so app.py
    can run predictions without knowing the training internals."""

    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.scaler = bundle.get("scaler")
        self.feature_columns = bundle["feature_columns"]
        self.task_type = bundle["task_type"]
        self.model_name = bundle["model_name"]
        self.target_column = bundle["target_column"]

    @classmethod
    def from_file(cls, path: str):
        return cls(ModelTrainer.load_model(path))

    def predict_row(self, row: dict) -> dict:
        """row: {feature_name: value, ...} — missing features default to 0."""
        values = [[float(row.get(col, 0) or 0) for col in self.feature_columns]]
        X = pd.DataFrame(values, columns=self.feature_columns)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        prediction = self.model.predict(X)[0]

        result = {
            "target_column": self.target_column,
            "prediction": prediction.item() if hasattr(prediction, "item") else prediction,
        }

        if self.task_type == "classification" and hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(X)[0]
                classes = self.model.classes_
                result["probabilities"] = {
                    str(c): round(float(p), 4) for c, p in zip(classes, proba)
                }
            except Exception:
                pass

        return result
