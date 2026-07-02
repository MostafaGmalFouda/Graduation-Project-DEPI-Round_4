import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)


class ModelEvaluator:
    """Computes and formats metrics for a fitted model's test-set predictions."""

    @staticmethod
    def evaluate_classification(y_true, y_pred, labels=None) -> dict:
        labels = labels or sorted(set(list(y_true) + list(y_pred)))
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return {
            "task_type": "classification",
            "accuracy": round(float(acc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),
            "labels": [str(l) for l in labels],
        }

    @staticmethod
    def evaluate_regression(y_true, y_pred) -> dict:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_true, y_pred)
        return {
            "task_type": "regression",
            "mae": round(float(mae), 4),
            "mse": round(float(mse), 4),
            "rmse": round(rmse, 4),
            "r2_score": round(float(r2), 4),
        }

    @staticmethod
    def evaluate(task_type: str, y_true, y_pred, labels=None) -> dict:
        if task_type == "classification":
            return ModelEvaluator.evaluate_classification(y_true, y_pred, labels=labels)
        return ModelEvaluator.evaluate_regression(y_true, y_pred)

    @staticmethod
    def feature_importance(model, feature_columns: list) -> list:
        """Returns a sorted [{feature, importance}] list, or [] if the model
        doesn't expose one (e.g. KNN, SVM without a linear kernel)."""
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)

        if importances is None:
            return []

        ranked = sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": round(float(i), 4)} for f, i in ranked]
