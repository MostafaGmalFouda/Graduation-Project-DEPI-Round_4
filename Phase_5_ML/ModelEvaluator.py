import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class ModelEvaluator:
    """
    ModelEvaluator Class

    Responsible for measuring how good a trained model actually is.
    Supports BOTH classification and regression metrics — the caller
    simply asks for whichever metric makes sense for their task type
    (e.g. accuracy() for classification, regression_metrics() for
    regression), so this single class serves both task families
    without needing two separate evaluator implementations.

    Design notes:
        - evaluate() runs predictions ONCE and caches them in
          self.y_pred, so calling multiple individual metric methods
          afterward (accuracy(), f1_score(), etc.) doesn't re-run
          inference repeatedly.
        - All metric methods gracefully degrade for the wrong task type
          by raising a clear error rather than producing a nonsensical
          number (e.g. calling .accuracy() on regression predictions).

    Attributes:
        model (object)      : The trained model to evaluate.
        X_test (DataFrame)  : Test features.
        y_test (Series)     : True test labels/values.
        y_pred (Series)     : Cached predictions (populated by evaluate()).

    Methods:
        evaluate()                 : Run predictions and cache them.
        accuracy()                 : Classification accuracy.
        precision()                : Classification precision (weighted).
        recall()                   : Classification recall (weighted).
        f1_score()                 : Classification F1 score (weighted).
        roc_auc()                  : ROC-AUC score (binary/multiclass-ovr).
        confusion_matrix()         : Confusion matrix as a DataFrame.
        classification_report()    : Full per-class precision/recall/F1 report.
        regression_metrics()       : MAE, MSE, RMSE, R².
        get_all_metrics()          : Convenience — returns every relevant
                                      metric for the detected task type.
    """

    def __init__(self, model=None, X_test: pd.DataFrame = None, y_test: pd.Series = None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None

    # ── Run predictions once, cache them ─────────────────────────────────
    def evaluate(self) -> pd.Series:
        """
        Run the model's predictions on X_test and cache them in
        self.y_pred for reuse by every other metric method.

        Returns:
            pd.Series (or np.ndarray): The predicted values/labels.
        """
        self._validate_ready()
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def _ensure_predictions(self) -> None:
        """Run evaluate() automatically if predictions haven't been cached yet."""
        if self.y_pred is None:
            self.evaluate()

    def _validate_ready(self) -> None:
        if self.model is None:
            raise RuntimeError("No model assigned. Set self.model before evaluating.")
        if self.X_test is None or self.y_test is None:
            raise RuntimeError("X_test and y_test must both be set before evaluating.")

    # ── Classification metrics ───────────────────────────────────────────
    def accuracy(self) -> float:
        """
        Fraction of predictions that exactly match the true label.

        Returns:
            float: Accuracy score in [0, 1].
        """
        self._ensure_predictions()
        return float(accuracy_score(self.y_test, self.y_pred))

    def precision(self, average: str = "weighted") -> float:
        """
        Precision = of everything the model predicted as a given class,
        how much was actually correct. "weighted" average accounts for
        class imbalance by weighting each class's score by its support.

        Args:
            average (str): scikit-learn averaging strategy
                ("weighted", "macro", "micro", "binary").

        Returns:
            float: Precision score in [0, 1].
        """
        self._ensure_predictions()
        return float(precision_score(self.y_test, self.y_pred, average=average, zero_division=0))

    def recall(self, average: str = "weighted") -> float:
        """
        Recall = of everything that actually belongs to a given class,
        how much did the model correctly catch.

        Args:
            average (str): scikit-learn averaging strategy.

        Returns:
            float: Recall score in [0, 1].
        """
        self._ensure_predictions()
        return float(recall_score(self.y_test, self.y_pred, average=average, zero_division=0))

    def f1_score(self, average: str = "weighted") -> float:
        """
        Harmonic mean of precision and recall — a single number that
        balances both, useful when there's a trade-off between the two
        (e.g. imbalanced classification).

        Args:
            average (str): scikit-learn averaging strategy.

        Returns:
            float: F1 score in [0, 1].
        """
        self._ensure_predictions()
        return float(f1_score(self.y_test, self.y_pred, average=average, zero_division=0))

    def roc_auc(self) -> float:
        """
        Area Under the ROC Curve — measures how well the model separates
        classes across all possible decision thresholds, not just the
        default 0.5 cutoff. Requires the model to support predict_proba().

        Returns:
            float: ROC-AUC score in [0, 1] (0.5 = random guessing).
        """
        self._validate_ready()
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                "roc_auc() requires a model with predict_proba() "
                "(e.g. LogisticRegression, RandomForestClassifier)."
            )

        proba = self.model.predict_proba(self.X_test)
        n_classes = proba.shape[1]

        if n_classes == 2:
            return float(roc_auc_score(self.y_test, proba[:, 1]))
        return float(roc_auc_score(self.y_test, proba, multi_class="ovr"))

    def confusion_matrix(self) -> pd.DataFrame:
        """
        Build the confusion matrix as a labeled DataFrame — rows are
        true classes, columns are predicted classes.

        Returns:
            pd.DataFrame: Square matrix labeled with the unique class
            values found in y_test.
        """
        self._ensure_predictions()
        labels = sorted(pd.unique(self.y_test))
        cm = confusion_matrix(self.y_test, self.y_pred, labels=labels)
        return pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in labels],
            columns=[f"pred_{l}" for l in labels],
        )

    def classification_report(self) -> dict:
        """
        Full per-class breakdown of precision, recall, F1 and support,
        plus overall accuracy and weighted/macro averages.

        Returns:
            dict: scikit-learn's classification_report, as a dict
            (output_dict=True).
        """
        self._ensure_predictions()
        return classification_report(self.y_test, self.y_pred, output_dict=True, zero_division=0)

    # ── Regression metrics ───────────────────────────────────────────────
    def regression_metrics(self) -> dict:
        """
        Compute the standard set of regression error metrics.

        Returns:
            dict: {
                "mae": float   -> Mean Absolute Error (average magnitude
                                   of errors, same units as the target),
                "mse": float   -> Mean Squared Error (penalizes large
                                   errors more heavily),
                "rmse": float  -> Root Mean Squared Error (same units as
                                   the target, easier to interpret than MSE),
                "r2": float    -> R² score (proportion of variance
                                   explained by the model; 1.0 = perfect)
            }
        """
        self._ensure_predictions()

        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(self.y_test, self.y_pred)),
        }

    # ── Convenience: everything at once ──────────────────────────────────
    def get_all_metrics(self, task_type: str = "classification") -> dict:
        """
        Convenience method that returns every relevant metric for the
        given task type in one call — handy for populating a results
        dashboard or report without calling each metric individually.

        Args:
            task_type (str): "classification" or "regression".

        Returns:
            dict: For classification -> {accuracy, precision, recall,
                  f1_score, confusion_matrix, classification_report,
                  roc_auc (if supported by the model)}.
                  For regression -> output of regression_metrics().
        """
        self._ensure_predictions()

        if task_type == "classification":
            metrics = {
                "accuracy": self.accuracy(),
                "precision": self.precision(),
                "recall": self.recall(),
                "f1_score": self.f1_score(),
                "confusion_matrix": self.confusion_matrix().to_dict(),
                "classification_report": self.classification_report(),
            }
            if hasattr(self.model, "predict_proba"):
                try:
                    metrics["roc_auc"] = self.roc_auc()
                except ValueError:
                    # roc_auc can fail on certain edge cases (e.g. only
                    # one class present in y_test) — skip gracefully.
                    metrics["roc_auc"] = None
            return metrics

        elif task_type == "regression":
            return self.regression_metrics()

        else:
            raise ValueError("task_type must be 'classification' or 'regression'")
