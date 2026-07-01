import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for server/headless use (Flask, notebooks without display)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve


class ModelVisualizer:
    """
    ModelVisualizer Class

    Responsible for turning model results into visual plots — the
    "make ML debuggable" piece of the Explainable AI platform. Mirrors
    the role Phase 2's DataVisualizer plays for raw data, but focused on
    MODEL performance and behaviour instead of dataset exploration.

    Design notes:
        - Every plot_* method returns the matplotlib Figure it created
          (instead of just calling plt.show()), so callers — a Flask
          route, a Jupyter notebook, a report generator — can decide
          for themselves whether to display it, embed it in HTML, or
          save it to disk via save_plot().
        - show_plots() and save_plot() centralize the "what do I do with
          a finished figure" logic so individual plot methods stay
          focused purely on drawing.

    Methods:
        plot_confusion_matrix(cm)                       : Heatmap of a confusion matrix.
        plot_roc_curve(fpr, tpr)                          : ROC curve with AUC annotation.
        plot_precision_recall(precision, recall)          : Precision-Recall curve.
        plot_feature_importance(importance)               : Horizontal bar chart of feature importances.
        plot_learning_curve(history)                      : Train vs validation score over training size.
        plot_actual_vs_predicted(y_true, y_pred)          : Scatter plot for regression diagnostics.
        plot_distribution(data, column)                    : Histogram of a single column (e.g. residuals).
        plot_correlation_matrix(data)                      : Correlation heatmap of model inputs.
        plot_metrics_history(history)                      : Line chart of metrics across training epochs/folds.
        show_plots()                                       : Display all currently open figures.
        save_plot(fig, path)                                : Persist a figure to disk.
    """

    def __init__(self):
        sns.set_theme(style="whitegrid")

    # ── Classification: Confusion Matrix ─────────────────────────────────
    def plot_confusion_matrix(self, cm: pd.DataFrame, title: str = "Confusion Matrix"):
        """
        Draw a confusion matrix as an annotated heatmap.

        Args:
            cm (pd.DataFrame): Confusion matrix, e.g. the output of
                ModelEvaluator.confusion_matrix().
            title (str): Plot title.

        Returns:
            matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        fig.tight_layout()
        return fig

    # ── Classification: ROC Curve ────────────────────────────────────────
    def plot_roc_curve(self, fpr=None, tpr=None, y_true=None, y_score=None, title: str = "ROC Curve"):
        """
        Draw the ROC (Receiver Operating Characteristic) curve, with the
        Area Under the Curve (AUC) annotated in the legend.

        Can be called two ways:
            - With pre-computed fpr/tpr arrays, OR
            - With raw y_true/y_score, in which case fpr/tpr are computed
              internally via sklearn.metrics.roc_curve.

        Args:
            fpr (array-like, optional): False positive rates.
            tpr (array-like, optional): True positive rates.
            y_true (array-like, optional): Ground-truth binary labels.
            y_score (array-like, optional): Predicted probabilities/scores
                for the positive class.
            title (str): Plot title.

        Returns:
            matplotlib.figure.Figure
        """
        if fpr is None or tpr is None:
            if y_true is None or y_score is None:
                raise ValueError("Provide either (fpr, tpr) or (y_true, y_score).")
            fpr, tpr, _ = roc_curve(y_true, y_score)

        roc_auc_value = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC curve (AUC = {roc_auc_value:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random guess")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        fig.tight_layout()
        return fig

    # ── Classification: Precision-Recall Curve ───────────────────────────
    def plot_precision_recall(self, precision=None, recall=None, y_true=None, y_score=None, title: str = "Precision-Recall Curve"):
        """
        Draw the Precision-Recall curve — often more informative than
        ROC for imbalanced classification problems, since it focuses on
        the positive class without being affected by a large number of
        true negatives.

        Args:
            precision, recall (array-like, optional): Pre-computed curve points.
            y_true, y_score (array-like, optional): Raw inputs to compute
                the curve internally if precision/recall aren't given.
            title (str): Plot title.

        Returns:
            matplotlib.figure.Figure
        """
        if precision is None or recall is None:
            if y_true is None or y_score is None:
                raise ValueError("Provide either (precision, recall) or (y_true, y_score).")
            precision, recall, _ = precision_recall_curve(y_true, y_score)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, color="#16a34a", lw=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        fig.tight_layout()
        return fig

    # ── Feature importance ───────────────────────────────────────────────
    def plot_feature_importance(self, importance: pd.DataFrame, top_n: int = 15, title: str = "Feature Importance"):
        """
        Draw a horizontal bar chart of the top N most important features.

        Args:
            importance (pd.DataFrame): Must contain columns "feature" and
                "importance" (e.g. built from a model's
                feature_importances_ or coef_ attribute).
            top_n (int): Number of top features to display.
            title (str): Plot title.

        Returns:
            matplotlib.figure.Figure
        """
        if not {"feature", "importance"}.issubset(importance.columns):
            raise ValueError("importance DataFrame must contain 'feature' and 'importance' columns")

        ranked = importance.sort_values("importance", ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.35)))
        sns.barplot(data=ranked, x="importance", y="feature", color="#2563eb", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        return fig

    # ── Learning curve ───────────────────────────────────────────────────
    def plot_learning_curve(self, history: dict = None, model=None, X=None, y=None, cv: int = 5, title: str = "Learning Curve"):
        """
        Draw the learning curve — training score vs. validation score
        as a function of training set size. A key diagnostic for
        overfitting (large gap between train/val) vs underfitting (both
        scores low and converged).

        Can be called two ways:
            - With a pre-computed `history` dict, OR
            - With model/X/y, in which case sklearn's learning_curve()
              is run internally.

        Args:
            history (dict, optional): {
                "train_sizes": list[int],
                "train_scores": list[float],
                "val_scores": list[float]
            }
            model, X, y: Used to compute the learning curve internally
                when `history` isn't provided.
            cv (int): Number of cross-validation folds (only used when
                computing internally).
            title (str): Plot title.

        Returns:
            matplotlib.figure.Figure
        """
        if history is None:
            if model is None or X is None or y is None:
                raise ValueError("Provide either `history` or (model, X, y).")
            train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv=cv, n_jobs=-1)
            history = {
                "train_sizes": train_sizes.tolist(),
                "train_scores": train_scores.mean(axis=1).tolist(),
                "val_scores": val_scores.mean(axis=1).tolist(),
            }

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(history["train_sizes"], history["train_scores"], "o-", color="#2563eb", label="Training score")
        ax.plot(history["train_sizes"], history["val_scores"], "o-", color="#dc2626", label="Validation score")
        ax.set_xlabel("Training set size")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    # ── Regression diagnostic: actual vs predicted ──────────────────────
    def plot_actual_vs_predicted(self, y_true, y_pred, title: str = "Actual vs Predicted"):
        """
        Scatter plot of actual vs. predicted values for a regression
        model, with a diagonal reference line representing a perfect
        prediction (y_pred == y_true).

        Args:
            y_true (array-like): Ground-truth target values.
            y_pred (array-like): Model's predicted values.
            title (str): Plot title.

        Returns:
            matplotlib.figure.Figure
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, alpha=0.5, color="#2563eb", edgecolor="none")

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color="#dc2626", lw=2, linestyle="--", label="Perfect prediction")

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    # ── Generic distribution plot ────────────────────────────────────────
    def plot_distribution(self, data: pd.DataFrame, column: str, title: str = None):
        """
        Histogram (with KDE) of a single column — useful for inspecting
        residual distributions, prediction confidence scores, or any
        other model-derived numeric column.

        Args:
            data (pd.DataFrame): DataFrame containing the column to plot.
            column (str): Name of the column to plot.
            title (str, optional): Plot title (defaults to the column name).

        Returns:
            matplotlib.figure.Figure
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(data[column].dropna(), kde=True, color="#2563eb", ax=ax)
        ax.set_title(title or f"Distribution of {column}")
        ax.set_xlabel(column)
        fig.tight_layout()
        return fig

    # ── Correlation heatmap of model inputs ─────────────────────────────
    def plot_correlation_matrix(self, data: pd.DataFrame, title: str = "Feature Correlation Matrix"):
        """
        Correlation heatmap across all numeric columns of the given
        DataFrame — useful for spotting multicollinearity among model
        inputs, which can destabilize linear models' coefficients.

        Args:
            data (pd.DataFrame): Dataset (typically the model's feature set).
            title (str): Plot title.

        Returns:
            matplotlib.figure.Figure
        """
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValueError("No numeric columns found to compute a correlation matrix")

        corr = numeric_data.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        return fig

    # ── Metrics-over-time / metrics-over-folds line chart ───────────────
    def plot_metrics_history(self, history: dict, title: str = "Metrics History"):
        """
        Line chart tracking one or more metrics over successive training
        iterations, epochs, or cross-validation folds — useful for
        spotting trends like performance degradation or convergence.

        Args:
            history (dict): {metric_name: list[float], ...}, e.g.
                {"accuracy": [0.81, 0.85, 0.88], "loss": [0.5, 0.3, 0.2]}.
            title (str): Plot title.

        Returns:
            matplotlib.figure.Figure
        """
        if not history:
            raise ValueError("history dict cannot be empty")

        fig, ax = plt.subplots(figsize=(8, 5))
        for metric_name, values in history.items():
            ax.plot(range(1, len(values) + 1), values, marker="o", label=metric_name)

        ax.set_xlabel("Iteration / Fold")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    # ── Display / persistence helpers ───────────────────────────────────
    def show_plots(self) -> None:
        """
        Display all currently open matplotlib figures.

        Note: only works in environments with a display/backend that
        supports rendering (e.g. Jupyter). In a headless server context
        (e.g. Flask), use save_plot() instead and serve the saved image.

        Returns:
            None
        """
        plt.show()

    def save_plot(self, fig, path: str, dpi: int = 150) -> None:
        """
        Save a matplotlib figure to disk as an image file.

        Args:
            fig (matplotlib.figure.Figure): The figure to save (returned
                by any plot_* method above).
            path (str): Destination file path (e.g. "plots/roc_curve.png").
            dpi (int): Resolution of the saved image.

        Returns:
            None
        """
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
