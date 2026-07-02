import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BG_COLOR = "#0b0b18"
GRID_COLOR = "#1f1f3a"
TEXT_COLOR = "#e0e0e0"
ACCENT_PURPLE = "#a855f7"
ACCENT_CYAN = "#06d49d"


class ModelVisualizer:
    def __init__(self, plots_dir: str = None):
        self.plots_dir = plots_dir or os.path.join(BASE_DIR, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def _new_fig(self, figsize=(7, 5.5)):
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(ACCENT_CYAN)
        return fig, ax

    def _save(self, fig, filename):
        path = os.path.join(self.plots_dir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=150, facecolor=BG_COLOR)
        plt.close(fig)
        return path

    def plot_confusion_matrix(self, cm, labels, filename: str = "confusion_matrix.png") -> str:
        cm = np.array(cm)
        fig, ax = self._new_fig(figsize=(6, 5.5))
        ax.imshow(cm, cmap="viridis")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", color=TEXT_COLOR)
        ax.set_yticklabels(labels, color=TEXT_COLOR)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        return self._save(fig, filename)

    def plot_feature_importance(self, importance: list, filename: str = "feature_importance.png") -> str:
        items = importance[:15][::-1]
        feats = [i["feature"] for i in items]
        vals = [i["importance"] for i in items]
        fig, ax = self._new_fig(figsize=(8, max(4, len(feats) * 0.35)))
        ax.barh(feats, vals, color=ACCENT_CYAN)
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance")
        return self._save(fig, filename)

    def plot_actual_vs_predicted(self, y_true, y_pred, filename: str = "actual_vs_predicted.png") -> str:
        fig, ax = self._new_fig(figsize=(6.5, 6))
        ax.scatter(y_true, y_pred, alpha=0.6, color=ACCENT_PURPLE, edgecolors="none")
        lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
        ax.plot(lims, lims, color=ACCENT_CYAN, linestyle="--", linewidth=1.5)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        return self._save(fig, filename)

    def plot_model_comparison(self, results: list, metric: str = "score", filename: str = "model_comparison.png") -> str:
        """results: [{'model': name, 'score': value}, ...] — used by AutoML (User Mode)."""
        names = [r["model"] for r in results]
        scores = [r[metric] for r in results]
        colors = [ACCENT_CYAN if s == max(scores) else ACCENT_PURPLE for s in scores]
        fig, ax = self._new_fig(figsize=(8, 5))
        ax.bar(names, scores, color=colors)
        ax.set_title("Model Comparison")
        ax.set_ylabel(metric)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        return self._save(fig, filename)
