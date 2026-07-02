import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Same cyberpunk theme used by Phase_2/DataVisualizer, for visual consistency.
BG_COLOR = "#0b0b18"
GRID_COLOR = "#1f1f3a"
TEXT_COLOR = "#e0e0e0"
ACCENT_PURPLE = "#a855f7"
ACCENT_CYAN = "#06d49d"
PALETTE = [ACCENT_CYAN, ACCENT_PURPLE, "#a248eccf", "#8959f9c8", "#0ee98ed8"]


class NLPVisualizer:
    def __init__(self, plots_dir: str = None):
        self.plots_dir = plots_dir or os.path.join(BASE_DIR, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def _new_fig(self, figsize=(9, 5)):
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

    def plot_word_frequency(self, word_freq: list, filename: str = "word_frequency.png") -> str:
        words = [w["word"] for w in word_freq][::-1]
        counts = [w["count"] for w in word_freq][::-1]
        fig, ax = self._new_fig(figsize=(9, max(4, len(words) * 0.32)))
        ax.barh(words, counts, color=ACCENT_CYAN)
        ax.set_title("Top Words by Frequency")
        ax.set_xlabel("Count")
        return self._save(fig, filename)

    def plot_keywords(self, keywords: list, filename: str = "keywords.png") -> str:
        terms = [k["term"] for k in keywords][::-1]
        scores = [k["score"] for k in keywords][::-1]
        fig, ax = self._new_fig(figsize=(9, max(4, len(terms) * 0.32)))
        ax.barh(terms, scores, color=ACCENT_PURPLE)
        ax.set_title("Top Keywords (TF-IDF weighted)")
        ax.set_xlabel("Score")
        return self._save(fig, filename)

    def plot_sentiment_distribution(self, distribution: dict, filename: str = "sentiment_distribution.png") -> str:
        labels = list(distribution.keys())
        values = list(distribution.values())
        colors = {"positive": ACCENT_CYAN, "negative": "#f43f5e", "neutral": "#a855f7"}
        bar_colors = [colors.get(l, ACCENT_PURPLE) for l in labels]
        fig, ax = self._new_fig(figsize=(6, 5))
        ax.bar(labels, values, color=bar_colors)
        ax.set_title("Sentiment Distribution")
        ax.set_ylabel("Documents")
        return self._save(fig, filename)

    def plot_confusion_matrix(self, cm, labels, filename: str = "nlp_confusion_matrix.png") -> str:
        cm = np.array(cm)
        fig, ax = self._new_fig(figsize=(6, 5))
        im = ax.imshow(cm, cmap="mako" if False else "viridis")
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
