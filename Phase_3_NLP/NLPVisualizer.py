import os
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Same cyberpunk theme used by Phase_2/DataVisualizer, for visual consistency.
BG_COLOR = "#0b0b18"
GRID_COLOR = "#1f1f3a"
TEXT_COLOR = "#e0e0e0"
ACCENT_PURPLE = "#a855f7"
ACCENT_CYAN = "#06d49d"
PALETTE = [ACCENT_CYAN, ACCENT_PURPLE, "#a248eccf", "#8959f9c8", "#0ee98ed8"]


def _neon_color_func(word=None, font_size=None, position=None, orientation=None,
                      font_path=None, random_state=None):
    return random.choice(PALETTE)


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

    def plot_bigrams(self, bigrams: list, filename: str = "bigrams.png") -> str:
        terms = [b["term"] for b in bigrams][::-1]
        counts = [b["score"] for b in bigrams][::-1]
        fig, ax = self._new_fig(figsize=(9, max(4, len(terms) * 0.32)))
        ax.barh(terms, counts, color=ACCENT_CYAN)
        ax.set_title("Top Bigrams")
        ax.set_xlabel("Count")
        return self._save(fig, filename)

    def plot_trigrams(self, trigrams: list, filename: str = "trigrams.png") -> str:
        terms = [t["term"] for t in trigrams][::-1]
        counts = [t["score"] for t in trigrams][::-1]
        fig, ax = self._new_fig(figsize=(9, max(4, len(terms) * 0.32)))
        ax.barh(terms, counts, color=ACCENT_PURPLE)
        ax.set_title("Top Trigrams")
        ax.set_xlabel("Count")
        return self._save(fig, filename)

    def plot_document_length_histogram(self, lengths: list, filename: str = "document_length.png") -> str:
        fig, ax = self._new_fig(figsize=(8, 5))
        bins = min(30, max(5, len(set(lengths)))) if lengths else 5
        ax.hist(lengths or [0], bins=bins, color=ACCENT_PURPLE, edgecolor=BG_COLOR)
        ax.set_title("Document Length Distribution")
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Documents")
        return self._save(fig, filename)

    def plot_vocab_before_after(self, before: int, after: int, filename: str = "vocab_before_after.png") -> str:
        fig, ax = self._new_fig(figsize=(5, 5))
        ax.bar(["Before", "After"], [before, after], color=[ACCENT_PURPLE, ACCENT_CYAN])
        ax.set_title("Vocabulary Size — Before vs After")
        ax.set_ylabel("Unique Terms")
        return self._save(fig, filename)

    def plot_avg_words_before_after(self, before: float, after: float, filename: str = "avg_words_before_after.png") -> str:
        fig, ax = self._new_fig(figsize=(5, 5))
        ax.bar(["Before", "After"], [before, after], color=[ACCENT_PURPLE, ACCENT_CYAN])
        ax.set_title("Average Words / Document — Before vs After")
        ax.set_ylabel("Avg Words")
        return self._save(fig, filename)

    def plot_wordcloud(self, word_frequency: list, filename: str = "wordcloud.png") -> str:
        """word_frequency: the same list produced by NLPAnalyzer.word_frequency()
        — [{"word": ..., "count": ...}, ...]"""
        freqs = {item["word"]: item["count"] for item in word_frequency if item.get("count", 0) > 0}
        if not freqs:
            freqs = {"no_data": 1}

        wc = WordCloud(
            width=900, height=450,
            background_color=BG_COLOR,
            color_func=_neon_color_func,
            prefer_horizontal=0.9,
        ).generate_from_frequencies(freqs)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor(BG_COLOR)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Word Cloud", color=ACCENT_CYAN)
        return self._save(fig, filename)