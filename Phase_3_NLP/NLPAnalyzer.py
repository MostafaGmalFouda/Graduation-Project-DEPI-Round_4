import re
import pandas as pd

from Phase_3_NLP.TextPreprocessor import TextPreprocessor
from Phase_3_NLP.TextVectorizer import TextVectorizer

# Compact, dependency-free sentiment lexicon. This is a LIGHTWEIGHT, rule-based
# signal only — it is NOT a trained model, on purpose. Training real
# classifiers (Logistic Regression / Naive Bayes / SVC) on a label column is
# Phase 5's job (ML page); Phase 3 no longer duplicates that.
_POSITIVE_WORDS = set("""
good great excellent amazing awesome love loved loving best wonderful
fantastic perfect happy positive nice beautiful brilliant impressive
enjoy enjoyed enjoyable fun exciting excited superb outstanding
recommend recommended favorite favourite delightful pleasant satisfied
satisfying charming remarkable terrific fabulous
""".split())

_NEGATIVE_WORDS = set("""
bad terrible awful horrible hate hated hating worst poor disappointing
disappointed sad negative boring dull ugly annoying awful pathetic
mediocre waste worse fail failed failure broken useless
frustrating frustrated angry disgusting weak lousy dreadful painful
""".split())


class NLPAnalyzer:
    """
    Runs text-understanding analysis on a chosen text column of a DataFrame,
    and prepares it to feed Phase 6 (RAG chatbot).

    Two flows share this same engine:
      - User mode      : analyze(auto=True) — sensible defaults, single call.
      - Developer mode : analyze(auto=False, method=..., ngram_range=..., ...)
        — control over the vectorizer/keyword extraction only. No classifier
        training happens here anymore (see Phase 5 for that).
    """

    def __init__(self, df: pd.DataFrame, text_column: str):
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataset.")
        self.df = df
        self.text_column = text_column
        self.texts = df[text_column].fillna("").astype(str)
        self.preprocessor = TextPreprocessor(self.texts)

    # ── Text statistics ──────────────────────────────────────────────
    def text_statistics(self) -> dict:
        return self.preprocessor.stats()

    # ── Keyword extraction ───────────────────────────────────────────
    def top_keywords(self, method: str = "tfidf", ngram_range: tuple = (1, 1), top_n: int = 20) -> list:
        cleaned = self.preprocessor.clean_all(remove_stopwords=True)
        cleaned = [c for c in cleaned if c.strip()] or ["empty"]
        vectorizer = TextVectorizer(method=method, ngram_range=ngram_range, max_features=5000)
        return vectorizer.top_terms(cleaned, top_n=top_n)

    # ── Word frequency (raw counts, for bar chart) ───────────────────
    def word_frequency(self, top_n: int = 20) -> list:
        counts = {}
        for tokens in self.preprocessor.tokenize(remove_stopwords=True):
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"word": w, "count": c} for w, c in ranked]

    # ── Rule-based sentiment (no label column needed) ────────────────
    def lexicon_sentiment(self) -> dict:
        results = []
        for text in self.texts:
            tokens = re.findall(r"[a-zA-Z']+", text.lower())
            pos = sum(1 for t in tokens if t in _POSITIVE_WORDS)
            neg = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
            if pos > neg:
                label = "positive"
            elif neg > pos:
                label = "negative"
            else:
                label = "neutral"
            results.append(label)

        series = pd.Series(results)
        counts = series.value_counts().to_dict()
        total = len(series)
        return {
            "method": "lexicon",
            "labels": results,
            "distribution": {k: int(v) for k, v in counts.items()},
            "distribution_pct": {k: round(v / total * 100, 1) for k, v in counts.items()} if total else {},
            "dominant_sentiment": series.value_counts().idxmax() if total else None,
        }

    # ── Raw text sample, kept so Phase 6 (RAG) can retrieve actual content
    #    instead of only a statistical summary ─────────────────────────
    def raw_text_samples(self, max_samples: int = 300) -> list:
        """Non-empty raw texts (not cleaned/stopword-stripped), capped so the
        session context file doesn't grow unbounded on huge datasets."""
        samples = [t for t in self.texts if t.strip()]
        return samples[:max_samples]

    # ── Full pipeline (used by both User & Developer routes) ─────────
    def analyze(
        self,
        auto: bool = True,
        method: str = "tfidf",
        ngram_range: tuple = (1, 1),
        top_n: int = 20,
        include_sentiment: bool = True,
    ) -> dict:
        result = {
            "text_column": self.text_column,
            "statistics": self.text_statistics(),
            "keywords": self.top_keywords(method="tfidf" if auto else method,
                                           ngram_range=(1, 1) if auto else ngram_range,
                                           top_n=top_n),
            "word_frequency": self.word_frequency(top_n=top_n),
        }

        if include_sentiment:
            result["sentiment"] = self.lexicon_sentiment()

        return result