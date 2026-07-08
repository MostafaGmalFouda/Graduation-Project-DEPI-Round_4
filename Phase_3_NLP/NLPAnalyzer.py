import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from Phase_3_NLP.TextPreprocessor import TextPreprocessor
from Phase_3_NLP.TextVectorizer import TextVectorizer

# Compact, dependency-free sentiment lexicon (rule-based fallback used when
# the dataset has no label column to train a real classifier on).
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

CLASSIFIER_FACTORY = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "naive_bayes": lambda: MultinomialNB(),
    "linear_svc": lambda: LinearSVC(),
}


class NLPAnalyzer:
    """
    Runs the full NLP analysis on a chosen text column of a DataFrame.

    Two flows share this same engine:
      - User mode   : analyze(auto=True) — sensible defaults, single call.
      - Developer mode: analyze(auto=False, ...) — full control over the
        vectorizer, classifier, ngram range, and (optionally) a label
        column to train real sentiment/topic classification instead of the
        rule-based lexicon fallback.
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

    # ── Trainable sentiment / text classification ────────────────────
    def train_classifier(
        self,
        label_column: str,
        method: str = "tfidf",
        ngram_range: tuple = (1, 1),
        classifier: str = "logistic_regression",
        test_size: float = 0.2,
    ) -> dict:
        if label_column not in self.df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset.")

        cleaned = self.preprocessor.clean_all(remove_stopwords=True)
        labels = self.df[label_column].astype(str)

        mask = pd.Series(cleaned).str.strip().astype(bool) & labels.notna()
        cleaned = pd.Series(cleaned)[mask].tolist()
        labels = labels[mask].tolist()

        if len(set(labels)) < 2:
            raise ValueError("Label column needs at least 2 distinct classes to train a classifier.")
        if len(labels) < 5:
            raise ValueError(
                f"Only {len(labels)} usable rows after cleaning — need at least 5 to train/test a classifier."
            )

        vectorizer = TextVectorizer(method=method, ngram_range=ngram_range, max_features=8000)
        X = vectorizer.fit_transform(cleaned)

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=42,
            stratify=labels if min(pd.Series(labels).value_counts()) > 1 else None,
        )

        clf_fn = CLASSIFIER_FACTORY.get(classifier, CLASSIFIER_FACTORY["logistic_regression"])
        model = clf_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )
        labels_order = sorted(set(labels))
        cm = confusion_matrix(y_test, preds, labels=labels_order)

        self.trained_model = model
        self.trained_vectorizer = vectorizer
        self.trained_label_order = labels_order

        return {
            "method": "trained_classifier",
            "classifier": classifier,
            "vectorizer": method,
            "ngram_range": ngram_range,
            "accuracy": round(float(acc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),
            "labels": labels_order,
            "test_size": len(y_test),
            "train_size": len(y_train),
        }

    def predict(self, texts: list) -> list:
        """Predict labels for new text(s) using the last trained classifier."""
        if not hasattr(self, "trained_model"):
            raise RuntimeError("No trained classifier available. Call train_classifier() first.")
        cleaned = [TextPreprocessor.clean_text(t) for t in texts]
        X = self.trained_vectorizer.transform(cleaned)
        return self.trained_model.predict(X).tolist()

    # ── Full pipeline (used by both User & Developer routes) ─────────
    def analyze(
        self,
        auto: bool = True,
        label_column: str = None,
        method: str = "tfidf",
        ngram_range: tuple = (1, 1),
        classifier: str = "logistic_regression",
        top_n: int = 20,
    ) -> dict:
        result = {
            "text_column": self.text_column,
            "statistics": self.text_statistics(),
            "keywords": self.top_keywords(method="tfidf" if auto else method,
                                           ngram_range=(1, 1) if auto else ngram_range,
                                           top_n=top_n),
            "word_frequency": self.word_frequency(top_n=top_n),
        }

        if label_column:
            result["sentiment"] = self.train_classifier(
                label_column=label_column,
                method=method,
                ngram_range=ngram_range,
                classifier=classifier,
            )
        else:
            result["sentiment"] = self.lexicon_sentiment()

        return result
