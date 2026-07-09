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

# Words/contractions that negate whatever sentiment word follows within
# _NEGATION_WINDOW tokens. Without this, "isn't good" / "wasn't great" /
# "didn't enjoy" score as POSITIVE (the scorer only ever sees "good" /
# "great" / "enjoy" and has no idea they were negated) — this single-handedly
# skews any review-style dataset (full of "isn't", "wasn't", "didn't")
# heavily toward positive, since negative reviews still use positive words,
# just negated ("wasn't great", "isn't worth it").
_NEGATORS = set("""
not no never neither nor cannot can't dont don't doesn't didn't isn't
aren't wasn't weren't hasn't haven't hadn't wouldn't couldn't shouldn't
won't hardly barely rarely scarcely
""".split())
_NEGATION_WINDOW = 3  # how many tokens ahead a negator can still flip


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

    # ── N-grams (bigrams / trigrams) ─────────────────────────────────
    def top_ngrams(self, n: int = 2, top_n: int = 20) -> list:
        """Top n-grams (n=2 bigrams, n=3 trigrams, ...) ranked by raw count,
        computed on the fully cleaned (stopwords removed, lemmatized) text so
        phrases like 'customer service' aren't diluted by filler words."""
        cleaned = self.preprocessor.clean_all(remove_stopwords=True)
        cleaned = [c for c in cleaned if c.strip()] or ["empty"]
        vectorizer = TextVectorizer(method="count", ngram_range=(n, n), max_features=5000)
        return vectorizer.top_terms(cleaned, top_n=top_n)

    def top_bigrams(self, top_n: int = 20) -> list:
        return self.top_ngrams(2, top_n=top_n)

    def top_trigrams(self, top_n: int = 20) -> list:
        return self.top_ngrams(3, top_n=top_n)

    # ── Document length distribution (for histogram) ─────────────────
    def document_length_distribution(self) -> list:
        """Raw word counts per document (before any cleaning), one entry
        per row — feeds a histogram of how long documents typically are."""
        return [len(t.split()) for t in self.texts]

    # ── Rule-based sentiment (no label column needed) ────────────────
    def lexicon_sentiment(self, n_examples: int = 3) -> dict:
        results = []
        scores = []  # signed strength per text: positive-word hits minus negative-word hits
        for text in self.texts:
            tokens = re.findall(r"[a-zA-Z']+", text.lower())
            pos = 0
            neg = 0
            negate_countdown = 0  # >0 means "the next sentiment word we hit gets flipped"
            for tok in tokens:
                if tok in _NEGATORS:
                    negate_countdown = _NEGATION_WINDOW
                    continue

                is_pos = tok in _POSITIVE_WORDS
                is_neg = tok in _NEGATIVE_WORDS
                if is_pos or is_neg:
                    flip = negate_countdown > 0
                    if is_pos:
                        neg += 1 if flip else 0
                        pos += 0 if flip else 1
                    else:
                        pos += 1 if flip else 0
                        neg += 0 if flip else 1
                    negate_countdown = 0  # negation "spends" itself on the first sentiment word it reaches
                elif negate_countdown > 0:
                    negate_countdown -= 1

            if pos > neg:
                label = "positive"
            elif neg > pos:
                label = "negative"
            else:
                label = "neutral"
            results.append(label)
            scores.append(pos - neg)

        series = pd.Series(results)
        counts = series.value_counts().to_dict()
        total = len(series)

        # Rank the ORIGINAL (uncleaned) texts by score so "what's the most
        # negative review?" / "most positive?" has an actual answer instead
        # of only an aggregate distribution. Ties broken by original order.
        # Empty/whitespace-only texts are excluded — they carry no signal.
        ranked = sorted(
            (
                (score, i, text)
                for i, (score, text) in enumerate(zip(scores, self.texts))
                if text and text.strip()
            ),
            key=lambda item: item[0],
        )

        def _examples(items):
            return [
                {"text": text[:300], "score": score}
                for score, _, text in items
            ]

        top_negative = _examples(ranked[:n_examples])
        top_positive = _examples(list(reversed(ranked[-n_examples:])) if ranked else [])

        return {
            "method": "lexicon",
            "labels": results,
            "scores": scores,
            "distribution": {k: int(v) for k, v in counts.items()},
            "distribution_pct": {k: round(v / total * 100, 1) for k, v in counts.items()} if total else {},
            "dominant_sentiment": series.value_counts().idxmax() if total else None,
            "top_negative": top_negative,
            "top_positive": top_positive,
        }

    # ── Raw text sample, kept so Phase 6 (RAG) can retrieve actual content
    #    instead of only a statistical summary ─────────────────────────
    def raw_text_samples(self, max_samples: int = 300) -> list:
        """Non-empty raw texts (not cleaned/stopword-stripped), capped so the
        session context file doesn't grow unbounded on huge datasets."""
        samples = [t for t in self.texts if t.strip()]
        return samples[:max_samples]

    # ── Before vs After cleaning comparison ───────────────────────────
    def before_after_comparison(self, n_samples: int = 5) -> dict:
        """
        Compares raw-ish text (HTML/URLs/punctuation stripped, but stopwords
        kept, nothing lemmatized — 'before') against the fully cleaned text
        used everywhere else in this class (stopwords removed, lemmatized —
        'after'). Returns aggregate stats for both plus a handful of
        side-by-side original/cleaned samples for the report table.
        """
        before_cleaned = self.preprocessor.clean_all(remove_stopwords=False)
        after_cleaned = self.preprocessor.clean_all(remove_stopwords=True)

        def _agg(cleaned_list):
            lengths = [len(c.split()) for c in cleaned_list]
            vocab = set()
            for c in cleaned_list:
                vocab.update(c.split())
            return {
                "avg_word_count": round(sum(lengths) / len(lengths), 2) if lengths else 0,
                "vocabulary_size": len(vocab),
            }

        samples = []
        for original, cleaned in zip(self.texts, after_cleaned):
            if original.strip():
                samples.append({"original": original[:300], "cleaned": cleaned[:300]})
            if len(samples) >= n_samples:
                break

        return {
            "before": _agg(before_cleaned),
            "after": _agg(after_cleaned),
            "samples": samples,
        }

    # ── Export cleaned text as a new column, for downstream reuse ────
    def export_dataframe(self) -> pd.DataFrame:
        """Returns a copy of the original dataframe with a new
        '<text_column>_clean' column added (stopwords removed, lemmatized),
        without touching the original column — for saving as
        processed_dataset.csv so Phase 5 (ML) or a report can reuse it."""
        cleaned = self.preprocessor.clean_all(remove_stopwords=True)
        out = self.df.copy()
        out[f"{self.text_column}_clean"] = cleaned
        return out

    # ── Full pipeline (used by both User & Developer routes) ─────────
    def analyze(
        self,
        auto: bool = True,
        method: str = "tfidf",
        ngram_range: tuple = (1, 1),
        top_n: int = 20,
        include_sentiment: bool = True,
        include_trigrams: bool = False,
    ) -> dict:
        ngram_top_n = min(top_n, 20)
        result = {
            "text_column": self.text_column,
            "statistics": self.text_statistics(),
            "keywords": self.top_keywords(method="tfidf" if auto else method,
                                           ngram_range=(1, 1) if auto else ngram_range,
                                           top_n=top_n),
            "word_frequency": self.word_frequency(top_n=top_n),
            "bigrams": self.top_bigrams(top_n=ngram_top_n),
            "document_lengths": self.document_length_distribution(),
            "before_after": self.before_after_comparison(),
        }

        if include_trigrams:
            result["trigrams"] = self.top_trigrams(top_n=ngram_top_n)

        if include_sentiment:
            result["sentiment"] = self.lexicon_sentiment()

        return result