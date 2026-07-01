import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class TextVectorizer:
    """
    TextVectorizer Class

    Responsible for converting cleaned text into numerical vector
    representations that downstream ML models, RAG components or
    similarity-search engines can consume. Supports the two classic
    vectorization methods from scikit-learn:

        - TF-IDF (Term Frequency–Inverse Document Frequency)
        - BoW   (Bag of Words / raw token counts)

    Design notes:
        - The fitted vectorizer object AND the corpus it was fitted on
          are cached on the instance, so:
            * transform_text() can project NEW, unseen text (e.g. a RAG
              query) into the SAME vector space without re-fitting.
            * get_top_terms() can recompute corpus-wide statistics on
              demand without needing the caller to pass the column again.

    Attributes:
        data (pd.DataFrame)      : Dataset containing the text column.
        _tfidf (TfidfVectorizer) : Cached fitted TF-IDF vectorizer (or None).
        _bow (CountVectorizer)   : Cached fitted BoW vectorizer (or None).
        _last_method (str)       : "tfidf" or "bow" — whichever was fitted last.
        _last_corpus (list[str]) : The exact list of texts used in the last fit,
                                    needed to recompute aggregate statistics.

    Methods:
        fit_tfidf(text_col, max_features)  : Fit + transform using TF-IDF.
        fit_bow(text_col, max_features)    : Fit + transform using Bag of Words.
        transform_text(texts)              : Transform NEW text using the
                                              already-fitted vectorizer.
        get_feature_names()                : Return the vocabulary learned by the
                                              last fitted vectorizer.
        get_top_terms(n)                   : Return the n highest average-weighted
                                              terms across the corpus (TF-IDF only).
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._tfidf = None          # fitted TfidfVectorizer
        self._bow = None            # fitted CountVectorizer
        self._last_method = None    # "tfidf" | "bow"
        self._last_corpus = None    # list[str] used in the last fit_*() call

    # ── TF-IDF vectorization ────────────────────────────────────────────
    def fit_tfidf(self, text_col: str, max_features: int = 500) -> np.ndarray:
        """
        Fit a TF-IDF vectorizer on the text column and return the
        resulting matrix.

        TF-IDF weighs words by how informative they are: common words
        that appear across almost every document get a LOW score, while
        words that are frequent in a specific document but rare overall
        get a HIGH score. This usually outperforms raw Bag-of-Words for
        tasks like search, similarity ranking and text classification.

        Args:
            text_col (str): Name of the column containing cleaned text.
            max_features (int): Cap on vocabulary size — keeps only the
                top N terms by corpus frequency. Prevents the matrix from
                exploding in size on large datasets.

        Returns:
            np.ndarray: Dense TF-IDF matrix of shape
            (n_documents, n_features).
        """
        if text_col not in self.data.columns:
            raise ValueError(f"Column '{text_col}' not found in data")

        texts = self.data[text_col].astype(str).tolist()

        self._tfidf = TfidfVectorizer(max_features=max_features)
        matrix = self._tfidf.fit_transform(texts)

        self._last_method = "tfidf"
        self._last_corpus = texts

        return matrix.toarray()

    # ── Bag of Words vectorization ──────────────────────────────────────
    def fit_bow(self, text_col: str, max_features: int = 500) -> np.ndarray:
        """
        Fit a Bag-of-Words (raw token count) vectorizer on the text
        column and return the resulting matrix.

        Simpler and faster than TF-IDF, and sometimes preferable when raw
        frequency matters more than relative importance (e.g. word-cloud
        generation, simple keyword counting).

        Args:
            text_col (str): Name of the column containing cleaned text.
            max_features (int): Cap on vocabulary size.

        Returns:
            np.ndarray: Dense count matrix of shape
            (n_documents, n_features).
        """
        if text_col not in self.data.columns:
            raise ValueError(f"Column '{text_col}' not found in data")

        texts = self.data[text_col].astype(str).tolist()

        self._bow = CountVectorizer(max_features=max_features)
        matrix = self._bow.fit_transform(texts)

        self._last_method = "bow"
        self._last_corpus = texts

        return matrix.toarray()

    # ── Transform new/unseen text using the fitted vectorizer ──────────
    def transform_text(self, texts) -> np.ndarray:
        """
        Transform new text (not part of the original fitting corpus)
        using the vectorizer that was most recently fitted.

        Why this matters for RAG: when a user asks a question, that
        question must be projected into the SAME vector space as the
        indexed documents — fitting a brand-new vectorizer on just the
        query would produce a meaningless, incompatible vector space.

        Args:
            texts (str | list[str]): A single string or a list of strings
                to transform.

        Returns:
            np.ndarray: Dense matrix of shape (n_texts, n_features),
            using the vocabulary learned during the last fit_* call.
        """
        if self._last_method is None:
            raise RuntimeError(
                "No vectorizer has been fitted yet. "
                "Call fit_tfidf() or fit_bow() first."
            )

        if isinstance(texts, str):
            texts = [texts]

        vectorizer = self._tfidf if self._last_method == "tfidf" else self._bow
        return vectorizer.transform(texts).toarray()

    # ── Inspect the learned vocabulary ──────────────────────────────────
    def get_feature_names(self) -> list:
        """
        Return the vocabulary (feature names) learned by the most
        recently fitted vectorizer.

        Returns:
            list[str]: Sorted list of vocabulary terms.
        """
        if self._last_method is None:
            raise RuntimeError(
                "No vectorizer has been fitted yet. "
                "Call fit_tfidf() or fit_bow() first."
            )

        vectorizer = self._tfidf if self._last_method == "tfidf" else self._bow
        return list(vectorizer.get_feature_names_out())

    # ── Top weighted terms (TF-IDF only) ─────────────────────────────────
    def get_top_terms(self, n: int = 20) -> list:
        """
        Return the top N terms ranked by their average TF-IDF score
        across the whole corpus — useful as a quick "what is this
        dataset about?" signal, or to populate a word-cloud / bar chart.

        Args:
            n (int): Number of top terms to return.

        Returns:
            list[tuple[str, float]]: (term, avg_score) pairs, sorted
            descending by score.
        """
        if self._last_method != "tfidf" or self._tfidf is None:
            raise RuntimeError(
                "get_top_terms() requires a fitted TF-IDF vectorizer. "
                "Call fit_tfidf() first."
            )

        feature_names = self._tfidf.get_feature_names_out()
        matrix = self._tfidf.transform(self._last_corpus)
        avg_scores = np.asarray(matrix.mean(axis=0)).ravel()

        ranked = sorted(
            zip(feature_names, avg_scores), key=lambda pair: pair[1], reverse=True
        )
        return ranked[:n]
