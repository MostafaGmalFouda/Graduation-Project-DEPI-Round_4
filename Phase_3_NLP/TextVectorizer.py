from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class TextVectorizer:
    """
    Thin wrapper around scikit-learn's TF-IDF / Bag-of-Words vectorizers,
    so both the automatic (User) and manual (Developer) NLP flows share the
    exact same vectorization code path.

    method       : "tfidf" | "count"
    ngram_range  : e.g. (1, 1) unigrams, (1, 2) uni+bigrams
    max_features : cap on vocabulary size (keeps things fast on big corpora)
    """

    def __init__(self, method: str = "tfidf", ngram_range: tuple = (1, 1), max_features: int = 5000):
        self.method = method if method in ("tfidf", "count") else "tfidf"
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vectorizer = self._build()

    def _build(self):
        params = dict(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=1,
        )
        if self.method == "tfidf":
            return TfidfVectorizer(**params)
        return CountVectorizer(**params)

    def fit_transform(self, documents):
        return self.vectorizer.fit_transform(documents)

    def transform(self, documents):
        return self.vectorizer.transform(documents)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def top_terms(self, documents, top_n: int = 20) -> list:
        """
        Returns the top_n terms across the whole corpus, ranked by summed
        TF-IDF / count weight — used for keyword extraction & word-frequency
        charts.
        """
        matrix = self.fit_transform(documents)
        scores = matrix.sum(axis=0).A1
        terms = self.get_feature_names()
        ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
        return [{"term": t, "score": round(float(s), 4)} for t, s in ranked[:top_n]]