import re
import string
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── One-time NLTK resource download ──────────────────────────────────────────
# These resources are required for stopword removal, tokenization and
# lemmatization. We download them quietly and only once; if they already
# exist locally, nltk skips the download automatically.
for _resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(_resource)
    except LookupError:
        nltk.download(_resource, quiet=True)


class TextPreprocessor:
    """
    TextPreprocessor Class

    Responsible for cleaning and normalizing raw text columns before any
    NLP analysis, vectorization or RAG indexing happens. This is the first
    stage of the NLP pipeline (Phase 3) — every other module
    (NLPAnalyzer, TextVectorizer, TextChunker) expects text that has
    already passed through here.

    Design notes:
        - Every method receives a DataFrame + the name of the text column,
          and returns a NEW DataFrame with an extra processed column
          (the original column is never mutated in place). This keeps the
          pipeline composable: each step can be chained or skipped freely.
        - All heavy NLP resources (stopwords list, lemmatizer) are loaded
          ONCE in __init__ instead of inside every method call, for
          performance on large datasets.

    Attributes:
        data (pd.DataFrame)      : The dataset holding the text column(s).
        language (str)           : Language used for stopwords (default "english").
        stop_words (set)         : Cached set of stopwords for fast lookup.
        lemmatizer (WordNetLemmatizer) : Cached lemmatizer instance.

    Methods:
        lowercase(text_col)                    : Lowercase all text.
        remove_punctuation_and_urls(text_col)  : Strip punctuation, URLs, emails, numbers.
        remove_stopwords(text_col)             : Remove common stopwords.
        tokenize(text_col)                     : Split text into list-of-words column.
        apply_lemmatization(token_col)         : Reduce tokens to their dictionary root form.
        clean_pipeline(text_col)               : Run the full recommended cleaning sequence in one call.
    """

    def __init__(self, data: pd.DataFrame, language: str = "english"):
        self.data = data
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    # ── Stage 1: Lowercasing ─────────────────────────────────────────────
    def lowercase(self, text_col: str) -> pd.DataFrame:
        """
        Lowercase all text in the given column.

        Why: Avoids treating "Data" and "data" as two different tokens
        further down the pipeline (TF-IDF, NER, sentiment, etc.).

        Args:
            text_col (str): Name of the column containing raw text.

        Returns:
            pd.DataFrame: Copy of self.data with the column lowercased.
        """
        if text_col not in self.data.columns:
            raise ValueError(f"Column '{text_col}' not found in data")

        df = self.data.copy()
        df[text_col] = df[text_col].astype(str).str.lower()
        self.data = df
        return df

    # ── Stage 2: Punctuation / URL / noise removal ──────────────────────
    def remove_punctuation_and_urls(self, text_col: str) -> pd.DataFrame:
        """
        Strip punctuation, URLs, email addresses, HTML tags and standalone
        numbers from the text column.

        Why: These tokens rarely carry semantic meaning for classic NLP
        tasks (sentiment, NER, TF-IDF) and only add noise / inflate the
        vocabulary size unnecessarily.

        Args:
            text_col (str): Name of the column containing text.

        Returns:
            pd.DataFrame: Copy of self.data with the column cleaned.
        """
        if text_col not in self.data.columns:
            raise ValueError(f"Column '{text_col}' not found in data")

        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        email_pattern = re.compile(r"\S+@\S+")
        html_pattern = re.compile(r"<.*?>")
        number_pattern = re.compile(r"\b\d+\b")
        punctuation_table = str.maketrans("", "", string.punctuation)

        def _clean(text: str) -> str:
            text = url_pattern.sub(" ", text)
            text = email_pattern.sub(" ", text)
            text = html_pattern.sub(" ", text)
            text = number_pattern.sub(" ", text)
            text = text.translate(punctuation_table)
            # collapse multiple spaces left behind by the substitutions
            return re.sub(r"\s+", " ", text).strip()

        df = self.data.copy()
        df[text_col] = df[text_col].astype(str).apply(_clean)
        self.data = df
        return df

    # ── Stage 3: Stopword removal ───────────────────────────────────────
    def remove_stopwords(self, text_col: str) -> pd.DataFrame:
        """
        Remove common stopwords (e.g. "the", "is", "and") from the text
        column.

        Why: Stopwords appear in almost every document, so they add no
        discriminative value to vectorization (TF-IDF/BoW) and just dilute
        the signal in word-frequency-based analysis.

        Args:
            text_col (str): Name of the column containing text.

        Returns:
            pd.DataFrame: Copy of self.data with stopwords removed.
        """
        if text_col not in self.data.columns:
            raise ValueError(f"Column '{text_col}' not found in data")

        def _remove(text: str) -> str:
            words = text.split()
            filtered = [w for w in words if w.lower() not in self.stop_words]
            return " ".join(filtered)

        df = self.data.copy()
        df[text_col] = df[text_col].astype(str).apply(_remove)
        self.data = df
        return df

    # ── Stage 4: Tokenization ───────────────────────────────────────────
    def tokenize(self, text_col: str) -> pd.DataFrame:
        """
        Split each row of text into a list of tokens (words).

        Why: Downstream steps like lemmatization, n-gram extraction and
        some vectorizers operate on token lists rather than raw strings.

        Args:
            text_col (str): Name of the column containing text.

        Returns:
            pd.DataFrame: Copy of self.data with a new "<text_col>_tokens"
            column containing list-of-string tokens.
        """
        if text_col not in self.data.columns:
            raise ValueError(f"Column '{text_col}' not found in data")

        df = self.data.copy()
        df[f"{text_col}_tokens"] = df[text_col].astype(str).apply(word_tokenize)
        self.data = df
        return df

    # ── Stage 5: Lemmatization ──────────────────────────────────────────
    def apply_lemmatization(self, token_col: str) -> pd.DataFrame:
        """
        Reduce each token to its dictionary base form
        (e.g. "running" -> "run", "better" -> "good").

        Why: Lemmatization groups inflected forms of a word together,
        which shrinks vocabulary size and improves the quality of
        downstream vectorization and topic/keyword analysis compared to
        treating every inflection as a separate word.

        Args:
            token_col (str): Name of the column containing a list of tokens
                (typically the output of tokenize()).

        Returns:
            pd.DataFrame: Copy of self.data with a new
            "<token_col>_lemmatized" column.
        """
        if token_col not in self.data.columns:
            raise ValueError(f"Column '{token_col}' not found in data")

        def _lemmatize(tokens) -> list:
            if not isinstance(tokens, list):
                return tokens
            return [self.lemmatizer.lemmatize(tok) for tok in tokens]

        df = self.data.copy()
        df[f"{token_col}_lemmatized"] = df[token_col].apply(_lemmatize)
        self.data = df
        return df

    # ── Convenience: Full pipeline in one call ──────────────────────────
    def clean_pipeline(self, text_col: str) -> pd.DataFrame:
        """
        Run the full recommended cleaning sequence on a text column in one
        call:
            lowercase -> remove_punctuation_and_urls -> remove_stopwords
            -> tokenize -> apply_lemmatization

        This is a convenience wrapper for the common case where the
        caller just wants "clean text, ready for analysis" without
        manually chaining every stage.

        Args:
            text_col (str): Name of the raw text column to process.

        Returns:
            pd.DataFrame: Fully processed DataFrame with intermediate
            columns: text_col (cleaned), "<text_col>_tokens",
            "<text_col>_tokens_lemmatized".
        """
        self.lowercase(text_col)
        self.remove_punctuation_and_urls(text_col)
        self.remove_stopwords(text_col)
        self.tokenize(text_col)
        self.apply_lemmatization(f"{text_col}_tokens")
        return self.data

    def get_data(self) -> pd.DataFrame:
        """
        Return the current state of the processed data.

        Returns:
            pd.DataFrame
        """
        return self.data
