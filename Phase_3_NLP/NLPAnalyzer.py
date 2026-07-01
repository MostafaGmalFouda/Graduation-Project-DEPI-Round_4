import pandas as pd
import nltk
from nltk import ngrams
from textblob import TextBlob

# spaCy is used for Named Entity Recognition (NER). It is imported lazily
# inside extract_named_entities() so the rest of this module still works
# even on machines where the spaCy English model hasn't been downloaded yet.


class NLPAnalyzer:
    """
    NLPAnalyzer Class

    Responsible for extracting analytical insights from already-cleaned
    text (output of TextPreprocessor). While TextPreprocessor answers
    "how do I clean this text?", NLPAnalyzer answers "what can I learn
    from this text?".

    Attributes:
        data (pd.DataFrame): Dataset containing the text/token column(s).
        _nlp                : Lazily-loaded spaCy pipeline (None until first
                               NER call, to avoid paying the load cost when
                               NER is never used).

    Methods:
        compute_text_lengths(text_col)        : Word/char length stats per row.
        analyze_sentiment(text_col)            : Polarity + subjectivity per row.
        extract_named_entities(text_col)       : Named entities (people, orgs, locations...).
        extract_ngrams(token_col, n)           : Most frequent n-grams across the corpus.
        get_corpus_summary(text_col)           : High-level statistics for the whole column.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._nlp = None  # spaCy pipeline, loaded on first use

    # ── Text length statistics ──────────────────────────────────────────
    def compute_text_lengths(self, text_col: str) -> pd.DataFrame:
        """
        Compute character count, word count and average word length for
        every row in the text column.

        Why: Text length is a cheap but surprisingly useful feature —
        e.g. spam messages, fake reviews or bot-generated text often have
        distinct length distributions compared to genuine text.

        Args:
            text_col (str): Name of the column containing text.

        Returns:
            pd.DataFrame: Copy of self.data with 3 new columns:
                "<text_col>_char_count", "<text_col>_word_count",
                "<text_col>_avg_word_len".
        """
        if text_col not in self.data.columns:
            raise ValueError(f"Column '{text_col}' not found in data")

        df = self.data.copy()
        texts = df[text_col].astype(str)

        df[f"{text_col}_char_count"] = texts.str.len()
        df[f"{text_col}_word_count"] = texts.str.split().apply(len)
        df[f"{text_col}_avg_word_len"] = (
            df[f"{text_col}_char_count"] / df[f"{text_col}_word_count"].replace(0, 1)
        ).round(2)

        self.data = df
        return df

    # ── Sentiment analysis ──────────────────────────────────────────────
    def analyze_sentiment(self, text_col: str) -> pd.DataFrame:
        """
        Compute sentiment polarity and subjectivity for every row using
        TextBlob.

        - polarity    : float in [-1.0, 1.0] -> negative ... positive
        - subjectivity: float in [0.0, 1.0]  -> objective ... subjective
        - sentiment_label: human-readable bucket derived from polarity
          ("positive" / "neutral" / "negative")

        Why: Sentiment is one of the most common NLP features requested
        in EDA/BI tools — it turns free text (reviews, comments, tickets)
        into a quantifiable signal that can be plotted, filtered or
        correlated with other columns.

        Args:
            text_col (str): Name of the column containing text.

        Returns:
            pd.DataFrame: Copy of self.data with 3 new columns:
                "<text_col>_polarity", "<text_col>_subjectivity",
                "<text_col>_sentiment_label".
        """
        if text_col not in self.data.columns:
            raise ValueError(f"Column '{text_col}' not found in data")

        def _polarity(text: str) -> float:
            return TextBlob(str(text)).sentiment.polarity

        def _subjectivity(text: str) -> float:
            return TextBlob(str(text)).sentiment.subjectivity

        def _label(polarity: float) -> str:
            if polarity > 0.05:
                return "positive"
            elif polarity < -0.05:
                return "negative"
            return "neutral"

        df = self.data.copy()
        df[f"{text_col}_polarity"] = df[text_col].astype(str).apply(_polarity)
        df[f"{text_col}_subjectivity"] = df[text_col].astype(str).apply(_subjectivity)
        df[f"{text_col}_sentiment_label"] = df[f"{text_col}_polarity"].apply(_label)

        self.data = df
        return df

    # ── Named Entity Recognition ────────────────────────────────────────
    def extract_named_entities(self, text_col: str) -> dict:
        """
        Extract named entities (people, organizations, locations, dates,
        etc.) from every row using spaCy, and return both per-row and
        aggregated results.

        Why: NER turns unstructured text into structured, queryable
        information — e.g. finding every company name mentioned across
        thousands of support tickets or news articles.

        Lazily loads spaCy's small English model ("en_core_web_sm") on
        first call. If the model is not installed, raises a clear,
        actionable error instead of a cryptic stack trace.

        Args:
            text_col (str): Name of the column containing text.

        Returns:
            dict: {
                "per_row": list[list[dict]]  -> entities found in each row,
                                                  each dict = {"text", "label"},
                "entity_counts": dict[str, int] -> overall frequency of each
                                                     entity text across the corpus,
                "label_counts": dict[str, int]  -> frequency per entity TYPE
                                                     (PERSON, ORG, GPE, DATE, ...)
            }
        """
        if text_col not in self.data.columns:
            raise ValueError(f"Column '{text_col}' not found in data")

        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except OSError as exc:
                raise RuntimeError(
                    "spaCy model 'en_core_web_sm' is not installed.\n"
                    "Install it with:\n"
                    "    python -m spacy download en_core_web_sm"
                ) from exc

        per_row = []
        entity_counts: dict = {}
        label_counts: dict = {}

        for text in self.data[text_col].astype(str):
            doc = self._nlp(text)
            row_entities = []
            for ent in doc.ents:
                row_entities.append({"text": ent.text, "label": ent.label_})
                entity_counts[ent.text] = entity_counts.get(ent.text, 0) + 1
                label_counts[ent.label_] = label_counts.get(ent.label_, 0) + 1
            per_row.append(row_entities)

        return {
            "per_row": per_row,
            "entity_counts": dict(
                sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
            ),
            "label_counts": dict(
                sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
            ),
        }

    # ── N-gram frequency analysis ───────────────────────────────────────
    def extract_ngrams(self, token_col: str, n: int = 2) -> list:
        """
        Extract the most frequent n-grams (default: bigrams) across the
        entire corpus.

        Why: N-grams reveal recurring phrases ("customer service",
        "out of stock") that single-word frequency analysis misses —
        very useful for topic discovery and keyword extraction.

        Args:
            token_col (str): Name of the column containing a list of
                tokens (typically output of TextPreprocessor.tokenize()).
            n (int): Size of the n-gram (2 = bigram, 3 = trigram, ...).

        Returns:
            list[tuple]: Sorted list of (ngram_tuple, frequency), most
            frequent first.
        """
        if token_col not in self.data.columns:
            raise ValueError(f"Column '{token_col}' not found in data")
        if n < 1:
            raise ValueError("n must be >= 1")

        freq: dict = {}
        for tokens in self.data[token_col]:
            if not isinstance(tokens, list):
                continue
            for gram in ngrams(tokens, n):
                freq[gram] = freq.get(gram, 0) + 1

        return sorted(freq.items(), key=lambda x: x[1], reverse=True)

    # ── Corpus-level summary ────────────────────────────────────────────
    def get_corpus_summary(self, text_col: str) -> dict:
        """
        Produce a high-level statistical summary of the entire text
        column — useful as a quick health-check before deeper analysis.

        Args:
            text_col (str): Name of the column containing text.

        Returns:
            dict: {
                "total_documents": int,
                "empty_documents": int,
                "avg_word_count": float,
                "max_word_count": int,
                "min_word_count": int,
                "vocabulary_size": int  -> number of unique words across the corpus
            }
        """
        if text_col not in self.data.columns:
            raise ValueError(f"Column '{text_col}' not found in data")

        texts = self.data[text_col].astype(str)
        word_counts = texts.str.split().apply(len)
        vocabulary = set(word for text in texts for word in text.split())

        return {
            "total_documents": len(texts),
            "empty_documents": int((texts.str.strip() == "").sum()),
            "avg_word_count": round(float(word_counts.mean()), 2),
            "max_word_count": int(word_counts.max()),
            "min_word_count": int(word_counts.min()),
            "vocabulary_size": len(vocabulary),
        }
