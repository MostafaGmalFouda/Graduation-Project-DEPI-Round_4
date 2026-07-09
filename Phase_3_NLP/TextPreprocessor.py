import re
import pandas as pd

# NLTK lemmatization: reduces word variants (running/runs/ran -> run) so
# frequency/keyword counts reflect the REAL concept instead of splitting it
# across multiple surface forms. This is the fix for noisy/irrelevant
# keywords. Falls back to no-op (identity) if nltk/wordnet isn't available,
# so the app doesn't crash if the corpus hasn't been downloaded yet.
try:
    import nltk
    from nltk.stem import WordNetLemmatizer

    def _ensure_wordnet():
        for resource in ("corpora/wordnet", "corpora/omw-1.4"):
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.download(resource.split("/")[-1], quiet=True)

    _ensure_wordnet()
    _lemmatizer = WordNetLemmatizer()
except Exception:
    _lemmatizer = None


def _lemmatize(word: str) -> str:
    return _lemmatizer.lemmatize(word) if _lemmatizer else word


# Small, dependency-free English stopword list (avoids needing nltk downloads
# at runtime). Good enough for frequency / keyword analysis purposes.
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my
myself no nor not of off on once only or other ought our ours ourselves
out over own same shan't she she'd she'll she's should shouldn't so some
such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under
until up very was wasn't we we'd we'll we're we've were weren't what
what's when when's where where's which while who who's whom why why's
with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves it's im
""".split())

# Generic filler words that dominate keyword lists without adding analytical
# value, regardless of domain (unlike, say, "amazon" or "shipping" which are
# dataset-specific — add those separately per-project if needed).
EXTRA_STOPWORDS = set("""
one use used using would like get got also really much
just even well see saw make made know knew think thought
going go said say us thing things time though still way another
can could
""".split())

STOPWORDS = STOPWORDS.union(EXTRA_STOPWORDS)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\u0600-\u06FF\s]")
_MULTISPACE_RE = re.compile(r"\s+")
# Keeps the apostrophe INSIDE the token during tokenization, so contractions
# like "don't" / "wasn't" / "wouldn't" still read as "don't" / "wasn't" /
# "wouldn't" at the moment they're checked against STOPWORDS (which stores
# those exact apostrophe forms). The old approach stripped all punctuation
# (via _NON_ALPHA_RE) BEFORE the stopword check ever ran, so "don't" became
# "don t" first and could never match "don't" in STOPWORDS - leaving
# meaningless leftover fragments like "don", "wasn", "isn", "wouldn", "hasn"
# polluting word frequency, keywords, n-grams and the word cloud (especially
# bad on review-style text, which is full of negations).
_WORD_RE = re.compile(r"[a-zA-Z\u0600-\u06FF']+")


class TextPreprocessor:
    """
    Cleans and tokenizes raw text so it can be vectorized / analyzed.

    Handles both English and Arabic text reasonably (keeps Arabic unicode
    range, strips HTML/urls/punctuation/digits, lowercases Latin text,
    collapses whitespace). Stopword removal AND lemmatization only apply to
    English (WordNetLemmatizer silently no-ops on non-English tokens).
    """

    def __init__(self, texts):
        # Accept a pandas Series, list, or single string.
        if isinstance(texts, pd.Series):
            self.texts = texts.fillna("").astype(str).tolist()
        elif isinstance(texts, (list, tuple)):
            self.texts = [str(t) if t is not None else "" for t in texts]
        else:
            self.texts = [str(texts)]

    @staticmethod
    def clean_text(text: str, remove_stopwords: bool = True) -> str:
        if not text:
            return ""
        text = _HTML_TAG_RE.sub(" ", text)
        text = _URL_RE.sub(" ", text)
        text = text.lower()

        # Tokenize keeping apostrophes attached (e.g. "don't" stays "don't")
        # so contractions can be matched against STOPWORDS *before* the
        # apostrophe is stripped. Stripping punctuation first would turn
        # "don't" into "don t", which never matches "don't" in STOPWORDS.
        raw_tokens = _WORD_RE.findall(text)

        if remove_stopwords:
            tokens = []
            for raw in raw_tokens:
                if raw in STOPWORDS:
                    continue
                # Safe to drop the apostrophe now that the contraction check
                # above has already run (e.g. "dogs'" -> "dogs").
                t = raw.strip("'")
                # min length 2 -> 2, so 2-letter noise like "im"/"ok" is dropped too
                if not t or t in STOPWORDS or len(t) <= 2:
                    continue
                tokens.append(_lemmatize(t))
            return " ".join(tokens)

        return " ".join(raw.strip("'") for raw in raw_tokens)

    def clean_all(self, remove_stopwords: bool = True) -> list:
        return [self.clean_text(t, remove_stopwords=remove_stopwords) for t in self.texts]

    def tokenize(self, remove_stopwords: bool = True) -> list:
        """Returns a list of token-lists, one per document."""
        return [c.split() for c in self.clean_all(remove_stopwords=remove_stopwords)]

    def stats(self) -> dict:
        """Basic corpus-level statistics used for both the report and the RAG context."""
        raw_lengths = [len(t.split()) for t in self.texts]
        char_lengths = [len(t) for t in self.texts]
        vocab = set()
        for tokens in self.tokenize(remove_stopwords=True):
            vocab.update(tokens)

        return {
            "documents": len(self.texts),
            "avg_word_count": round(sum(raw_lengths) / len(raw_lengths), 2) if raw_lengths else 0,
            "max_word_count": max(raw_lengths) if raw_lengths else 0,
            "min_word_count": min(raw_lengths) if raw_lengths else 0,
            "avg_char_count": round(sum(char_lengths) / len(char_lengths), 2) if char_lengths else 0,
            "vocabulary_size": len(vocab),
            "empty_documents": sum(1 for t in self.texts if not t.strip()),
        }