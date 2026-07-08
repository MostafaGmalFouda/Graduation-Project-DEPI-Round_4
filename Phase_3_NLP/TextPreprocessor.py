import re
import pandas as pd

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

one two three also get gets got getting go goes going gone went like
likes liked just really actually quite pretty rather even well much many
lot lots way ways thing things something anything everything nothing
someone anyone everyone else still yet already almost always never ever
maybe perhaps though although however therefore thus hence upon within
without across among around behind beside besides towards toward via
per etc us let us make made making makes take takes taking took put puts
putting say says said saying tell tells told telling see sees seeing saw
seen look looks looked looking know knows knew knowing think thinks
thought thinking come comes came coming want wants wanted wanting use
uses used using find finds found finding give gives gave giving
new old first last long own am is can could would should will shall must
don doesn didn wasn isn aren weren wouldn couldn shouldn won hasn haven
ve re ll nt
""".split())

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\u0600-\u06FF\s]")
_MULTISPACE_RE = re.compile(r"\s+")


class TextPreprocessor:
    """
    Cleans and tokenizes raw text so it can be vectorized / analyzed.

    Handles both English and Arabic text reasonably (keeps Arabic unicode
    range, strips HTML/urls/punctuation/digits, lowercases Latin text,
    collapses whitespace). Stopword removal only applies to English.
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
        text = _NON_ALPHA_RE.sub(" ", text)
        text = _MULTISPACE_RE.sub(" ", text).strip()
        if remove_stopwords:
            tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
            return " ".join(tokens)
        return text

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
