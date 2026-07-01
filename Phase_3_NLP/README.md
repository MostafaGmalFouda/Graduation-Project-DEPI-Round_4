# 🔷 Phase 3 — NLP Layer (Text Intelligence)

Adds natural-language understanding on top of the EDA core (Phase 1 +
Phase 2). Turns raw, messy text columns into clean, analyzable, and
vectorized data — the foundation that **Phase 4 (RAG)** builds on top of.

## 📦 Modules

| Module | Responsibility |
|---|---|
| `TextPreprocessor` | Cleans raw text: lowercase, strip punctuation/URLs/emails, remove stopwords, tokenize, lemmatize |
| `NLPAnalyzer` | Extracts insight from clean text: text length stats, sentiment (polarity/subjectivity), Named Entity Recognition, n-grams, corpus summary |
| `TextVectorizer` | Converts text into numeric vectors via TF-IDF or Bag-of-Words, for use in ML models or similarity tasks |

## 🔄 Data Flow

```
[ Raw Text Column ] → [ TextPreprocessor.clean_pipeline() ]
   → [ NLPAnalyzer ] → (sentiment, entities, n-grams, stats)
   → [ TextVectorizer ] → (TF-IDF / BoW matrix)
```

## 🚀 Quick Start

```python
import pandas as pd
from Phase_3_NLP.TextPreprocessor import TextPreprocessor
from Phase_3_NLP.NLPAnalyzer import NLPAnalyzer
from Phase_3_NLP.TextVectorizer import TextVectorizer

df = pd.DataFrame({"review": ["I love this product!", "Terrible quality, avoid."]})

# 1. Clean the text
pre = TextPreprocessor(df)
clean_df = pre.clean_pipeline("review")

# 2. Analyze it
analyzer = NLPAnalyzer(clean_df)
sentiment_df = analyzer.analyze_sentiment("review")
entities = analyzer.extract_named_entities("review")

# 3. Vectorize it
vectorizer = TextVectorizer(clean_df)
tfidf_matrix = vectorizer.fit_tfidf("review", max_features=500)
```

## ⚙️ Setup

```bash
pip install -r requirements.txt
python -m textblob.download_corpora
python -m spacy download en_core_web_sm
```

## 🧩 Integration Notes

- `TextPreprocessor.clean_pipeline()` is the recommended entry point —
  it runs the full recommended cleaning sequence in one call.
- `NLPAnalyzer.extract_named_entities()` lazily loads the spaCy model
  on first use, so importing the class never requires spaCy to already
  be configured.
- `TextVectorizer.transform_text()` reuses the SAME fitted vocabulary
  from `fit_tfidf()`/`fit_bow()` — this is what lets Phase 4's RAG
  query embedding stay consistent with the indexed corpus.
