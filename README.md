<div align="center">

# 🧠 Explainable AI Model Debugger
### DEPI Graduation Project — Round 4

![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?style=flat-square&logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-purple?style=flat-square&logo=plotly)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=flat-square&logo=flask)
![Status](https://img.shields.io/badge/Status-Phase%206%20In%20Progress-brightgreen?style=flat-square)

</div>

---

## 🚀 Project Vision

A full-scale **Explainable AI & Model Debugging Platform** that transforms machine learning systems from black-box models into **transparent, interpretable, and debuggable** AI systems.

The platform is built incrementally across multiple phases — each adding a new intelligence layer on top of the last.

---

## ✅ Progress Overview

| Phase | Title | Status |
|-------|-------|--------|
| **Phase 1** | Data Intelligence Layer (EDA Pipeline) | ✅ Complete |
| **Phase 2** | Advanced Visualization Engine + Automatic EDA Dashboard | ✅ Complete (actively extended — see [Recent Updates](#-recent-updates)) |
| **Phase 3** | NLP Analysis (Text Preprocessing, Vectorization, Visualization) | ✅ Complete |
| **Phase 4** | *(reserved / merged into Phase 6 RAG work)* | — |
| **Phase 5** | Machine Learning Pipeline (Training, Evaluation, Prediction) | ✅ Complete |
| **Phase 6** | RAG Chatbot — "BRight AI" data assistant | 🔄 In Progress |

---

## 🧱 System Architecture

```
┌─────────────────────────────────┐
│         User Interface          │
│    (Flask App + HTML/CSS/JS)    │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│        EDA Pipeline Core        │
│       (EDAPipeline Engine)      │
└────────────────┬────────────────┘
                 │
 ┌───────────────┼───────────────┐
 │               │               │
 ▼               ▼               ▼
DataLoader   DataValidator   DataPreprocessor
                                  │
                                  ▼
                           OutlierHandler
                                  │
                                  ▼
                           ReportGenerator
                                  │
                                  ▼
                          DataVisualizer  ← Phase 2
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
              Static Plots (PNG)      Interactive Plots (HTML)
                                              │
                                              ▼
                                  Automatic EDA Dashboard
                              (Quick Actions + Balance Analysis)
                                              │
        ┌─────────────────────────────────────┼─────────────────────────┐
        ▼                                     ▼                         ▼
  Phase 3 — NLP Engine                Phase 5 — ML Pipeline      Phase 6 — RAG Chatbot
  TextPreprocessor                    ModelFactory                ContextManager
  TextVectorizer                     ModelTrainer                 VectorStore (FAISS)
  NLPAnalyzer                        ModelEvaluator                Chatbot ("BRight AI")
  NLPVisualizer                      Predictor
                                     ModelVisualizer
```

---

## 📁 Project Structure

```
Graduation-Project-DEPI-Round_4/
│
├── Phase_1/                        # EDA Pipeline Engine
│   ├── DataLoader.py
│   ├── DataValidator.py
│   ├── DataPreprocessor.py
│   ├── OutlierHandler.py
│   ├── ReportGenerator.py
│   ├── EDAPipeline.py
│   ├── __init__.py
│   └── template.html
│
├── Phase_2/                        # Visualization Engine + Automatic Dashboard
│   ├── DataVisualizer.py           # Charts, KPI strip, Quick Actions, Balance Analysis
│   ├── __init__.py
│   └── plots/
│
├── Phase_3_NLP/                    # Text / NLP Engine
│   ├── TextPreprocessor.py
│   ├── TextVectorizer.py
│   ├── NLPAnalyzer.py
│   ├── NLPVisualizer.py
│   ├── __init__.py
│   └── plots/
│
├── Phase_5_ML/                     # Machine Learning Pipeline
│   ├── MLPipeline.py
│   ├── ModelFactory.py
│   ├── ModelTrainer.py
│   ├── ModelEvaluator.py
│   ├── ModelVisualizer.py
│   ├── Predictor.py
│   ├── __init__.py
│   ├── plots/
│   └── saved_models/               # Trained model .pkl files (served via /ml/download-model)
│
├── Phase_6/                        # RAG Chatbot — "BRight AI"
│   ├── context/                    # Dataset / EDA / model / conversation context builders
│   ├── rag/                        # document_builder, embeddings, vector_store, chatbot, llm, prompt
│   ├── requirements.txt
│   └── README.md                   # Chatbot-specific setup (Ollama / Claude backend)
│
├── templates/
│   ├── index.html                  # Main UI
│   ├── phase2_ui.html              # EDA + Automatic Dashboard UI
│   ├── nlp_ui.html                 # NLP phase UI
│   ├── ml_ui.html                  # ML phase UI
│   ├── eda_ui.html
│   ├── pipeline_ui.html
│   └── components/
│       └── chatbot.html            # Floating "BRight AI" chat widget
│
├── static/
│   ├── css/
│   └── scripts/
│       └── phase2_logic.js
│
├── reports/
│   ├── final_report.html
│   ├── bi_report.html
│   └── detailed_report.html
│
├── app.py                          # Flask Application (all phase routes)
├── Main.ipynb                      # Jupyter Notebook
├── requirements.txt
└── README.md
```

---

## 🔷 Phase 1 — Data Intelligence Layer

The backbone of the system. Handles everything from raw data ingestion to automated report generation.

### 🔄 Data Flow

```
[ Raw Dataset ] → [ DataLoader ] → [ DataValidator ]
→ [ DataPreprocessor ] → [ OutlierHandler ] → [ ReportGenerator ] → [ HTML Report ]
```

### 🧩 Components

| Module | Responsibility |
|--------|---------------|
| `DataLoader` | Reads and structures datasets from CSV/Excel |
| `DataValidator` | Checks data quality, missing values, types |
| `DataPreprocessor` | Cleans, encodes, and normalizes data |
| `OutlierHandler` | Detects and handles anomalies statistically |
| `ReportGenerator` | Produces automated HTML insight reports |
| `EDAPipeline` | Orchestrates the full workflow end-to-end |

### ✔ Phase 1 Deliverables
- Fully functional EDA pipeline
- Data validation & quality checks
- Outlier detection engine
- Automated HTML reporting
- Basic Flask UI

---

## 🔷 Phase 2 — Advanced Visualization Engine

Introduces a rich, interactive visualization layer built on top of the Phase 1 pipeline. The `DataVisualizer` class supports both **static** (Matplotlib/Seaborn) and **interactive** (Plotly) charts, with automatic type detection for numerical and categorical columns.

### 📊 Visualization Capabilities

**General / Multivariate**
- Summary Dashboard (overview of all columns)
- Correlation Heatmap

**Numerical × Numerical**
- 2D Scatter Plot (interactive HTML)
- 3D Scatter Plot (interactive HTML)
- Joint Distribution Plot

**Categorical × Categorical**
- Stacked Bar Chart
- Cross-Tabulation Heatmap
- Violin Plot by Category
- Facet Grid (multi-histogram)
- Bubble Chart (interactive HTML)

### 🆕 What Changed in Phase 2

- **`Phase_2/DataVisualizer.py`** — New visualization engine with full chart suite
- **`Phase_2/plots/`** — 10 generated charts (PNG + interactive HTML)
- **`reports/bi_report.html`** — New BI-style report
- **`reports/detailed_report.html`** — New detailed analysis report
- **`app.py`** — Updated Flask routes to serve Phase 2 outputs
- **`templates/index.html`** — Updated UI to display visualizations
- **`static/css/style.css`** — Refreshed styling
- **`static/scripts/pipeline_logic.js`** — Extended pipeline interactions
- Removed legacy `phase2_ui.html`, `phase2_style.css`, `phase2_logic.js`

---

## 🔷 Phase 3 — NLP Engine

Adds a text-analysis layer for datasets with free-text columns (reviews, comments, descriptions, etc.).

| Module | Responsibility |
|--------|---------------|
| `TextPreprocessor` | Cleans, tokenizes, and normalizes raw text |
| `TextVectorizer` | Converts text into numeric features (TF-IDF / Bag-of-Words) |
| `NLPAnalyzer` | Extracts sentiment, word frequency, and key-phrase insights |
| `NLPVisualizer` | Word clouds, frequency bar charts, and sentiment distribution plots |

---

## 🔷 Phase 5 — Machine Learning Pipeline

Turns the cleaned dataset into a trained, evaluated, and downloadable model — no manual scikit-learn code needed.

| Module | Responsibility |
|--------|---------------|
| `MLPipeline` | Orchestrates the end-to-end training workflow |
| `ModelFactory` | Builds the requested model (classification / regression) with sane defaults |
| `ModelTrainer` | Splits data, trains the model, and persists it to `Phase_5_ML/saved_models/` |
| `ModelEvaluator` | Computes accuracy/F1/RMSE/etc. depending on task type |
| `ModelVisualizer` | Confusion matrices, ROC curves, feature importance plots |
| `Predictor` | Runs inference with a saved model on new rows |

Every trained model is saved as `model_<model_id>.pkl` and can be pulled back down via the **Download Model** action (see Recent Updates below).

---

## 🔷 Phase 6 — RAG Chatbot ("BRight AI")

A retrieval-augmented chat assistant, embedded as a floating widget (`templates/components/chatbot.html`) on every phase page, that answers questions grounded in the *user's own* uploaded dataset, EDA results, and trained models — not generic knowledge.

| Module | Responsibility |
|--------|---------------|
| `Phase_6/context/*` | Builds structured context objects (dataset shape, EDA findings, NLP results, trained-model metrics, conversation history) |
| `Phase_6/rag/document_builder.py` | Turns that context into retrievable text documents |
| `Phase_6/rag/vector_store.py` | Embeds and indexes those documents (FAISS) |
| `Phase_6/rag/chatbot.py` | Ties retrieval (Finds the most relevant context for a given question) + prompt + LLM together into `chatbot.ask(...)` |
| `Phase_6/rag/llm.py` | Pluggable LLM backend — local Ollama (`llama3.2`, free/offline) or Anthropic Claude (`CHATBOT_LLM=claude` + `ANTHROPIC_API_KEY`) |

See `Phase_6/README.md` for backend-specific setup steps.

---

## 🆕 Recent Updates

### ⚖️🤖📦 Automatic EDA Dashboard — Quick Actions & Balance Analysis

The Phase 2 **Automatic EDA Dashboard** (`Phase_2/DataVisualizer.py → generate_automatic_dashboard`) got a bigger header and a new **Quick Actions** bar with four working shortcuts, plus a brand-new analysis section:

| Action | What it does |
|--------|--------------|
| ⚖️ **Measurements for Unbalanced Data** | Jumps to a new dashboard section that measures every categorical (and class-like numeric) column: majority-class %, minority-class %, and an **imbalance ratio**, tagged `Balanced` / `Moderate` / `Imbalanced`. A new **"Imbalanced Cols"** KPI card summarizes it at a glance — so skewed target/feature columns are visible *before* modeling, not discovered after a model quietly ignores the minority class. |
| 🤖 **Explanation AI** | Opens the existing floating "BRight AI" chat widget (reaches through the iframe to `#chat-toggle` on the parent page) so the person can ask questions about the dashboard they're looking at. |
| 🧹 **Download Data After Cleaning** | Calls the new `GET /download-clean-data` route, which streams the current session's cleaned DataFrame as `cleaned_data.csv`. |
| 📦 **Download Model** | Calls the new `GET /ml/download-model` route, which returns the most recently trained model (`model_<id>.pkl` from `Phase_5_ML/saved_models/`) as an attachment. Accepts an optional `?model_id=` to fetch a specific one. |

Both download actions fetch first and show a small toast message (e.g. *"No trained model available yet — train one in Phase 5 first"*) instead of failing silently if nothing's ready yet.

**Files touched:**
- `Phase_2/DataVisualizer.py` — `_analyze_balance()` method, balance-analysis section HTML/CSS, enlarged header, Quick Actions bar, and its JS (`bgDownload`, `bgOpenExplainAI`, toast).
- `app.py` — `/download-clean-data` and `/ml/download-model` routes; `session['last_model_id']` is now stored after every successful training run so the download button always knows which model to fetch.

---

## 🛠️ Technologies Used

| Layer | Technologies |
|-------|-------------|
| Language | Python 3.12+ |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| NLP | Custom TextPreprocessor/Vectorizer, TF-IDF |
| Machine Learning | scikit-learn-based ModelFactory/Trainer/Evaluator |
| RAG / Chatbot | LangChain, FAISS, HuggingFace Embeddings, Ollama (`llama3.2`) or Anthropic Claude |
| Web | Flask, HTML5, CSS3, JavaScript |
| Analysis | OOP, Statistical Methods |
| Reporting | Jinja2 Templates, HTML Reports |

---

## 👥 Project Team

| Name |
|------|
| Mostafa Fathalla |
| Mostafa Gamal Fouda |
| Mariam Gaber |
| Tasneem Radwan |
| Samuel Adel |
| Abdelhamid Ibrahim |

---

<div align="center">

*Building transparent AI — one phase at a time.*

</div>
