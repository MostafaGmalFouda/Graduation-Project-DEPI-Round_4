<div align="center">

# 🧠 Explainable AI Model Debugger
### DEPI Graduation Project — Round 4

![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?style=flat-square&logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-purple?style=flat-square&logo=plotly)
![Status](https://img.shields.io/badge/Status-Phase%205%20Complete-brightgreen?style=flat-square)

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
| **Phase 2** | Advanced Visualization Engine | ✅ Complete |
| **Phase 3** | NLP Engine (Text Intelligence) | ✅ Complete |
| **Phase 4** | RAG & LLM Engine (Ask Your Data) | ✅ Complete |
| **Phase 5** | Machine Learning Pipeline (User &amp; Developer Modes) | ✅ Complete |
| Phase 6 | Explainability (SHAP / LIME) | 🔜 Upcoming |
| Phase 7 | Full Deployment (API + Cloud) | 🔜 Upcoming |

---

## 🧱 System Architecture

```
┌─────────────────────────────────┐
│         User Interface          │
│    (Flask App + HTML/CSS/JS)    │
│   User Mode  ⇄  Developer Mode  │
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


        Clean Text Column (any phase above)
                     │
                     ▼
              TextPreprocessor  ← Phase 3 (NLP)
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
  NLPAnalyzer   TextVectorizer   TextChunker  ← Phase 4 (RAG)
  (sentiment,                         │
   NER, n-grams)                      ▼
                              VectorStoreManager (ChromaDB)
                                       │
                                       ▼
                               RAGOrchestrator → Claude LLM
                                       │
                                       ▼
                              Grounded Natural-Language Answer


        Clean Data (Phase 1)
                     │
                     ▼
               ModelFactory   ← Phase 5 (ML)
                     │
                     ▼
               ModelTrainer ──► cross-val / grid / random search
                     │
                     ▼
              ModelEvaluator ──► accuracy / F1 / ROC-AUC / RMSE / R²
                     │
              ┌──────┴──────┐
              ▼             ▼
          Predictor   ModelVisualizer
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
├── Phase_2/                        # Visualization Engine
│   ├── DataVisualizer.py
│   ├── __init__.py
│   └── plots/
│
├── Phase_3_NLP/                     # NLP Engine (text intelligence)
│   ├── TextPreprocessor.py
│   ├── NLPAnalyzer.py
│   ├── TextVectorizer.py
│   ├── requirements.txt
│   └── __init__.py
│
├── Phase_4_RAG/                     # RAG & LLM Engine (ask your data)
│   ├── TextChunker.py
│   ├── VectorStoreManager.py
│   ├── RAGOrchestrator.py
│   ├── requirements.txt
│   └── __init__.py
│
├── Phase_5_ML/                      # Machine Learning Pipeline
│   ├── ModelFactory.py
│   ├── ModelTrainer.py
│   ├── ModelEvaluator.py
│   ├── Predictor.py
│   ├── ModelVisualizer.py
│   ├── MLPipeline.py
│   ├── requirements.txt
│   ├── plots/                       # Generated ML plots served to the UI
│   └── __init__.py
│
├── templates/
│   ├── index.html                  # Main UI shell (2-item sidebar: EDA Platform / AI Pipeline)
│   ├── phase2_ui.html               # Visualization dashboard (auto-continues to AI Pipeline)
│   ├── ai_pipeline_ui.html          # Unified ML → NLP → RAG connected journey (New)
│   ├── phase3_ui.html               # Standalone NLP & RAG page (kept for direct access)
│   ├── phase4_ui.html               # Standalone ML page (kept for direct access)
│   └── pipeline_ui.html
│
├── static/
│   ├── css/
│   │   ├── style.css
│   │   ├── app_shell.css
│   │   ├── phase2_style.css
│   │   ├── ai_pipeline_style.css     # New — unified ML/NLP/RAG theme
│   │   ├── phase3_style.css
│   │   ├── phase4_style.css
│   │   └── pipeline_style.css
│   └── scripts/
│       ├── app_shell.js             # Sidebar nav — now just EDA Platform / AI Pipeline
│       ├── phase2_logic.js          # Auto-continues to AI Pipeline in User Mode
│       ├── ai_pipeline_logic.js     # New — unified ML → NLP → RAG controller
│       ├── phase3_logic.js
│       ├── phase4_logic.js
│       └── pipeline_logic.js
│
├── session_data/
│   ├── models/                      # Saved/downloaded trained models (joblib)
│   └── vector_store/                # Per-session ChromaDB RAG indexes
│
├── reports/
│   ├── final_report.html
│   ├── bi_report.html
│   └── detailed_report.html
│
├── app.py                          # Flask Application (all routes)
├── requirements.txt                # New — full project dependencies
├── Main.ipynb                      # Jupyter Notebook
├── train.csv
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

## 🔗 The Connected Journey

BRight AI is no longer five separate tools — it's **one continuous platform**.
The sidebar collapses to two stops, and the platform carries your dataset
between them automatically:

```
EDA Platform (Phase 1 + 2)  ──────►  AI Pipeline (ML → NLP → RAG)
   Upload → Clean → Visualize             Train → Analyze Text → Ask Questions
```

**The User/Developer choice is asked once**, at the very first screen, and
applies to the *entire* journey:

- **User Mode** — fully automated end-to-end. Upload a file once, and the
  platform walks itself through cleaning, visualization, model training,
  text analysis, and RAG indexing — landing you on a working chatbot with
  zero extra clicks. Short countdown banners (skippable) connect each step.
- **Developer Mode** — full manual control at every single stage: every
  preprocessing option, every model and hyperparameter, every tuning
  strategy, every chunking/embedding setting — all exposed, all editable.
  "Continue" buttons (not auto-redirects) move between stages so technical
  users stay in the driver's seat.

A mode toggle in the top bar still lets anyone switch mid-session if they
change their mind.

### 🔷 Phase 1 + 2 — EDA Platform

Upload → Validate → Clean → Visualize, exactly as before. In User Mode,
finishing the pipeline auto-advances into Phase 2's dashboard, which in turn
auto-advances into the AI Pipeline once it's rendered.

### 🔷 AI Pipeline — Machine Learning → NLP → RAG

One page (`/ai-pipeline`), three connected stages, sharing the same dataset
end-to-end — no re-uploading, no losing context between stages.

```
[ Clean Data ] → ModelFactory → ModelTrainer → ModelEvaluator → Predictor
                                                                     │
        ┌────────────────────────────────────────────────────────────┘
        ▼
[ Text Column ] → TextPreprocessor → NLPAnalyzer → TextVectorizer
        │
        ▼
TextChunker → VectorStoreManager (ChromaDB) → RAGOrchestrator → Claude
        │
        ▼
   💬 Chatbot — ask anything about your dataset
```

| Stage | What happens | User Mode | Developer Mode |
|---|---|---|---|
| **Machine Learning** | Pick a target → train a model → see metrics, plots, and live predictions | Top-recommended model auto-trained | Pick any model, set hyperparameters, enable grid/random tuning, set CV folds &amp; test size |
| **NLP** | Clean the text column → sentiment, entities, n-grams → TF-IDF/BoW vectors | Full pipeline runs automatically | Toggle individual cleaning steps, n-gram size, NER on/off, TF-IDF vs BoW, max features |
| **RAG / Chatbot** | Chunk → embed → index → ask questions, grounded in retrieved context | Index builds automatically; chatbot opens with suggested questions | Tune chunk size, overlap, and number of retrieved chunks (k) |

A **master journey tracker** (ML → NLP → RAG) sits above every stage so you
always know where you are and can jump back to a completed stage at any time.

> ⚙️ The chatbot requires an `ANTHROPIC_API_KEY` environment variable set on
> the server. See [console.anthropic.com](https://console.anthropic.com/).

#### Components reused under the hood

| Module | Lives in | Responsibility |
|---|---|---|
| `ModelFactory`, `ModelTrainer`, `ModelEvaluator`, `Predictor`, `ModelVisualizer` | `Phase_5_ML/` | Model recommendation, training, evaluation, prediction, plots |
| `TextPreprocessor`, `NLPAnalyzer`, `TextVectorizer` | `Phase_3_NLP/` | Text cleaning, sentiment/NER/n-grams, vectorization |
| `TextChunker`, `VectorStoreManager`, `RAGOrchestrator` | `Phase_4_RAG/` | Chunking, embeddings + ChromaDB, grounded LLM answers |

These modules are unchanged from their original design — the unified
`/ai-pipeline` page simply orchestrates calls to the same underlying JSON
APIs (`/phase4/*`, `/phase3/*`) that power the connected journey end-to-end.

---

## 🛠️ Technologies Used

| Layer | Technologies |
|-------|-------------|
| Language | Python 3.12+ |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web | Flask, HTML5, CSS3, JavaScript |
| NLP | NLTK, spaCy, TextBlob, scikit-learn (TF-IDF/BoW) |
| RAG / LLM | sentence-transformers, ChromaDB, Anthropic Claude API |
| Machine Learning | scikit-learn (RandomForest, SVM, KNN, Logistic/Linear Regression, DecisionTree, KMeans, DBSCAN) |
| Analysis | OOP, Statistical Methods |
| Reporting | Jinja2 Templates, HTML Reports |

---

## ⚙️ Setup &amp; Run

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r Phase_3_NLP/requirements.txt
pip install -r Phase_4_RAG/requirements.txt
pip install -r Phase_5_ML/requirements.txt

# 2. One-time NLP resource downloads
python -m textblob.download_corpora
python -m spacy download en_core_web_sm

# 3. Set your Anthropic API key (required for Phase 4 RAG answers)
export ANTHROPIC_API_KEY="sk-ant-..."        # https://console.anthropic.com/

# 4. Run the app
python app.py
```

Then open `http://127.0.0.1:5000` and pick **User Mode** or **Developer Mode** once — that choice carries through the entire connected journey. Use the sidebar to jump between **EDA Platform** and **AI Pipeline**, or just let User Mode carry you through automatically from upload to chatbot.

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
