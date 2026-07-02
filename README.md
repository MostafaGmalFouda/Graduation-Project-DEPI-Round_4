<div align="center">

# 🧠 Explainable AI Model Debugger
### DEPI Graduation Project — Round 4

![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?style=flat-square&logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-purple?style=flat-square&logo=plotly)
![Status](https://img.shields.io/badge/Status-Phase%202%20Complete-brightgreen?style=flat-square)

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
| Phase 3 | Machine Learning Models + Debugging | 🔜 Upcoming |
| Phase 4 | Explainable AI (SHAP / LIME) | 🔜 Upcoming |
| Phase 5 | Full Deployment (API + Cloud) | 🔜 Upcoming |

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
│       ├── summary_dashboard.png
│       ├── correlation_heatmap.png
│       ├── scatter_2d_Age_vs_RestingBP.html
│       ├── scatter_3d_Age_RestingBP_Cholesterol.html
│       ├── bubble_chart_Age_vs_RestingBP.html
│       ├── joint_plot_Age_vs_RestingBP.png
│       ├── violin_Age_by_Sex.png
│       ├── facet_grid_by_Sex.png
│       ├── stacked_bar_Sex_by_ChestPainType.png
│       └── cross_tab_Sex_vs_ChestPainType.png
│
├── templates/
│   ├── index.html                  # Main UI
│   └── pipeline_ui.html            # Pipeline interface
│
├── static/
│   ├── css/
│   │   ├── style.css
│   │   └── pipeline_style.css
│   └── scripts/
│       └── pipeline_logic.js
│
├── reports/
│   ├── final_report.html
│   ├── bi_report.html              # New in Phase 2
│   └── detailed_report.html        # New in Phase 2
│
├── app.py                          # Flask Application
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

## 🛠️ Technologies Used

| Layer | Technologies |
|-------|-------------|
| Language | Python 3.12+ |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
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
