from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, Response, session
import os
import io
import re
import json
import time
import uuid
import pickle
from datetime import datetime, timezone
import pandas as pd
from werkzeug.utils import secure_filename
import numpy as np

# Import custom modules from Phase_1
from Phase_1.DataLoader import DataLoader
from Phase_1.ReportGenerator import ReportGenerator
from Phase_1.EDAPipeline import EDAPipeline
from Phase_1.DataPreprocessor import DataPreprocessor
from Phase_1.OutlierHandler import OutlierHandler

# Import Phase_2 DataVisualizer
from Phase_2.DataVisualizer import DataVisualizer
 
# Import Phase_6 
from Phase_6.context.context_manager import ChatContext
from Phase_6.context.context_manager import save_context
from Phase_6.context.context_manager import load_context
from Phase_6.context.context_manager import new_context_filename
from Phase_6.rag.document_builder import build_documents
from Phase_6.rag.chatbot import Chatbot
from Phase_6.rag.vector_store import VectorStore

# Import Phase_3 NLP
from Phase_3_NLP.NLPAnalyzer import NLPAnalyzer
from Phase_3_NLP.NLPVisualizer import NLPVisualizer

# Import Phase_5 ML
from Phase_5_ML.MLPipeline import MLPipeline
from Phase_5_ML.ModelTrainer import ModelTrainer
from Phase_5_ML.ModelFactory import ModelFactory
from Phase_5_ML.Predictor import Predictor
from Phase_5_ML.Featureschema import FeatureSchema

chatbot = Chatbot()

# ── App Setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = "bright_ai_secret_key_2024"

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder to store uploaded files
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Folder to store generated reports
REPORTS_FOLDER = os.path.join(BASE_DIR, 'reports')
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Folder to store session DataFrames (pickled)
SESSION_DATA_FOLDER = os.path.join(BASE_DIR, 'session_data')
os.makedirs(SESSION_DATA_FOLDER, exist_ok=True)

# Allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}


# ── Session data helpers ─────────────────────────────────────────────────────

def save_session_df(df: pd.DataFrame, key: str) -> str:
    """Pickle a DataFrame to disk and return the file path."""
    fname = f"{key}_{uuid.uuid4().hex}.pkl"
    path = os.path.join(SESSION_DATA_FOLDER, fname)
    df.to_pickle(path)
    return path

def load_session_df(path: str) -> pd.DataFrame:
    """Load a pickled DataFrame from disk."""
    if path and os.path.exists(path):
        return pd.read_pickle(path)
    return None

def get_clean_df() -> pd.DataFrame:
    """Return the clean (preprocessed) DataFrame stored in the session, or None."""
    path = session.get('clean_df_path')
    return load_session_df(path)

def get_raw_df() -> pd.DataFrame:
    """Return the raw DataFrame stored in the session, or None."""
    path = session.get('raw_df_path')
    return load_session_df(path)


def get_pipeline_log() -> list:
    """
    Read back the pipeline stage/console log written by the last completed
    /pipeline-stream run for this session (see the `send()` closure there).
    Lets /eda rebuild the exact same stages panel + console log after
    navigating away and back, instead of it only existing during the one
    live SSE run. Returns [] if no run has happened yet (or the file is
    missing/corrupt for any reason — never let a bad log break the page).
    """
    path = session.get('pipeline_log_path')
    if not path or not os.path.exists(path):
        return []
    entries = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except (ValueError, TypeError):
                    continue
    except OSError:
        return []
    return entries


# ── Data purpose helpers (drives text-column cleaning + Phase locking) ───────

def get_data_purpose() -> str:
    """
    The purpose chosen on the intro/EDA card ('general' | 'ml' | 'nlp'),
    persisted at pipeline-run time. This is NOT a Developer Mode knob — it's
    collected in both modes because it decides how free-text columns get
    cleaned during Phase 1 (encoded away for ML vs. kept as real text for NLP).
    """
    purpose = session.get('data_purpose', 'general')
    return purpose if purpose in ('general', 'ml', 'nlp') else 'general'


def purpose_lock_check(target: str):
    """
    Returns a human-readable lock message if `target` ('nlp' or 'ml') should
    be LOCKED given how the current dataset was actually processed, or None
    if it's unlocked and safe to use.

    Why: choosing "ML" as the data purpose fully encodes/removes free-text
    columns during Phase 1, so the NLP page would have nothing real left to
    analyze. Choosing "NLP" keeps text columns raw/unencoded, so the ML page
    can't train on them as-is. 'general' never locks either page.
    """
    purpose = get_data_purpose()
    if purpose == 'general':
        return None

    if target == 'nlp' and purpose == 'ml':
        return ("This dataset was processed for Machine Learning, so free-text "
                "columns were encoded into numbers and are no longer available "
                "for NLP analysis. Re-run Phase 1 and set the data purpose to "
                "'NLP / Text Analysis' to unlock this page.")

    if target == 'ml' and purpose == 'nlp':
        return ("This dataset was processed for NLP, so text columns were kept "
                "as raw text instead of being encoded for model training. "
                "Re-run Phase 1 and set the data purpose to 'Machine Learning' "
                "to unlock this page.")

    return None


# ── Chat context helpers (per-session, isolates users from each other) ───────

def get_session_id() -> str:
    """Stable id for this browser session — used as the chatbot's vector-cache key."""
    sid = session.get('session_id')
    if not sid:
        sid = uuid.uuid4().hex
        session['session_id'] = sid
    return sid


def get_chat_context() -> ChatContext:
    """Load this session's ChatContext from disk, or an empty one if none exists."""
    path = session.get('context_path')
    return load_context(path)


def save_chat_context(ctx: ChatContext):
    """Persist this session's ChatContext, creating a file the first time."""
    path = session.get('context_path')
    if not path:
        path = os.path.join(SESSION_DATA_FOLDER, new_context_filename())
        session['context_path'] = path
    save_context(ctx, path)


def clear_session_data():
    """
    Wipe everything belonging to the CURRENT session: raw/clean DataFrames,
    column schema, and chat context — both the files on disk and the paths
    kept in the Flask session. Also drops the chatbot's cached vector index
    for this session so old answers can never leak into the next dataset.
    """
    for key in ('raw_df_path', 'clean_df_path', 'schema_path', 'context_path', 'pipeline_log_path',
                'nlp_state_path', 'ml_state_path', 'phase2_state_path'):
        path = session.pop(key, None)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    session.pop('dataset_filename', None)
    session.pop('report_view_url', None)
    session.pop('report_download_url', None)
    session['pipeline_done'] = False
    # A fresh dataset means the previous data_purpose lock no longer applies —
    # whatever purpose gets chosen on the NEXT pipeline run should decide
    # which of /nlp or /ml stays unlocked, not a purpose left over from a
    # dataset that no longer exists.
    session.pop('data_purpose', None)
    chatbot.forget(get_session_id())


# ── Schema helpers ────────────────────────────────────────────────────────────
def compute_column_schema(df: pd.DataFrame) -> dict:
    """
    Classify every column as 'num' or 'cat' based on its CURRENT dtype.
    Used to snapshot the schema BEFORE any encoding/type-conversion step
    changes a column's dtype (e.g. Label/One-Hot encoding turns a
    categorical column into integers).
    """
    return {col: ("num" if df[col].dtype.kind in "iufcb" else "cat") for col in df.columns}


def save_column_schema(schema: dict) -> str:
    """Persist a column-type schema (dict) to disk and return its path."""
    fname = f"schema_{uuid.uuid4().hex}.json"
    path = os.path.join(SESSION_DATA_FOLDER, fname)
    with open(path, 'w') as f:
        json.dump(schema, f)
    return path


# ── Generic per-page state persistence (Phase 3/5/2 "stay as I left it") ────
# Same pattern as pipeline_log_path / schema_path above: the Flask session
# cookie only holds a PATH, the actual JSON blob lives on disk. This is what
# lets /nlp, /ml and /phase2 rebuild the exact UI + results a person left
# behind after navigating to another page and back — without needing a
# refresh or a re-run — while still resetting cleanly on a real refresh
# (session cookie survives that) or a brand new upload (clear_session_data
# below deletes these files and pops the session keys).

def save_json_blob(prefix: str, data: dict) -> str:
    fname = f"{prefix}_{uuid.uuid4().hex}.json"
    path = os.path.join(SESSION_DATA_FOLDER, fname)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, default=str)
    return path


def load_json_blob(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def get_page_state(session_key: str) -> dict:
    """Read back the whole saved-state dict for a page (empty dict if none)."""
    return load_json_blob(session.get(session_key)) or {}


def update_page_state(session_key: str, entry_key: str, entry_value: dict, filename_prefix: str):
    """
    Merge one entry (keyed by e.g. a chart_type, or just '_last') into this
    page's saved-state dict and persist it. Reuses the SAME file on disk
    across calls (looked up via the path already in session) instead of
    writing a new file every run, so repeatedly generating charts/analyses
    doesn't leak files on disk.
    """
    path = session.get(session_key)
    state = load_json_blob(path) or {}
    state[entry_key] = entry_value
    if not path:
        path = os.path.join(SESSION_DATA_FOLDER, f"{filename_prefix}_{uuid.uuid4().hex}.json")
        session[session_key] = path
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, default=str)


def load_column_schema(path: str) -> dict:
    """Load a persisted column-type schema, or None if unavailable."""
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def get_original_schema() -> dict:
    """
    Return the pre-encoding column-type schema stored in the session, or None
    if it was never saved (e.g. data produced by an older code path).
    """
    path = session.get('schema_path')
    return load_column_schema(path)


def resolve_column_type(col_name: str, df: pd.DataFrame, original_schema: dict) -> str:
    """
    Decide whether a column should be reported as 'num' or 'cat' for
    visualization purposes:
      - If the column existed BEFORE encoding, trust its ORIGINAL type
        (e.g. a categorical column stays 'cat' even after Label/One-Hot
        encoding turned it into integers/binary columns).
      - If the column is new (e.g. a One-Hot expansion like
        'category_A', 'category_B'), it is always treated as categorical
        (binary 0/1 indicator), regardless of its numeric dtype.
      - If no original schema is available at all, fall back to detecting
        from the current dtype (legacy behavior).
    """
    if original_schema is None:
        return "num" if df[col_name].dtype.kind in "iufcb" else "cat"

    if col_name in original_schema:
        return original_schema[col_name]

    # Column not in the original schema → it's almost certainly a
    # One-Hot-expanded indicator column. Treat it as categorical.
    return "cat"


# ── Chart description helpers ────────────────────────────────────────────────
# The chatbot can only be as accurate as the text it's given. A generic
# "chart generated successfully" message forces the LLM to GUESS what the
# chart shows — which is exactly how it gave a wrong answer about a strong
# correlation being weak. These helpers compute the REAL numbers so the
# chatbot always has grounded facts to answer from.

def _corr_strength_label(value: float) -> str:
    v = abs(value)
    if v >= 0.8:
        strength = "very strong"
    elif v >= 0.6:
        strength = "strong"
    elif v >= 0.4:
        strength = "moderate"
    elif v >= 0.2:
        strength = "weak"
    else:
        strength = "very weak / negligible"
    direction = "positive" if value > 0 else ("negative" if value < 0 else "no")
    return f"{strength} {direction} correlation"


def describe_pairwise_correlation(df: pd.DataFrame, columns=None) -> str:
    """Real correlation numbers between the given (or all numeric) columns."""
    numeric_df = df.select_dtypes(include="number")
    if columns:
        numeric_df = numeric_df[[c for c in columns if c in numeric_df.columns]]

    if numeric_df.shape[1] < 2:
        return "Not enough numeric columns to compute a correlation."

    corr = numeric_df.corr()
    cols = corr.columns
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.isna(val):
                continue
            pairs.append(f"{cols[i]} vs {cols[j]}: correlation = {val:.2f} ({_corr_strength_label(val)})")

    return "; ".join(pairs) if pairs else "Could not compute correlation for these columns."


def describe_crosstab(df: pd.DataFrame, col1: str, col2: str) -> str:
    """Real most/least common category combinations."""
    if col1 not in df.columns or col2 not in df.columns:
        return f"Columns {col1}/{col2} not found."
    try:
        ct = pd.crosstab(df[col1], df[col2])
        if ct.empty:
            return "No data to cross-tabulate."
        stacked = ct.stack()
        top = stacked.idxmax()
        top_count = stacked.max()
        return (
            f"Cross-tabulation of {col1} vs {col2}. Most common combination: "
            f"{col1}={top[0]}, {col2}={top[1]} ({int(top_count)} rows)."
        )
    except Exception as e:
        return f"Could not compute cross-tabulation: {e}"


def describe_group_stats(df: pd.DataFrame, num_col: str, cat_col: str) -> str:
    """Real per-category mean/median for a numeric column."""
    if num_col not in df.columns or cat_col not in df.columns:
        return f"Columns {num_col}/{cat_col} not found."
    try:
        grouped = df.groupby(cat_col)[num_col].agg(["mean", "median", "count"]).round(2)
        parts = [
            f"{cat}: mean={row['mean']}, median={row['median']}, n={int(row['count'])}"
            for cat, row in grouped.iterrows()
        ]
        return f"{num_col} by {cat_col} — " + "; ".join(parts)
    except Exception as e:
        return f"Could not compute group statistics: {e}"


TARGET_NAME_HINTS = [
    'target', 'label', 'class', 'y', 'outcome', 'churn', 'result',
    'status', 'category', 'rating', 'score', 'price', 'sales',
    'survived', 'default', 'fraud', 'response', 'diagnosis', 'purchase',
]


def analyze_columns_summary(df: pd.DataFrame, max_cols: int = 12) -> dict:
    """
    Quick per-column glance (type, cardinality, a one-line stat) for the
    EDA insights page. Capped at max_cols so wide datasets don't blow up
    the page — the rest are just counted.
    """
    rows = []
    cols = list(df.columns)
    for col in cols[:max_cols]:
        s = df[col]
        is_numeric = s.dtype.kind in "iufc"
        unique_count = int(s.nunique(dropna=True))
        if is_numeric:
            col_type = "Numeric"
            try:
                detail = f"mean {s.mean():.2f} · std {s.std():.2f} · range [{s.min():.2f}, {s.max():.2f}]"
            except Exception:
                detail = f"{unique_count} unique values"
        else:
            col_type = "Categorical / Text"
            mode_vals = s.mode(dropna=True)
            top_val = str(mode_vals.iloc[0]) if not mode_vals.empty else "—"
            if len(top_val) > 28:
                top_val = top_val[:25] + "..."
            detail = f"most common: '{top_val}' · {unique_count} unique values"
        rows.append({"name": col, "type": col_type, "detail": detail})

    return {"rows": rows, "shown": len(rows), "total": len(cols), "hidden": max(len(cols) - len(rows), 0)}


def suggest_target_and_model(df: pd.DataFrame) -> dict:
    """
    Lightweight, heuristic-only target-column and model-family suggestion —
    NOT a trained recommendation. It pattern-matches on column names,
    cardinality, and dataset size to point the user toward a sensible
    starting point before they head into the ML phase; always framed as a
    starting hint, not a guarantee.
    """
    result = {
        "target_col": None, "target_reason": "",
        "task_type": None, "models": [], "model_reason": "",
    }
    if df is None or df.shape[1] == 0:
        return result

    cols = list(df.columns)

    # 1) Name-based match first — most reliable signal when present.
    target_col = None
    for hint in TARGET_NAME_HINTS:
        for c in cols:
            lc = c.lower()
            if lc == hint or lc.endswith("_" + hint) or lc.startswith(hint + "_"):
                target_col = c
                break
        if target_col:
            break

    if target_col:
        result["target_reason"] = (
            f"Column name '{target_col}' strongly suggests it's the outcome you'd want to predict."
        )
    else:
        # 2) Fall back to a low-cardinality column (classic classification-
        # target shape), else the numeric column most correlated with the
        # rest (classic regression-target shape).
        candidate = None
        for c in cols:
            nun = df[c].nunique(dropna=True)
            if 2 <= nun <= 10:
                candidate = c
                break
        if candidate:
            target_col = candidate
            result["target_reason"] = (
                f"'{target_col}' has only {df[target_col].nunique()} distinct values — "
                "a common shape for a classification target."
            )
        else:
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.shape[1] >= 2:
                try:
                    avg_corr = numeric_df.corr(numeric_only=True).abs().mean().sort_values(ascending=False)
                except Exception:
                    avg_corr = pd.Series(dtype=float)
                if not avg_corr.empty:
                    target_col = avg_corr.index[0]
                    result["target_reason"] = (
                        f"'{target_col}' is, on average, the most correlated numeric column with the "
                        "rest of the dataset — often a sign it's the value the others help explain."
                    )
        if not target_col and cols:
            target_col = cols[-1]
            result["target_reason"] = (
                "No strong signal in the column names or shapes — defaulting to the last column. "
                "Please confirm this is actually the outcome you want to predict."
            )

    result["target_col"] = target_col
    if target_col is None:
        return result

    # 3) Task type + model family, based on target shape and dataset size.
    nunique = df[target_col].nunique(dropna=True)
    is_numeric = df[target_col].dtype.kind in "iufc"
    n_rows = df.shape[0]

    if is_numeric and nunique > 10:
        result["task_type"] = "Regression"
        if n_rows < 1000:
            result["models"] = ["Linear Regression", "Ridge / Lasso Regression", "Decision Tree Regressor"]
            result["model_reason"] = "Smaller dataset — simpler regressors are less likely to overfit."
        else:
            result["models"] = ["Random Forest Regressor", "Gradient Boosting Regressor", "Linear Regression (baseline)"]
            result["model_reason"] = "Enough rows to support ensemble/boosted regressors for stronger performance."
    elif nunique == 2:
        result["task_type"] = "Binary Classification"
        if n_rows < 1000:
            result["models"] = ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"]
            result["model_reason"] = "Smaller dataset — simpler classifiers tend to generalize better."
        else:
            result["models"] = ["Random Forest Classifier", "Gradient Boosting Classifier", "Logistic Regression (baseline)"]
            result["model_reason"] = "Enough rows for ensemble/boosted classifiers to outperform simpler baselines."
    else:
        result["task_type"] = "Multiclass Classification"
        result["models"] = ["Random Forest Classifier", "Gradient Boosting Classifier", "Multinomial Logistic Regression"]
        result["model_reason"] = f"{nunique} distinct classes — tree-based ensembles handle multiclass well out of the box."

    return result


def describe_dataset_highlights(df: pd.DataFrame) -> str:
    """A short, factual overview used for the summary/automatic dashboards."""
    numeric_df = df.select_dtypes(include="number")
    parts = [f"{df.shape[0]} rows, {df.shape[1]} columns."]
    if numeric_df.shape[1] >= 2:
        parts.append("Correlations: " + describe_pairwise_correlation(df))
    return " ".join(parts)


def build_eda_insights(raw_df: pd.DataFrame, clean_df: pd.DataFrame, data_purpose: str) -> dict:
    """
    Real, data-driven summary for the /eda_result page: what the dataset
    actually looks like after cleaning, what the pipeline changed compared
    to the raw upload, and a plain-language recommendation for what to do
    next — computed straight from the dataframes instead of being hardcoded.
    """
    insights = {
        "highlights": "",
        "rows_before": None, "rows_after": None, "rows_removed": 0,
        "cols_before": None, "cols_after": None, "cols_removed": 0,
        "missing_before": 0, "missing_after": 0,
        "numeric_cols": 0, "categorical_cols": 0,
        "recommendation": "",
        "column_summary": None,
        "target_suggestion": None,
    }
    if clean_df is None:
        return insights

    insights["highlights"] = describe_dataset_highlights(clean_df)
    insights["column_summary"] = analyze_columns_summary(clean_df)
    insights["target_suggestion"] = suggest_target_and_model(clean_df)
    insights["rows_after"] = int(clean_df.shape[0])
    insights["cols_after"] = int(clean_df.shape[1])
    insights["missing_after"] = int(clean_df.isna().sum().sum())
    insights["numeric_cols"] = int(clean_df.select_dtypes(include="number").shape[1])
    insights["categorical_cols"] = int(clean_df.shape[1] - insights["numeric_cols"])

    if raw_df is not None:
        insights["rows_before"] = int(raw_df.shape[0])
        insights["cols_before"] = int(raw_df.shape[1])
        insights["rows_removed"] = max(insights["rows_before"] - insights["rows_after"], 0)
        insights["cols_removed"] = max(insights["cols_before"] - insights["cols_after"], 0)
        insights["missing_before"] = int(raw_df.isna().sum().sum())

    if data_purpose == "ml":
        insights["recommendation"] = (
            "This dataset was cleaned for Machine Learning — categorical columns "
            "were fully encoded into numbers, so it's ready to go straight into "
            "the ML phase for model training."
        )
    elif data_purpose == "nlp":
        insights["recommendation"] = (
            "This dataset was cleaned for NLP — free-text columns were kept as "
            "real, readable text instead of being encoded away, so it's ready "
            "for the NLP phase (sentiment, topics, text statistics)."
        )
    elif insights["numeric_cols"] and insights["categorical_cols"]:
        insights["recommendation"] = (
            "This dataset has a mix of numeric and categorical columns. Head to "
            "Visualization to explore distributions and relationships first, or "
            "re-run Phase 1 with a specific purpose (ML or NLP) once you know "
            "what you'll build next."
        )
    elif insights["numeric_cols"]:
        insights["recommendation"] = (
            "This dataset is now fully numeric — a good candidate for the ML "
            "phase, or Visualization if you want to inspect distributions first."
        )
    else:
        insights["recommendation"] = (
            "This dataset is mostly text/categorical — Visualization or the "
            "NLP phase are good next steps."
        )

    return insights


# ── Shared sidebar context ────────────────────────────────────────────────────

@app.context_processor
def inject_sidebar_context():
    """
    Injected into EVERY template render automatically — components/sidebar.html
    is included on every phase page (EDA, Visualization, NLP, ML), and it needs
    to know whether a finished ("clean") dataset exists so it can show the
    "Enter Results" shortcut, without every single route having to remember to
    pass that flag in manually.

    Source of truth is the clean DataFrame actually existing on disk
    (get_clean_df), NOT a session boolean — pipeline_done can't be flipped to
    True from inside the SSE generator (it runs outside the request context),
    so the pickle file's existence is the only thing that's always accurate.
    This also means it naturally disappears after /clear-data, /session/reset,
    or a brand new /process upload, since all three remove or replace the
    clean_df_path — no extra bookkeeping needed.
    """
    return {
        'sidebar_has_clean': get_clean_df() is not None
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """
    Boot/mode-select gate — the very first thing shown when the app opens.
    Just the intro animation + User/Developer mode cards. Choosing a mode
    saves it via /set-mode and redirects the browser to /eda.
    """
    return render_template('intro.html')


@app.route('/eda')
def eda():
    """
    The real EDA workspace (former main-app). Reached only after the
    intro/mode-select gate at '/'. Sidebar is shown from the start since
    the mode has already been chosen by this point.
    """
    mode = session.get('mode', 'user')
    has_dataset = (
        get_raw_df() is not None or
        get_clean_df() is not None
    )

    # Same "did the pipeline already finish" state /eda_result uses. Passing
    # it here too lets index.html render its finished-report card straight
    # from the server on every load — including after navigating away and
    # back — instead of relying on the live SSE 'done' event, which only
    # ever fires once, during the run itself.
    clean_df = get_clean_df()
    has_clean = clean_df is not None
    stats = {"rows": clean_df.shape[0], "cols": clean_df.shape[1]} if has_clean else None

    return render_template(
        "index.html",
        active_page="eda",
        mode=mode,
        has_dataset=has_dataset,
        has_clean=has_clean,
        stats=stats,
        report_view_url=session.get('report_view_url'),
        report_download_url=session.get('report_download_url'),
        dataset_filename=session.get('dataset_filename'),
        pipeline_log=get_pipeline_log() if has_clean else [],
    )
@app.route('/eda_result')
def eda_page():
    mode = session.get("mode", "user")
    raw_df = get_raw_df()
    has_raw = raw_df is not None
    clean_df = get_clean_df()
    has_clean = clean_df is not None
    data_purpose = session.get('data_purpose', 'general')

    stats = None
    insights = None
    if has_clean:
        stats = {"rows": clean_df.shape[0], "cols": clean_df.shape[1]}
        insights = build_eda_insights(raw_df, clean_df, data_purpose)

    return render_template(
        "eda_ui.html",
        mode=mode,
        active_page='eda',
        has_raw=has_raw,
        has_clean=has_clean,
        stats=stats,
        insights=insights,
        data_purpose=data_purpose,
        report_view_url=session.get('report_view_url'),
        report_download_url=session.get('report_download_url'),
    )

@app.route('/process', methods=['POST'])
def process():
    """Upload file → raw preview. Stores raw DF in session."""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files['file']
    action_mode = request.form.get('action_type')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            loader = DataLoader(filepath)
            df = loader.load()

            if df is None or df.empty:
                return jsonify({"status": "error", "message": "Empty or invalid file"}), 400

            # ── New upload: wipe anything left over from a previous dataset ──
            # (old raw/clean DFs, schema, and — importantly — the old chat
            # context, so the chatbot never answers with stale data)
            clear_session_data()

            # ── Persist raw DF in session ──────────────────────────────
            raw_path = save_session_df(df, 'raw')
            session['raw_df_path'] = raw_path
            session['dataset_filename'] = file.filename

            ctx = ChatContext()
            ctx.update_raw_dataset(df)
            save_chat_context(ctx)

            # ── Generate preview report ────────────────────────────────
            reporter = ReportGenerator(df)
            mode = "basic" if action_mode == "summary" else "detailed"
            out_file = "final_report.html" if mode == "basic" else "detailed_report.html"
            output_path = os.path.join(REPORTS_FOLDER, out_file)
            reporter.generate_report(mode=mode, file_name=output_path)

            preview_html = df.head(5).to_html(classes='preview-table', index=False)

            return jsonify({
                "status": "success",
                "preview": preview_html,
                "report_name": out_file,
                "view_url": f"/view/{out_file}",
                "download_url": f"/download/{out_file}",
                "rows": df.shape[0],
                "cols": df.shape[1],
                "columns": list(df.columns),
            })

        except Exception as e:
            print("ERROR:", e)
            return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "error", "message": "Unsupported file format"}), 400


@app.route('/clear-data', methods=['POST'])
def clear_data():
    """Remove this session's data (dataset + chat context) so the next upload starts fresh."""
    clear_session_data()
    return jsonify({"status": "ok"})


@app.route('/session/reset', methods=['POST'])
def session_reset():
    """
    Called by the frontend on a real browser refresh (F5 / reload), never on
    normal in-app navigation. Wipes this session completely: dataset files,
    chat context, and the Flask session cookie itself. The chatbot must go
    back to "no data loaded yet" answers until something new is uploaded.
    """
    clear_session_data()
    session.clear()
    return jsonify({"status": "ok"})


@app.route('/pipeline-stream', methods=['POST'])
def pipeline_stream():
    """
    SSE endpoint: runs full preprocessing pipeline on the raw DF
    already stored in the session (no file re-upload needed).
    Falls back to accepting a file if raw DF is not in session.
    At the end, the clean DF is persisted so Phase 2 can use it.

    FIX: Flask session is only accessible inside the request context.
    SSE generators run OUTSIDE the request context, so we must update
    session BEFORE returning the Response, and pass the clean_path
    into the generator via a closure variable — not via session.

    Collected in BOTH modes (not a Developer Mode knob):
        data_purpose          general|ml|nlp           default general
            → 'nlp' forces text_action='keep' so free-text columns survive
              cleaning; 'ml'/'general' fall back to the developer/default
              text_action ('drop'). Also persisted to session['data_purpose']
              so /nlp and /ml can lock themselves out when the dataset was
              cleaned for the other purpose (see purpose_lock_check()).

    Developer Mode (optional form fields — all default to the original
    automatic User Mode behavior when absent):
        null_threshold        float  0.0–1.0          default 0.4
        null_fill_strategy    median|mean|mode         default median
        do_type_conversion    true|false               default true
        do_remove_duplicates  true|false               default true
        exclude_columns       comma-separated          default ''
        text_action           drop|hash|keep           default drop
        text_unique_threshold float  0.0–1.0          default 0.6
        encoding_method       none|label|onehot        default none
            → 'none' triggers auto_encode (smart User Mode behaviour)
            → 'label'/'onehot' overrides with uniform manual encoding
        outlier_method        iqr|zscore               default iqr
        zscore_threshold      float  2.0–4.0           default 3.0
        outlier_strategy      cap|remove               default cap
    """
    def _float(key, default, lo=None, hi=None):
        try:
            v = float(request.form.get(key, default))
        except (TypeError, ValueError):
            v = default
        if lo is not None: v = max(v, lo)
        if hi is not None: v = min(v, hi)
        return v

    def _bool(key, default='true'):
        return request.form.get(key, default) == 'true'
    
    def _choice(key, choices, default):
        v = request.form.get(key, default)
        return v if v in choices else default
    
    # ── Read Developer Mode overrides (safe defaults = old behavior) ───────
    try:
        null_threshold = float(request.form.get('null_threshold', 0.4))
    except (TypeError, ValueError):
        null_threshold = 0.4
    null_threshold = min(max(null_threshold, 0.0), 1.0)

    null_fill_strategy = request.form.get('null_fill_strategy', 'median')
    if null_fill_strategy not in ('median', 'mean', 'mode'):
        null_fill_strategy = 'median'

    do_type_conversion = request.form.get('do_type_conversion', 'true') == 'true'
    do_remove_duplicates = request.form.get('do_remove_duplicates', 'true') == 'true'

    exclude_columns_raw = request.form.get('exclude_columns', '')
    exclude_columns = [c.strip() for c in exclude_columns_raw.split(',') if c.strip()]

     
    # ── Data purpose (general | ml | nlp) ───────────────────────────────────
    # Not a Developer Mode knob — collected in both modes from the "What will
    # you mainly use this dataset for?" selector. It decides how free-text
    # columns get cleaned:
    #   ml      → text columns get fully dropped/encoded away (old default)
    #   nlp     → text columns are KEPT as real, readable text
    #   general → same as ml (safe default when the person isn't sure yet)
    data_purpose = request.form.get('data_purpose', 'general')
    if data_purpose not in ('general', 'ml', 'nlp'):
        data_purpose = 'general'

    if data_purpose == 'nlp':
        text_action = 'keep'
    else:
        text_action = _choice('text_action', ('drop', 'hash', 'keep'), 'drop')
    text_unique_threshold = _float('text_unique_threshold', 0.6, 0.0, 1.0)

    encoding_method = request.form.get('encoding_method', 'none')
    if encoding_method not in ('none', 'label', 'onehot'):
        encoding_method = 'none'

    outlier_method = request.form.get('outlier_method', 'iqr')
    if outlier_method not in ('iqr', 'zscore'):
        outlier_method = 'iqr'

    try:
        zscore_threshold = float(request.form.get('zscore_threshold', 3.0))
    except (TypeError, ValueError):
        zscore_threshold = 3.0
    zscore_threshold = min(max(zscore_threshold, 2.0), 4.0)

    outlier_strategy = request.form.get('outlier_strategy', 'cap')
    if outlier_strategy not in ('cap', 'remove'):
        outlier_strategy = 'cap'

    # ── Load raw DF (still inside request context) ─────────────────────────
    df_raw = get_raw_df()

    if df_raw is None:
        # Fallback: file was sent directly
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No data available. Please upload a file first."}), 400
        clear_session_data()
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        loader = DataLoader(filepath)
        df_raw = loader.load()
        raw_path = save_session_df(df_raw, 'raw')
        session['raw_df_path'] = raw_path
        session['dataset_filename'] = file.filename
        _ctx = ChatContext()
        _ctx.update_raw_dataset(df_raw)
    else:
        _ctx = get_chat_context()

    # ── Reserve a clean_path slot in session NOW (inside request context) ──
    # We pre-generate the filename so the generator can write to it,
    # and the session already knows the path before streaming starts.
    clean_fname = f"clean_{uuid.uuid4().hex}.pkl"
    clean_path = os.path.join(SESSION_DATA_FOLDER, clean_fname)
    session['clean_df_path'] = clean_path
    session['pipeline_done'] = False

    # Reserve the report filename NOW too (same reason as clean_path above):
    # the generator runs OUTSIDE the request context and can't write back
    # into session, so report_view_url/report_download_url were never
    # actually being persisted — /eda and /eda_result always saw None for
    # them on any load after the live SSE event. Deciding the filename here
    # instead, while we still have a real request/session, fixes that.
    report_fname = f"report_{uuid.uuid4().hex[:8]}.html"
    session['report_view_url'] = f"/view/{report_fname}"
    session['report_download_url'] = f"/download/{report_fname}"

    # Reserve a log file path too — the generator below runs OUTSIDE the
    # request context, so it can't write to `session`, but it CAN append to
    # a plain file at a path we decide now. This is what lets /eda rebuild
    # the exact same pipeline dashboard (stages + console log) after
    # navigating away and back, instead of it only ever existing for the
    # one live SSE run.
    log_fname = f"pipelinelog_{uuid.uuid4().hex}.jsonl"
    log_path = os.path.join(SESSION_DATA_FOLDER, log_fname)
    session['pipeline_log_path'] = log_path
    open(log_path, "w", encoding="utf-8").close()  # start empty, before streaming begins
    # Lock in the purpose for THIS clean dataset — /nlp and /ml check this on
    # every request so a page whose columns were cleaned away for the other
    # purpose stays locked until Phase 1 is re-run with a matching purpose.
    session['data_purpose'] = data_purpose

    # ── Snapshot the ORIGINAL column schema (num/cat) from the RAW data ────
    # This must happen BEFORE type conversion / encoding, since those steps
    # can change a column's dtype (e.g. Label/One-Hot turns text into
    # numbers). Phase 2 visualization relies on this snapshot so categorical
    # columns are still recognized as categorical even after encoding.
    original_schema = compute_column_schema(df_raw)
    schema_path = save_column_schema(original_schema)
    session['schema_path'] = schema_path

    # Make sure this session's chat context is persisted to a known path
    # BEFORE streaming starts (generator runs outside the request context).
    save_chat_context(_ctx)
    _ctx_path = session['context_path']

    def send(stage, message, progress):
        entry = {
            "stage": stage, "message": message, "progress": progress,
            "time": datetime.now().strftime("%H:%M:%S"),
        }
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass
        return f"data: {json.dumps({'stage': stage, 'message': message, 'progress': progress, 'type': 'progress'})}\n\n"

    # Capture variables needed inside generator (avoids any session access)
    _df_raw = df_raw
    _clean_path = clean_path
    _report_fname = report_fname
    _log_path = log_path
    _null_threshold = null_threshold
    _null_fill_strategy = null_fill_strategy
    _do_type_conversion = do_type_conversion
    _do_remove_duplicates = do_remove_duplicates
    _exclude_columns = exclude_columns
    _text_action          = text_action
    _text_unique_threshold = text_unique_threshold
    _encoding_method = encoding_method
    _outlier_method = outlier_method
    _zscore_threshold = zscore_threshold
    _outlier_strategy = outlier_strategy
    _chat_ctx = _ctx
    _chat_ctx_path = _ctx_path

    def generate():
        try:
            yield send("Data Validation", "Engine started. Accessing data stream...", 8)
            time.sleep(0.6)

            yield send("Data Validation", f"File locked. Detected {_df_raw.shape[0]} rows.", 18)
            time.sleep(0.6)

            preprocessor = DataPreprocessor(_df_raw)

            # Step 1 — Exclude user-specified columns
            if _exclude_columns:
                yield send("Preprocessing", f"Excluding {len(_exclude_columns)} column(s) per configuration...", 22)
                preprocessor.exclude_columns(_exclude_columns)
                time.sleep(0.3)

            # Step 2 — Clean empty strings, empty rows, constant columns
            # (Phase 1: whitespace-only cells are normalized to real NaN so
            # handle_nulls() can actually see them, fully-empty rows are
            # dropped, and zero-information columns are removed before they
            # ever reach null-handling / encoding. Must run BEFORE
            # handle_nulls — EDAPipeline.run_pipeline() follows this same
            # order.)
            yield send("Preprocessing", "Cleaning empty cells, empty rows, and constant columns...", 28)
            preprocessor.clean_empty_strings()
            preprocessor.remove_empty_rows()
            preprocessor.remove_constant_columns()
            _chat_ctx.log_eda("Cleaned empty cells/rows and removed constant columns.")
            time.sleep(0.3)

            # Step 3 — Drop numeric ID columns (e.g. PassengerId)
            # yield send("Preprocessing", "Detecting and dropping numeric ID columns...", 20)
            # preprocessor.drop_id_columns()
            # time.sleep(0.2)

            # Step 4 — Handle nulls
            yield send("Preprocessing", "Scanning for missing values (Nulls)...", 35)
            preprocessor.handle_nulls(threshold=_null_threshold, fill_strategy=_null_fill_strategy)
            _chat_ctx.log_eda("Handled missing values.")

            # Step 5 — Handle high-cardinality text columns
            action_label = {"drop": "Dropping", "hash": "Hashing", "keep": "Keeping"}.get(_text_action, "Dropping")
            yield send(
                "Preprocessing",
                f"Analyzing text columns — {action_label} ID-like columns (>{_text_unique_threshold:.0%} unique)...",
                38
            )
            preprocessor.handle_text_columns(
                unique_threshold=_text_unique_threshold,
                action=_text_action
            )
            time.sleep(0.4)

            # Step 6 — Type conversion
            if _do_type_conversion:
                yield send("Preprocessing", "Applying smart type conversion...", 48)
                preprocessor.convert_types()
                _chat_ctx.log_eda("Converted data types.")
            else:
                yield send("Preprocessing", "Skipping type conversion (disabled)...", 48)

            # Step 7 — Remove duplicates
            if _do_remove_duplicates:
                preprocessor.remove_duplicates()
                _chat_ctx.log_eda("Removed duplicate rows.")

            # Step 8 — Encoding
            if _encoding_method == "none":
                # USER MODE — smart auto-encoding
                yield send("Preprocessing", "Auto-encoding categorical columns (smart mode)...", 62)
                preprocessor.auto_encode(onehot_max_unique=10)
                _chat_ctx.log_eda("Encoded categorical columns.")
            else:
                # DEVELOPER MODE — manual uniform encoding
                enc_label = "Label" if _encoding_method == "label" else "One-Hot"
                yield send("Preprocessing", f"Encoding categorical columns ({enc_label})...", 62)
                preprocessor.encode_categoricals(_encoding_method)
                _chat_ctx.log_eda("Encoded categorical columns.")
            time.sleep(0.4)


            # Step 9 — Outlier detection & handling
            # NOTE: OutlierHandler.__init__ only accepts `data` — it does
            # NOT accept an `original_schema` kwarg. Passing one through
            # here previously would raise TypeError the moment this line
            # actually ran. Removed to match the real OutlierHandler API
            # (see DataPreprocessor.handle_outliers docstring).
            method_label = "IQR" if _outlier_method == "iqr" else f"Z-Score (threshold={_zscore_threshold})"
            strategy_label = "Capping" if _outlier_strategy == "cap" else "Removal"
            yield send("Outlier Detection", f"Analyzing statistical distribution ({method_label} / {strategy_label})...", 75)
            preprocessor.handle_outliers(
                method=_outlier_method,
                strategy=_outlier_strategy,
                zscore_threshold=_zscore_threshold,
            )
            _chat_ctx.log_eda("Handled outliers.")
            clean_data = preprocessor.get_clean_data()
            time.sleep(0.6)
            
            # Step 10 — Generate report
            yield send("Report Generated", "Synthesizing intelligence report...", 95)
            out_file = _report_fname
            output_path = os.path.join(REPORTS_FOLDER, out_file)
            ReportGenerator(clean_data).generate_report(mode="detailed", file_name=output_path)

            # Step 11 — Persist clean DF to pre-agreed path
            # (session was already updated BEFORE the generator started)
            clean_data.to_pickle(_clean_path)

            # Make the CLEAN data queryable in the chatbot, alongside the
            # raw snapshot captured at upload time.
            _chat_ctx.update_clean_dataset(clean_data)
            # Save the context back to disk (outside request context, so we
            # write directly to the path captured before streaming began).
            save_context(_chat_ctx, _chat_ctx_path)
            # ✅ FIX: pipeline_done is already True in session (set before streaming).
            # We can't update session here (outside request context), so we rely on
            # the existence of the pickle file as the source of truth.
            # The /phase2/status endpoint checks get_clean_df() which checks the file.

            done_entry = {
                "stage": "Report Generated", "message": "Complete", "progress": 100,
                "time": datetime.now().strftime("%H:%M:%S"),
            }
            try:
                with open(_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(done_entry) + "\n")
                    f.write(json.dumps({
                        "stage": "Report Generated", "message": "Final Intelligence Deployed",
                        "progress": 100, "time": done_entry["time"],
                    }) + "\n")
            except OSError:
                pass

            yield f"data: {json.dumps({'done': True, 'stage': 'Report Generated', 'message': 'Complete', 'progress': 100, 'view_url': f'/view/{out_file}', 'download_url': f'/download/{out_file}', 'rows': len(clean_data), 'cols': len(clean_data.columns)})}\n\n"

        except Exception as e:
            yield send("Error", f"Engine failure: {str(e)}", 0)

    return Response(generate(), mimetype='text/event-stream')


# ── Static file routes ────────────────────────────────────────────────────────
@app.route('/view/<filename>')
def view(filename):
    return send_from_directory(REPORTS_FOLDER, filename)


@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(REPORTS_FOLDER, filename)
    return send_file(file_path, as_attachment=True)


# save mode 
@app.route("/set-mode", methods=["POST"])
def set_mode():

    data = request.get_json()

    session["mode"] = data.get("mode", "user")

    return jsonify({
        "status": "success"
    })

# ════════════════════════════════════════════════════════════
# PHASE 2 — Data Visualization Routes (uses clean DF from session)
# ════════════════════════════════════════════════════════════

PLOTS_FOLDER = os.path.join(BASE_DIR, 'Phase_2', 'plots')
os.makedirs(PLOTS_FOLDER, exist_ok=True)

NLP_PLOTS_FOLDER = os.path.join(BASE_DIR, 'Phase_3_NLP', 'plots')
os.makedirs(NLP_PLOTS_FOLDER, exist_ok=True)

ML_PLOTS_FOLDER = os.path.join(BASE_DIR, 'Phase_5_ML', 'plots')
os.makedirs(ML_PLOTS_FOLDER, exist_ok=True)

ML_MODELS_FOLDER = os.path.join(BASE_DIR, 'Phase_5_ML', 'saved_models')
os.makedirs(ML_MODELS_FOLDER, exist_ok=True)

# Lightweight registry for the explicit "Save Model" action in the ML GUI.
# Every trained model is ALREADY written to ML_MODELS_FOLDER automatically
# right after training (see MLPipeline._finish / ModelTrainer.save_model) —
# this registry doesn't duplicate that write. It just lets the person give
# a trained run a friendly name and confirms it's been "saved" in a way the
# UI can show back to them (a minimal saved-models list), without touching
# the training/eval flow itself.
ML_MODEL_REGISTRY_PATH = os.path.join(ML_MODELS_FOLDER, 'registry.json')


def _load_model_registry() -> dict:
    if not os.path.exists(ML_MODEL_REGISTRY_PATH):
        return {}
    try:
        with open(ML_MODEL_REGISTRY_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_model_registry(registry: dict) -> None:
    with open(ML_MODEL_REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)


def _is_textlike_dtype(series: pd.Series) -> bool:
    """
    Robust text-column check across pandas versions.
    Plain `dtype == object` breaks the moment a column is the newer pandas
    `string` / StringDtype / ArrowDtype (e.g. pandas 3.x with
    `future.infer_string` enabled, or any lib that flips that global
    option) — those compare False against `object` even though they hold
    text. `is_string_dtype` covers both worlds, so we OR the two checks.
    """
    try:
        return series.dtype == object or pd.api.types.is_string_dtype(series)
    except Exception:
        return False


def detect_text_columns(df: pd.DataFrame) -> list:
    """Heuristic: text-like columns whose cells look like free text (reviews,
    comments) rather than short categorical labels."""
    text_cols = []
    for col in df.columns:
        if _is_textlike_dtype(df[col]):
            sample = df[col].dropna().astype(str).head(200)
            if sample.empty:
                continue
            avg_words = sample.str.split().str.len().mean()
            nunique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
            if avg_words and avg_words >= 4 and nunique_ratio > 0.05:
                text_cols.append(col)
    return text_cols

@app.route("/phase2")
def phase2():

    mode = session.get("mode", "user")

    has_dataset = get_raw_df() is not None or get_clean_df() is not None

    return render_template(
    "phase2_ui.html",
    mode=mode,
    active_page="visualization",
    has_dataset=has_dataset
)


@app.route('/phase2/status', methods=['GET'])
def phase2_status():
    """Returns whether the pipeline has been run and clean data is available."""
    clean_df = get_clean_df()
    if clean_df is not None:
        original_schema = get_original_schema()
        return jsonify({
            "status": "ready",
            "rows": clean_df.shape[0],
            "cols": clean_df.shape[1],
            "columns": [
                {"name": col, "type": resolve_column_type(col, clean_df, original_schema)}
                for col in clean_df.columns
            ],
            # Every chart already generated this session, keyed by chart_type
            # — lets /phase2 rebuild the whole gallery + axis selections on
            # a revisit instead of starting blank.
            "last_results": get_page_state('phase2_state_path'),
        })
    return jsonify({"status": "no_data"})


@app.route('/phase2/detect-columns', methods=['POST'])
def phase2_detect_columns():
    """
    If clean DF exists in session → use it (no file needed).
    Otherwise accept an uploaded file, run full preprocessing, then return schema.
    """
    clean_df = get_clean_df()

    if clean_df is None:
        # No preprocessed data yet — need a file
        file = request.files.get('file')
        if not file:
            return jsonify({"status": "error", "message": "No preprocessed data available. Please run the pipeline first."}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            loader = DataLoader(filepath)
            df_raw = loader.load()
            if df_raw is None or df_raw.empty:
                return jsonify({"status": "error", "message": "Empty or invalid file."}), 400

            # New file being fed in directly → treat as a brand new dataset
            clear_session_data()
            ctx = ChatContext()
            ctx.update_raw_dataset(df_raw)

            # Snapshot the ORIGINAL schema before any type-changing step
            original_schema = compute_column_schema(df_raw)
            schema_path = save_column_schema(original_schema)
            session['schema_path'] = schema_path

            # Full preprocessing
            preprocessor = DataPreprocessor(df_raw)
            # Phase 1: normalize empty/whitespace-only text to NaN, drop
            # fully-empty rows, and drop zero-information (constant)
            # columns — must run BEFORE handle_nulls so it can see them.
            preprocessor.clean_empty_strings()
            preprocessor.remove_empty_rows()
            preprocessor.remove_constant_columns()
            ctx.log_eda("Cleaned empty cells/rows and removed constant columns.")
            preprocessor.handle_nulls()
            ctx.log_eda("Handled missing values.")
            # preprocessor.drop_id_columns()
            preprocessor.handle_text_columns(unique_threshold=0.6)          
            preprocessor.convert_types()
            ctx.log_eda("Converted data types.")
            preprocessor.remove_duplicates()
            ctx.log_eda("Removed duplicate rows.")
            preprocessor.auto_encode()  
            ctx.log_eda("Encoded categorical columns.")

            # NOTE: OutlierHandler.__init__ only accepts `data`, not
            # `original_schema` — dropped from this call to match the real
            # API (see DataPreprocessor.handle_outliers docstring).
            preprocessor.handle_outliers()
            clean_data = preprocessor.get_clean_data()
            ctx.log_eda("Handled outliers.")

            # Save to session
            clean_path = save_session_df(clean_data, 'clean')
            session['clean_df_path'] = clean_path
            session['pipeline_done'] = True
            clean_df = clean_data
            ctx.update_clean_dataset(clean_data)
            save_chat_context(ctx)

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    try:
        original_schema = get_original_schema()
        columns = []
        for col in clean_df.columns:
            col_type = resolve_column_type(col, clean_df, original_schema)
            columns.append({"name": col, "type": col_type})

        return jsonify({
            "status": "success",
            "columns": columns,
            "rows": clean_df.shape[0],
            "cols": clean_df.shape[1],
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/phase2/generate', methods=['POST'])
def phase2_generate():
    """
    Uses the clean (preprocessed) DF from session.
    Falls back to uploaded file if session data missing.
    """
    chart_type = request.form.get('chart_type', '')

    # Snapshot every non-file form field now (used below to persist this
    # chart's exact settings, so /phase2 can restore the same axis/column
    # picks on the next page load — not just the resulting image).
    params_out = {}
    for k in request.form.keys():
        if k == 'chart_type':
            continue
        vals = request.form.getlist(k)
        params_out[k] = vals if len(vals) > 1 else vals[0]

    # Get clean DF from session
    df = get_clean_df()

    if df is None:
        # Fallback: file was sent
        file = request.files.get('file')
        if not file:
            return jsonify({"status": "error", "message": "No data available. Run the pipeline first."}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        loader = DataLoader(filepath)
        df = loader.load()
        if df is None or df.empty:
            return jsonify({"status": "error", "message": "Empty dataset."}), 400

    try:
        viz = DataVisualizer(df)
        viz.plots_dir = PLOTS_FOLDER
        ctx = get_chat_context()

        output_path = None
        plot_type = "png"

        if chart_type == "summary_dashboard":
            output_path = viz.generate_summary_dashboard(save=True)
            ctx.log_visualization(chart_type, describe_dataset_highlights(df))

        elif chart_type == "correlation_heatmap":
            output_path = viz.plot_correlation_heatmap(save=True)
            ctx.log_visualization(
                chart_type,
                f"Correlation heatmap. {describe_pairwise_correlation(df)}"
            )

        elif chart_type == "scatter_2d":
            col1 = request.form.get("col1", "")
            col2 = request.form.get("col2", "")
            color_col = request.form.get("color_col") or None
            output_path = viz.plot_scatter_2d(col1, col2, color_col=color_col, save=True)
            plot_type = "html"
            ctx.log_visualization(
                chart_type,
                f"2D scatter plot of {col1} vs {col2}. {describe_pairwise_correlation(df, [col1, col2])}"
            )

        elif chart_type == "scatter_3d":
            col1 = request.form.get("col1", "")
            col2 = request.form.get("col2", "")
            col3 = request.form.get("col3", "")
            color_col = request.form.get("color_col") or None
            output_path = viz.plot_scatter_3d(col1, col2, col3, color_col=color_col, save=True)
            plot_type = "html"
            ctx.log_visualization(
                chart_type,
                f"3D scatter plot of {col1}, {col2}, {col3}. {describe_pairwise_correlation(df, [col1, col2, col3])}"
            )

        elif chart_type == "joint_plot":
            col1 = request.form.get("col1", "")
            col2 = request.form.get("col2", "")
            kind = request.form.get("kind", "scatter")
            output_path = viz.plot_joint_plot(col1, col2, kind=kind, save=True)
            ctx.log_visualization(
                chart_type,
                f"Joint plot ({kind}) of {col1} vs {col2}. {describe_pairwise_correlation(df, [col1, col2])}"
            )
        elif chart_type == "stacked_bar":
            col1 = request.form.get("col1", "")
            col2 = request.form.get("col2", "")
            normalize = request.form.get("normalize", "false") == "true"
            output_path = viz.plot_stacked_bar(col1, col2, normalize=normalize, save=True)
            ctx.log_visualization(
                chart_type,
                f"Stacked bar chart of {col1} vs {col2}. {describe_crosstab(df, col1, col2)}"
            )

        elif chart_type == "cross_tabulation":
            col1 = request.form.get("col1", "")
            col2 = request.form.get("col2", "")
            output_path = viz.plot_cross_tabulation(col1, col2, save=True)
            ctx.log_visualization(
                chart_type,
                describe_crosstab(df, col1, col2)
            )

        elif chart_type == "violin_plot":
            num_col = request.form.get("num_col", "")
            cat_col = request.form.get("cat_col", "")
            output_path = viz.plot_violin_plot_by_category(num_col, cat_col, save=True)
            ctx.log_visualization(
                chart_type,
                f"Violin plot. {describe_group_stats(df, num_col, cat_col)}"
            )

        elif chart_type == "facet_grid":
            num_cols = request.form.getlist("num_cols")
            cat_col = request.form.get("cat_col", "")
            output_path = viz.plot_facet_grid(num_cols, cat_col, save=True)
            facet_desc = "; ".join(describe_group_stats(df, nc, cat_col) for nc in num_cols)
            ctx.log_visualization(chart_type, f"Facet grid by {cat_col}. {facet_desc}")

        elif chart_type == "bubble_chart":
            x = request.form.get("x", "")
            y = request.form.get("y", "")
            size = request.form.get("size", "")
            color = request.form.get("color") or None
            output_path = viz.plot_bubble_chart(x, y, size, color=color, save=True)
            plot_type = "html"
            ctx.log_visualization(
                chart_type,
                f"Bubble chart of {x} vs {y} (size={size}). {describe_pairwise_correlation(df, [x, y, size])}"
            )
        elif chart_type == "automatic_dashboard":
            output_path = viz.generate_automatic_dashboard(save=True)
            plot_type = "html"
            ctx.log_visualization(chart_type, describe_dataset_highlights(df))

        else:
            return jsonify({"status": "error", "message": f"Unknown chart type: {chart_type}"}), 400

        if not output_path or not os.path.exists(output_path):
            return jsonify({"status": "error", "message": "Plot file was not created."}), 500

        plot_filename = os.path.basename(output_path)

        save_chat_context(ctx)

        result_payload = {
            "status": "success",
            "chart_type": chart_type,
            "plot_type": plot_type,
            "view_url": f"/phase2/view/{plot_filename}",
            "download_url": f"/phase2/download/{plot_filename}",
        }

        # Persist this chart (its exact form choices + the resulting image)
        # keyed by chart_type, so /phase2/status can hand it back on the
        # next page load and the whole gallery reappears without
        # regenerating anything.
        update_page_state('phase2_state_path', chart_type,
                           {"params": params_out, "result": result_payload}, "phase2_state")

        return jsonify(result_payload)

    except Exception as e:
        print("Phase 2 Error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/phase2/view/<filename>')
def phase2_view(filename):
    return send_from_directory(PLOTS_FOLDER, filename)


@app.route('/phase2/download/<filename>')
def phase2_download(filename):
    file_path = os.path.join(PLOTS_FOLDER, filename)
    return send_file(file_path, as_attachment=True)

#ChatBot
@app.route("/chat/status")
def chat_status():
    ctx = get_chat_context()
    return jsonify({
        "dataset": ctx.dataset_context.get_context(),
        "eda": ctx.eda_context.get_context(),
        "visualization": ctx.visualization_context.get_context(),
    })


@app.route("/documents")
def documents():
    ctx = get_chat_context()
    docs = build_documents(ctx)

    return jsonify([
        {"source": doc.metadata, "content": doc.page_content}
        for doc in docs
    ])


@app.route("/rag/test")
def rag_test():
    ctx = get_chat_context()
    docs = build_documents(ctx)

    vector = VectorStore(chatbot.embeddings)
    if not docs:
        return jsonify([])

    vector.build(docs)
    results = vector.search("missing values")

    return jsonify([
        {"source": doc.metadata, "content": doc.page_content}
        for doc in results
    ])


@app.route("/chat", methods=["GET", "POST"])
def chat():

    if request.method == "GET":
        return jsonify({"message": "Chat endpoint is working"})

    data = request.json
    question = data["question"]

    session_id = get_session_id()
    ctx = get_chat_context()

    answer = chatbot.ask(question, ctx, session_id)

    # chatbot.ask() appended this turn to ctx.conversation_context — save it
    # so the NEXT question in this session can reference it as memory.
    save_chat_context(ctx)

    return jsonify({
        "answer": answer
    })

# ════════════════════════════════════════════════════════════
# PHASE 3 — NLP Routes
# ════════════════════════════════════════════════════════════
def purpose_locks_out(target: str) -> bool:
    """
    Once a dataset has been explicitly set up for NLP or ML during the EDA
    step, that choice is final for this dataset — the other path stays
    locked so cleaning done for one (raw text kept vs. fully encoded)
    is never silently reused for the other. 'general' locks nothing.
    """
    purpose = session.get('data_purpose', 'general')
    return purpose in ('nlp', 'ml') and purpose != target

@app.route('/nlp')
def nlp_page():
    mode = session.get("mode", "user")
    locked = purpose_locks_out('nlp')

    nlp_state = {"has_data": False}
    if not locked:
        df = get_raw_df()
        if df is None:
            df = get_clean_df()
        if df is not None:
            text_cols = detect_text_columns(df)
            all_object_cols = [c for c in df.columns if _is_textlike_dtype(df[c])]
            nlp_state = {
                "has_data": True,
                "rows": df.shape[0],
                "text_columns": text_cols,
                "all_text_like_columns": all_object_cols,
                "candidates": text_cols if text_cols else all_object_cols,
                "columns": list(df.columns),
            }
            # Restore the last analysis run (dev-mode field choices + full
            # results) so navigating away and back shows exactly what was
            # there before — only if it was run on a column that still
            # exists on the CURRENT dataset (stale otherwise).
            saved = get_page_state('nlp_state_path').get('_last')
            if saved and saved.get('params', {}).get('text_column') in df.columns:
                nlp_state['last_run'] = saved
    has_dataset = get_raw_df() is not None or get_clean_df() is not None
    return render_template("nlp_ui.html", mode=mode, active_page='nlp',
                            locked=locked, nlp_state=nlp_state,has_dataset=has_dataset)


@app.route('/nlp/status', methods=['GET'])
def nlp_status():
    """Returns candidate text columns from the RAW dataset (free text is
    usually stripped/encoded away by the Phase 1 clean pipeline)."""
    lock_message = purpose_lock_check('nlp')
    if lock_message:
        return jsonify({"status": "locked", "message": lock_message})

    df = get_raw_df()
    if df is None:
        df = get_clean_df()
    if df is None:
        return jsonify({"status": "no_data"})

    text_cols = detect_text_columns(df)
    all_object_cols = [c for c in df.columns if df[c].dtype == object]

    return jsonify({
        "status": "ready",
        "rows": df.shape[0],
        "text_columns": text_cols,
        "all_text_like_columns": all_object_cols,
        "columns": list(df.columns),
    })


@app.route('/nlp/analyze', methods=['POST'])
def nlp_analyze():
    lock_message = purpose_lock_check('nlp')
    if lock_message:
        return jsonify({"status": "locked", "message": lock_message})

    data = request.form
    text_column = data.get("text_column")
    if not text_column:
        return jsonify({"status": "error", "message": "text_column is required."}), 400

    df = get_raw_df()
    if df is None:
        df = get_clean_df()
    if df is None:
        return jsonify({"status": "error", "message": "No data available. Run Phase 1 first."}), 400
    if text_column not in df.columns:
        return jsonify({"status": "error", "message": f"Column '{text_column}' not found."}), 400

    auto = data.get("auto", "true") == "true"
    method = data.get("method", "tfidf")
    ngram_max = int(data.get("ngram_max", 1))
    top_n = int(data.get("top_n", 20))
    include_sentiment = data.get("include_sentiment", "true") == "true"
    include_trigrams = data.get("include_trigrams", "false") == "true"

    try:
        analyzer = NLPAnalyzer(df, text_column)
        result = analyzer.analyze(
            auto=auto,
            method=method,
            ngram_range=(1, max(1, ngram_max)),
            top_n=top_n,
            include_sentiment=include_sentiment,
            include_trigrams=include_trigrams,
        )

        # Unique suffix per run: without this, every analysis overwrote the
        # SAME filename (e.g. "keywords.png"), so the browser kept showing
        # the old cached image at that URL even after re-running the
        # analysis — and two sessions running NLP at the same time would
        # clobber each other's plot files on disk.
        run_id = uuid.uuid4().hex[:8]

        viz = NLPVisualizer(plots_dir=NLP_PLOTS_FOLDER)
        plots = {
            "word_frequency": os.path.basename(
                viz.plot_word_frequency(result["word_frequency"], filename=f"word_frequency_{run_id}.png")
            ),
            "keywords": os.path.basename(
                viz.plot_keywords(result["keywords"], filename=f"keywords_{run_id}.png")
            ),
            "word_cloud": os.path.basename(
                viz.plot_wordcloud(result["word_frequency"], filename=f"wordcloud_{run_id}.png")
            ),
            "bigrams": os.path.basename(
                viz.plot_bigrams(result["bigrams"], filename=f"bigrams_{run_id}.png")
            ),
            "document_length": os.path.basename(
                viz.plot_document_length_histogram(result["document_lengths"], filename=f"doclength_{run_id}.png")
            ),
        }

        if "trigrams" in result:
            plots["trigrams"] = os.path.basename(
                viz.plot_trigrams(result["trigrams"], filename=f"trigrams_{run_id}.png")
            )

        before_after = result["before_after"]
        plots["vocab_before_after"] = os.path.basename(
            viz.plot_vocab_before_after(
                before_after["before"]["vocabulary_size"],
                before_after["after"]["vocabulary_size"],
                filename=f"vocab_ba_{run_id}.png",
            )
        )
        plots["avg_words_before_after"] = os.path.basename(
            viz.plot_avg_words_before_after(
                before_after["before"]["avg_word_count"],
                before_after["after"]["avg_word_count"],
                filename=f"avgwords_ba_{run_id}.png",
            )
        )

        summary_line = ""
        if include_sentiment:
            sentiment = result["sentiment"]
            plots["sentiment_distribution"] = os.path.basename(
                viz.plot_sentiment_distribution(sentiment["distribution"], filename=f"sentiment_{run_id}.png")
            )
            summary_line = (
                f"Lexicon-based sentiment on '{text_column}': "
                f"{sentiment['distribution']} ({sentiment['dominant_sentiment']} dominant). "
            )
            # Concrete examples, not just the aggregate counts above — this is
            # what lets the chatbot answer "what's the most negative review?"
            # instead of only reporting percentages.
            if sentiment.get("top_negative"):
                neg_lines = "; ".join(
                    f"\"{ex['text']}\" (score {ex['score']})" for ex in sentiment["top_negative"]
                )
                summary_line += f"Most negative example(s): {neg_lines}. "
            if sentiment.get("top_positive"):
                pos_lines = "; ".join(
                    f"\"{ex['text']}\" (score {ex['score']})" for ex in sentiment["top_positive"]
                )
                summary_line += f"Most positive example(s): {pos_lines}. "

        stats = result["statistics"]
        keyword_line = ", ".join(k["term"] for k in result["keywords"][:10])
        bigram_line = ", ".join(b["term"] for b in result["bigrams"][:10])
        trigram_line = ", ".join(t["term"] for t in result.get("trigrams", [])[:10])
        ba = result["before_after"]

        description = (
            f"{stats['documents']} documents, avg {stats['avg_word_count']} words each "
            f"(min {stats['min_word_count']}, max {stats['max_word_count']}), "
            f"vocabulary size {stats['vocabulary_size']}, {stats['empty_documents']} empty documents. "
            f"Top keywords: {keyword_line}. "
            f"Top bigrams (2-word phrases): {bigram_line}. "
            + (f"Top trigrams (3-word phrases): {trigram_line}. " if trigram_line else "")
            + f"Cleaning impact: vocabulary went from {ba['before']['vocabulary_size']} to "
              f"{ba['after']['vocabulary_size']} unique words, average length from "
              f"{ba['before']['avg_word_count']} to {ba['after']['avg_word_count']} words. "
            + summary_line
        )

        ctx = get_chat_context()
        ctx.log_nlp(text_column, description)
        # Feed the actual text content (capped sample) so the chatbot can
        # answer content questions, not just questions about the summary.
        ctx.log_nlp_raw_texts(text_column, analyzer.raw_text_samples())
        save_chat_context(ctx)

        result["plots"] = plots
        result["status"] = "success"

        # Persist this run (the exact form choices + the full result) so
        # /nlp can restore it on the next page load without re-analyzing.
        update_page_state('nlp_state_path', '_last', {
            "params": {
                "text_column": text_column,
                "auto": auto,
                "method": method,
                "ngram_max": ngram_max,
                "top_n": top_n,
                "include_sentiment": include_sentiment,
                "include_trigrams": include_trigrams,
            },
            "result": result,
        }, "nlp_state")

        return jsonify(result)

    except Exception as e:
        print("NLP Error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/nlp/view/<filename>')
def nlp_view(filename):
    return send_from_directory(NLP_PLOTS_FOLDER, filename)


@app.route('/nlp/download/<filename>')
def nlp_download(filename):
    return send_file(os.path.join(NLP_PLOTS_FOLDER, filename), as_attachment=True)


@app.route('/nlp/export', methods=['POST'])
def nlp_export():
    """Download the dataset with a new '<text_column>_clean' column added
    (stopwords removed, lemmatized) — the original column is left untouched,
    so this is safe to feed into Phase 5 (ML) or keep as a report artifact."""
    lock_message = purpose_lock_check('nlp')
    if lock_message:
        return jsonify({"status": "locked", "message": lock_message})

    data = request.form
    text_column = data.get("text_column")
    if not text_column:
        return jsonify({"status": "error", "message": "text_column is required."}), 400

    df = get_raw_df()
    if df is None:
        df = get_clean_df()
    if df is None:
        return jsonify({"status": "error", "message": "No data available. Run Phase 1 first."}), 400
    if text_column not in df.columns:
        return jsonify({"status": "error", "message": f"Column '{text_column}' not found."}), 400

    try:
        analyzer = NLPAnalyzer(df, text_column)
        out_df = analyzer.export_dataframe()
        buf = io.BytesIO()
        out_df.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(
            buf,
            as_attachment=True,
            download_name="processed_dataset.csv",
            mimetype="text/csv",
        )
    except Exception as e:
        print("NLP Export Error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# ════════════════════════════════════════════════════════════
# PHASE 5 — ML Routes
# ════════════════════════════════════════════════════════════

@app.route('/ml')
def ml_page():
    mode = session.get("mode", "user")
    locked = purpose_locks_out('ml')

    ml_state = {"has_data": False}
    if not locked:
        df = get_clean_df()
        if df is not None:
            trainer = ModelTrainer(df)
            guessed_target = trainer.guess_target_column()
            task_type = trainer.detect_task_type(guessed_target)
            ml_state = {
                "has_data": True,
                "rows": df.shape[0],
                "cols": df.shape[1],
                "columns": list(df.columns),
                "guessed_target": guessed_target,
                "guessed_task_type": task_type,
                "available_models": {
                    "classification": ModelFactory.available_models("classification"),
                    "regression": ModelFactory.available_models("regression"),
                },
            }
            # Restore the last training run (target/model/hyperparam choices
            # + full trained-model results) so navigating away and back
            # shows the same trained model instead of an empty form — only
            # if it was trained against a target that still exists on the
            # CURRENT clean dataset (stale otherwise, e.g. after re-running
            # Phase 1 with different columns).
            saved = get_page_state('ml_state_path').get('_last')
            if saved and saved.get('params', {}).get('target_column') in df.columns:
                ml_state['last_run'] = saved
    has_dataset = get_raw_df() is not None or get_clean_df() is not None
    return render_template("ml_ui.html", mode=mode, active_page='ml',
                            locked=locked, ml_state=ml_state,has_dataset=has_dataset)


@app.route('/ml/status', methods=['GET'])
def ml_status():
    lock_message = purpose_lock_check('ml')
    if lock_message:
        return jsonify({"status": "locked", "message": lock_message})

    df = get_clean_df()
    if df is None:
        return jsonify({"status": "no_data"})

    trainer = ModelTrainer(df)
    guessed_target = trainer.guess_target_column()
    task_type = trainer.detect_task_type(guessed_target)

    # Transparency check: any ORIGINAL (as-uploaded) column that has no
    # trace at all in the clean/encoded data was dropped upstream in Phase 1
    # (excluded manually, dropped as an ID column, or dropped as
    # high-cardinality free text) — not by anything in Phase 5. Surfacing
    # this here means the person always knows why a feature they expect to
    # see isn't in the target/feature pickers, instead of it just vanishing.
    raw_df = get_raw_df()
    missing_raw_columns = FeatureSchema.missing_raw_columns(raw_df, df) if raw_df is not None else []

    return jsonify({
        "status": "ready",
        "rows": df.shape[0],
        "cols": df.shape[1],
        "columns": list(df.columns),
        "guessed_target": guessed_target,
        "guessed_task_type": task_type,
        "available_models": {
            "classification": ModelFactory.available_models("classification"),
            "regression": ModelFactory.available_models("regression"),
        },
        "hyperparam_specs": ModelFactory.HYPERPARAM_SPECS,
        "raw_column_count": len(raw_df.columns) if raw_df is not None else None,
        "missing_raw_columns": missing_raw_columns,
    })


@app.route('/ml/recommend', methods=['POST'])
def ml_recommend():
    """
    User Mode, step 1: cross-validate the candidate pool for the chosen
    target and return a ranked list + the top pick, WITHOUT training or
    saving a final model. The frontend shows this as a "Recommended: X"
    card and lets the person accept it or choose a different candidate
    before /ml/train actually trains anything.
    """
    lock_message = purpose_lock_check('ml')
    if lock_message:
        return jsonify({"status": "locked", "message": lock_message})

    df = get_clean_df()
    if df is None:
        return jsonify({"status": "error", "message": "No preprocessed data available. Run Phase 1 first."}), 400

    target_column = request.form.get("target_column") or None
    if not target_column:
        return jsonify({"status": "error", "message": "Please choose a target column first."}), 400

    try:
        pipeline = MLPipeline(df, plots_dir=ML_PLOTS_FOLDER)
        result = pipeline.recommend_models(target_column=target_column)
        result["status"] = "success"
        return jsonify(result)
    except Exception as e:
        print("ML Recommend Error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/ml/feature_schema', methods=['GET'])
def ml_feature_schema():
    """
    Returns the ORIGINAL (pre-encoding) identity of EVERY column in the
    clean/encoded dataset — e.g. a One-Hot group like 'Gender_Male'/
    'Gender_Female' collapses into one 'Gender' entry with ['Male',
    'Female'] as options, and a Label-Encoded column reports its real
    category names instead of codes. Used by the target picker, the
    Developer Mode feature checklist, and the prediction form in both
    modes, so people never have to read or type raw encoded values.

    NOTE: this always computes the schema for ALL columns and ignores any
    '?columns=' filter. Filtering used to happen server-side based on a
    comma-separated column list built into the URL — with a wide one-hot
    encoded dataset (many categorical columns each exploded into several
    columns) that list can be long enough to be silently truncated by the
    browser/an intermediary, which used to come back as "only some of my
    features showed up". Returning everything and letting the frontend
    narrow it down client-side removes that failure mode entirely.
    """
    lock_message = purpose_lock_check('ml')
    if lock_message:
        return jsonify({"status": "locked", "message": lock_message})

    clean_df = get_clean_df()
    if clean_df is None:
        return jsonify({"status": "error", "message": "No preprocessed data available. Run Phase 1 first."}), 400

    raw_df = get_raw_df()  # may be None on older sessions — schema falls back to numeric-only

    try:
        schema = FeatureSchema.build(raw_df, clean_df)
        features = FeatureSchema.to_logical_features(schema, list(clean_df.columns))
        return jsonify({"status": "success", "features": features})
    except Exception as e:
        print("Feature Schema Error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/ml/train', methods=['POST'])
def ml_train():
    lock_message = purpose_lock_check('ml')
    if lock_message:
        return jsonify({"status": "locked", "message": lock_message})

    df = get_clean_df()
    if df is None:
        return jsonify({"status": "error", "message": "No preprocessed data available. Run Phase 1 first."}), 400

    data = request.form
    mode = data.get("mode", "user")

    try:
        pipeline = MLPipeline(df, plots_dir=ML_PLOTS_FOLDER)

        if mode == "user":
            target_column = data.get("target_column") or None
            if not target_column:
                return jsonify({"status": "error", "message": "Please choose a target column first."}), 400
            # Set when the person accepted the recommendation or picked an
            # alternative from the /ml/recommend step. If absent, run_auto
            # falls back to searching the candidate pool itself.
            model_name = data.get("model_name") or None
            result = pipeline.run_auto(target_column=target_column, model_name=model_name)
        else:
            target_column = data.get("target_column")
            model_name = data.get("model_name", "random_forest")
            task_type = data.get("task_type") or None
            feature_columns = request.form.getlist("feature_columns") or None
            test_size = float(data.get("test_size", 0.2))
            scale = data.get("scale", "false") == "true"

            if not target_column:
                return jsonify({"status": "error", "message": "Please choose a target column first."}), 400

            resolved_task_type = task_type or pipeline.trainer.detect_task_type(target_column)
            spec = ModelFactory.hyperparam_spec(resolved_task_type, model_name)

            hyperparams = {}
            for field in spec:
                raw_val = data.get(field["name"])
                if raw_val in (None, ""):
                    continue
                if field["type"] == "int":
                    hyperparams[field["name"]] = int(raw_val)
                elif field["type"] == "float":
                    hyperparams[field["name"]] = float(raw_val)
                elif field["type"] == "bool":
                    hyperparams[field["name"]] = raw_val == "true"
                else:  # select / string options e.g. kernel, weights
                    hyperparams[field["name"]] = raw_val

            result = pipeline.run_manual(
                target_column=target_column,
                model_name=model_name,
                task_type=task_type,
                feature_columns=feature_columns,
                test_size=test_size,
                scale=scale,
                hyperparams=hyperparams,
            )

        description = (
            f"Trained {result['model_name']} ({result['task_type']}) on target "
            f"'{result['target_column']}' using {len(result['feature_columns'])} features. "
            f"Metrics: {result['metrics']}."
        )
        ctx = get_chat_context()
        ctx.log_model(
            model_id=result["model_id"],
            model_name=result["model_name"],
            task_type=result["task_type"],
            target_column=result["target_column"],
            metrics=result["metrics"],
            mode=result["mode"],
        )
        ctx.log_prediction({"type": "training_run", "summary": description})
        save_chat_context(ctx)

        session[f"ml_model_path_{result['model_id']}"] = os.path.join(ML_MODELS_FOLDER, f"model_{result['model_id']}.pkl")

        result["status"] = "success"

        # Persist this run (exact form choices + the full trained-model
        # result) so /ml can restore it on the next page load without
        # re-training. Saved regardless of mode — target_column and
        # model_name always end up set one way or another above.
        run_params = {
            "mode": mode,
            "target_column": result.get("target_column"),
            "model_name": result.get("model_name"),
            "task_type": result.get("task_type"),
            "feature_columns": result.get("feature_columns"),
        }
        if mode != "user":
            run_params.update({
                "test_size": data.get("test_size", 0.2),
                "scale": data.get("scale", "false") == "true",
                "hyperparams": hyperparams,
            })
        update_page_state('ml_state_path', '_last', {
            "params": run_params,
            "result": result,
        }, "ml_state")

        return jsonify(result)

    except Exception as e:
        print("ML Error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/ml/predict', methods=['POST'])
def ml_predict():
    data = request.get_json(force=True)
    model_id = data.get("model_id")
    row = data.get("row", {})

    model_path = session.get(f"ml_model_path_{model_id}")
    if not model_path or not os.path.exists(model_path):
        model_path = os.path.join(ML_MODELS_FOLDER, f"model_{model_id}.pkl")
        if not os.path.exists(model_path):
            return jsonify({"status": "error", "message": "Model not found. Train a model first."}), 404

    try:
        predictor = Predictor.from_file(model_path)
        prediction = predictor.predict_row(row)

        # De-encode the prediction back to the ORIGINAL target label — e.g.
        # a Label-Encoded target trained as 0/1 should come back as "No"/"Yes",
        # never as the raw numeric code the model actually predicted.
        if predictor.task_type == "classification":
            raw_df = get_raw_df()
            clean_df = get_clean_df()
            if raw_df is not None and clean_df is not None and predictor.target_column in clean_df.columns:
                schema = FeatureSchema.build(raw_df, clean_df)
                target_info = schema.get(predictor.target_column)
                if target_info and target_info["type"] == "categorical_label":
                    value_map = target_info["value_map"]  # encoded code (str) -> original label
                    raw_pred = prediction["prediction"]
                    if isinstance(raw_pred, float) and raw_pred.is_integer():
                        key = str(int(raw_pred))
                    else:
                        key = str(raw_pred)
                    if key in value_map:
                        prediction["prediction"] = value_map[key]
                    if "probabilities" in prediction:
                        prediction["probabilities"] = {
                            value_map.get(k, k): v for k, v in prediction["probabilities"].items()
                        }

        ctx = get_chat_context()
        ctx.log_prediction({"type": "single_prediction", **prediction, "input": row})
        save_chat_context(ctx)

        prediction["status"] = "success"
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/ml/view/<filename>')
def ml_view(filename):
    return send_from_directory(ML_PLOTS_FOLDER, filename)


@app.route('/ml/download/<filename>')
def ml_download(filename):
    return send_file(os.path.join(ML_PLOTS_FOLDER, filename), as_attachment=True)

@app.route('/eda/download-data')
def eda_download_data():
    """Download the cleaned dataset (post Phase 1 pipeline) as a CSV file."""
    df = get_clean_df()
    if df is None:
        return jsonify({"status": "error", "message": "No cleaned data available yet."}), 404
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name="cleaned_data.csv",
        mimetype="text/csv",
    )

@app.route('/ml/download-model/<model_id>')
def ml_download_model(model_id):
    """Download the trained model (features, scaler, and metadata included) as a .pkl file."""
    # model_id is a uuid4 hex fragment generated server-side (see
    # MLPipeline._finish) — safe to use directly in a path, but guard
    # against path traversal regardless since it comes from the URL.
    safe_id = re.sub(r'[^a-zA-Z0-9]', '', model_id)
    model_path = os.path.join(ML_MODELS_FOLDER, f"model_{safe_id}.pkl")
    if not os.path.exists(model_path):
        return jsonify({"status": "error", "message": "Model not found."}), 404
    return send_file(model_path, as_attachment=True, download_name=f"model_{safe_id}.pkl")


@app.route('/ml/save-model', methods=['POST'])
def ml_save_model():
    """
    Explicit "Save Model" action from the ML results GUI (Developer Mode
    and User Mode both hit this — it's mode-agnostic). The trained model
    artifact (.pkl bundling the model, scaler, feature_columns, task_type,
    target_column, labels) is already written to disk automatically right
    after training via MLPipeline._finish -> ModelTrainer.save_model, so
    this endpoint never re-trains or re-pickles anything. Its only job is
    to: confirm that artifact genuinely exists, attach a friendly display
    name (or a sensible generated default) plus a small metadata snapshot
    to ML_MODEL_REGISTRY_PATH, and give the UI an explicit success/failure
    signal it can show the person, all without touching the training or
    evaluation flow.
    """
    model_id = (request.form.get("model_id") or "").strip()
    if not model_id:
        return jsonify({"status": "error", "message": "Missing model_id — train a model first."}), 400

    # model_id is expected to be the uuid4 hex fragment MLPipeline._finish
    # generates; stripped down regardless since it ends up in a filesystem
    # path.
    safe_id = re.sub(r'[^a-zA-Z0-9]', '', model_id)
    model_path = os.path.join(ML_MODELS_FOLDER, f"model_{safe_id}.pkl")
    if not os.path.exists(model_path):
        return jsonify({"status": "error", "message": "Model artifact not found on disk. Train it again before saving."}), 404

    try:
        bundle = ModelTrainer.load_model(model_path)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Could not read the trained model file: {e}"}), 500

    custom_name = (request.form.get("model_name") or "").strip()
    default_name = f"{bundle.get('model_name', 'model')}_{bundle.get('target_column', safe_id)}"
    display_name = custom_name or default_name

    registry = _load_model_registry()
    registry[safe_id] = {
        "model_id": safe_id,
        "display_name": display_name,
        "model_name": bundle.get("model_name"),
        "task_type": bundle.get("task_type"),
        "target_column": bundle.get("target_column"),
        "feature_count": len(bundle.get("feature_columns") or []),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "filename": f"model_{safe_id}.pkl",
    }

    try:
        _save_model_registry(registry)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Could not persist the saved-models list: {e}"}), 500

    return jsonify({"status": "success", "model_id": safe_id, "saved_name": display_name})


@app.route('/ml/saved-models', methods=['GET'])
def ml_saved_models():
    """Lists every model explicitly saved via the 'Save Model' button, most recently saved first."""
    registry = _load_model_registry()
    items = sorted(registry.values(), key=lambda r: r.get("saved_at", ""), reverse=True)
    return jsonify({"status": "success", "models": items})


if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)