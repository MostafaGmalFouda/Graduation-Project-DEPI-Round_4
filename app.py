from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, Response, session
import os
import json
import time
import uuid
import pickle
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
    for key in ('raw_df_path', 'clean_df_path', 'schema_path', 'context_path'):
        path = session.pop(key, None)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    session['pipeline_done'] = False
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


def describe_dataset_highlights(df: pd.DataFrame) -> str:
    """A short, factual overview used for the summary/automatic dashboards."""
    numeric_df = df.select_dtypes(include="number")
    parts = [f"{df.shape[0]} rows, {df.shape[1]} columns."]
    if numeric_df.shape[1] >= 2:
        parts.append("Correlations: " + describe_pairwise_correlation(df))
    return " ".join(parts)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', active_page='eda', hide_sidebar_initially=True)


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

     
    text_action           = _choice('text_action', ('drop','hash','keep'), 'drop')
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
        return f"data: {json.dumps({'stage': stage, 'message': message, 'progress': progress, 'type': 'progress'})}\n\n"

    # Capture variables needed inside generator (avoids any session access)
    _df_raw = df_raw
    _clean_path = clean_path
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
                yield send("Preprocessing", f"Excluding {len(_exclude_columns)} column(s) per configuration...", 25)
                preprocessor.exclude_columns(_exclude_columns)
                time.sleep(0.3)
            
            # Step 2 — Drop numeric ID columns (e.g. PassengerId)
            # yield send("Preprocessing", "Detecting and dropping numeric ID columns...", 20)
            # preprocessor.drop_id_columns()
            # time.sleep(0.2)

            # Step 3 — Handle nulls
            yield send("Preprocessing", "Scanning for missing values (Nulls)...", 35)
            preprocessor.handle_nulls(threshold=_null_threshold, fill_strategy=_null_fill_strategy)
            _chat_ctx.log_eda("Handled missing values.")
            # Step 4 — Handle high-cardinality text columns  
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

            # Step 5 — Type conversion
            if _do_type_conversion:
                yield send("Preprocessing", "Applying smart type conversion...", 48)
                preprocessor.convert_types()
                _chat_ctx.log_eda("Converted data types.")
            else:
                yield send("Preprocessing", "Skipping type conversion (disabled)...", 48)

            # Step 6 — Remove duplicates
            if _do_remove_duplicates:
                preprocessor.remove_duplicates()
                _chat_ctx.log_eda("Removed duplicate rows.")
            # Step 7 — Encoding
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

            clean_data = preprocessor.get_clean_data()

            # Step 8 — Outlier detection & handling
            method_label = "IQR" if _outlier_method == "iqr" else f"Z-Score (threshold={_zscore_threshold})"
            strategy_label = "Capping" if _outlier_strategy == "cap" else "Removal"
            yield send("Outlier Detection", f"Analyzing statistical distribution ({method_label} / {strategy_label})...", 75)
            outlier_handler = OutlierHandler(clean_data)
            if _outlier_method == "iqr":
                outlier_handler.detect_iqr()
            else:
                outlier_handler.detect_zscore(threshold=_zscore_threshold)

            if _outlier_strategy == "cap":
                clean_data = outlier_handler.cap_outliers(_outlier_method, zscore_threshold=_zscore_threshold)
                _chat_ctx.log_eda("Handled outliers.")
            else:
                clean_data = outlier_handler.remove_outliers(_outlier_method, zscore_threshold=_zscore_threshold)
            time.sleep(0.6)
            
            # Step 9 — Generate report
            yield send("Report Generated", "Synthesizing intelligence report...", 95)
            out_file = f"report_{uuid.uuid4().hex[:8]}.html"
            output_path = os.path.join(REPORTS_FOLDER, out_file)
            ReportGenerator(clean_data).generate_report(mode="detailed", file_name=output_path)

            # Step 10 — Persist clean DF to pre-agreed path
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


def detect_text_columns(df: pd.DataFrame) -> list:
    """Heuristic: object columns whose cells look like free text (reviews,
    comments) rather than short categorical labels."""
    text_cols = []
    for col in df.columns:
        if df[col].dtype == object:
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

    return render_template(
        "phase2_ui.html",
        mode=mode,
        active_page='visualization'
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
            ]
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
            clean_data = preprocessor.get_clean_data()
            outlier_handler = OutlierHandler(clean_data)
            outlier_handler.detect_iqr()
            clean_data = outlier_handler.cap_outliers()
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

        return jsonify({
            "status": "success",
            "chart_type": chart_type,
            "plot_type": plot_type,
            "view_url": f"/phase2/view/{plot_filename}",
            "download_url": f"/phase2/download/{plot_filename}",
        })

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

@app.route('/nlp')
def nlp_page():
    mode = session.get("mode", "user")
    return render_template("nlp_ui.html", mode=mode, active_page='nlp')


@app.route('/nlp/status', methods=['GET'])
def nlp_status():
    """Returns candidate text columns from the RAW dataset (free text is
    usually stripped/encoded away by the Phase 1 clean pipeline)."""
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
    label_column = data.get("label_column") or None
    method = data.get("method", "tfidf")
    ngram_max = int(data.get("ngram_max", 1))
    classifier = data.get("classifier", "logistic_regression")
    top_n = int(data.get("top_n", 20))

    try:
        analyzer = NLPAnalyzer(df, text_column)
        result = analyzer.analyze(
            auto=auto,
            label_column=label_column,
            method=method,
            ngram_range=(1, max(1, ngram_max)),
            classifier=classifier,
            top_n=top_n,
        )

        viz = NLPVisualizer(plots_dir=NLP_PLOTS_FOLDER)
        plots = {
            "word_frequency": os.path.basename(viz.plot_word_frequency(result["word_frequency"])),
            "keywords": os.path.basename(viz.plot_keywords(result["keywords"])),
        }
        sentiment = result["sentiment"]
        if sentiment.get("method") == "lexicon":
            plots["sentiment_distribution"] = os.path.basename(
                viz.plot_sentiment_distribution(sentiment["distribution"])
            )
            summary_line = (
                f"Lexicon-based sentiment on '{text_column}': "
                f"{sentiment['distribution']} ({sentiment['dominant_sentiment']} dominant)."
            )
        else:
            plots["confusion_matrix"] = os.path.basename(
                viz.plot_confusion_matrix(sentiment["confusion_matrix"], sentiment["labels"])
            )
            summary_line = (
                f"Trained {sentiment['classifier']} classifier on '{text_column}' vs '{label_column}': "
                f"accuracy={sentiment['accuracy']}, f1={sentiment['f1_score']}."
            )

        stats = result["statistics"]
        keyword_line = ", ".join(k["term"] for k in result["keywords"][:10])
        description = (
            f"{stats['documents']} documents, avg {stats['avg_word_count']} words each, "
            f"vocabulary size {stats['vocabulary_size']}. Top keywords: {keyword_line}. {summary_line}"
        )

        ctx = get_chat_context()
        ctx.log_nlp(text_column, description)
        save_chat_context(ctx)

        result["plots"] = plots
        result["status"] = "success"
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


# ════════════════════════════════════════════════════════════
# PHASE 5 — ML Routes
# ════════════════════════════════════════════════════════════

@app.route('/ml')
def ml_page():
    mode = session.get("mode", "user")
    return render_template("ml_ui.html", mode=mode, active_page='ml')


@app.route('/ml/status', methods=['GET'])
def ml_status():
    df = get_clean_df()
    if df is None:
        return jsonify({"status": "no_data"})

    trainer = ModelTrainer(df)
    guessed_target = trainer.guess_target_column()
    task_type = trainer.detect_task_type(guessed_target)

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
    })


@app.route('/ml/train', methods=['POST'])
def ml_train():
    df = get_clean_df()
    if df is None:
        return jsonify({"status": "error", "message": "No preprocessed data available. Run Phase 1 first."}), 400

    data = request.form
    mode = data.get("mode", "user")

    try:
        pipeline = MLPipeline(df, plots_dir=ML_PLOTS_FOLDER)

        if mode == "user":
            target_column = data.get("target_column") or None
            result = pipeline.run_auto(target_column=target_column)
        else:
            target_column = data.get("target_column")
            model_name = data.get("model_name", "random_forest")
            task_type = data.get("task_type") or None
            feature_columns = request.form.getlist("feature_columns") or None
            test_size = float(data.get("test_size", 0.2))
            scale = data.get("scale", "false") == "true"

            hyperparams = {}
            n_estimators = data.get("n_estimators")
            if n_estimators:
                hyperparams["n_estimators"] = int(n_estimators)
            max_depth = data.get("max_depth")
            if max_depth:
                hyperparams["max_depth"] = int(max_depth)
            n_neighbors = data.get("n_neighbors")
            if n_neighbors:
                hyperparams["n_neighbors"] = int(n_neighbors)
            c_param = data.get("C")
            if c_param:
                hyperparams["C"] = float(c_param)

            if not target_column:
                return jsonify({"status": "error", "message": "target_column is required in Developer mode."}), 400

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


if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)