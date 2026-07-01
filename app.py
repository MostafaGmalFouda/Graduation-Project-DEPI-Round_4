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

# Import Phase_3 NLP modules
from Phase_3_NLP.TextPreprocessor import TextPreprocessor
from Phase_3_NLP.NLPAnalyzer import NLPAnalyzer
from Phase_3_NLP.TextVectorizer import TextVectorizer

# Import Phase_4 RAG modules
from Phase_4_RAG.TextChunker import TextChunker
from Phase_4_RAG.VectorStoreManager import VectorStoreManager
from Phase_4_RAG.RAGOrchestrator import RAGOrchestrator

# Import Phase_5 ML modules
from Phase_5_ML.ModelFactory import ModelFactory
from Phase_5_ML.ModelTrainer import ModelTrainer
from Phase_5_ML.ModelEvaluator import ModelEvaluator
from Phase_5_ML.Predictor import Predictor
from Phase_5_ML.ModelVisualizer import ModelVisualizer
from Phase_5_ML.MLPipeline import MLPipeline


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

# Folder to store trained ML models (joblib)
MODELS_FOLDER = os.path.join(BASE_DIR, 'session_data', 'models')
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Folder to store ML visualization plots (Phase 5)
ML_PLOTS_FOLDER = os.path.join(BASE_DIR, 'Phase_5_ML', 'plots')
os.makedirs(ML_PLOTS_FOLDER, exist_ok=True)

# Folder used by the RAG vector store (Phase 4) — one persistent Chroma DB per session
VECTOR_STORE_FOLDER = os.path.join(BASE_DIR, 'session_data', 'vector_store')
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', mode=session.get('mode'))


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

            # ── Persist raw DF in session ──────────────────────────────
            raw_path = save_session_df(df, 'raw')
            session['raw_df_path'] = raw_path
            # Clear any previously stored clean df so phase2 knows it needs preprocessing
            session.pop('clean_df_path', None)
            session['pipeline_done'] = False

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
    """Remove session data so next upload starts fresh."""
    # Delete persisted pickle/json files
    for key in ('raw_df_path', 'clean_df_path', 'schema_path'):
        path = session.pop(key, None)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
    session['pipeline_done'] = False
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
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        loader = DataLoader(filepath)
        df_raw = loader.load()
        raw_path = save_session_df(df_raw, 'raw')
        session['raw_df_path'] = raw_path

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
            else:
                yield send("Preprocessing", "Skipping type conversion (disabled)...", 48)

            # Step 6 — Remove duplicates
            if _do_remove_duplicates:
                preprocessor.remove_duplicates()

            # Step 7 — Encoding
            if _encoding_method == "none":
                # USER MODE — smart auto-encoding
                yield send("Preprocessing", "Auto-encoding categorical columns (smart mode)...", 62)
                preprocessor.auto_encode(onehot_max_unique=10)
            else:
                # DEVELOPER MODE — manual uniform encoding
                enc_label = "Label" if _encoding_method == "label" else "One-Hot"
                yield send("Preprocessing", f"Encoding categorical columns ({enc_label})...", 62)
                preprocessor.encode_categoricals(_encoding_method)
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


@app.route("/phase2")
def phase2():

    mode = session.get("mode", "user")

    return render_template(
        "phase2_ui.html",
        mode=mode
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

            # Snapshot the ORIGINAL schema before any type-changing step
            original_schema = compute_column_schema(df_raw)
            schema_path = save_column_schema(original_schema)
            session['schema_path'] = schema_path

            # Full preprocessing
            preprocessor = DataPreprocessor(df_raw)
            preprocessor.handle_nulls()
            # preprocessor.drop_id_columns()
            preprocessor.handle_text_columns(unique_threshold=0.6)          
            preprocessor.convert_types()
            preprocessor.remove_duplicates()
            preprocessor.auto_encode()  
            clean_data = preprocessor.get_clean_data()
            outlier_handler = OutlierHandler(clean_data)
            outlier_handler.detect_iqr()
            clean_data = outlier_handler.cap_outliers()

            # Save to session
            clean_path = save_session_df(clean_data, 'clean')
            session['clean_df_path'] = clean_path
            session['pipeline_done'] = True
            clean_df = clean_data

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

        output_path = None
        plot_type = "png"

        if chart_type == "summary_dashboard":
            output_path = viz.generate_summary_dashboard(save=True)

        elif chart_type == "correlation_heatmap":
            output_path = viz.plot_correlation_heatmap(save=True)

        elif chart_type == "scatter_2d":
            col1 = request.form.get("col1", "")
            col2 = request.form.get("col2", "")
            color_col = request.form.get("color_col") or None
            output_path = viz.plot_scatter_2d(col1, col2, color_col=color_col, save=True)
            plot_type = "html"

        elif chart_type == "scatter_3d":
            col1 = request.form.get("col1", "")
            col2 = request.form.get("col2", "")
            col3 = request.form.get("col3", "")
            color_col = request.form.get("color_col") or None
            output_path = viz.plot_scatter_3d(col1, col2, col3, color_col=color_col, save=True)
            plot_type = "html"

        elif chart_type == "joint_plot":
            col1 = request.form.get("col1", "")
            col2 = request.form.get("col2", "")
            kind = request.form.get("kind", "scatter")
            output_path = viz.plot_joint_plot(col1, col2, kind=kind, save=True)

        elif chart_type == "stacked_bar":
            col1 = request.form.get("col1", "")
            col2 = request.form.get("col2", "")
            normalize = request.form.get("normalize", "false") == "true"
            output_path = viz.plot_stacked_bar(col1, col2, normalize=normalize, save=True)

        elif chart_type == "cross_tabulation":
            col1 = request.form.get("col1", "")
            col2 = request.form.get("col2", "")
            output_path = viz.plot_cross_tabulation(col1, col2, save=True)

        elif chart_type == "violin_plot":
            num_col = request.form.get("num_col", "")
            cat_col = request.form.get("cat_col", "")
            output_path = viz.plot_violin_plot_by_category(num_col, cat_col, save=True)

        elif chart_type == "facet_grid":
            num_cols = request.form.getlist("num_cols")
            cat_col = request.form.get("cat_col", "")
            output_path = viz.plot_facet_grid(num_cols, cat_col, save=True)

        elif chart_type == "bubble_chart":
            x = request.form.get("x", "")
            y = request.form.get("y", "")
            size = request.form.get("size", "")
            color = request.form.get("color") or None
            output_path = viz.plot_bubble_chart(x, y, size, color=color, save=True)
            plot_type = "html"

        elif chart_type == "automatic_dashboard":
            output_path = viz.generate_automatic_dashboard(save=True)
            plot_type = "html"

        else:
            return jsonify({"status": "error", "message": f"Unknown chart type: {chart_type}"}), 400

        if not output_path or not os.path.exists(output_path):
            return jsonify({"status": "error", "message": "Plot file was not created."}), 500

        plot_filename = os.path.basename(output_path)

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


# ════════════════════════════════════════════════════════════
# PHASE 3 — NLP & RAG Routes (text intelligence + Q&A over data)
# ════════════════════════════════════════════════════════════

# In-memory cache: one VectorStoreManager + RAGOrchestrator per Flask
# session id, so each user's RAG index stays isolated. Kept in-process
# (not pickled) because a fitted vectorizer/vector-db connection isn't
# cleanly picklable — acceptable for a single-process dev/demo deployment.
_RAG_REGISTRY = {}


def _session_id() -> str:
    """Stable per-browser-session id, created once and stored in the Flask session."""
    if 'sid' not in session:
        session['sid'] = uuid.uuid4().hex
    return session['sid']


def get_nlp_df() -> pd.DataFrame:
    """Return the NLP-processed DataFrame stored in the session, or None."""
    path = session.get('nlp_df_path')
    return load_session_df(path)


@app.route('/ai-pipeline')
def ai_pipeline():
    """
    Unified AI Pipeline entry point — the connected ML -> NLP -> RAG
    journey. Replaces separately-navigated Phase 3 / Phase 4 pages with
    a single continuous flow: train a model, then move straight into
    text intelligence on the same dataset, then land on the RAG chatbot.

    All existing /phase3/* and /phase4/* JSON API routes are reused
    as-is under the hood — this route only serves the new combined
    template that orchestrates calls to them in sequence.
    """
    mode = session.get("mode", "user")
    return render_template("ai_pipeline_ui.html", mode=mode)


@app.route('/phase3')
def phase3():
    """Phase 3 entry point — NLP & RAG module, respects the global User/Developer mode."""
    mode = session.get("mode", "user")
    return render_template("phase3_ui.html", mode=mode)


@app.route('/phase3/detect-columns', methods=['POST'])
def phase3_detect_columns():
    """
    Accept an uploaded file (or reuse the clean DF already in session)
    and return its column list, so the UI can let the user pick a text
    column to run NLP/RAG on.
    """
    df = get_clean_df()

    if df is None:
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            return jsonify({"status": "error", "message": "No data available. Upload a file or run Phase 1 first."}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df = DataLoader(filepath).load()
            if df is None or df.empty:
                return jsonify({"status": "error", "message": "Empty or invalid file."}), 400
            raw_path = save_session_df(df, 'raw')
            session['raw_df_path'] = raw_path
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    # Only text-like (object) columns make sense for NLP/RAG
    text_columns = [c for c in df.columns if df[c].dtype == object]
    all_columns = list(df.columns)

    return jsonify({
        "status": "success",
        "rows": df.shape[0],
        "cols": df.shape[1],
        "columns": all_columns,
        "text_columns": text_columns,
    })


@app.route('/phase3/clean-text', methods=['POST'])
def phase3_clean_text():
    """
    Run the TextPreprocessor cleaning pipeline on the chosen text column.
    Developer Mode lets the caller toggle individual cleaning steps;
    User Mode always runs the full recommended pipeline.
    """
    text_col = request.form.get('text_column')
    df = get_clean_df() if get_clean_df() is not None else get_raw_df()

    if df is None or not text_col or text_col not in df.columns:
        return jsonify({"status": "error", "message": "No data or invalid text column."}), 400

    steps = request.form.get('steps', 'all')  # 'all' or comma list: lowercase,clean,stopwords,tokenize,lemmatize

    try:
        pre = TextPreprocessor(df)

        if steps == 'all':
            result = pre.clean_pipeline(text_col)
        else:
            chosen = set(s.strip() for s in steps.split(','))
            if 'lowercase' in chosen:
                pre.lowercase(text_col)
            if 'clean' in chosen:
                pre.remove_punctuation_and_urls(text_col)
            if 'stopwords' in chosen:
                pre.remove_stopwords(text_col)
            if 'tokenize' in chosen or 'lemmatize' in chosen:
                pre.tokenize(text_col)
            if 'lemmatize' in chosen:
                pre.apply_lemmatization(f"{text_col}_tokens")
            result = pre.get_data()

        nlp_path = save_session_df(result, 'nlp')
        session['nlp_df_path'] = nlp_path
        session['nlp_text_col'] = text_col

        preview_cols = [text_col] + [c for c in result.columns if c.startswith(f"{text_col}_token")]
        preview_cols = [c for c in preview_cols if c in result.columns]
        preview_html = result[preview_cols].head(8).to_html(classes='preview-table', index=False)

        return jsonify({
            "status": "success",
            "preview": preview_html,
            "rows": result.shape[0],
            "cols": result.shape[1],
        })

    except Exception as e:
        print("Phase3 clean-text error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/phase3/analyze', methods=['POST'])
def phase3_analyze():
    """
    Run NLPAnalyzer on the cleaned text column: lengths, sentiment,
    named entities, n-grams, and a corpus-level summary — all in one call.
    """
    df = get_nlp_df()
    text_col = session.get('nlp_text_col')

    if df is None or not text_col:
        return jsonify({"status": "error", "message": "Run text cleaning first."}), 400

    ngram_n = int(request.form.get('ngram_n', 2))
    run_ner = request.form.get('run_ner', 'true') == 'true'

    try:
        analyzer = NLPAnalyzer(df)

        lengths_df = analyzer.compute_text_lengths(text_col)
        sentiment_df = analyzer.analyze_sentiment(text_col)

        summary = analyzer.get_corpus_summary(text_col)

        sentiment_counts = sentiment_df[f"{text_col}_sentiment_label"].value_counts().to_dict()

        token_col = f"{text_col}_tokens_lemmatized"
        if token_col not in sentiment_df.columns:
            token_col = f"{text_col}_tokens"

        ngrams_result = []
        if token_col in sentiment_df.columns:
            top_ngrams = analyzer.extract_ngrams(token_col, n=ngram_n)[:15]
            ngrams_result = [{"ngram": " ".join(g), "count": c} for g, c in top_ngrams]

        entities_result = {}
        if run_ner:
            try:
                ner = analyzer.extract_named_entities(text_col)
                entities_result = {
                    "label_counts": ner["label_counts"],
                    "top_entities": list(ner["entity_counts"].items())[:15],
                }
            except RuntimeError as ner_err:
                entities_result = {"error": str(ner_err)}

        # persist enriched df (with length/sentiment columns) back to session
        session['nlp_df_path'] = save_session_df(sentiment_df, 'nlp')

        return jsonify({
            "status": "success",
            "corpus_summary": summary,
            "sentiment_distribution": sentiment_counts,
            "avg_polarity": round(float(sentiment_df[f"{text_col}_polarity"].mean()), 3),
            "avg_subjectivity": round(float(sentiment_df[f"{text_col}_subjectivity"].mean()), 3),
            "top_ngrams": ngrams_result,
            "entities": entities_result,
            "sample_rows": sentiment_df[[text_col, f"{text_col}_sentiment_label", f"{text_col}_polarity"]]
                .head(8).to_dict(orient='records'),
        })

    except Exception as e:
        print("Phase3 analyze error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/phase3/vectorize', methods=['POST'])
def phase3_vectorize():
    """Fit TF-IDF or Bag-of-Words on the cleaned text column and return top terms."""
    df = get_nlp_df()
    text_col = session.get('nlp_text_col')
    method = request.form.get('method', 'tfidf')
    max_features = int(request.form.get('max_features', 300))

    if df is None or not text_col:
        return jsonify({"status": "error", "message": "Run text cleaning first."}), 400

    try:
        vectorizer = TextVectorizer(df)
        if method == 'bow':
            matrix = vectorizer.fit_bow(text_col, max_features=max_features)
            top_terms = []
        else:
            matrix = vectorizer.fit_tfidf(text_col, max_features=max_features)
            top_terms = [{"term": t, "score": round(float(s), 4)} for t, s in vectorizer.get_top_terms(20)]

        return jsonify({
            "status": "success",
            "method": method,
            "matrix_shape": list(matrix.shape),
            "vocabulary_size": len(vectorizer.get_feature_names()),
            "top_terms": top_terms,
        })

    except Exception as e:
        print("Phase3 vectorize error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/phase3/rag/status', methods=['GET'])
def phase3_rag_status():
    """
    Lightweight pre-flight check the frontend calls when the RAG/chatbot
    stage loads, so a missing ANTHROPIC_API_KEY is surfaced clearly
    BEFORE the person builds an index and starts typing questions —
    rather than failing silently on the first chat message with no
    context for why.
    """
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return jsonify({
        "status": "success",
        "llm_configured": has_key,
        "message": (
            "RAG chatbot is ready." if has_key else
            "ANTHROPIC_API_KEY is not set on the server. The index can "
            "still be built, but the chatbot won't be able to generate "
            "answers until an administrator sets this environment "
            "variable and restarts the app."
        ),
    })


@app.route('/phase3/rag/build-index', methods=['POST'])
def phase3_rag_build_index():
    """
    Build (or rebuild) the RAG vector index from the cleaned text column:
        TextChunker → VectorStoreManager (embed + store in ChromaDB)
    """
    df = get_nlp_df()
    text_col = session.get('nlp_text_col')

    if df is None or not text_col:
        return jsonify({"status": "error", "message": "Run text cleaning first."}), 400

    chunk_size = int(request.form.get('chunk_size', 120))
    chunk_overlap = int(request.form.get('chunk_overlap', 20))
    split_mode = request.form.get('split_mode', 'token')  # 'token' | 'character'

    sid = _session_id()
    persist_dir = os.path.join(VECTOR_STORE_FOLDER, sid)
    os.makedirs(persist_dir, exist_ok=True)

    try:
        chunker = TextChunker(df)
        if split_mode == 'character':
            chunks_df = chunker.split_by_character(text_col, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            chunks_df = chunker.split_by_token(text_col, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if chunks_df.empty:
            return jsonify({"status": "error", "message": "No text chunks were produced — check the selected column."}), 400

        store = VectorStoreManager()
        store.initialize_vector_db(
            collection_name=f"collection_{sid}",
            persist_directory=persist_dir,
        )

        vectors = store.generate_embeddings(chunks_df["chunk_text"].tolist())
        metadata = chunks_df[["source_row", "chunk_id"]].to_dict(orient="records")
        store.insert_vectors(vectors, metadata, documents=chunks_df["chunk_text"].tolist())

        rag = RAGOrchestrator()
        rag.set_vector_store(store)

        _RAG_REGISTRY[sid] = {"store": store, "rag": rag}
        session['rag_ready'] = True
        session['rag_chunk_count'] = int(len(chunks_df))

        return jsonify({
            "status": "success",
            "chunk_count": int(len(chunks_df)),
            "embedding_dim": int(vectors.shape[1]),
        })

    except Exception as e:
        print("Phase3 RAG build-index error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/phase3/rag/ask', methods=['POST'])
def phase3_rag_ask():
    """Ask a natural-language question against the built RAG index."""
    sid = _session_id()
    entry = _RAG_REGISTRY.get(sid)

    if entry is None or not session.get('rag_ready'):
        return jsonify({"status": "error", "message": "No RAG index built yet. Build the index first."}), 400

    question = request.form.get('question', '').strip()
    k = int(request.form.get('k', 5))

    if not question:
        return jsonify({"status": "error", "message": "Question cannot be empty."}), 400

    try:
        retrieved = entry["store"].similarity_search(question, k=k)
        prompt = entry["rag"].build_prompt(question, retrieved)
        answer = entry["rag"].generate_answer(prompt)

        return jsonify({
            "status": "success",
            "answer": answer,
            "sources": [
                {"text": r["text"][:240], "distance": round(float(r["distance"]), 4)}
                for r in retrieved
            ],
        })

    except EnvironmentError as e:
        # Missing ANTHROPIC_API_KEY — surface a clear, actionable message
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        print("Phase3 RAG ask error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# ════════════════════════════════════════════════════════════
# PHASE 4 — Machine Learning Routes (User & Developer Modes)
# ════════════════════════════════════════════════════════════

# In-memory cache: one MLPipeline per browser session id, since a
# trained scikit-learn model + the live MLPipeline orchestrator aren't
# things we round-trip through the session cookie — same pattern as
# the RAG registry above.
_ML_REGISTRY = {}


@app.route('/phase4')
def phase4():
    """Phase 4 entry point — Machine Learning module, respects the global User/Developer mode."""
    mode = session.get("mode", "user")
    return render_template("phase4_ui.html", mode=mode)


@app.route('/phase4/detect-columns', methods=['POST'])
def phase4_detect_columns():
    """
    Use the clean DF from Phase 1 session if available, otherwise accept
    a fresh file upload. Returns the column list + a numeric/categorical
    hint so the UI can warn before an unsuitable target is chosen.
    """
    df = get_clean_df()

    if df is None:
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            return jsonify({"status": "error", "message": "No data available. Upload a file or run Phase 1 first."}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df = DataLoader(filepath).load()
            if df is None or df.empty:
                return jsonify({"status": "error", "message": "Empty or invalid file."}), 400
            raw_path = save_session_df(df, 'raw')
            session['raw_df_path'] = raw_path
            # also stash it as "clean" so ML routes downstream can reuse it directly
            session['clean_df_path'] = save_session_df(df, 'clean')
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    columns = [
        {"name": col, "dtype": str(df[col].dtype), "n_unique": int(df[col].nunique())}
        for col in df.columns
    ]

    return jsonify({
        "status": "success",
        "rows": df.shape[0],
        "cols": df.shape[1],
        "columns": columns,
    })


@app.route('/phase4/recommend', methods=['POST'])
def phase4_recommend():
    """User Mode step 1: given a target column, return ranked model recommendations."""
    df = get_clean_df()
    target = request.form.get('target') or None

    if df is None:
        return jsonify({"status": "error", "message": "No data available."}), 400

    try:
        task_type = MLPipeline._infer_task_type(df[target]) if target else "clustering"
        factory = ModelFactory(task_type=task_type)
        recommendations = factory.recommend_models(task_type, df)

        return jsonify({
            "status": "success",
            "task_type": task_type,
            "recommendations": recommendations,
        })

    except Exception as e:
        print("Phase4 recommend error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/phase4/available-models', methods=['GET'])
def phase4_available_models():
    """Developer Mode helper: list every model name supported per task type."""
    factory = ModelFactory()
    return jsonify({"status": "success", "models": factory.available_models()})


@app.route('/phase4/train', methods=['POST'])
def phase4_train():
    """
    Single entry point for BOTH modes:
      - User Mode      : only 'target' is required — top recommended model
                          is auto-selected and trained with defaults.
      - Developer Mode : 'target', 'model_name', optional 'params' (JSON),
                          optional 'tuning' (JSON), and 'cv' are honored.
    """
    df = get_clean_df()
    if df is None:
        return jsonify({"status": "error", "message": "No data available. Run Phase 1 first."}), 400

    mode = request.form.get('run_mode', session.get('mode', 'user'))
    target = request.form.get('target') or None
    test_size = float(request.form.get('test_size', 0.2))

    sid = _session_id()

    try:
        if mode == 'developer':
            model_name = request.form.get('model_name', 'random_forest')
            params_raw = request.form.get('params', '{}')
            tuning_raw = request.form.get('tuning', '')
            cv = int(request.form.get('cv', 5))

            params = json.loads(params_raw) if params_raw else {}
            tuning = json.loads(tuning_raw) if tuning_raw else None

            # Guard against a stale/mismatched model choice (e.g. the
            # frontend was loaded before the target column was picked,
            # or a request was crafted by hand). Without this check,
            # asking ModelFactory for a classifier on a continuous target
            # (or vice versa) reaches all the way into sklearn's .fit()
            # and crashes with a cryptic "Unknown label type: continuous"
            # error instead of a clean, actionable message.
            inferred_task_type = MLPipeline._infer_task_type(df[target]) if target else "clustering"
            factory_check = ModelFactory(task_type=inferred_task_type)
            supported_models = factory_check.available_models().get(inferred_task_type, [])

            if model_name not in supported_models:
                return jsonify({
                    "status": "error",
                    "message": (
                        f"'{model_name.replace('_', ' ')}' does not support the detected task type "
                        f"'{inferred_task_type}' for this target column. "
                        f"Compatible models: {', '.join(m.replace('_', ' ') for m in supported_models)}."
                    ),
                }), 400

            pipeline = MLPipeline(mode='developer')
            result = pipeline.run_developer_mode(
                df, target=target, model_name=model_name, params=params,
                test_size=test_size, cv=cv, tuning=tuning,
            )
        else:
            pipeline = MLPipeline(mode='user')
            result = pipeline.run_user_mode(df, target=target, test_size=test_size)

        _ML_REGISTRY[sid] = pipeline
        session['ml_ready'] = True
        session['ml_task_type'] = pipeline.task_type
        session['ml_target'] = target

        # numpy/pandas types aren't directly JSON serializable — sanitize
        safe_result = json.loads(json.dumps(result, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))

        return jsonify({"status": "success", "result": safe_result})

    except Exception as e:
        print("Phase4 train error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/phase4/predict', methods=['POST'])
def phase4_predict():
    """Single-record prediction using the currently trained model for this session."""
    sid = _session_id()
    pipeline = _ML_REGISTRY.get(sid)

    if pipeline is None or pipeline.predictor is None:
        return jsonify({"status": "error", "message": "No trained model yet. Train a model first."}), 400

    sample_raw = request.form.get('sample', '{}')

    try:
        sample = json.loads(sample_raw)
        result = pipeline.predictor.predict_single(sample)
        explanation = pipeline.predictor.explain_prediction(sample)

        safe_result = {
            "prediction": result["prediction"] if isinstance(result["prediction"], (int, float, str)) else str(result["prediction"]),
            "probabilities": result.get("probabilities"),
            "top_contributing_features": explanation.get("top_contributing_features"),
        }
        return jsonify({"status": "success", "result": safe_result})

    except Exception as e:
        print("Phase4 predict error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/phase4/plot', methods=['POST'])
def phase4_plot():
    """
    Generate a ModelVisualizer plot for the currently trained model and
    save it as a PNG the frontend can <img> directly.
    """
    sid = _session_id()
    pipeline = _ML_REGISTRY.get(sid)

    if pipeline is None or pipeline.evaluator is None:
        return jsonify({"status": "error", "message": "No trained model yet. Train a model first."}), 400

    plot_type = request.form.get('plot_type', 'confusion_matrix')

    try:
        viz = pipeline.visualizer
        fig = None

        if plot_type == 'confusion_matrix':
            cm = pipeline.evaluator.confusion_matrix()
            fig = viz.plot_confusion_matrix(cm)

        elif plot_type == 'roc_curve':
            if not hasattr(pipeline.evaluator.model, 'predict_proba'):
                return jsonify({"status": "error", "message": "Selected model does not support ROC curves."}), 400
            y_test = pipeline.evaluator.y_test
            proba = pipeline.evaluator.model.predict_proba(pipeline.evaluator.X_test)
            classes = getattr(pipeline.evaluator.model, "classes_", None)
            if classes is not None and len(classes) == 2:
                fig = viz.plot_roc_curve(y_true=(y_test == classes[1]).astype(int), y_score=proba[:, 1])
            else:
                return jsonify({"status": "error", "message": "ROC curve currently supports binary classification only."}), 400

        elif plot_type == 'feature_importance':
            model = pipeline.evaluator.model
            X_test = pipeline.evaluator.X_test
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({"feature": X_test.columns, "importance": model.feature_importances_})
            elif hasattr(model, 'coef_'):
                coef = np.ravel(model.coef_)
                importance_df = pd.DataFrame({"feature": X_test.columns, "importance": np.abs(coef)})
            else:
                return jsonify({"status": "error", "message": "Selected model has no feature importance to plot."}), 400
            fig = viz.plot_feature_importance(importance_df)

        elif plot_type == 'actual_vs_predicted':
            y_test = pipeline.evaluator.y_test
            y_pred = pipeline.evaluator.evaluate()
            fig = viz.plot_actual_vs_predicted(y_test, y_pred)

        elif plot_type == 'correlation_matrix':
            fig = viz.plot_correlation_matrix(pipeline.evaluator.X_test)

        else:
            return jsonify({"status": "error", "message": f"Unknown plot type: {plot_type}"}), 400

        out_name = f"{plot_type}_{uuid.uuid4().hex[:8]}.png"
        out_path = os.path.join(ML_PLOTS_FOLDER, out_name)
        viz.save_plot(fig, out_path)

        return jsonify({"status": "success", "view_url": f"/phase4/view/{out_name}"})

    except Exception as e:
        print("Phase4 plot error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/phase4/view/<filename>')
def phase4_view(filename):
    return send_from_directory(ML_PLOTS_FOLDER, filename)


@app.route('/phase4/download-model', methods=['GET'])
def phase4_download_model():
    """Persist the currently trained model to disk via joblib and serve it for download."""
    sid = _session_id()
    pipeline = _ML_REGISTRY.get(sid)

    if pipeline is None or pipeline.trainer is None or pipeline.trainer.model is None:
        return jsonify({"status": "error", "message": "No trained model yet."}), 400

    model_path = os.path.join(MODELS_FOLDER, f"model_{sid}.joblib")
    pipeline.trainer.save_model(model_path)
    return send_file(model_path, as_attachment=True, download_name="trained_model.joblib")


if __name__ == '__main__':
    app.run(debug=True)