from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, Response, session
import os
import json
import time
import uuid
import pickle
import pandas as pd
from werkzeug.utils import secure_filename

# Import custom modules from Phase_1
from Phase_1.DataLoader import DataLoader
from Phase_1.ReportGenerator import ReportGenerator
from Phase_1.EDAPipeline import EDAPipeline
from Phase_1.DataPreprocessor import DataPreprocessor
from Phase_1.OutlierHandler import OutlierHandler

# Import Phase_2 DataVisualizer
from Phase_2.DataVisualizer import DataVisualizer


# Initialize Flask app
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
    return render_template('index.html')


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
        null_threshold    : float  (0.0–1.0)         default 0.4
        null_fill_strategy: 'median'|'mean'|'mode'    default 'median'
        do_type_conversion: 'true'|'false'            default 'true'
        do_remove_duplicates: 'true'|'false'          default 'true'
        exclude_columns   : comma-separated string    default '' (none)
        encoding_method   : 'none'|'label'|'onehot'   default 'none'
        outlier_method    : 'iqr' | 'zscore'          default 'iqr'
        zscore_threshold  : float (2.5–3.5)           default 3.0
        outlier_strategy  : 'cap'  | 'remove'         default 'cap'
    """
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

            if _exclude_columns:
                yield send("Preprocessing", f"Excluding {len(_exclude_columns)} column(s) per configuration...", 25)
                preprocessor.exclude_columns(_exclude_columns)
                time.sleep(0.3)

            yield send("Preprocessing", "Scanning for missing values (Nulls)...", 35)
            preprocessor.handle_nulls(threshold=_null_threshold, fill_strategy=_null_fill_strategy)

            if _do_type_conversion:
                yield send("Preprocessing", "Applying smart type conversion...", 48)
                preprocessor.convert_types()
            else:
                yield send("Preprocessing", "Skipping type conversion (disabled)...", 48)

            if _do_remove_duplicates:
                preprocessor.remove_duplicates()

            if _encoding_method != "none":
                enc_label = "Label" if _encoding_method == "label" else "One-Hot"
                yield send("Preprocessing", f"Encoding categorical columns ({enc_label})...", 58)
                preprocessor.encode_categoricals(_encoding_method)

            clean_data = preprocessor.get_clean_data()
            time.sleep(0.4)

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

            yield send("Report Generated", "Synthesizing intelligence report...", 95)

            out_file = f"report_{uuid.uuid4().hex[:8]}.html"
            output_path = os.path.join(REPORTS_FOLDER, out_file)
            ReportGenerator(clean_data).generate_report(mode="detailed", file_name=output_path)

            # ── Persist clean DF to the pre-agreed path ────────────────
            # (session was already updated BEFORE the generator started)
            clean_data.to_pickle(_clean_path)

            yield f"data: {json.dumps({'done': True, 'stage': 'Report Generated', 'message': 'Complete', 'progress': 100, 'view_url': f'/view/{out_file}', 'download_url': f'/download/{out_file}', 'rows': len(clean_data), 'cols': len(clean_data.columns)})}\n\n"

        except Exception as e:
            yield send("Error", f"Engine failure: {str(e)}", 0)

    return Response(generate(), mimetype='text/event-stream')


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
            preprocessor.convert_types()
            preprocessor.remove_duplicates()
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


if __name__ == '__main__':
    app.run(debug=True)