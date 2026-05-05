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
            })

        except Exception as e:
            print("ERROR:", e)
            return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "error", "message": "Unsupported file format"}), 400


@app.route('/clear-data', methods=['POST'])
def clear_data():
    """Remove session data so next upload starts fresh."""
    # Delete persisted pickle files
    for key in ('raw_df_path', 'clean_df_path'):
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
    """
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

    def send(stage, message, progress):
        return f"data: {json.dumps({'stage': stage, 'message': message, 'progress': progress, 'type': 'progress'})}\n\n"

    # Capture variables needed inside generator (avoids any session access)
    _df_raw = df_raw
    _clean_path = clean_path

    def generate():
        try:
            yield send("Data Validation", "Engine started. Accessing data stream...", 10)
            time.sleep(0.8)

            yield send("Data Validation", f"File locked. Detected {_df_raw.shape[0]} rows.", 25)
            time.sleep(0.8)

            yield send("Preprocessing", "Scanning for missing values (Nulls)...", 35)
            preprocessor = DataPreprocessor(_df_raw)
            preprocessor.handle_nulls()

            yield send("Preprocessing", "Applying smart type conversion...", 50)
            preprocessor.convert_types()
            preprocessor.remove_duplicates()
            clean_data = preprocessor.get_clean_data()
            time.sleep(0.4)

            yield send("Outlier Detection", "Analyzing statistical distribution...", 75)
            outlier_handler = OutlierHandler(clean_data)
            outlier_handler.detect_iqr()
            clean_data = outlier_handler.cap_outliers()
            time.sleep(0.8)

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


# ════════════════════════════════════════════════════════════
# PHASE 2 — Data Visualization Routes (uses clean DF from session)
# ════════════════════════════════════════════════════════════

PLOTS_FOLDER = os.path.join(BASE_DIR, 'Phase_2', 'plots')
os.makedirs(PLOTS_FOLDER, exist_ok=True)


@app.route('/phase2')
def phase2():
    return render_template('phase2_ui.html')


@app.route('/phase2/status', methods=['GET'])
def phase2_status():
    """Returns whether the pipeline has been run and clean data is available."""
    clean_df = get_clean_df()
    if clean_df is not None:
        return jsonify({
            "status": "ready",
            "rows": clean_df.shape[0],
            "cols": clean_df.shape[1],
            "columns": [
                {"name": col, "type": "num" if clean_df[col].dtype.kind in "iufcb" else "cat"}
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
        columns = []
        for col in clean_df.columns:
            col_type = "num" if clean_df[col].dtype.kind in "iufcb" else "cat"
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