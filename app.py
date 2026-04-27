from flask import Flask, render_template, request, jsonify, send_from_directory,send_file,Response
import os
import json
import time
import uuid
from werkzeug.utils import secure_filename


# Import custom modules from Phase_1
from Phase_1.DataLoader import DataLoader
from Phase_1.ReportGenerator import ReportGenerator
from Phase_1.EDAPipeline import EDAPipeline


# Initialize Flask app
app = Flask(__name__)
# app.secret_key = "mariam_bright_key"

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder to store uploaded files
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Folder to store generated reports
REPORTS_FOLDER = os.path.join(BASE_DIR, 'reports')
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Allowed file extensions
def allowed_file(filename):
    """Check allowed file extensions"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}


# Home route → renders main HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and processing
@app.route('/process', methods=['POST'])
def process():
    # Check if file exists in request
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    
    file = request.files['file']
    action_mode = request.form.get('action_type')  # 'summary' or 'predict'
    
    # Validate file
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Step 1: Load data using DataLoader
            loader = DataLoader(filepath)
            df = loader.load()
            
            # Check if data is empty or invalid
            if df is None or df.empty:
                return jsonify({"status": "error", "message": "Empty or invalid file"}), 400
            
            # Step 2: Initialize report generator
            reporter = ReportGenerator(df)
            
            # Determine report type based on user input
            if action_mode == "summary":
                mode = "basic"
            elif action_mode == "predict":
                mode = "detailed"
            else:
                mode = "basic"
            
            # Define output report filename
            out_file = "final_report.html" if mode == "basic" else "detailed_report.html"
            
            # Full path for saving the report
            output_path = os.path.join(REPORTS_FOLDER, out_file)
            
            # Generate the report file
            reporter.generate_report(mode=mode, file_name=output_path)
            
            # Create a preview of the first 5 rows
            preview_html = df.head(5).to_html(classes='preview-table', index=False)            
            
            # Return response to frontend
            return jsonify({
                "status": "success",
                "preview": preview_html,
                "report_name": out_file,
                "view_url": f"/view/{out_file}",
                "download_url": f"/download/{out_file}",
                "rows": df.shape[0],
                "cols": df.shape[1]
            })
        
        except Exception as e:
            # Print error in terminal for debugging
            print("ERROR:", e)
            return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "error", "message": "Unsupported file format"}), 400

@app.route('/pipeline-stream', methods=['POST'])
def pipeline_stream():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    def send(stage, message, progress):
        return f"data: {json.dumps({'stage': stage, 'message': message, 'progress': progress, 'type': 'progress'})}\n\n"

    def generate():
        try:
            pipeline = EDAPipeline(filepath)
            yield send("Data Validation", "Engine started. Accessing data stream...", 10)
            time.sleep(0.8)
            
            raw_data = pipeline.loader.get_data()
            yield send("Data Validation", f"File locked. Detected {raw_data.shape[0]} rows.", 25)
            time.sleep(0.8)
            yield send("Preprocessing", "Scanning for missing values (Nulls)...", 35)
            pipeline.preprocessor.handle_nulls()
            yield send("Preprocessing", "Applying smart type conversion...", 50)
            pipeline.preprocessor.convert_types()
            pipeline.preprocessor.remove_duplicates()
            clean_data = pipeline.preprocessor.get_clean_data()
            time.sleep(0.4)
            yield send("Outlier Detection", "Analyzing statistical distribution...", 75)
            pipeline.outlier_handler = pipeline.outlier_handler.__class__(clean_data)
            pipeline.outlier_handler.detect_iqr()
            clean_data = pipeline.outlier_handler.cap_outliers()
            time.sleep(0.8)
            yield send("Report Generated", "Synthesizing intelligence report...", 95)
            out_file = f"report_{uuid.uuid4().hex[:8]}.html"
            output_path = os.path.join(REPORTS_FOLDER, out_file)
            ReportGenerator(clean_data).generate_report(mode="detailed", file_name=output_path)

            yield f"data: {json.dumps({'done': True, 'stage': 'Report Generated', 'message': 'Complete', 'progress': 100, 'view_url': f'/view/{out_file}', 'download_url': f'/download/{out_file}', 'rows': len(clean_data), 'cols': len(clean_data.columns)})}\n\n"
        except Exception as e:
            yield send("Error", f"Engine failure: {str(e)}", 0)

    return Response(generate(), mimetype='text/event-stream')
    
# View report in browser
@app.route('/view/<filename>')
def view(filename):
    return send_from_directory(REPORTS_FOLDER, filename)


# Download report (Save As)
@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(REPORTS_FOLDER, filename)
    return send_file(file_path, as_attachment=True)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)