"""
Single-file Flask web app for fabric TPI (threads-per-inch) analysis.
Drop-in replacement/enhancement of the provided Colab script so you can
upload an image from your browser and see the results (original image,
FFT magnitude image, and detected warp/weft counts or TPI when PPI is
provided).

How to run:
1. Create a virtualenv and activate it.
2. pip install flask numpy opencv-python-headless matplotlib scipy
3. python fabric_tpi_flask_app.py
4. Open http://127.0.0.1:5000 in your browser.

Note: This is a minimal working example meant for local use or light demo
hosting. For production, add authentication, request size limits, secure
file handling, and asynchronous workers if large images are expected.
"""

from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string, flash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'replace-with-a-secure-random-key'

# --- CORE ANALYSIS FUNCTION (adapted to return results and save images) ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_fabric_tpi(image_path, ppi=None, result_basename='result'):
    """Analyzes a fabric image and returns metrics + saves a result image.
    Returns a dict: { 'weft_freq': float, 'warp_freq': float, 'weft_tpi': float|None, 'warp_tpi': float|None, 'result_image': path }
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = img.shape

    # --- AUTO ROI: center square ---
    roi_size = min(h, w)
    start_h = (h - roi_size) // 2
    start_w = (w - roi_size) // 2
    roi = img[start_h : start_h + roi_size, start_w : start_w + roi_size]

    # --- FFT PROCESS ---
    f_transform = np.fft.fft2(roi)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log1p(np.abs(f_transform_shifted))

    center_x, center_y = roi_size // 2, roi_size // 2
    vertical_line = magnitude_spectrum[:, center_x]
    horizontal_line = magnitude_spectrum[center_y, :]

    smoothed_vertical_line = gaussian_filter1d(vertical_line, sigma=2)
    smoothed_horizontal_line = gaussian_filter1d(horizontal_line, sigma=2)

    # tune peak detection thresholds for typical images
    weft_peaks, _ = find_peaks(smoothed_vertical_line, prominence=0.1, distance=5)
    warp_peaks, _ = find_peaks(smoothed_horizontal_line, prominence=0.1, distance=5)

    center_ignore_radius = max(2, roi_size // 50)
    weft_freq = 0.0
    warp_freq = 0.0
    weft_peak_y = None
    warp_peak_x = None

    valid_weft = np.where(np.abs(weft_peaks - center_y) > center_ignore_radius)[0]
    if valid_weft.size > 0:
        valid_peak_locations = weft_peaks[valid_weft]
        distances = np.abs(valid_peak_locations - center_y)
        weft_freq = float(np.min(distances))
        weft_peak_y = int(valid_peak_locations[np.argmin(distances)])

    valid_warp = np.where(np.abs(warp_peaks - center_x) > center_ignore_radius)[0]
    if valid_warp.size > 0:
        valid_peak_locations = warp_peaks[valid_warp]
        distances = np.abs(valid_peak_locations - center_x)
        warp_freq = float(np.min(distances))
        warp_peak_x = int(valid_peak_locations[np.argmin(distances)])

    weft_tpi = None
    warp_tpi = None
    if ppi and ppi > 0:
        roi_inches = roi_size / float(ppi)
        if roi_inches > 0:
            weft_tpi = weft_freq / roi_inches if weft_freq > 0 else 0.0
            warp_tpi = warp_freq / roi_inches if warp_freq > 0 else 0.0

    # --- Visualization and saving result image ---
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f'Full Image ({w}x{h}) with ROI')
    ax1.axis('off')
    rect = plt.Rectangle((start_w, start_h), roi_size, roi_size,
                         edgecolor='red', facecolor='none', linewidth=2)
    ax1.add_patch(rect)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(magnitude_spectrum, cmap='gray')
    if warp_peak_x is not None:
        ax2.plot(warp_peak_x, center_y, 'ro', markersize=8, label='Warp Peak')
    if weft_peak_y is not None:
        ax2.plot(center_x, weft_peak_y, 'bo', markersize=8, label='Weft Peak')
    ax2.set_title('FFT Magnitude Spectrum')
    ax2.legend()
    ax2.axis('off')

    plt.tight_layout()

    result_image_path = os.path.join(RESULTS_FOLDER, f"{result_basename}.png")
    fig.savefig(result_image_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    return {
        'weft_freq': weft_freq,
        'warp_freq': warp_freq,
        'weft_tpi': weft_tpi,
        'warp_tpi': warp_tpi,
        'result_image': os.path.relpath(result_image_path, BASE_DIR)
    }


# --- FLASK ROUTES ---

INDEX_HTML = """
<!doctype html>
<title>Fabric TPI Analyzer</title>
<h1>Fabric TPI Analyzer</h1>
<p>Upload a clear, well-lit fabric macro image. The app will auto-select a center square ROI and compute warp/weft thread peaks from the FFT.</p>
<form method=post enctype=multipart/form-data action="/analyze">
  <label>Image: <input type=file name=file required></label><br><br>
  <label>Pixels Per Inch (optional): <input type=text name=ppi placeholder="e.g. 300"></label><br><br>
  <input type=submit value='Analyze'>
</form>
<hr>
<p>Results are saved in the static/results folder.</p>
"""

RESULT_HTML = """
<!doctype html>
<title>Analysis Result</title>
<h1>Analysis Result</h1>
<p><strong>Original file:</strong> {{ filename }}</p>
<ul>
  <li>Weft (raw pixels): {{ weft_freq }}</li>
  <li>Warp (raw pixels): {{ warp_freq }}</li>
  {% if weft_tpi is not none and warp_tpi is not none %}
  <li>Weft TPI: {{ weft_tpi }}</li>
  <li>Warp TPI: {{ warp_tpi }}</li>
  {% endif %}
</ul>
<div>
  <h3>Visualization</h3>
  <img src="/{{ result_image }}" style="max-width:90%;height:auto;border:1px solid #ccc;">
</div>
<br>
<a href="/">← Back</a>
"""


@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


@app.route('/analyze', methods=['POST'])
def upload_and_analyze():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(saved_path)

        # parse ppi if provided
        ppi_val = None
        ppi_field = request.form.get('ppi', '').strip()
        if ppi_field:
            try:
                ppi_val = float(ppi_field)
            except ValueError:
                ppi_val = None

        base = os.path.splitext(filename)[0]
        result_basename = f"{base}_analysis"
        try:
            result = analyze_fabric_tpi(saved_path, ppi=ppi_val, result_basename=result_basename)
        except Exception as e:
            flash(f'Analysis error: {e}')
            return redirect(url_for('index'))

        return render_template_string(RESULT_HTML,
                                      filename=filename,
                                      weft_freq=f"{result['weft_freq']:.2f}",
                                      warp_freq=f"{result['warp_freq']:.2f}",
                                      weft_tpi=(f"{result['weft_tpi']:.2f}" if result['weft_tpi'] is not None else None),
                                      warp_tpi=(f"{result['warp_tpi']:.2f}" if result['warp_tpi'] is not None else None),
                                      result_image=result['result_image'])

    else:
        flash('File type not allowed')
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
