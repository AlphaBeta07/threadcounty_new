from flask import Flask, render_template, request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def analyze_fabric_tpi(image_path, ppi=None):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, "Error: Could not load image."

    h, w = img.shape
    roi_size = min(h, w)
    start_h = (h - roi_size) // 2
    start_w = (w - roi_size) // 2
    roi = img[start_h:start_h+roi_size, start_w:start_w+roi_size]

    f_transform = np.fft.fft2(roi)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log1p(np.abs(f_transform_shifted))

    center_x, center_y = roi_size // 2, roi_size // 2
    vertical_line = magnitude_spectrum[:, center_x]
    horizontal_line = magnitude_spectrum[center_y, :]

    smoothed_vertical_line = gaussian_filter1d(vertical_line, sigma=2)
    smoothed_horizontal_line = gaussian_filter1d(horizontal_line, sigma=2)

    weft_peaks, _ = find_peaks(smoothed_vertical_line, prominence=0.1, distance=5)
    warp_peaks, _ = find_peaks(smoothed_horizontal_line, prominence=0.1, distance=5)

    center_ignore_radius = roi_size // 50
    weft_freq = warp_freq = 0
    weft_peak_y = warp_peak_x = None

    valid_weft = np.where(np.abs(weft_peaks - center_y) > center_ignore_radius)[0]
    if valid_weft.size > 0:
        valid_peak_locations = weft_peaks[valid_weft]
        distances = np.abs(valid_peak_locations - center_y)
        weft_freq = np.min(distances)
        weft_peak_y = valid_peak_locations[np.argmin(distances)]

    valid_warp = np.where(np.abs(warp_peaks - center_x) > center_ignore_radius)[0]
    if valid_warp.size > 0:
        valid_peak_locations = warp_peaks[valid_warp]
        distances = np.abs(valid_peak_locations - center_x)
        warp_freq = np.min(distances)
        warp_peak_x = valid_peak_locations[np.argmin(distances)]

    # --- Visualization ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Fabric Image with ROI')
    plt.axis('off')
    rect = plt.Rectangle((start_w, start_h), roi_size, roi_size, edgecolor='red', facecolor='none', linewidth=2)
    plt.gca().add_patch(rect)

    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    if warp_freq > 0:
        plt.plot(warp_peak_x, center_y, 'ro', label='Warp')
    if weft_freq > 0:
        plt.plot(center_x, weft_peak_y, 'bo', label='Weft')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()

    plot_filename = f"result_{os.path.basename(image_path)}.png"
    plot_path = os.path.join(app.config['RESULT_FOLDER'], plot_filename)
    plt.savefig(plot_path)
    plt.close()

    return weft_freq, warp_freq, f"results/{plot_filename}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'fabric_image' not in request.files:
            return render_template('index.html', error="No file uploaded.")
        file = request.files['fabric_image']
        if file.filename == '':
            return render_template('index.html', error="Please select an image.")

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        weft, warp, plot_path = analyze_fabric_tpi(filepath)
        if weft is None:
            return render_template('index.html', error=plot_path)

        return render_template('index.html', weft=weft, warp=warp, plot_image=plot_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
