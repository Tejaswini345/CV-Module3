#!/usr/bin/env python3
# README
# ------
# Computer Vision - Module 3
#
# What this does:
# This is the Flask server for the Module 3 assignment. It runs a small
# local website where we can pick an image from the dataset/ folder, pick a
# blur type/size, and it shows the image blurred two different ways:
#   1) spatial convolution (blur_core.convolve_spatial)
#   2) Fourier domain multiplication (blur_core.convolve_fourier)
# and the difference between the two, to prove they give the same result.
# The actual math is all in blur_core.py, this file just connects it to
# the webpage.
#
# How to run it:
# 1) Put your images inside the dataset/ folder. 
# 2) In the terminal: pip install -r requirements.txt
# 3) In the terminal: python app.py
# 4) It should open http://127.0.0.1:5000 in your browser automatically.
# =============================================================================

import os
import threading
import webbrowser

from flask import Flask, render_template, request, jsonify

import blur_core as bc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
WORK_SIZE = 220          # fixed working resolution, keeps the FFT fast
MAX_IMAGES = 10

app = Flask(__name__)


def list_dataset_images():
    if not os.path.isdir(DATASET_DIR):
        return []
    files = sorted(
        f for f in os.listdir(DATASET_DIR)
        if os.path.splitext(f)[1].lower() in ALLOWED_EXT
    )
    return files[:MAX_IMAGES]


@app.route("/")
def index():
    images = list_dataset_images()
    return render_template("index.html", images=images, max_images=MAX_IMAGES)


@app.route("/api/blur", methods=["POST"])
def api_blur():
    data = request.get_json(force=True)
    filename = data.get("filename")
    ktype = data.get("ktype", "gauss")
    ksize = int(data.get("ksize", 15))
    sigma = float(data.get("sigma", 3.0))
    colorspace = data.get("colorspace", "gray")   # "gray" or "color"

    images = list_dataset_images()
    if filename not in images:
        return jsonify({"error": "Unknown image. Refresh the page."}), 400
    if ksize % 2 == 0:
        ksize += 1

    path = os.path.join(DATASET_DIR, filename)
    h = bc.make_kernel("box" if ktype == "box" else "gaussian", ksize, sigma)

    if colorspace == "color":
        f = bc.load_image_color(path, WORK_SIZE)
        g_spatial = bc.convolve_spatial_color(f, h)
        g_fourier, _, _, _, P = bc.convolve_fourier_color(f, h, return_spectra=True)
        metrics = bc.compare(g_spatial, g_fourier)
        payload = {
            "original": bc.color_to_data_url(f),
            "spatial": bc.color_to_data_url(g_spatial),
            "fourier": bc.color_to_data_url(g_fourier),
            "diff": bc.diff_to_data_url_color(g_spatial, g_fourier, gain=1e5),
        }
    else:
        f = bc.load_image_gray(path, WORK_SIZE)
        g_spatial = bc.convolve_spatial(f, h)
        g_fourier, F, Hf, G, P = bc.convolve_fourier(f, h, return_spectra=True)
        metrics = bc.compare(g_spatial, g_fourier)
        payload = {
            "original": bc.gray_to_data_url(f),
            "spatial": bc.gray_to_data_url(g_spatial),
            "fourier": bc.gray_to_data_url(g_fourier),
            "diff": bc.diff_to_data_url(g_spatial, g_fourier, gain=1e5),
        }

    payload["metrics"] = {
        "max": metrics["max"],
        "rms": metrics["rmse"],
        "psnr": metrics["psnr"],
    }
    payload["info"] = (
        f"{filename} · {'color' if colorspace=='color' else 'grayscale'} · "
        f"{WORK_SIZE}×{WORK_SIZE} · {ksize}×{ksize} "
        f"{'box' if ktype=='box' else 'Gaussian'} blur · FFT grid {P}×{P}"
    )
    return jsonify(payload)


def _open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    n = len(list_dataset_images())
    print("=" * 60)
    if n == 0:
        print("No images found in dataset/ — add up to 10 image files")
        print("(.png, .jpg, .jpeg, .bmp) to that folder, then rerun.")
    else:
        print(f"Found {n} image(s) in dataset/")
        if n < MAX_IMAGES:
            print(f"(add up to {MAX_IMAGES - n} more to fill out the dropdown)")

    # When Render (or any host) runs this app, it sets the PORT env var and
    # expects the server to listen on 0.0.0.0 at that port. Locally, PORT
    # isn't set, so this falls back to 127.0.0.1:5000 like before, and opens
    # the browser automatically since that only makes sense on your own
    # machine.
    port = int(os.environ.get("PORT", 5000))
    is_local = "PORT" not in os.environ

    print(f"Starting server on port {port}")
    print("=" * 60)

    if is_local:
        threading.Timer(1.0, _open_browser).start()
        app.run(host="127.0.0.1", port=port, debug=False)
    else:
        app.run(host="0.0.0.0", port=port, debug=False)