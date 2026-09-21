# Module 3 — Spatial Blur vs. Fourier Blur

Computer Vision.
Shows that blurring an image by direct spatial convolution and blurring it by multiplying Fourier spectra produce the same result, up to floating-point rounding error.

## Scripts

- `blur_core.py` — the algorithm: kernel construction, direct spatial
  convolution, FFT-based convolution, metrics, image encoding. No filtering
  shortcuts (no `scipy.signal.convolve`, no `cv2.GaussianBlur`) — both routes
  are implemented from scratch so the comparison isn't hidden inside a
  library call.
- `app.py` — a small Flask server. Reads whatever images are in `dataset/`,
  runs both blur routes on the one you pick, and returns the results to the
  page as PNGs plus the difference metrics.
- `templates/index.html` — the web page itself.
- `dataset/` — put your 10 images here (see `dataset/README.txt`).

## Instructions on How to Run

```bash
pip install -r requirements.txt
python app.py
```

This starts a local server at `http://127.0.0.1:5000` and opens it in your
browser automatically. Select an image, pick a blur type/size/strength, and the
four panels (original, spatial result, Fourier result, difference) update by
calling back into `app.py`, which does the actual computation in Python.


## How the comparison works

For a chosen kernel `h` and image `f`:

- **Spatial route** (`convolve_spatial`): direct 2-D convolution, computed on
  the pixels with the kernel reflected (true convolution, not correlation),
  zero-padded borders.
- **Fourier route** (`convolve_fourier`): both `f` and `h` are zero-padded to
  a power of two at least `N + K - 1` on each axis (so multiplying their DFTs
  gives *linear* convolution rather than a wrapped-around circular one), then
  `IFFT2(FFT2(f) * FFT2(h))`.

The page reports the largest per-pixel difference, the RMS difference, and
PSNR between the two results. Expect the largest difference to be on the
order of `1e-12`–`1e-13` on a 0–255 scale — that's floating-point rounding,
not a disagreement between the two methods.
