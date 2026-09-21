# =============================================================================
#  blur_core.py
#  CSc 8830 Computer Vision - Module 3
#
#  The actual algorithm code. Everything here is plain NumPy - no filtering
#  shortcuts like scipy.signal.convolve or cv2.GaussianBlur - so the equality
#  being demonstrated is not hidden inside a library call.
#
#  Two independent ways to blur an image:
#     convolve_spatial  - direct convolution, computed on the pixels
#     convolve_fourier  - FFT2 -> multiply -> IFFT2
#
#  Both take the same image f and kernel h and are expected to return the same
#  result, up to floating-point rounding error.
# =============================================================================

import io
import base64

import numpy as np
from PIL import Image


# -----------------------------------------------------------------------------
# Kernel construction. Normalised to sum to 1 so mean brightness is preserved.
# -----------------------------------------------------------------------------
def make_kernel(kind: str, ksize: int, sigma: float) -> np.ndarray:
    if ksize % 2 == 0:
        raise ValueError("kernel size must be odd")

    if kind == "box":
        h = np.ones((ksize, ksize), dtype=np.float64)
    else:  # gaussian
        c = (ksize - 1) / 2.0
        ax = np.arange(ksize, dtype=np.float64) - c
        xx, yy = np.meshgrid(ax, ax)
        h = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))

    return h / h.sum()


# -----------------------------------------------------------------------------
# Route 1: spatial convolution, computed directly on the pixels.
# The kernel is reflected (c - u, c - v), which is what makes this convolution
# rather than correlation. Borders are treated as zero.
# -----------------------------------------------------------------------------
def convolve_spatial(f: np.ndarray, h: np.ndarray) -> np.ndarray:
    H, W = f.shape
    K = h.shape[0]
    c = (K - 1) // 2

    fp = np.zeros((H + 2 * c, W + 2 * c), dtype=np.float64)
    fp[c:c + H, c:c + W] = f

    g = np.zeros((H, W), dtype=np.float64)
    for u in range(K):
        for v in range(K):
            w = h[u, v]
            if w == 0.0:
                continue
            g += w * fp[c + (c - u): c + (c - u) + H,
                        c + (c - v): c + (c - v) + W]
    return g


# -----------------------------------------------------------------------------
# Route 2: Fourier convolution. Both f and h are zero-padded to a power of two
# at least N + K - 1 so that circular convolution (which is what multiplying
# DFTs gives you) equals the linear convolution computed above.
# -----------------------------------------------------------------------------
def convolve_fourier(f: np.ndarray, h: np.ndarray, return_spectra: bool = False):
    H, W = f.shape
    K = h.shape[0]
    c = (K - 1) // 2

    P = 1
    while P < max(H, W) + K - 1:
        P <<= 1

    fp = np.zeros((P, P), dtype=np.float64)
    fp[:H, :W] = f
    hp = np.zeros((P, P), dtype=np.float64)
    hp[:K, :K] = h

    F = np.fft.fft2(fp)
    Hf = np.fft.fft2(hp)
    G = F * Hf                          # <-- the whole filtering step
    g_full = np.real(np.fft.ifft2(G))

    r = (np.arange(H) + c) % P
    s = (np.arange(W) + c) % P
    g = g_full[np.ix_(r, s)]

    if return_spectra:
        return g, F, Hf, G, P
    return g


# -----------------------------------------------------------------------------
# Color version: run the spatial/Fourier routes on each of R, G, B separately
# and stack the results back into an (N, N, 3) array. The math being
# demonstrated (convolution = multiplication in frequency) is a per-channel
# statement, so this is just three independent grayscale runs, not a new
# algorithm.
# -----------------------------------------------------------------------------
def convolve_spatial_color(f: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.stack(
        [convolve_spatial(f[:, :, c], h) for c in range(f.shape[2])], axis=-1
    )


def convolve_fourier_color(f: np.ndarray, h: np.ndarray, return_spectra: bool = False):
    if not return_spectra:
        return np.stack(
            [convolve_fourier(f[:, :, c], h) for c in range(f.shape[2])], axis=-1
        )
    outs, Fs, Hs, Gs, P = [], [], [], [], None
    for c in range(f.shape[2]):
        g, F, Hf, G, P = convolve_fourier(f[:, :, c], h, return_spectra=True)
        outs.append(g); Fs.append(F); Hs.append(Hf); Gs.append(G)
    return np.stack(outs, axis=-1), Fs, Hs, Gs, P


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def compare(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    rmse = float(np.sqrt(np.mean(d ** 2)))
    return {
        "max": float(d.max()),
        "rmse": rmse,
        "psnr": None if rmse == 0 else 20.0 * np.log10(255.0 / rmse),
    }


# -----------------------------------------------------------------------------
# Image loading: grayscale, letterboxed into an N x N float64 array
# -----------------------------------------------------------------------------
def load_image_gray(path: str, N: int) -> np.ndarray:
    im = Image.open(path).convert("L")
    im.thumbnail((N, N), Image.LANCZOS)
    canvas = Image.new("L", (N, N), 0)
    canvas.paste(im, ((N - im.width) // 2, (N - im.height) // 2))
    return np.asarray(canvas, dtype=np.float64)


# -----------------------------------------------------------------------------
# Same idea, kept in RGB: letterboxed into an (N, N, 3) float64 array.
# -----------------------------------------------------------------------------
def load_image_color(path: str, N: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    im.thumbnail((N, N), Image.LANCZOS)
    canvas = Image.new("RGB", (N, N), (0, 0, 0))
    canvas.paste(im, ((N - im.width) // 2, (N - im.height) // 2))
    return np.asarray(canvas, dtype=np.float64)


# -----------------------------------------------------------------------------
# Encoding results back to PNG data-URLs for the browser
# -----------------------------------------------------------------------------
def _png_data_url(mode_array_uint8: np.ndarray) -> str:
    im = Image.fromarray(mode_array_uint8, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def gray_to_data_url(arr: np.ndarray) -> str:
    a = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    return _png_data_url(a)


def diff_to_data_url(a: np.ndarray, b: np.ndarray, gain: float) -> str:
    d = np.clip(np.abs(a - b) * gain, 0, 255).astype(np.uint8)
    return _png_data_url(d)


def kernel_to_data_url(h: np.ndarray) -> str:
    mx = h.max() if h.max() > 0 else 1.0
    a = np.clip(np.round(255.0 * h / mx), 0, 255).astype(np.uint8)
    return _png_data_url(a)


# -----------------------------------------------------------------------------
# Color equivalents of the encoders above. Same clip/round logic, just kept
# as three channels and saved with PIL mode "RGB" instead of "L".
# -----------------------------------------------------------------------------
def _png_data_url_rgb(arr_uint8: np.ndarray) -> str:
    im = Image.fromarray(arr_uint8, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def color_to_data_url(arr: np.ndarray) -> str:
    a = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    return _png_data_url_rgb(a)


def diff_to_data_url_color(a: np.ndarray, b: np.ndarray, gain: float) -> str:
    d = np.clip(np.abs(a - b) * gain, 0, 255).astype(np.uint8)
    return _png_data_url_rgb(d)


# violet ramp, matches the original web demo
_RAMP_STOPS = np.array([
    [8, 11, 16], [62, 38, 104], [150, 110, 232], [238, 231, 252]
], dtype=np.float64)


def _ramp(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    s = t * (len(_RAMP_STOPS) - 1)
    i = np.clip(np.floor(s).astype(int), 0, len(_RAMP_STOPS) - 2)
    frac = (s - i)[..., None]
    return _RAMP_STOPS[i] + (_RAMP_STOPS[i + 1] - _RAMP_STOPS[i]) * frac


def spectrum_to_data_url(X: np.ndarray) -> str:
    mag = np.log1p(np.abs(np.fft.fftshift(X)))
    mx = mag.max() if mag.max() > 0 else 1.0
    rgb = _ramp(mag / mx).astype(np.uint8)
    im = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"