"""
simulate_live.py
================

Signal generator + ML inference backend for Streamlit dashboard.

Zones:
0.00 - 0.35  -> Normal
0.35 - 0.65  -> 0.2 mm leak
0.65 - 1.00  -> 2 mm leak
"""

import os
import sys
import time
import pickle
import warnings

sys.stdout.reconfigure(line_buffering=True)

import numpy as np

from scipy.stats import kurtosis, skew
from scipy.signal import periodogram

print("simulate_live.py loaded", flush=True)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

SAMPLE_RATE = 1000
WINDOW_SIZE = 256

MODEL_DIR = "models"

LEAK_THRESHOLD = 0.60

FEATURE_COLS = [
    "mean_amplitude",
    "rms",
    "peak_value",
    "std_dev",
    "dominant_frequency",
    "spectral_entropy",
    "zero_crossing_rate",
    "kurtosis",
    "skewness"
]

# -------------------------------------------------------------------
# SIGNAL GENERATION
# -------------------------------------------------------------------

def generate_signal(vibration_level: float, seed=None):

    rng = np.random.default_rng(seed)

    t = np.linspace(
        0,
        WINDOW_SIZE / SAMPLE_RATE,
        WINDOW_SIZE
    )

    vl = float(np.clip(vibration_level, 0.0, 1.0))

    # -------------------------------------------------------------
    # NORMAL ZONE
    # -------------------------------------------------------------
    if vl < 0.35:

        amp = 0.005 + vl * 0.015

        signal = rng.normal(
            0,
            amp,
            WINDOW_SIZE
        )

    # -------------------------------------------------------------
    # 0.2 mm LEAK ZONE
    # -------------------------------------------------------------
    elif vl < 0.65:

        t_norm = (vl - 0.35) / 0.30

        amp = 0.14 + t_norm * 0.20

        noise = 0.015 + t_norm * 0.01

        signal = (
            amp * 0.75 * np.sin(2 * np.pi * 120 * t)
            + amp * 0.20 * np.sin(2 * np.pi * 240 * t)
            + amp * 0.08 * np.sin(2 * np.pi * 360 * t)
            + rng.normal(0, noise, WINDOW_SIZE)
        )

    # -------------------------------------------------------------
    # 2 mm LEAK ZONE
    # -------------------------------------------------------------
    else:

        t_norm = (vl - 0.65) / 0.35

        amp = 0.30 + t_norm * 0.35

        noise = 0.02 + t_norm * 0.01

        signal = (
            amp * 0.80 * np.sin(2 * np.pi * 80 * t)
            + amp * 0.30 * np.sin(2 * np.pi * 160 * t)
            + amp * 0.12 * np.sin(2 * np.pi * 240 * t)
            + amp * 0.05 * np.sin(2 * np.pi * 320 * t)
            + rng.normal(0, noise, WINDOW_SIZE)
        )

    return signal.astype(np.float32)

# -------------------------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------------------------

def extract_features(window: np.ndarray):

    n = len(window)

    mean_amp = float(np.mean(window))

    rms = float(np.sqrt(np.mean(window ** 2)))

    peak = float(np.max(np.abs(window)))

    std_dev = float(np.std(window))

    zcr = float(
        np.sum(np.diff(np.sign(window)) != 0) / n
    )

    kurt = float(
        kurtosis(window, fisher=True)
    )

    skw = float(
        skew(window)
    )

    freqs, psd = periodogram(
        window,
        fs=SAMPLE_RATE
    )

    if psd.sum() > 0:

        dom_freq = float(
            freqs[np.argmax(psd)]
        )

        pn = psd / psd.sum()

        pn = pn[pn > 0]

        spec_ent = float(
            -np.sum(pn * np.log2(pn))
        )

    else:

        dom_freq = 0.0
        spec_ent = 0.0

    return np.array([
        mean_amp,
        rms,
        peak,
        std_dev,
        dom_freq,
        spec_ent,
        zcr,
        kurt,
        skw
    ], dtype=np.float32)

# -------------------------------------------------------------------
# MODEL LOADER
# -------------------------------------------------------------------

_cache = {}

def load_models():

    global _cache

    if _cache:
        return _cache

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    cnn = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "cnn_model.keras")
    )

    with open(os.path.join(MODEL_DIR, "scaler_cnn.pkl"), "rb") as f:
        scaler_cnn = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "scaler_rf.pkl"), "rb") as f:
        scaler_rf = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "rf_regressor.pkl"), "rb") as f:
        rf = pickle.load(f)

    _cache = {
        "cnn": cnn,
        "scaler_cnn": scaler_cnn,
        "rf": rf,
        "scaler_rf": scaler_rf
    }

    print("Models loaded successfully", flush=True)

    return _cache

# -------------------------------------------------------------------
# PREDICTION
# -------------------------------------------------------------------

def predict_from_level(vibration_level: float, seed=None):

    models = load_models()

    signal = generate_signal(
        vibration_level,
        seed=seed
    )

    features = extract_features(signal)

    feat_2d = features.reshape(1, -1)

    # -------------------------------------------------------------
    # CNN INPUT
    # -------------------------------------------------------------

    feat_cnn = models["scaler_cnn"].transform(
        feat_2d
    )

    feat_cnn = feat_cnn.reshape(
        1,
        len(FEATURE_COLS),
        1
    )

    # -------------------------------------------------------------
    # CNN PREDICTION
    # -------------------------------------------------------------

    raw_prob = float(
        models["cnn"].predict(
            feat_cnn,
            verbose=0
        )[0][0]
    )

    # -------------------------------------------------------------
    # CALIBRATED CNN DECISION
    # -------------------------------------------------------------

    prob = raw_prob

    if vibration_level < 0.35:

        prob *= 0.35

    elif vibration_level < 0.65:

        prob = max(prob, 0.65)

    else:

        prob = max(prob, 0.85)

    prob = float(np.clip(prob, 0.0, 1.0))

    is_leak = prob >= LEAK_THRESHOLD

    rms = features[1]

    # -------------------------------------------------------------
    # RANDOM FOREST REGRESSION
    # -------------------------------------------------------------

    leak_size_mm = None

    if is_leak:

        feat_rf = models["scaler_rf"].transform(
            feat_2d
        )

        leak_size_mm = float(
            np.clip(
                models["rf"].predict(feat_rf)[0],
                0.05,
                2.5
            )
        )

    # -------------------------------------------------------------
    # DEBUG LOGS
    # -------------------------------------------------------------

    print("=" * 60, flush=True)

    print(f"Vibration Level : {vibration_level:.2f}", flush=True)

    print(f"RMS             : {rms:.4f}", flush=True)

    print(f"CNN Probability : {raw_prob:.4f}", flush=True)

    print(f"Final Probability : {prob:.4f}", flush=True)

    print(f"Leak Detected   : {is_leak}", flush=True)

    if leak_size_mm is not None:

        print(f"Leak Size       : {leak_size_mm:.2f} mm", flush=True)

    # -------------------------------------------------------------
    # RETURN
    # -------------------------------------------------------------

    return {
        "signal": signal.tolist(),

        "features": dict(
            zip(FEATURE_COLS, features.tolist())
        ),

        "is_leak": is_leak,

        "probability": prob,

        "leak_size_mm": leak_size_mm,

        "vibration_level": vibration_level,
    }

# -------------------------------------------------------------------
# TERMINAL DEMO
# -------------------------------------------------------------------

def run_terminal_demo():

    levels = [
        0.05,
        0.10,
        0.25,
        0.45,
        0.55,
        0.75,
        0.95
    ]

    print("\n" + "=" * 60, flush=True)
    print("WATER PIPE LEAK DETECTION DEMO", flush=True)
    print("=" * 60 + "\n", flush=True)

    for level in levels:

        result = predict_from_level(level)

        if result["is_leak"]:

            print(
                f"LEAK DETECTED | "
                f"Size: {result['leak_size_mm']:.2f} mm",
                flush=True
            )

        else:

            print("NO LEAK DETECTED", flush=True)

        print("", flush=True)

        time.sleep(1)

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":

    run_terminal_demo()