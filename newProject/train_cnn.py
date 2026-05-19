"""
Step 1a: Train the 1D CNN Binary Classifier
============================================
Trains a 1D CNN on extracted vibration features to classify:
  - 0 → Normal (no leak)
  - 1 → Leak (0.2mm or 2mm)

Saves:
  models/cnn_model.keras
  models/scaler_cnn.pkl
  models/label_encoder_cnn.pkl
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH   = "data/water_leakage_dataset.csv"
MODEL_DIR   = "models"
EPOCHS      = 40
BATCH_SIZE  = 64
TEST_SIZE   = 0.2

FEATURE_COLS = [
    "mean_amplitude", "rms", "peak_value", "std_dev",
    "dominant_frequency", "spectral_entropy",
    "zero_crossing_rate", "kurtosis", "skewness"
]

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load & prepare data ───────────────────────────────────────────────────────
print("=" * 55)
print("  1D CNN — Binary Leak Classifier — Training")
print("=" * 55)

df = pd.read_csv(DATA_PATH)
print(f"\n[Data] Loaded {len(df)} samples")
print(f"[Data] Label distribution:\n{df['leak_label'].value_counts().to_string()}\n")

X = df[FEATURE_COLS].values.astype(np.float32)
y = (df["leak_label"] != "normal").astype(np.int32).values   # 0=normal, 1=leak

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for 1D CNN: (samples, timesteps=9, channels=1)
X_cnn = X_scaled.reshape(-1, len(FEATURE_COLS), 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_cnn, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
)

print(f"[Split] Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── Build 1D CNN ─────────────────────────────────────────────────────────────
def build_cnn(input_shape):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(1, activation="sigmoid")(x)   # binary output
    return models.Model(inp, out, name="LeakDetector_1DCNN")

model = build_cnn((len(FEATURE_COLS), 1))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ── Callbacks ────────────────────────────────────────────────────────────────
cb_list = [
    callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
    callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "cnn_model.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1
    )
]

# ── Train ────────────────────────────────────────────────────────────────────
print("\n[Training] Starting ...\n")
history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb_list,
    verbose=1
)

# ── Evaluate ─────────────────────────────────────────────────────────────────
print("\n[Evaluation] Test set results:")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Loss    : {loss:.4f}")
print(f"  Accuracy: {acc * 100:.2f}%")

y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
print("\n[Classification Report]")
print(classification_report(y_test, y_pred, target_names=["Normal", "Leak"]))

# ── Save artefacts ───────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "scaler_cnn.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Save feature list so inference knows the order
with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "wb") as f:
    pickle.dump(FEATURE_COLS, f)

# Save training history plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"],     label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy"); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"],     label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss"); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "cnn_training_history.png"), dpi=120)
plt.close()

print(f"\n[Saved] models/cnn_model.keras")
print(f"[Saved] models/scaler_cnn.pkl")
print(f"[Saved] models/feature_cols.pkl")
print(f"[Saved] models/cnn_training_history.png")
print("\n  CNN training complete!\n")
