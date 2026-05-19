"""
Step 1b: Train the Random Forest Regressor
============================================
Trains a Random Forest on the LEAK-ONLY samples to estimate leak size.
  Input  : 9 vibration features (same as CNN)
  Output : Leak size in mm  (range 0.1 – 2.0 mm)

Saves:
  models/rf_regressor.pkl
  models/scaler_rf.pkl
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Config ───────────────────────────────────────────────────────────────────
SEED      = 42
DATA_PATH = "data/water_leakage_dataset.csv"
MODEL_DIR = "models"

FEATURE_COLS = [
    "mean_amplitude", "rms", "peak_value", "std_dev",
    "dominant_frequency", "spectral_entropy",
    "zero_crossing_rate", "kurtosis", "skewness"
]

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load & filter leak-only samples ──────────────────────────────────────────
print("=" * 55)
print("  Random Forest Regressor — Leak Size Estimation")
print("=" * 55)

df = pd.read_csv(DATA_PATH)

# Keep only leak samples for regression
df_leak = df[df["leak_label"] != "normal"].copy()
print(f"\n[Data] Total samples      : {len(df)}")
print(f"[Data] Leak-only samples  : {len(df_leak)}")
print(f"[Data] Leak size range    : {df_leak['leak_size_mm'].min():.2f} – {df_leak['leak_size_mm'].max():.2f} mm")
print(f"[Data] Leak size values   : {sorted(df_leak['leak_size_mm'].unique())}\n")

X = df_leak[FEATURE_COLS].values.astype(np.float32)
y = df_leak["leak_size_mm"].values.astype(np.float32)

# Scale
scaler_rf = StandardScaler()
X_scaled = scaler_rf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED
)

print(f"[Split] Train: {len(X_train)}  |  Test: {len(X_test)}\n")

# ── Train Random Forest ───────────────────────────────────────────────────────
print("[Training] Fitting Random Forest Regressor ...")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=SEED
)
rf.fit(X_train, y_train)
print("[Training] Done.\n")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = rf.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("[Evaluation] Test set results:")
print(f"  MAE  : {mae:.4f} mm")
print(f"  RMSE : {rmse:.4f} mm")
print(f"  R²   : {r2:.4f}")

# 5-fold CV
cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring="r2", n_jobs=-1)
print(f"\n[Cross-Val] R² per fold  : {[f'{s:.4f}' for s in cv_scores]}")
print(f"[Cross-Val] Mean R²      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Feature importances ───────────────────────────────────────────────────────
importances = rf.feature_importances_
idx = np.argsort(importances)[::-1]

print("\n[Feature Importances]")
for i in idx:
    print(f"  {FEATURE_COLS[i]:25s}: {importances[i]:.4f}")

plt.figure(figsize=(8, 4))
plt.bar([FEATURE_COLS[i] for i in idx], importances[idx], color="#1a5c4b")
plt.xticks(rotation=35, ha="right", fontsize=8)
plt.title("Random Forest — Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "rf_feature_importance.png"), dpi=120)
plt.close()

# ── Actual vs Predicted plot ──────────────────────────────────────────────────
plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, alpha=0.4, color="#c8873a", edgecolors="none", s=20)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "k--", lw=1)
plt.xlabel("Actual (mm)"); plt.ylabel("Predicted (mm)")
plt.title(f"Actual vs Predicted  |  R²={r2:.4f}")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "rf_actual_vs_predicted.png"), dpi=120)
plt.close()

# ── Save ──────────────────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "rf_regressor.pkl"), "wb") as f:
    pickle.dump(rf, f)

with open(os.path.join(MODEL_DIR, "scaler_rf.pkl"), "wb") as f:
    pickle.dump(scaler_rf, f)

print(f"\n[Saved] models/rf_regressor.pkl")
print(f"[Saved] models/scaler_rf.pkl")
print(f"[Saved] models/rf_feature_importance.png")
print(f"[Saved] models/rf_actual_vs_predicted.png")
print("\n  RF Regressor training complete!\n")
