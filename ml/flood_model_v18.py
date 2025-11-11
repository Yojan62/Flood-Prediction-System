#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HydroFusion v18.1 (clean + fixes)
- Open-Meteo archive fetch
- LightGBM base learner (log target)
- Residual LSTM (peak-weighted)
- Robust early stopping & logging fixes
- NaN/Inf-safe metrics & predictions
- SHAP plots kept
- Predict CSV + plots + pickle bundle

Run:
  python flood_model_v18_1.py --lat 23.81 --lon 90.41 --start 2020-01-01 --end 2025-12-31 --lead 0 --quick
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Silence overly chatty libs first
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # TF info/warn off (errors only)
os.environ.setdefault("OMP_NUM_THREADS", "4")

# LightGBM & SHAP
import lightgbm as lgb
import shap

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# TensorFlow (CPU ok)
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Plotting
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Utility: small, targeted stderr filter for LightGBM "No further splits" spam
# -----------------------------------------------------------------------------

class _StderrFilter(io.TextIOBase):
    def __init__(self, underlying):
        self._under = underlying

    def write(self, s):
        if "[LightGBM] [Warning] No further splits with positive gain" in s:
            return len(s)
        return self._under.write(s)

    def flush(self):
        return self._under.flush()


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="HydroFusion v18.1")
    p.add_argument("--lat", type=float, default=23.81)
    p.add_argument("--lon", type=float, default=90.41)
    p.add_argument("--start", type=str, default="2020-01-01")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--lead", type=int, default=0, help="lead days (prediction horizon)")
    p.add_argument("--quick", action="store_true", help="faster training")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Environment report
# -----------------------------------------------------------------------------

def report_env():
    try:
        tfver = tf.__version__
    except Exception:
        tfver = "unknown"

    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception:
        gpus = []

    try:
        bi = tf.sysconfig.get_build_info()
    except Exception:
        bi = {}

    print(f"TensorFlow: {tfver}")
    print(f"GPUs detected: {[d.name for d in gpus] if gpus else []}")
    print(f"Build info: {bi}")


# -----------------------------------------------------------------------------
# Data fetching (Open-Meteo archive). We’ll derive a synthetic discharge proxy.
# IMPORTANT: If you already have a true discharge series, plug it in where
# we create `target` and drop the synthetic step.
# -----------------------------------------------------------------------------

OM_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

def fetch_openmeteo_archive(lat: float, lon: float,
                            start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily archive for basic meteo vars (precip, temp, etc.)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "precipitation_sum",
            "rain_sum",
            "temperature_2m_max",
            "temperature_2m_min",
            "windspeed_10m_max",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration"
        ],
        "timezone": "UTC",
    }
    r = requests.get(OM_ARCHIVE_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    if "daily" not in data or "time" not in data["daily"]:
        raise RuntimeError("Open-Meteo daily archive: missing 'daily' or 'time' in response")

    daily = pd.DataFrame(data["daily"])
    daily["time"] = pd.to_datetime(daily["time"])
    daily = daily.set_index("time").sort_index()
    # Rename columns for convenience
    daily = daily.rename(columns={
        "precipitation_sum": "precip",
        "rain_sum": "rain",
        "temperature_2m_max": "tmax",
        "temperature_2m_min": "tmin",
        "windspeed_10m_max": "wind",
        "shortwave_radiation_sum": "swrad",
        "et0_fao_evapotranspiration": "et0",
    })
    return daily


def engineer_features(daily: pd.DataFrame, lead_days: int) -> pd.DataFrame:
    """Create features & a synthetic discharge-like target as placeholder.
    Replace this synthetic target with real discharge if you have it!
    """
    df = daily.copy()

    # Basic derived vars
    df["tmean"] = (df["tmax"] + df["tmin"]) / 2.0
    df["diurnal_range"] = df["tmax"] - df["tmin"]
    df["precip_roll3"] = df["precip"].rolling(3, min_periods=1).sum()
    df["precip_roll7"] = df["precip"].rolling(7, min_periods=1).sum()
    df["rain_roll7"] = df["rain"].rolling(7, min_periods=1).sum()
    df["et0_roll7"] = df["et0"].rolling(7, min_periods=1).sum()
    df["tmean_roll7"] = df["tmean"].rolling(7, min_periods=1).mean()

    # Lags that often help for hydrology
    for l in [1, 2, 3, 7, 14]:
        df[f"precip_lag{l}"] = df["precip"].shift(l)
        df[f"rain_lag{l}"] = df["rain"].shift(l)
        df[f"tmean_lag{l}"] = df["tmean"].shift(l)

    # --- Synthetic discharge proxy (replace with observed discharge if available) ---
    # A simple kernel over recent precip & rain with evap penalty
    discharge = (
        2.5 * df["precip_roll7"] + 3.0 * df["rain_roll7"] +
        0.3 * df["wind"] - 0.8 * df["et0_roll7"] + 0.1 * df["tmean_roll7"]
    )
    discharge = discharge.clip(lower=0)
    # Add some smoothness + small noise
    discharge = discharge.ewm(span=3, adjust=False).mean()
    discharge = discharge + 0.05 * discharge.std() * np.random.RandomState(7).randn(len(discharge))
    discharge = discharge.clip(lower=0)

    # Shift target by lead days if predicting ahead
    df["target"] = discharge.shift(-lead_days)

    df = df.dropna()
    return df


def time_clip(start: str, end: str) -> Tuple[str, str]:
    """Avoid future end date."""
    today = dt.date.today().isoformat()
    if end > today:
        print(f"Warning: requested end_date {end} is in the future. Clipping to {today}.")
        end = today
    return start, end


# -----------------------------------------------------------------------------
# Metrics (NaN-safe)
# -----------------------------------------------------------------------------

def sanitize_array(a: np.ndarray, fill: float = 0.0) -> np.ndarray:
    if a is None:
        return np.array([], dtype=float)
    a = np.asarray(a, dtype=float)
    a = np.where(np.isfinite(a), a, np.nan)
    if np.any(np.isnan(a)):
        # replace NaN with fill to keep metrics defined
        a = np.where(np.isnan(a), fill, a)
    return a


def metric_pack(y_true: np.ndarray, y_pred: np.ndarray, thr: float) -> Dict[str, float]:
    y_true = sanitize_array(y_true)
    y_pred = sanitize_array(y_pred)
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "r2_peak": np.nan, "thr": thr}

    # basic
    mae = mean_absolute_error(y_true, y_pred)

    # rmse (manual to avoid sklearn 'squared' arg)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # r2
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = np.nan

    # peak-only r2
    mask = y_true >= thr
    if mask.sum() >= 3:
        try:
            r2_peak = float(r2_score(y_true[mask], y_pred[mask]))
        except Exception:
            r2_peak = np.nan
    else:
        r2_peak = np.nan

    return {"mae": float(mae), "rmse": rmse, "r2": r2, "r2_peak": float(r2_peak), "thr": float(thr)}


# -----------------------------------------------------------------------------
# LightGBM (OOF + full model) — log target
# -----------------------------------------------------------------------------

def train_lgb_oof_log(
    X: np.ndarray,
    y: np.ndarray,
    weight: Optional[np.ndarray] = None,
    quick: bool = False,
    random_state: int = 42
) -> Tuple[np.ndarray, lgb.LGBMRegressor]:

    n = X.shape[0]
    oof_pred = np.zeros(n, dtype=float)

    params = dict(
        n_estimators=(700 if not quick else 300),
        learning_rate=0.03 if not quick else 0.05,
        max_depth=-1,
        num_leaves=64 if not quick else 48,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_samples=20,
        min_split_gain=0.0,
        objective="regression",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,                   # silence info
        # small positive gain threshold discourages futile splits
        min_gain_to_split=1e-6,
    )

    kf = KFold(n_splits=4, shuffle=True, random_state=random_state)

    # We’ll filter LightGBM stderr spam during CV
    with redirect_lgb_stderr():
        for i, (tr, va) in enumerate(kf.split(X), 1):
            Xtr, Xva = X[tr], X[va]
            ytr, yva = y[tr], y[va]
            wtr = None if weight is None else weight[tr]

            ytr_log = np.log1p(np.maximum(ytr, 0.0))
            yva_log = np.log1p(np.maximum(yva, 0.0))

            model = lgb.LGBMRegressor(**params)

            # Use early stopping only here (has eval_set)
            model.fit(
                Xtr, ytr_log,
                sample_weight=wtr,
                eval_set=[(Xva, yva_log)],
                eval_metric="l2",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=(30 if quick else 60), verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            oof_pred[va] = np.expm1(np.clip(model.predict(Xva, raw_score=False), a_min=-50, a_max=50))

    # Train a "full" model without early-stopping callbacks
    y_log = np.log1p(np.maximum(y, 0.0))
    model_full = lgb.LGBMRegressor(**params)
    with redirect_lgb_stderr():
        model_full.fit(
            X, y_log,
            sample_weight=weight,
            # no eval_set, no callbacks -> avoids early-stopping error
        )

    return oof_pred, model_full


from contextlib import contextmanager, redirect_stderr

@contextmanager
def redirect_lgb_stderr():
    """Filter the LightGBM split warning spam while preserving real errors."""
    try:
        filt = _StderrFilter(sys.stderr)
        with redirect_stderr(filt):
            yield
    finally:
        pass


# -----------------------------------------------------------------------------
# Residual LSTM (peak-weighted)
# -----------------------------------------------------------------------------

def make_lstm(seq_len: int, nfeat: int, lr: float = 1e-3) -> keras.Model:
    inp = layers.Input(shape=(seq_len, nfeat))
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def build_seq_dataset(df: pd.DataFrame, features: List[str], target_col: str,
                      seq_len: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Make overlapping sequences [t-seq_len+1 ... t] -> y_t."""
    Xall = df[features].values.astype("float32")
    yall = df[target_col].values.astype("float32")

    xs, ys = [], []
    for t in range(seq_len - 1, len(df)):
        xs.append(Xall[t - (seq_len - 1): t + 1])
        ys.append(yall[t])
    if not xs:
        return np.zeros((0, seq_len, len(features)), dtype="float32"), np.zeros((0,), dtype="float32")
    return np.stack(xs, 0), np.array(ys)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def save_backtest_plot(dates, y_true, y_base, y_final, out_png="v18_backtest.png"):
    plt.figure(figsize=(12, 4))
    plt.plot(dates, y_true, label="Observed")
    plt.plot(dates, y_base, label="Base (LGB)")
    if y_final is not None:
        plt.plot(dates, y_final, label="Hybrid (LGB+LSTM)")
    plt.legend()
    plt.title("Backtest")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_peaks_plot(dates, y_true, y_pred, thr, out_png="v18_backtest_peaks.png"):
    m = y_true >= thr
    plt.figure(figsize=(12, 4))
    plt.plot(dates, y_true, label="Observed")
    plt.plot(dates, y_pred, label="Prediction")
    plt.scatter(np.array(dates)[m], y_true[m], label="Observed Peaks", marker="o")
    plt.axhline(thr, linestyle="--", label=f"Peak thr={thr:.1f}")
    plt.legend()
    plt.title("Peaks focus")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_scatter(y_true, y_pred, out_png="v18_obs_vs_pred.png"):
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(y_true, y_pred, s=10)
    lim = (0, max(1.0, np.percentile(np.concatenate([y_true, y_pred]), 99.5)))
    plt.plot(lim, lim, "--")
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title("Observed vs Predicted")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# SHAP
# -----------------------------------------------------------------------------

def shap_summary_png(model: lgb.LGBMRegressor, X_sample: np.ndarray, feature_names: List[str],
                     out_png="v18_shap_summary.png"):
    """Compute + save SHAP summary for LightGBM base model."""
    try:
        explainer = shap.TreeExplainer(model.booster_)
    except Exception:
        explainer = shap.TreeExplainer(model)
    # limit sample to keep runtime sensible
    if X_sample.shape[0] > 2000:
        X_use = X_sample[:2000]
    else:
        X_use = X_sample
    shap_values = explainer.shap_values(X_use)
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X_use, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Training driver
# -----------------------------------------------------------------------------

@dataclass
class TrainResult:
    final_metrics: Dict[str, float]
    base_metrics: Dict[str, float]


def train_v18(lat: float, lon: float, start_date: str, end_date: str,
              lead_days: int = 0, quick: bool = False) -> TrainResult:
    print(f"HydroFusion v18.1 — Training {start_date} → {end_date} (lat={lat:.2f}, lon={lon:.2f})")
    start_date, end_date = time_clip(start_date, end_date)

    # 1) Fetch & features
    # Split by year-pages for logging clarity (mirrors your prints)
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)
    years = list(range(sd.year, ed.year + 1))
    chunks = []
    for y in years:
        y0 = pd.Timestamp(year=y, month=1, day=1)
        y1 = pd.Timestamp(year=y, month=12, day=31)
        a = max(y0, sd)
        b = min(y1, ed)
        print(f"→ Fetching archive {a.date()} → {b.date()}")
        df_y = fetch_openmeteo_archive(lat, lon, a.date().isoformat(), b.date().isoformat())
        chunks.append(df_y)
    daily = pd.concat(chunks).sort_index()
    df = engineer_features(daily, lead_days=lead_days)

    # 2) Train/test split (last 20% as test)
    n = len(df)
    if n < 30:
        raise RuntimeError("Very few rows after feature engineering; results may be unstable.")

    split_idx = int(n * 0.8)
    tr_df = df.iloc[:split_idx].copy()
    te_df = df.iloc[split_idx:].copy()

    # Decide peak threshold from train
    thr = float(np.nanpercentile(tr_df["target"], 90))

    # Feature list
    all_cols = [c for c in df.columns if c not in ["target"]]
    feature_names = all_cols

    Xtr = tr_df[feature_names].values
    ytr = tr_df["target"].values
    Xte = te_df[feature_names].values
    yte = te_df["target"].values

    print(f"Train {len(tr_df)}, Test {len(te_df)}")

    # Optional peak augmentation: duplicate rows above threshold with weights
    peak_mask = tr_df["target"].values >= thr
    w_base = np.ones(len(tr_df), dtype=float)
    w_base[peak_mask] = 2.0

    # 3) Scale features
    scaler = StandardScaler()
    scaler.fit(Xtr)  # fit only on train
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    print("\nBase LightGBM OOF (log-target) ...")

    # 4) Base LGB
    base_oof, lgb_full = train_lgb_oof_log(Xtr_s, ytr, weight=w_base, quick=quick)

    # Base predictions
    yte_base = np.expm1(np.clip(lgb_full.predict(Xte_s), a_min=-50, a_max=50))
    ytr_base = base_oof  # already inverted in function

    # 5) Residual LSTM
    print("\nTraining residual LSTM (peak-weighted)...")

    seq_len = 7
    # residuals on train part
    tr_df = tr_df.copy()
    tr_df["residual"] = tr_df["target"].values - ytr_base

    # LSTM features: you may include both raw features and base prediction
    lstm_feats = feature_names + []  # keep simple: same features
    # Scale features separately for LSTM (can reuse scaler too; here we reuse Xtr_s)
    tr_df_s = tr_df.copy()
    tr_df_s[lstm_feats] = Xtr_s

    # Build sequences (train)
    Xseq_tr, yseq_tr = build_seq_dataset(tr_df_s, lstm_feats, target_col="residual", seq_len=seq_len)

    # Sample weights (peaks heavier)
    y_tr_target_for_weight = tr_df["target"].values[seq_len - 1:]
    sw = np.where(y_tr_target_for_weight >= thr, 2.0, 1.0).astype("float32")

    # If not enough sequences, fallback to base
    use_lstm = len(Xseq_tr) >= 50

    val_split = 0.15
    epochs = 80 if not quick else 40
    batch_size = 32

    lstm_model = None
    if use_lstm:
        lstm_model = make_lstm(seq_len=seq_len, nfeat=len(lstm_feats), lr=(5e-4 if not quick else 1e-3))
        es = keras.callbacks.EarlyStopping(patience=8 if not quick else 5, restore_best_weights=True, monitor="val_loss")
        lstm_model.fit(
            Xseq_tr, yseq_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            sample_weight=sw,
            verbose=0,
            callbacks=[es]
        )

    # Predict residuals on TEST via rolling window
    # Prepare test DF with scaled features
    te_df_s = te_df.copy()
    te_df_s[lstm_feats] = Xte_s

    # Concatenate tail of train to seed sequences
    seed_tail = tr_df_s.iloc[-(seq_len - 1):][lstm_feats]
    seq_source = pd.concat([seed_tail, te_df_s[lstm_feats]], axis=0).reset_index(drop=True)

    if use_lstm:
        # Build test sequences aligned to test rows
        Xseq_te = []
        for t in range(seq_len - 1, (seq_len - 1) + len(te_df_s)):
            Xseq_te.append(seq_source.iloc[t - (seq_len - 1): t + 1].values.astype("float32"))
        Xseq_te = np.stack(Xseq_te, 0)
        yte_res = lstm_model.predict(Xseq_te, verbose=0).reshape(-1)
    else:
        yte_res = np.zeros(len(te_df_s), dtype="float32")

    # Final hybrid
    yte_final = yte_base + yte_res

    # 6) Metrics (base vs final)
    base_metrics = metric_pack(yte, yte_base, thr)
    final_metrics = metric_pack(yte, yte_final, thr)

    print("\n--- Base Model Metrics (Test) ---")
    for k, v in base_metrics.items():
        if k != "thr":
            print(f"{k}: {v:.6f}")
    print(f"thr: {base_metrics['thr']:.6f}")

    print("\n--- Final Hybrid Metrics (Test) ---")
    for k, v in final_metrics.items():
        if k != "thr":
            print(f"{k}: {v:.6f}")
    print(f"thr: {final_metrics['thr']:.6f}")

    # 7) Save outputs
    out_csv = "v18_test_predictions.csv"
    out_bundle = os.path.join("ml", "open_meteo_flood_model_v18.pkl")
    os.makedirs("ml", exist_ok=True)

    # Predictions CSV
    out_df = te_df.copy()
    out_df = out_df.assign(
        y_true=yte,
        y_base=yte_base,
        y_final=yte_final
    )
    out_df.to_csv(out_csv, index=True)
    print(f"Saved predictions to {out_csv}")

    # Plots
    save_backtest_plot(te_df.index, yte, yte_base, yte_final, out_png="v18_backtest.png")
    save_peaks_plot(te_df.index, yte, yte_final, thr, out_png="v18_backtest_peaks.png")
    save_scatter(yte, yte_final, out_png="v18_obs_vs_pred.png")
    print("Saved backtest plot to v18_backtest.png")
    print("Saved peaks plot to v18_backtest_peaks.png")
    print("Saved observed vs predicted scatter: v18_obs_vs_pred.png")

    # SHAP on base (use test-scaled features and feature names)
    try:
        print("Computing SHAP values for LightGBM (base)...")
        shap_summary_png(lgb_full, Xte_s, feature_names, out_png="v18_shap_summary.png")
        print("Saved SHAP summary to v18_shap_summary.png")
    except Exception as e:
        print(f"SHAP failed (skipped): {e}")

    # Bundle (scaler + lgb + lstm meta)
    bundle = {
        "scaler": scaler,
        "feature_names": feature_names,
        "lgb_model": lgb_full,
        "lstm_seq_len": seq_len,
        "lstm_feats": lstm_feats,
        "thr": thr,
        "meta": {
            "lat": lat, "lon": lon,
            "start": start_date, "end": end_date,
            "lead_days": lead_days,
            "timestamp": dt.datetime.utcnow().isoformat() + "Z"
        }
    }
    try:
        import pickle
        with open(out_bundle, "wb") as f:
            pickle.dump(bundle, f)
        print(f"Saved model bundle to {out_bundle}")
    except Exception as e:
        print(f"Bundle save failed (skipped): {e}")

    return TrainResult(final_metrics=final_metrics, base_metrics=base_metrics)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    report_env()

    # soft pin of numpy print
    np.set_printoptions(suppress=True, linewidth=120, precision=6)

    t0 = time.time()
    try:
        res = train_v18(
            lat=args.lat, lon=args.lon,
            start_date=args.start, end_date=args.end,
            lead_days=args.lead, quick=args.quick
        )
        print(f"✅ Training complete: {json.dumps({'final_metrics': res.final_metrics, 'base_metrics': res.base_metrics})}")
    except Exception as e:
        print("❌ Training failed:", repr(e))
        raise
    finally:
        print(f"Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
