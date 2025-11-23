#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HydroFusion v21 - Autoregressive Model (FIXED)

This model pivots away from the hybrid weather approach (v15-v20)
which failed to predict peaks (negative r2_peak).

This script implements a robust autoregressive model that *only* uses
historical river discharge data, which is the correct signal for this
fluvial (river-driven) system.

- Data: Fetches *only* 'river_discharge' from the Flood API.
- Features: Uses only discharge lags, rates, rolling avgs, and time features.
- Training: Trains 4 models (RF, GB, LGBM, CatBoost) in a competition.
- Peak Fix: Trains models *directly* (no GridSearchCV) and uses a high
  sample weight (alpha=25.0) to force peak prediction.
- Selection: Automatically selects the champion model based on R2_PEAK.

Usage:
    python flood_model_v21.py --mode train [--quick]
"""

import os, time, math, json, argparse, warnings
from datetime import date
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from dotenv import load_dotenv 
import sys 
import io 
warnings.filterwarnings("ignore")

from contextlib import contextmanager, redirect_stderr

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    
# optional libs
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not found. Skipping CatBoost model.")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False
    print("Warning: lightgbm not found. Skipping LGBM model.")

# ---------- config ----------
LAT = 23.81
LON = 90.41
DEFAULT_START = "2012-01-01" # Using longer history
DEFAULT_END = "2025-12-31"
LEAD_DAYS = 1
TIMEZONE = "auto"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FLOOD = "https://flood-api.open-meteo.com/v1/flood"

MODEL_DIR = "ml"
MODEL_FILE = "open_meteo_flood_model_v21.pkl"
FEATURES_JSON = "open_meteo_flood_features_v21.json"
PLOT_FULL = "v21_backtest.png"
PLOT_PEAKS = "v21_backtest_peaks.png"
PLOT_SHAP = "v21_shap_summary.png"
PLOT_BACKTEST = "v21_backtest.png"
PLOT_OBS_PRED = "v21_obs_vs_pred.png"
PRED_CSV = "v21_test_predictions.csv"

RANDOM_STATE = 42
FLOOD_Q = 0.90
PEAK_WEIGHT_ALPHA = 25.0 # High weight to force peak learning
TIME_SERIES_SPLITS = 4

# Global ARGS
ARGS = None

# ---------- utilities ----------
class _StderrFilter(io.TextIOBase):
    def __init__(self, underlying):
        self._under = underlying
    def write(self, s):
        if "[LightGBM] [Warning] No further splits with positive gain" in s:
            return len(s)
        return self._under.write(s)
    def flush(self):
        return self._under.flush()

@contextmanager
def redirect_lgb_stderr():
    try:
        filt = _StderrFilter(sys.stderr)
        with redirect_stderr(filt):
            yield
    finally:
        pass
        
def safe_get_json(url, params, timeout=60):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def nse(obs, sim):
    obs = np.array(obs); sim = np.array(sim)
    denom = np.sum((obs - obs.mean())**2)
    return 1 - np.sum((obs - sim)**2) / denom if denom != 0 else float("nan")

def combined_r2_peak(y_true, y_pred, alpha=0.7, q=FLOOD_Q):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    overall = r2_score(y_true, y_pred)
    thr = np.nanpercentile(y_true, q*100)
    mask = y_true >= thr
    peak = r2_score(y_true[mask], y_pred[mask]) if mask.sum() >= 2 else overall
    return alpha * overall + (1-alpha) * peak

COMBINED_SCORER = make_scorer(combined_r2_peak, greater_is_better=True)

# ---------- fetching (per-year) ----------
# --- v21: REMOVED fetch_archive_hourly ---

def fetch_flood_daily(lat, lon, start_date, end_date):
    """Fetches REAL historical river discharge (daily) from the Flood API."""
    print(f"Fetching river discharge data for {start_date} -> {end_date}")
    start = pd.to_datetime(start_date).date()
    end   = pd.to_datetime(end_date).date()
    today = date.today()
    if end > today:
        print(f"Warning: requested end_date {end} is in the future. Clipping to today {today}.")
        end = today
    
    # We'll fetch in 5-year chunks to be safe with the API
    dfs = []
    year = start.year
    while year <= end.year:
        seg_start = max(start, date(year, 1, 1))
        seg_end   = min(end, date(year + 4, 12, 31)) # 5-year chunks
        print(f"→ Fetching discharge segment {seg_start} → {seg_end}")
        params = {
            "latitude": lat, "longitude": lon, "daily": "river_discharge",
            "start_date": seg_start.isoformat(), "end_date": seg_end.isoformat(), "timezone": TIMEZONE
        }
        j = safe_get_json(OPEN_METEO_FLOOD, params)
        df_chunk = pd.DataFrame(j.get("daily", {}))
        if 'time' in df_chunk.columns:
            df_chunk = df_chunk.rename(columns={"time": "date", "river_discharge": "discharge_m3_s"})
            df_chunk['date'] = pd.to_datetime(df_chunk['date'])
            df_chunk = df_chunk.set_index('date').sort_index()
            dfs.append(df_chunk)
        year += 5 # Move to the next 5-year block
        time.sleep(0.5)

    if not dfs:
        return pd.DataFrame()
        
    df = pd.concat(dfs).sort_index()
    # Ensure no duplicate index entries
    df = df[~df.index.duplicated(keep='first')]
    return df

# ---------- features ----------
# --- v21: REMOVED build_daily_merged ---

def engineer_features(df_daily, lead_days=1):
    """Creates all autoregressive time-series features."""
    print("Engineering features...")
    d = df_daily.copy().sort_index()
    
    if 'discharge_m3_s' not in d.columns:
        raise ValueError("DataFrame must contain 'discharge_m3_s' from fetch_flood_daily")

    # Discharge lags (autoregressive features)
    d['dis_lag1'] = d['discharge_m3_s'].shift(1)
    d['dis_lag2'] = d['discharge_m3_s'].shift(2)
    d['dis_lag3'] = d['discharge_m3_s'].shift(3)
    d['dis_lag7'] = d['discharge_m3_s'].shift(7)
    
    # Discharge momentum and smoothing
    d['dis_rate'] = d['discharge_m3_s'] - d['dis_lag1']
    d['dis_roll_3'] = d['discharge_m3_s'].shift(1).rolling(3).mean()
    d['dis_roll_7'] = d['discharge_m3_s'].shift(1).rolling(7).mean()
    
    # Cyclical time features (seasonality)
    d['month'] = d.index.month
    d['day_of_year'] = d.index.dayofyear
    d['month_sin'] = np.sin(2*np.pi*d['month']/12)
    d['month_cos'] = np.cos(2*np.pi*d['month']/12)
    
    # --- This is the TARGET variable ---
    d['target'] = d['discharge_m3_s'].shift(-lead_days)

    # Drop rows that have NaNs from the lag/shift operations
    d = d.dropna(subset=['target','dis_lag7','dis_roll_7'])
    
    print(f"Total rows after feature engineering: {len(d)}")
    return d

# --- v21: REMOVED augment_with_surges ---
# (Not needed, as we are modeling discharge on discharge)

# ---------- feature list for tabular models ----------
FEATURE_LIST = [
 'discharge_m3_s','dis_lag1','dis_lag2','dis_lag3','dis_lag7',
 'dis_rate','dis_roll_3','dis_roll_7',
 'month_sin','month_cos','day_of_year'
]

def prepare_tabular_Xy(df, feature_list):
    """Prepares X and y, returning X, y, and the final list of columns used."""
    
    # Find constant columns BEFORE filling NaNs
    constant_cols = []
    for col in feature_list:
        if col in df.columns:
            if df[col].std() < 1e-6:
                constant_cols.append(col)
        
    if constant_cols:
        print(f"Warning: Found constant features, removing them: {constant_cols}")
    
    features_to_use = [f for f in feature_list if f not in constant_cols]
    
    X = df.reindex(columns=features_to_use).apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    y = df['target'].astype(np.float32)
    return X, y, features_to_use

# ---------- training workflow ----------
def train_v21(lat, lon, start, end, lead_days=1, quick=False):
    print(f"HydroFusion v21 — Training {start} → {end} (lat={lat:.2f}, lon={lon:.2f})")
    
    # --- 1. Fetch & Prepare Data ---
    df_flood  = fetch_flood_daily(lat, lon, start, end)
    
    if df_flood.empty:
        print(f"❌ No daily flood data found for the given range. Aborting.")
        return
        
    df = engineer_features(df_flood, lead_days=lead_days).apply(pd.to_numeric, errors='coerce')
    
    # Fill any NaNs created by feature engineering before splitting
    df = df.fillna(0)
    
    split = int(len(df)*0.8)
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()
    print(f"Train {len(train_df)}, Test {len(test_df)}")

    # --- 2. Define Features & Get Weights ---
    # We define the feature list *after* engineering
    GLOBAL_FEATURE_LIST = [
        'discharge_m3_s','dis_lag1','dis_lag2','dis_lag3','dis_lag7',
        'dis_rate','dis_roll_3','dis_roll_7',
        'month_sin','month_cos','day_of_year'
    ]

    thr = np.nanpercentile(train_df['target'].values, int(FLOOD_Q*100))
    print(f"Peak threshold (train {int(FLOOD_Q*100)}th pct) = {thr:.2f}")

    # Create sample weights.
    sw = np.where(train_df['target'] >= thr, 1.0 + PEAK_WEIGHT_ALPHA, 1.0)

    # --- 3. Prepare Tabular & Scaled Data ---
    X_train_full, y_train_full, features_used = prepare_tabular_Xy(train_df, GLOBAL_FEATURE_LIST)
    X_test_full,  y_test_full, _  = prepare_tabular_Xy(test_df, GLOBAL_FEATURE_LIST)
    
    FINAL_TABULAR_FEATURES = features_used
    print(f"Training on {len(FINAL_TABULAR_FEATURES)} features: {FINAL_TABULAR_FEATURES}")
    
    X_test_full = X_test_full.reindex(columns=FINAL_TABULAR_FEATURES, fill_value=0)

    scaler = StandardScaler()
    scaler.fit(X_train_full.fillna(0)) # Fit on all training data
    
    X_train = pd.DataFrame(scaler.transform(X_train_full.fillna(0)), index=X_train_full.index, columns=FINAL_TABULAR_FEATURES)
    X_test  = pd.DataFrame(scaler.transform(X_test_full.fillna(0)),  index=X_test_full.index,  columns=FINAL_TABULAR_FEATURES)
    
    # We will log-transform the target for stability
    y_train_fit = np.log1p(y_train_full.clip(lower=0))

    # --------------- STAGE 1: Train Models Directly ---------------
    models_to_train = {}
    
    print(f"Training models directly with sample_weight (alpha={PEAK_WEIGHT_ALPHA})...")
    
    # --- Model 1: RandomForest ---
    print("Training RandomForest...")
    rf_model = RandomForestRegressor(
        n_estimators=400 if not quick else 100,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train_fit, sample_weight=sw)
    models_to_train['rf'] = rf_model
    
    # --- Model 2: GradientBoosting ---
    print("Training GradientBoosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200 if not quick else 100,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    )
    gb_model.fit(X_train, y_train_fit, sample_weight=sw)
    models_to_train['gb'] = gb_model

    # --- Model 3: LightGBM ---
    if LGB_AVAILABLE:
        print("Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=(260 if quick else 520),
            learning_rate=0.05,
            num_leaves=48,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
        with redirect_lgb_stderr():
            lgb_model.fit(X_train, y_train_fit, sample_weight=sw)
        models_to_train['lgb'] = lgb_model
        
    # --- Model 4: CatBoost ---
    if CAT_AVAILABLE:
        print("Training CatBoost...")
        cb_model = CatBoostRegressor(
            iterations=400 if not quick else 150,
            learning_rate=0.05,
            depth=6,
            random_state=RANDOM_STATE,
            verbose=0,
            thread_count=-1
        )
        cb_model.fit(X_train, y_train_fit, sample_weight=sw)
        models_to_train['cat'] = cb_model
    
    print("Direct training complete.")

    # --------------- STAGE 2: Evaluate & Select Best Model ---------------
    
    y_test_vals = y_test_full.values
    test_preds = {}
    base_metrics = {}
    
    for name, model in models_to_train.items():
        # Predict on log scale
        pred_log = model.predict(X_test)
        # Inverse transform to original scale
        pred_orig = np.expm1(pred_log)
        
        test_preds[name] = pred_orig
        base_metrics[name] = metrics_with_peak(y_test_vals, pred_orig, thr)
        
    print("\n--- Base Model Metrics (Test Set) ---")
    metrics_df = pd.DataFrame(base_metrics).T.sort_values(by='r2_peak', ascending=False)
    print(metrics_df)

    # --- Select the best model based on R2_PEAK ---
    best_model_name = metrics_df.index[0]
    best_model = models_to_train[best_model_name]
    final_metrics = base_metrics[best_model_name]
    final_test_pred = test_preds[best_model_name]
    
    print(f"\n--- Champion Model Selected: {best_model_name.upper()} ---")
    print(pd.Series(final_metrics).to_frame('Score'))

    # --------------- SHAP (on Champion Model) ---------------
    if SHAP_AVAILABLE and best_model_name in ['lgb', 'cat', 'gb', 'rf']:
        try:
            print(f"\nComputing SHAP values for {best_model_name.upper()}...")
            
            # Use TreeExplainer for tree models
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer(X_test)
            
            plt.figure(figsize=(10,8))
            shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")
            plt.tight_layout(); plt.savefig(PLOT_SHAP); plt.close()
            print(f"Saved SHAP summary: {PLOT_SHAP}")
        except Exception as e:
            print(f"SHAP failed: {e}")

    # --------------- Save artifacts ---------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        'model': best_model,
        'model_name': best_model_name,
        'scaler': scaler,
        'features_tab': FINAL_TABULAR_FEATURES,
        'trained_range': {'start': start, 'end': end},
        'peak_threshold': thr,
        'lead_days': lead_days
    }
    joblib.dump(bundle, os.path.join(MODEL_DIR, MODEL_FILE))
    with open(os.path.join(MODEL_DIR, FEATURES_JSON), 'w') as f:
        json.dump({'features': FINAL_TABULAR_FEATURES, 'lead_days': lead_days}, f, indent=2)
    print(f"Saved model bundle to {os.path.join(MODEL_DIR, MODEL_FILE)}")

    # predictions CSV
    out_df = pd.DataFrame({'date': test_df.index, 'observed': y_test_vals})
    for name, pred in test_preds.items():
        out_df[f'{name}_pred'] = pred
    out_df['final_pred'] = final_test_pred
    out_df.to_csv(PRED_CSV, index=False)
    print(f"Saved predictions to {PRED_CSV}")

    # plots
    idx = test_df.index
    plt.figure(figsize=(14,6))
    plt.plot(idx, y_test_vals, label='Observed', linewidth=1.2)
    plt.plot(idx, final_test_pred, '--', label=f'Final Pred ({best_model_name.upper()})', linewidth=1.4)
    plt.axhline(thr, color='orange', linestyle=':', label=f'{int(FLOOD_Q*100)}th pct ({thr:.2f})')
    plt.legend(); plt.grid(True); plt.title('v21 Backtest — Observed vs Final Prediction')
    plt.tight_layout(); plt.savefig(PLOT_BACKTEST); plt.close()
    print(f"Saved backtest plot to {PLOT_BACKTEST}")

    plot_obs_vs_pred(y_test_vals, final_test_pred)

    thr_mask = y_test_vals >= thr
    if thr_mask.sum() > 0:
        plt.figure(figsize=(12,5))
        plt.plot(idx[thr_mask], y_test_vals[thr_mask], 'o-', label='Observed peaks')
        plt.plot(idx[thr_mask], final_test_pred[thr_mask], 'x--', label='Predicted peaks (final)')
        plt.legend(); plt.grid(True); plt.title('v21 Peaks')
        plt.tight_layout(); plt.savefig(PLOT_PEAKS); plt.close()
        print(f"Saved peaks plot to {PLOT_PEAKS}")

    return {'final_metrics': final_metrics, 'champion_model': best_model_name}

# -----------------------
# CLI
# -----------------------
def parse_args():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train'], default='train')
    p.add_argument('--lat', type=float, default=LAT)
    # --- FIX: Corrected typo 'p.add.add_argument' ---
    p.add_argument('--lon', type=float, default=LON)
    p.add_argument('--start', type=str, default=DEFAULT_START)
    p.add_argument('--end', type=str, default=DEFAULT_END)
    p.add_argument('--lead', type=int, default=1)
    p.add_argument('--quick', action='store_true', help='smaller models / fewer epochs for quick iteration')
    ARGS = p.parse_args()
    return ARGS

if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    res = train_v21(args.lat, args.lon, args.start, args.end, lead_days=args.lead, quick=args.quick)
    print("✅ Training v21 complete:", res)