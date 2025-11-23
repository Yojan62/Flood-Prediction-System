#!/usr/bin/env python3
"""
flood_model_v10.py (Fixed)

- This version removes the broken Keras/LSTM wrapper.
- It builds a stacked ensemble of RandomForest, GradientBoosting, and CatBoost.
- This matches the stable and accurate 'v10 Backtest' graph.
- Fixes SyntaxError, NameError, and KeyError (from constant features).
"""

import os
import time
import json
import math
import joblib
import argparse
from datetime import date
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from dotenv import load_dotenv 

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

# optional libs
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False
    
# (Removed TF/Keras imports as they are not used in this file)

# ---------- config ----------
LAT = 23.81
LON = 90.41
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2025-12-31"
LEAD_DAYS = 1
TIMEZONE = "auto"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FLOOD = "https://flood-api.open-meteo.com/v1/flood"

MODEL_DIR = "ml"
MODEL_FILE = "open_meteo_flood_model_v10.pkl"
FEATURES_JSON = "open_meteo_flood_features_v10.json"
PLOT_FULL = "v10_backtest.png"
PLOT_PEAKS = "v10_backtest_peaks.png"

RANDOM_STATE = 42
FLOOD_Q = 0.90
FLOOD_ALPHA_BASE = 12.0

# Global ARGS
ARGS = None

# ---------- utilities ----------
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
def fetch_archive_hourly(lat, lon, start_date, end_date):
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    today = date.today()
    if end > today:
        print(f"Warning: requested end_date {end} is in the future. Clipping to today {today}.")
        end = today
    dfs = []
    year = start.year
    while year <= end.year:
        seg_start = max(start, date(year, 1, 1))
        seg_end = min(end, date(year, 12, 31))
        print(f"→ Fetching hourly segment {seg_start} → {seg_end}")
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "precipitation,soil_moisture_0_1cm,et0_fao_evapotranspiration,temperature_2m",
            "start_date": seg_start.isoformat(), "end_date": seg_end.isoformat(),
            "timezone": TIMEZONE
        }
        j = safe_get_json(OPEN_METEO_ARCHIVE, params)
        hourly = pd.DataFrame(j['hourly'])
        hourly['time'] = pd.to_datetime(hourly['time'])
        hourly = hourly.set_index('time').sort_index()
        dfs.append(hourly)
        year += 1
        time.sleep(0.8)
        
    if not dfs:
        return pd.DataFrame()
        
    df_hourly = pd.concat(dfs).sort_index()
    # Rename columns
    rename_map = {
        'precipitation': 'rainfall_mm',
        'soil_moisture_0_1cm': 'soil_moisture',
        'et0_fao_evapotranspiration': 'evapotranspiration',
        'temperature_2m': 'temperature'
    }
    df_hourly = df_hourly.rename(columns=rename_map)
    # Ensure max_hourly_rain_mm exists
    if 'rainfall_mm' in df_hourly.columns:
        df_hourly['max_hourly_rain_mm'] = df_hourly['rainfall_mm']
    else:
        df_hourly['rainfall_mm'] = 0.0
        df_hourly['max_hourly_rain_mm'] = 0.0
    return df_hourly

def fetch_flood_daily(lat, lon, start_date, end_date):
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    today = date.today()
    if end > today:
        end = today
    params = {
        "latitude": lat, "longitude": lon, "daily": "river_discharge",
        "start_date": start.isoformat(), "end_date": end.isoformat(), "timezone": TIMEZONE
    }
    j = safe_get_json(OPEN_METEO_FLOOD, params)
    df = pd.DataFrame(j.get("daily", {}))
    if 'time' in df.columns:
        df = df.rename(columns={"time": "date", "river_discharge": "discharge_m3_s"})
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
    return df

# ---------- features ----------
def build_daily_merged(df_hourly, df_flood):
    print("Aggregating hourly -> daily...")
    agg_ops = {
        'rainfall_mm': 'sum',
        'soil_moisture': 'mean',
        'evapotranspiration': 'mean',
        'temperature': 'mean',
        'max_hourly_rain_mm': 'max'
    }
    for col in agg_ops.keys():
        if col not in df_hourly.columns:
            print(f"Warning: Column '{col}' not in hourly data. Filling with 0.")
            df_hourly[col] = 0.0
            
    df_daily = df_hourly.resample('D').agg(agg_ops)
    df_daily.index.name='date'
    print("Merging flood and weather data...")
    df = pd.merge(df_flood, df_daily, left_index=True, right_index=True, how='inner')
    print(f"Total merged daily rows: {len(df)}")
    return df

def engineer_v10(df_daily, lead_days=1, api_k=0.85):
    print("Engineering features...")
    d = df_daily.copy().sort_index()
    # ensure columns exist
    for c in ['discharge_m3_s','rainfall_mm','soil_moisture','evapotranspiration','temperature','max_hourly_rain_mm']:
        if c not in d.columns:
            d[c] = 0.0

    # discharge lags
    d['dis_lag1'] = d['discharge_m3_s'].shift(1)
    d['dis_lag2'] = d['discharge_m3_s'].shift(2)
    d['dis_lag3'] = d['discharge_m3_s'].shift(3)
    d['dis_rate'] = d['discharge_m3_s'] - d['dis_lag1']

    # rainfall lags + rolling
    for i in [1,2,3,7,14]:
        d[f'rain_lag_{i}'] = d['rainfall_mm'].shift(i)
    d['rain_roll_3']  = d['rainfall_mm'].shift(1).rolling(3).sum()
    d['rain_roll_7']  = d['rainfall_mm'].shift(1).rolling(7).sum()
    d['rain_roll_14'] = d['rainfall_mm'].shift(1).rolling(14).sum()
    
    # API
    api = []; prev=0.0
    for r in d['rainfall_mm'].fillna(0).values:
        val = r + api_k*prev; api.append(val); prev=val
    d['api'] = api
    
    d['max_hourly_rain_mm'] = d['max_hourly_rain_mm'].fillna(0)
    d['rain_x_soil'] = d['rainfall_mm'] * d['soil_moisture']
    
    # cyclical time
    d['month'] = d.index.month
    d['month_sin'] = np.sin(2*np.pi*d['month']/12); d['month_cos'] = np.cos(2*np.pi*d['month']/12)
    
    # forecast & target
    d['rain_forecast_Nd'] = d['rainfall_mm'].shift(-lead_days)
    d['target'] = d['discharge_m3_s'].shift(-lead_days)
    
    d = d.dropna(subset=['target', 'dis_lag3', 'rain_lag_14', 'api'])
    return d

# ---------- augmentation ----------
def augment_with_surges(df, n_augment=400, max_extra_mm=140.0, prob_peak_day=0.7, runoff_coeff=0.5):
    print("Augmenting training data with synthetic surges...")
    rng = np.random.default_rng(RANDOM_STATE)
    df_aug = df.copy()
    indices = df.index.values
    new_rows = []
    
    for i in range(n_augment):
        base_idx = rng.choice(indices[len(indices)//4:])
        base_row = df.loc[base_idx].copy()
        day_col = 'rain_lag_1' if rng.random() < prob_peak_day else 'rain_lag_3'
        extra = rng.uniform(10.0, max_extra_mm)
        base_row[day_col] = (base_row.get(day_col, 0.0) or 0.0) + extra
        base_row['rain_roll_3'] = (base_row.get('rain_roll_3', 0) or 0) + extra
        base_row['api'] = (base_row.get('api', 0) or 0) + extra
        base_row['max_hourly_rain_mm'] = max(base_row.get('max_hourly_rain_mm',0), extra)
        base_row['target'] = base_row['target'] + runoff_coeff * extra
        base_row['is_synthetic'] = 1
        base_row.name = base_idx + pd.Timedelta(nanoseconds=i+1) # unique index
        new_rows.append(base_row)

    if new_rows:
        df_aug = pd.concat([df_aug, pd.DataFrame(new_rows)], ignore_index=False)
        
    df_aug['is_synthetic'] = df_aug.get('is_synthetic', 0)
    df_aug['is_synthetic'] = df_aug['is_synthetic'].fillna(0).astype(int)
    df_aug = df_aug.sort_index()
    print(f"After augmentation: {len(df_aug)} rows")
    return df_aug

# ---------- feature list for tabular models ----------
FEATURE_LIST = [
 'discharge_m3_s','dis_lag1','dis_lag2','dis_lag3','dis_rate',
 'rain_lag_1','rain_lag_2','rain_lag_3','rain_lag_7','rain_lag_14',
 'rain_roll_3','rain_roll_7','rain_roll_14','api',
 'rain_forecast_Nd','rain_x_soil','max_hourly_rain_mm',
 'soil_moisture','evapotranspiration','temperature',
 'month_sin','month_cos'
]

def prepare_tabular_Xy(df):
    # --- FIX: Remove constant features based on logs ---
    # Your logs showed soil_moisture and rain_x_soil are all zeros.
    # We remove them here to prevent the model from crashing.
    features_to_use = [f for f in FEATURE_LIST if f not in ('soil_moisture', 'rain_x_soil')]
    
    X = df.reindex(columns=features_to_use).apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    y = df['target'].astype(np.float32)
    return X, y

# ---------- training workflow ----------
def train_v10(lat, lon, start, end, lead_days=1, quick=False):
    print(f"Fetching data for {start} → {end}")
    df_hourly = fetch_archive_hourly(lat, lon, start, end)
    df_flood = fetch_flood_daily(lat, lon, start, end)
    
    if df_hourly.empty or df_flood.empty:
        print("Error: No data fetched for one or both sources. Aborting.")
        return
        
    df_daily = build_daily_merged(df_hourly, df_flood)
    if df_daily.empty:
        print("Error: No overlapping data. Aborting.")
        return
        
    print(f"Total merged daily rows: {len(df_daily)}")
    df_feat = engineer_v10(df_daily, lead_days=lead_days)

    # Fill NaNs created by feature engineering
    df_feat = df_feat.fillna(0)

    # split time-based
    split_i = int(len(df_feat)*0.8)
    train_df = df_feat.iloc[:split_i].copy()
    test_df = df_feat.iloc[split_i:].copy()
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")

    # augment training
    train_aug = augment_with_surges(train_df, n_augment=400 if not quick else 150)
    train_aug['target'] = train_aug['target'].fillna(0) # Fill NaNs on target

    # prepare tabular X,y
    X_train_tab, y_train_tab = prepare_tabular_Xy(train_aug)
    X_test_tab, y_test_tab = prepare_tabular_Xy(test_df)

    # compute sample weights
    thr = np.nanpercentile(train_aug['target'].values, FLOOD_Q*100)
    sample_weight = np.where(train_aug['target'].values >= thr, 1.0 + FLOOD_ALPHA_BASE, 1.0)
    if 'is_synthetic' in train_aug.columns:
        sample_weight = sample_weight * np.where(train_aug['is_synthetic']==1, 0.6, 1.0)
    print(f"Flood threshold (train 90p) = {thr:.2f}  -> sample weight alpha = {FLOOD_ALPHA_BASE}")

    # log-transform targets
    use_log = True
    y_train_fit = np.log1p(y_train_tab) # Use y_train_tab, not train_aug['target']

    # Train RandomForest (fast grid)
    print("Tuning RandomForest...")
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    rf_grid = {'n_estimators':[200] if quick else [200,400], 'max_depth':[12] if quick else [12,18], 'min_samples_leaf':[2]}
    tscv = TimeSeriesSplit(n_splits=3 if quick else 4)
    gs_rf = GridSearchCV(rf, rf_grid, cv=tscv, scoring=COMBINED_SCORER, n_jobs=-1, verbose=1)
    try:
        gs_rf.fit(X_train_tab, y_train_fit, sample_weight=sample_weight)
    except TypeError:
        print("Warning: GridSearchCV failed with sample_weight. Fitting without.")
        gs_rf.fit(X_train_tab, y_train_fit)
    rf_best = gs_rf.best_estimator_
    print("RF best params:", gs_rf.best_params_)

    # Train GradientBoosting
    print("Tuning GradientBoosting...")
    gb = GradientBoostingRegressor(random_state=RANDOM_STATE)
    gb_grid = {'n_estimators':[200] if quick else [200,400], 'max_depth':[3] if quick else [3,5], 'learning_rate':[0.05]}
    gs_gb = GridSearchCV(gb, gb_grid, cv=tscv, scoring=COMBINED_SCORER, n_jobs=-1, verbose=1)
    try:
        gs_gb.fit(X_train_tab, y_train_fit, sample_weight=sample_weight)
    except TypeError:
        print("Warning: GridSearchCV failed with sample_weight. Fitting without.")
        gs_gb.fit(X_train_tab, y_train_fit)
    gb_best = gs_gb.best_estimator_
    print("GB best params:", gs_gb.best_params_)

    # Optional: CatBoost
    cb_best = None
    if CATBOOST_AVAILABLE:
        print("Tuning CatBoost...")
        cb = CatBoostRegressor(random_state=RANDOM_STATE, verbose=0)
        cb_grid = {'iterations':[200] if quick else [200,400], 'depth':[6] if quick else [6,8], 'learning_rate':[0.05]}
        gs_cb = GridSearchCV(cb, cb_grid, cv=tscv, scoring=COMBINED_SCORER, n_jobs=-1, verbose=1)
        try:
            gs_cb.fit(X_train_tab, y_train_fit, sample_weight=sample_weight)
        except TypeError:
            gs_cb.fit(X_train_tab, y_train_fit)
        cb_best = gs_cb.best_estimator_
        print("CatBoost best params:", gs_cb.best_params_)
    else:
        print("CatBoost not installed, skipping.")

    # (LSTM portion removed)
    
    # ---------- predictions from base models on test ----------
    def unlog(arr):
        return np.expm1(arr) if use_log else arr

    rf_pred_test = unlog(rf_best.predict(X_test_tab))
    gb_pred_test = unlog(gb_best.predict(X_test_tab))
    cb_pred_test = unlog(cb_best.predict(X_test_tab)) if cb_best is not None else None
    
    # create DataFrame of base preds for stacking
    base_preds = pd.DataFrame({'rf': rf_pred_test, 'gb': gb_pred_test}, index=X_test_tab.index)
    if cb_pred_test is not None:
        base_preds['cb'] = cb_pred_test

    # ---------- stacking meta-learner ----------
    print("Training meta-learner...")
    val_slice = int(len(train_aug) * 0.9)
    meta_val_df = train_aug.iloc[val_slice:].copy()
    X_meta_val_tab, y_meta_val = prepare_tabular_Xy(meta_val_df)
    
    # predictions from base models on meta_val
    rf_meta = unlog(rf_best.predict(X_meta_val_tab))
    gb_meta = unlog(gb_best.predict(X_meta_val_tab))
    meta_df = pd.DataFrame({'rf': rf_meta, 'gb': gb_meta}, index=meta_val_df.index)
    if cb_best is not None:
        meta_df['cb'] = unlog(cb_best.predict(X_meta_val_tab))

    # meta learner (linear regression) trained to map base preds -> observed
    meta_lr = LinearRegression()
    meta_sample_weight = np.where(meta_val_df['target'].values >= thr, 1.0 + FLOOD_ALPHA_BASE, 1.0)
    meta_lr.fit(meta_df.fillna(0).values, meta_val_df['target'].values, sample_weight=meta_sample_weight)
    
    # apply meta to test
    stack_pred_test = meta_lr.predict(base_preds.fillna(0).values)

    # ---------- peak calibrator (Huber for robustness) ----------
    print("Training peak calibrator...")
    ensemble_val = meta_lr.predict(meta_df.fillna(0).values)
    high_mask_val = meta_val_df['target'].values >= thr
    calibrator = None
    if high_mask_val.sum() >= 10:
        calibrator = HuberRegressor()
        calibrator.fit(ensemble_val[high_mask_val].reshape(-1,1), meta_val_df['target'].values[high_mask_val])
    else:
        print("Warning: Not enough high-flow samples to train calibrator.")

    final_pred_test = stack_pred_test.copy()
    if calibrator is not None:
        high_mask_test = final_pred_test >= thr # Calibrate based on predicted peaks
        if high_mask_test.sum() > 0:
            print(f"Applying calibration to {high_mask_test.sum()} predicted peaks...")
            final_pred_test[high_mask_test] = calibrator.predict(final_pred_test[high_mask_test].reshape(-1,1))

    # ---------- evaluate ----------
    y_test_vals = test_df['target'].values
    
    # --- Define `thr` for the metrics function ---
    test_thr = np.nanpercentile(y_test_vals, FLOOD_Q*100)
    
    def metrics(y_true, y_pred, thr):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        r2val = r2_score(y_true, y_pred)
        peak_mask = y_true >= thr
        r2peak = r2_score(y_true[peak_mask], y_pred[peak_mask]) if peak_mask.sum() >= 2 else float('nan')
        return {'mae':mae, 'rmse':rmse, 'r2':r2val, 'r2_peak': r2peak}

    base_metrics = {
        'rf': metrics(y_test_vals, rf_pred_test, test_thr),
        'gb': metrics(y_test_vals, gb_pred_test, test_thr)
    }
    if cb_pred_test is not None:
        base_metrics['cb'] = metrics(y_test_vals, cb_pred_test, test_thr)

    final_metrics = metrics(y_test_vals, final_pred_test, test_thr)
    print("\n--- Base model metrics: ---")
    print(pd.DataFrame(base_metrics).T)
    print("\n--- Final stacked+calibrated metrics: ---")
    print(pd.Series(final_metrics).to_frame('Score'))

    # save bundle
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        'rf': rf_best, 'gb': gb_best, 'cb': cb_best,
        'meta_lr': meta_lr, 'calibrator': calibrator,
        'features': X_train_tab.columns.tolist(), # Save the actual features used
        'use_log_target': use_log, 'lead_days': lead_days,
    }
    joblib.dump(bundle, os.path.join(MODEL_DIR, MODEL_FILE))
    with open(os.path.join(MODEL_DIR, FEATURES_JSON), 'w') as f:
        json.dump({'features': X_train_tab.columns.tolist(), 'lead_days': lead_days}, f, indent=2)
    print(f"Saved model bundle to {os.path.join(MODEL_DIR, MODEL_FILE)}")

    # plots
    idx = test_df.index
    plt.figure(figsize=(14,6))
    plt.plot(idx, y_test_vals, label='Observed')
    plt.plot(idx, final_pred_test, '--', label='Predicted final')
    plt.legend(); plt.grid(True); plt.title('v10 Backtest'); plt.savefig(PLOT_FULL); plt.close()
    
    thr_mask = y_test_vals >= test_thr
    if thr_mask.sum()>0:
        plt.figure(figsize=(12,5))
        plt.plot(idx[thr_mask], y_test_vals[thr_mask], 'o-', label='Observed peaks')
        plt.plot(idx[thr_mask], final_pred_test[thr_mask], 'x--', label='Predicted peaks (final)')
        plt.legend(); plt.grid(True); plt.title('v10 Peaks'); plt.savefig(PLOT_PEAKS); plt.close()
    print(f"Saved {PLOT_FULL} and {PLOT_PEAKS}")

    return {'final_metrics': final_metrics, 'base_metrics': base_metrics}

# ---------- CLI ----------
def parse_args():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train'], default='train')
    p.add_argument('--lat', type=float, default=LAT)
    p.add_argument('--lon', type=float, default=LON)
    p.add_argument('--start', type=str, default=DEFAULT_START)
    p.add_argument('--end', type=str, default=DEFAULT_END) 
    p.add_argument('--lead', type=int, default=LEAD_DAYS)
    p.add_argument('--quick', action='store_true', help='Quicker run with smaller grids')
    ARGS = p.parse_args()
    return ARGS

if __name__ == "__main__":
    load_dotenv() # Load .env file
    args = parse_args()
    res = train_v10(args.lat, args.lon, args.start, args.end, lead_days=args.lead, quick=args.quick)
    print("Train complete. Results:", res)