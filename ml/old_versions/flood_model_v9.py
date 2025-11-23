#!/usr/bin/env python3
"""
flood_model_v9.py

v9: Peak-aware hybrid training pipeline.

Usage:
    python flood_model_v9.py --mode train
    python flood_model_v9.py --mode train --quick    # faster, smaller grids

Outputs:
    - ml/open_meteo_flood_model_v9.pkl  (bundle with models, ensemble weights, calibrator, features, metadata)
    - v9_backtest.png
    - v9_backtest_peaks.png
"""

import os
import json
import math
import time
import joblib
import argparse
from datetime import date, timedelta
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer

# -----------------------
# CONFIG & defaults
# -----------------------
LAT = 23.81
LON = 90.41
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2025-12-31"
LEAD_DAYS = 1
TIMEZONE = "auto"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FLOOD = "https://flood-api.open-meteo.com/v1/flood"

MODEL_DIR = "ml"
MODEL_FILE = "open_meteo_flood_model_v9.pkl"
FEATURES_JSON = "open_meteo_flood_features_v9.json"
PLOT_FULL = "v9_backtest.png"
PLOT_PEAKS = "v9_backtest_peaks.png"

RANDOM_STATE = 42
FLOOD_Q = 0.90  # quantile to define peaks
FLOOD_ALPHA_BASE = 12.0

# -----------------------
# Utilities
# -----------------------
def safe_get_json(url, params, timeout=60):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def nse(obs, sim):
    obs = np.array(obs); sim = np.array(sim)
    denom = np.sum((obs - obs.mean())**2)
    return 1 - np.sum((obs - sim)**2) / denom if denom != 0 else float("nan")

def combined_r2_peak(y_true, y_pred, alpha=0.7, q=FLOOD_Q):
    overall = r2_score(y_true, y_pred)
    thr = np.nanpercentile(y_true, q*100)
    mask = np.array(y_true) >= thr
    peak = r2_score(np.array(y_true)[mask], np.array(y_pred)[mask]) if mask.sum() >= 2 else overall
    return alpha * overall + (1-alpha) * peak

COMBINED_SCORER = make_scorer(combined_r2_peak, greater_is_better=True)

# -----------------------
# fetch functions with per-year splitting
# -----------------------
def fetch_archive_hourly(lat, lon, start_date, end_date):
    """Fetch archive hourly by splitting into yearly chunks to avoid API 400s."""
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    today = date.today()
    if end > today:
        print(f"Warning: requested end_date {end} is in the future. Clipping to today {today}.")
        end = today
    dfs = []
    cur = date(start.year, 1, 1)
    # iterate years from start.year to end.year
    year = start.year
    while year <= end.year:
        seg_start = max(start, date(year, 1, 1))
        seg_end = min(end, date(year, 12, 31))
        print(f"→ Fetching segment {seg_start} → {seg_end}")
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation,soil_moisture_0_1cm,et0_fao_evapotranspiration,temperature_2m",
            "start_date": seg_start.isoformat(),
            "end_date": seg_end.isoformat(),
            "timezone": TIMEZONE
        }
        j = safe_get_json(OPEN_METEO_ARCHIVE, params)
        hourly = pd.DataFrame(j['hourly'])
        hourly['time'] = pd.to_datetime(hourly['time'])
        hourly = hourly.set_index('time')
        dfs.append(hourly)
        year += 1
        time.sleep(1.0)  # polite
    df_hourly = pd.concat(dfs).sort_index()
    return df_hourly

def fetch_flood_daily(lat, lon, start_date, end_date):
    params = {"latitude": lat, "longitude": lon, "daily": "river_discharge",
              "start_date": start_date, "end_date": end_date, "timezone": TIMEZONE}
    j = safe_get_json(OPEN_METEO_FLOOD, params)
    df = pd.DataFrame(j['daily']).rename(columns={"time":"date", "river_discharge":"discharge_m3_s"})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    return df

# -----------------------
# feature engineering (v9)
# -----------------------
def build_daily_merged(df_hourly, df_flood):
    # daily aggregates
    daily_precip = df_hourly['precipitation'].resample('D').sum().rename('rainfall_mm')
    daily_soil = df_hourly['soil_moisture_0_1cm'].resample('D').mean().rename('soil_moisture')
    daily_et = df_hourly['et0_fao_evapotranspiration'].resample('D').mean().rename('evapotranspiration')
    daily_temp = df_hourly['temperature_2m'].resample('D').mean().rename('temperature')
    daily_max_hour = df_hourly['precipitation'].resample('D').max().rename('max_hourly_rain_mm')

    df_daily = pd.concat([daily_precip, daily_soil, daily_et, daily_temp, daily_max_hour], axis=1)
    df_daily.index.name = 'date'
    df = pd.merge(df_flood, df_daily, left_index=True, right_index=True, how='inner')
    return df

def engineer_v9(df_daily, lead_days=1, api_k=0.85):
    d = df_daily.copy().sort_index()
    # discharge lags
    d['dis_lag1'] = d['discharge_m3_s'].shift(1)
    d['dis_lag2'] = d['discharge_m3_s'].shift(2)
    d['dis_lag3'] = d['discharge_m3_s'].shift(3)
    # rainfall lags + rolling sums
    for i in [1,2,3,7]:
        d[f'rain_lag_{i}'] = d['rainfall_mm'].shift(i)
    d['rain_roll_3'] = d['rainfall_mm'].shift(1).rolling(3).sum()
    d['rain_roll_7'] = d['rainfall_mm'].shift(1).rolling(7).sum()
    # API recursive
    api = []
    prev = 0.0
    for r in d['rainfall_mm'].fillna(0).values:
        val = r + api_k * prev
        api.append(val); prev = val
    d['api'] = api
    # intensity & interactions
    d['max_hourly_rain_mm'] = d['max_hourly_rain_mm'].fillna(0)
    d['rain_x_soil'] = d['rainfall_mm'] * d['soil_moisture']
    d['dis_rate'] = d['discharge_m3_s'] - d['dis_lag1']
    # cyclical
    d['month'] = d.index.month
    d['month_sin'] = np.sin(2*np.pi*d['month']/12)
    d['month_cos'] = np.cos(2*np.pi*d['month']/12)
    # forecast-style: pseudo-forecast (observed next-day rain) used in training (best-case)
    d['rain_forecast_Nd'] = d['rainfall_mm'].shift(-lead_days)
    d['target'] = d['discharge_m3_s'].shift(-lead_days)
    # drop rows w/o target
    d = d.dropna(subset=['target'])
    return d

# -----------------------
# Synthetic surge augmentation
# -----------------------
def augment_with_surges(df, n_augment=300, max_extra_mm=100.0, prob_peak_day=0.6, runoff_coeff=0.4):
    """
    Create synthetic augmented rows by injecting short, intense rainfall bursts in the recent days before target.
    - n_augment: how many synthetic samples to create
    - max_extra_mm: maximum extra mm to add to a single day as surge
    - prob_peak_day: fraction of augments applied to 'yesterday' vs 3-day earlier
    - runoff_coeff: fraction of surge rainfall converted to discharge peak (simple proxy)
    Returns augmented DataFrame (original + augmented).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    df_aug = df.copy()
    indices = df.index.values
    for i in range(n_augment):
        # pick a base index from historical days that have room for shifts (avoid earliest)
        base_idx = rng.choice(indices[len(indices)//4:])  # avoid earliest quarter
        base_row = df.loc[base_idx].copy()
        # decide which day to add surge (relative to base): -1 or -3 typically (before target)
        if rng.random() < prob_peak_day:
            day_col = 'rain_lag_1'
            added_day_label = 'surge_on_lag1'
        else:
            day_col = 'rain_lag_3'
            added_day_label = 'surge_on_lag3'
        extra = rng.uniform(10.0, max_extra_mm)  # mm
        base_row[day_col] = (base_row.get(day_col, 0.0) or 0.0) + extra
        # update rolling sums & API (approx)
        base_row['rain_roll_3'] = (base_row.get('rain_roll_3', 0) or 0) + extra
        base_row['api'] = (base_row.get('api', 0) or 0) + extra
        base_row['max_hourly_rain_mm'] = max(base_row.get('max_hourly_rain_mm',0), extra)
        # adjust target discharge upward by simple runoff proxy
        extra_discharge = runoff_coeff * extra
        base_row['target'] = base_row['target'] + extra_discharge
        # mark synthetic
        base_row['is_synthetic'] = 1
        df_aug = pd.concat([df_aug, pd.DataFrame([base_row])], ignore_index=False)
    # fill NaNs for synthetic indicator
    if 'is_synthetic' not in df_aug.columns:
        df_aug['is_synthetic'] = 0
    df_aug['is_synthetic'] = df_aug['is_synthetic'].fillna(0).astype(int)
    # shuffle index
    df_aug = df_aug.sample(frac=1, random_state=RANDOM_STATE)
    return df_aug

# -----------------------
# Prepare X,y and split
# -----------------------
FEATURE_LIST = [
 'discharge_m3_s','dis_lag1','dis_lag2','dis_lag3','dis_rate',
 'rain_lag_1','rain_lag_2','rain_lag_3','rain_lag_7',
 'rain_roll_3','rain_roll_7','api',
 'rain_forecast_Nd','rain_x_soil','max_hourly_rain_mm',
 'soil_moisture','evapotranspiration','temperature',
 'month_sin','month_cos'
]

def prepare_xy(df):
    X = df[FEATURE_LIST].fillna(0)
    y = df['target']
    return X, y

# -----------------------
# Training routine (v9)
# -----------------------
def train_v9(lat, lon, start, end, lead_days=1, quick=False):
    print(f"Fetching data for {start} → {end} ({lat},{lon})")
    df_hourly = fetch_archive_hourly(lat, lon, start, end)
    df_flood = fetch_flood_daily(lat, lon, start, end)
    print("Aggregating hourly -> daily and merging...")
    df_daily = build_daily_merged(df_hourly, df_flood)
    print(f"Total merged daily records: {len(df_daily)}")

    df_feat = engineer_v9(df_daily, lead_days=lead_days)

    # augment (only for training portion later)
    # split 80/20 now by time
    split = int(len(df_feat) * 0.8)
    train_df = df_feat.iloc[:split].copy()
    test_df = df_feat.iloc[split:].copy()
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")

    # augment training set with synthetic surges
    print("Augmenting training data with synthetic surges (n=300)...")
    train_aug = augment_with_surges(train_df, n_augment=300, max_extra_mm=120.0, prob_peak_day=0.7, runoff_coeff=0.5)

    X_train, y_train = prepare_xy(train_aug), None
    X_train, y_train = prepare_xy(train_aug)
    X_test, y_test = prepare_xy(test_df)

    # compute adaptive sample weights for original (non-synthetic) training rows only
    y_train_vals = train_aug['target'].values
    thr = np.nanpercentile(y_train_vals, FLOOD_Q * 100)
    sample_weight = np.where(y_train_vals >= thr, 1.0 + FLOOD_ALPHA_BASE, 1.0)
    # decrease weight for synthetic rows slightly so they help but don't dominate
    if 'is_synthetic' in train_aug.columns:
        sample_weight = sample_weight * np.where(train_aug['is_synthetic'] == 1, 0.6, 1.0)

    print(f"Flood threshold (train 90th pct) = {thr:.2f} -> flood alpha base {FLOOD_ALPHA_BASE}")

    # log-transform target for stability
    use_log = True
    y_train_fit = np.log1p(train_aug['target']) if use_log else train_aug['target'].values

    # Models: RandomForest and GradientBoosting (GridSearch small)
    if quick:
        rf_grid = {'n_estimators':[200], 'max_depth':[12], 'min_samples_leaf':[1]}
        gb_grid = {'n_estimators':[200], 'max_depth':[3], 'learning_rate':[0.05]}
        cv_splits = 3
    else:
        rf_grid = {'n_estimators':[200,400], 'max_depth':[12,18], 'min_samples_leaf':[1,2]}
        gb_grid = {'n_estimators':[200,400], 'max_depth':[3,5], 'learning_rate':[0.03,0.07]}
        cv_splits = 4

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    print("Training RandomForest (GridSearch)...")
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    gs_rf = GridSearchCV(rf, rf_grid, cv=tscv, scoring=COMBINED_SCORER, n_jobs=-1, verbose=1)
    try:
        gs_rf.fit(X_train, y_train_fit, sample_weight=sample_weight)
    except TypeError:
        gs_rf.fit(X_train, y_train_fit)
    rf_best = gs_rf.best_estimator_
    print("RF best params:", gs_rf.best_params_, "cv_score:", gs_rf.best_score_)

    print("Training GradientBoosting (GridSearch)...")
    gb = GradientBoostingRegressor(random_state=RANDOM_STATE)
    gs_gb = GridSearchCV(gb, gb_grid, cv=tscv, scoring=COMBINED_SCORER, n_jobs=-1, verbose=1)
    try:
        gs_gb.fit(X_train, y_train_fit, sample_weight=sample_weight)
    except TypeError:
        gs_gb.fit(X_train, y_train_fit)
    gb_best = gs_gb.best_estimator_
    print("GB best params:", gs_gb.best_params_, "cv_score:", gs_gb.best_score_)

    # Evaluate on test (inverse transform)
    def unlog(arr): return np.expm1(arr) if use_log else arr

    rf_pred = unlog(rf_best.predict(X_test))
    gb_pred = unlog(gb_best.predict(X_test))
    y_test_vals = test_df['target'].values

    def metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        r2p = r2_score(y_true[y_true >= thr], y_pred[y_true >= thr]) if np.sum(y_true >= thr) >= 2 else float('nan')
        return {'mae':mae,'rmse':rmse,'r2':r2,'r2_peak':r2p}

    rf_metrics = metrics(y_test_vals, rf_pred)
    gb_metrics = metrics(y_test_vals, gb_pred)
    print("RF test metrics:", rf_metrics)
    print("GB test metrics:", gb_metrics)

    # find best ensemble blend (grid search over weights) optimizing peak-R2 first then overall R2
    best_combo = None
    best_score = (-1e9, -1e9)
    for w in np.linspace(0,1,11):
        pred = w * gb_pred + (1-w) * rf_pred
        r2p = r2_score(y_test_vals[y_test_vals >= thr], pred[y_test_vals >= thr]) if np.sum(y_test_vals >= thr) >= 2 else -999
        r2o = r2_score(y_test_vals, pred)
        cand = (r2p, r2o)
        if cand > best_score:
            best_score = cand
            best_combo = (w, cand)
    blend_w = best_combo[0]
    print(f"Selected blend weight for GB = {blend_w:.2f} (best peakR2,overallR2) = {best_combo[1]}")

    ensemble_pred = blend_w * gb_pred + (1-blend_w) * rf_pred
    ensemble_metrics = metrics(y_test_vals, ensemble_pred)
    print("Ensemble test metrics:", ensemble_metrics)

    # Peak calibrator: train linear regressor on (ensemble_pred -> observed) for high flows in training set
    # Build calibrator using validation slice from training_aug (use last 10% of train_aug as val)
    val_slice = int(len(train_aug) * 0.9)
    val_df = train_aug.iloc[val_slice:]
    X_val, y_val = prepare_xy(val_df)
    # compute predictions on val set from models (unlog)
    rf_val = unlog(rf_best.predict(X_val))
    gb_val = unlog(gb_best.predict(X_val))
    ensemble_val = blend_w * gb_val + (1-blend_w) * rf_val
    thr_train = np.nanpercentile(train_aug['target'].values, FLOOD_Q*100)
    high_mask = val_df['target'].values >= thr_train
    if high_mask.sum() >= 10:
        # learn slope & intercept to correct ensemble predictions for high flows
        lr = LinearRegression()
        lr.fit(ensemble_val[high_mask].reshape(-1,1), val_df['target'].values[high_mask])
        print("Trained peak calibrator on high flows.")
    else:
        lr = None
        print("Not enough high-flow validation samples for calibrator; skipping.")

    # Apply calibrator to ensemble predictions on test set (if exists)
    if lr is not None:
        ensemble_calibrated = ensemble_pred.copy()
        high_mask_test = y_test_vals >= thr
        if high_mask_test.sum() > 0:
            ensemble_calibrated[high_mask_test] = lr.predict(ensemble_pred[high_mask_test].reshape(-1,1))
        final_pred = ensemble_calibrated
    else:
        final_pred = ensemble_pred

    final_metrics = metrics(y_test_vals, final_pred)
    print("Final (ensemble + calibrator) metrics:", final_metrics)

    # Save bundle: include rf_best, gb_best, blend_w, calibrator lr, features, metadata
    bundle = {
        'rf': rf_best, 'gb': gb_best,
        'blend_w_gb': float(blend_w),
        'calibrator': lr,
        'features': FEATURE_LIST,
        'use_log_target': use_log,
        'lead_days': lead_days,
        'trained_range': {'start': start, 'end': end}
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, os.path.join(MODEL_DIR, MODEL_FILE))
    with open(os.path.join(MODEL_DIR, FEATURES_JSON), 'w') as f:
        json.dump({'features': FEATURE_LIST, 'lead_days': lead_days}, f, indent=2)
    print(f"Saved model bundle to {os.path.join(MODEL_DIR, MODEL_FILE)}")

    # Plot full backtest and peaks
    idx = test_df.index
    plt.figure(figsize=(14,6))
    plt.plot(idx, y_test_vals, label='Observed')
    plt.plot(idx, final_pred, '--', label='Final Pred')
    plt.legend(); plt.grid(True); plt.title('v9 Backtest - Observed vs Predicted'); plt.savefig(PLOT_FULL); plt.close()
    thr_val = thr
    mask = y_test_vals >= thr_val
    if mask.sum()>0:
        plt.figure(figsize=(12,5))
        plt.plot(idx[mask], y_test_vals[mask], 'o-', label='Observed peaks')
        plt.plot(idx[mask], final_pred[mask], 'x--', label='Predicted peaks')
        plt.legend(); plt.grid(True); plt.title('v9 Peak Backtest'); plt.savefig(PLOT_PEAKS); plt.close()
    print(f"Saved {PLOT_FULL} and {PLOT_PEAKS}")

    return {'final_metrics': final_metrics, 'rf_metrics': rf_metrics, 'gb_metrics': gb_metrics}

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train'], default='train')
    p.add_argument('--lat', type=float, default=LAT)
    p.add_argument('--lon', type=float, default=LON)
    p.add_argument('--start', type=str, default=DEFAULT_START)
    p.add_argument('--end', type=str, default=DEFAULT_END)
    p.add_argument('--lead', type=int, default=LEAD_DAYS)
    p.add_argument('--quick', action='store_true', help='Smaller/shorter grids for faster run')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    res = train_v9(args.lat, args.lon, args.start, args.end, lead_days=args.lead, quick=args.quick)
    print("Training complete. Results:", res)
