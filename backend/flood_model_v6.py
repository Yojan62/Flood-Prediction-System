#!/usr/bin/env python3
"""
flood_model_v6.py

Train (and evaluate) a 1- to N-day-ahead river-discharge model for Dhaka using Open-Meteo data.

Features:
- Configurable lead_time_days (default 1 => predict next calendar day discharge)
- Collects flood daily + hourly weather archive, aggregates hourly->daily
- Feature engineering: lags, rolling stats, ARI (antecedent rainfall index),
  rainfall x soil interaction, discharge rate, cyclical month encodings
- Supports pseudo-forecast (perfect next-day observed rainfall) and archived forecasts
- Trains RandomForest / XGBoost / LightGBM (if available) with RandomizedSearchCV + TimeSeriesSplit
- Flood-weighted sample weighting and log1p target transform supported
- Outputs: model bundle (.pkl) with features & metadata, evaluation metrics, plots

Usage examples:
    python flood_model_v6.py --mode train
    python flood_model_v6.py --mode train --lead 2 --start 2018-01-01 --end 2024-12-31
"""

import argparse
import os
import json
import joblib
import math
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import randint, uniform
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# optional libs
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------
# CONFIG / defaults
# ---------------------------
OPEN_METEO_FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "auto"

DEFAULT_LAT = 23.81
DEFAULT_LON = 90.41
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_LEAD_DAYS = 1  # 1 day ahead (24h). Set to 2 for 48h, etc.

MODEL_OUTPUT_DIR = "ml"
MODEL_FILENAME = "open_meteo_flood_model_v6.pkl"
FEATURES_JSON = "open_meteo_flood_features_v6.json"
PLOT_FILENAME = "v6_backtest.png"

USE_LOG_TARGET_DEFAULT = True
EMPHASIZE_FLOODS_DEFAULT = True

# ---------------------------
# Helper functions & metrics
# ---------------------------
def nse(observed, simulated):
    """Nash-Sutcliffe Efficiency (higher is better, 1=perfect)"""
    obs = np.array(observed)
    sim = np.array(simulated)
    denom = np.sum((obs - obs.mean())**2)
    if denom == 0:
        return float("nan")
    return 1 - np.sum((obs - sim)**2) / denom

def safe_get_json(url, params, timeout=30):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ---------------------------
# Data fetching
# ---------------------------
def fetch_historical_data(lat, lon, start_date, end_date):
    """Fetch flood daily and hourly weather archive from Open-Meteo and return merged daily DataFrame."""
    # Flood daily
    flood_params = {
        "latitude": lat, "longitude": lon,
        "daily": "river_discharge",
        "start_date": start_date, "end_date": end_date,
        "timezone": TIMEZONE
    }
    print("Fetching flood daily data...")
    jf = safe_get_json(OPEN_METEO_FLOOD_URL, flood_params)
    if 'daily' not in jf:
        raise RuntimeError("Flood API returned unexpected JSON keys: {}".format(jf.keys()))
    df_flood = pd.DataFrame(jf['daily']).rename(columns={"time":"date", "river_discharge":"discharge_m3_s"})
    df_flood['date'] = pd.to_datetime(df_flood['date'])
    df_flood = df_flood.set_index('date').sort_index()

    # Archive hourly -> daily
    archive_params = {
        "latitude": lat, "longitude": lon,
        "hourly": ["precipitation", "soil_moisture_0_1cm", "et0_fao_evapotranspiration", "temperature_2m"],
        "start_date": start_date, "end_date": end_date, "timezone": TIMEZONE
    }
    print("Fetching hourly archive weather data (may take a moment)...")
    jw = safe_get_json(OPEN_METEO_ARCHIVE_URL, archive_params)
    if 'hourly' not in jw:
        raise RuntimeError("Archive API returned unexpected JSON keys: {}".format(jw.keys()))
    df_hour = pd.DataFrame(jw['hourly'])
    df_hour['time'] = pd.to_datetime(df_hour['time'])
    df_hour = df_hour.set_index('time').sort_index()

    # aggregate to daily
    df_daily = df_hour.resample('D').agg({
        'precipitation':'sum',
        'soil_moisture_0_1cm':'mean',
        'et0_fao_evapotranspiration':'mean',
        'temperature_2m':'mean'
    }).rename(columns={
        'precipitation':'rainfall_mm',
        'soil_moisture_0_1cm':'soil_moisture',
        'et0_fao_evapotranspiration':'evapotranspiration',
        'temperature_2m':'temperature'
    })
    df_daily.index.name='date'
    print(f"Archive aggregated to {len(df_daily)} daily rows.")

    # Merge inner join (only days with both)
    df = pd.merge(df_flood, df_daily, left_index=True, right_index=True, how='inner')
    print(f"Merged dataset has {len(df)} daily rows after inner join.")
    return df

# ---------------------------
# Feature engineering
# ---------------------------
def engineer_features(df, lead_days=1, archived_forecasts=None):
    """
    df: daily DataFrame (observed rainfall and discharge)
    lead_days: integer (1 => predict next day discharge, 2 => predict 2-days-ahead, etc.)
    archived_forecasts: optional dict mapping date -> forecasted rainfall for next calendar day (real forecast when available)
    Returns DataFrame with features and target
    """
    d = df.copy().sort_index()

    # Lags & rolling
    d['discharge_lag_1'] = d['discharge_m3_s'].shift(1)
    d['discharge_lag_2'] = d['discharge_m3_s'].shift(2)
    d['rainfall_lag_1'] = d['rainfall_mm'].shift(1)
    d['rainfall_3_day_sum'] = d['rainfall_mm'].shift(1).rolling(window=3).sum()
    d['rainfall_7_day_avg'] = d['rainfall_mm'].shift(1).rolling(window=7).mean()
    d['discharge_3_day_avg'] = d['discharge_m3_s'].shift(1).rolling(window=3).mean()
    d['discharge_rate_change'] = d['discharge_m3_s'] - d['discharge_lag_1']

    # ARI (antecedent rainfall index: exponential decay)
    decay = 0.7
    ari = np.zeros(len(d))
    for i in range(1, 8):
        ari += (decay ** (i-1)) * d['rainfall_mm'].shift(i).fillna(0).values
    d['ari_7'] = ari

    # interaction
    d['rain_x_soil'] = d['rainfall_mm'] * d['soil_moisture']

    # storm intensity / max hourly proxies are unavailable in daily archive; use 1-day max if you have hourly originally (not here)
    # month cyclical
    d['month'] = d.index.month
    d['month_sin'] = np.sin(2*np.pi*d['month']/12)
    d['month_cos'] = np.cos(2*np.pi*d['month']/12)

    # Prepare rainfall_forecast_Nh: training options
    # For training t we want feature = forecast issued at t for t+lead_days (i.e., forecast for the target day)
    if archived_forecasts is not None:
        # archived_forecasts: dict or pd.Series keyed by issue date (date) -> forecasted next-day rainfall for that issue date
        # Align: for date t, we want archived_forecasts[t] (forecast for t+lead_days)
        def lookup_forecast(issue_date):
            key = pd.Timestamp(issue_date).date()
            return archived_forecasts.get(key, np.nan)
        d['rainfall_forecast_Nd'] = d.index.to_series().apply(lambda dt_idx: lookup_forecast(dt_idx.date()))
    else:
        # pseudo-forecast: use observed rainfall on target day as "forecast" (optimistic upper bound)
        d['rainfall_forecast_Nd'] = d['rainfall_mm'].shift(-lead_days)

    # Target: observed discharge at t + lead_days (calendar day)
    d['target_discharge'] = d['discharge_m3_s'].shift(-lead_days)

    return d

# ---------------------------
# Prepare data for training
# ---------------------------
def prepare_xy(df_feat, feature_list):
    dfc = df_feat.copy()
    dfc = dfc.dropna(subset=['target_discharge'])
    X = dfc[feature_list].fillna(0)
    y = dfc['target_discharge']
    return X, y, dfc

# ---------------------------
# Model training (multiple algorithms)
# ---------------------------
def get_model_candidates():
    candidates = []
    # RandomForest candidate
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_param_dist = {
        'n_estimators': randint(100, 600),
        'max_depth': randint(6, 30),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', None]
    }
    candidates.append(('rf', rf, rf_param_dist))

    # XGBoost candidate (if available)
    if XGB_AVAILABLE:
        xg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        xg_param_dist = {
            'n_estimators': randint(100, 800),
            'max_depth': randint(3, 12),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.5, 0.5)
        }
        candidates.append(('xgb', xg, xg_param_dist))

    if LGB_AVAILABLE:
        lg = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
        lg_param_dist = {
            'n_estimators': randint(100, 800),
            'num_leaves': randint(16, 128),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4)
        }
        candidates.append(('lgb', lg, lg_param_dist))

    return candidates

def train_best_model(X_train, y_train, use_log=True, emphasize_floods=True, n_iter=40):
    y_train_for_fit = np.log1p(y_train) if use_log else y_train.values

    candidates = get_model_candidates()
    best_score = -np.inf
    best_model = None
    best_name = None
    best_cv_res = None

    tscv = TimeSeriesSplit(n_splits=4)

    for name, model_inst, param_dist in candidates:
        print(f"\n--- Tuning candidate: {name} ---")
        search = RandomizedSearchCV(model_inst, param_distributions=param_dist,
                                    n_iter=n_iter, cv=tscv, scoring='r2', n_jobs=-1, random_state=42, verbose=1)
        sample_weight = None
        if emphasize_floods:
            thr = np.nanpercentile(y_train.values, 90)
            sample_weight = np.where(y_train.values >= thr, 1.0 + 5.0, 1.0)
            print(f"  (emphasize floods: weighting >90p flows by +5)")

        # Fit (RandomizedSearchCV doesn't accept sample_weight directly; pass via fit_params)
        fit_kwargs = {'sample_weight': sample_weight} if sample_weight is not None else {}
        try:
            search.fit(X_train, y_train_for_fit, **fit_kwargs)
        except TypeError:
            # older sklearn or estimator doesn't accept fit_params - fallback to fit without sample_weight
            search.fit(X_train, y_train_for_fit)
        print(" Best params:", search.best_params_)
        print(" Best CV score (r2):", search.best_score_)
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = search.best_estimator_
            best_name = name
            best_cv_res = search.cv_results_

    print(f"\nSelected best model: {best_name} with CV r2 = {best_score:.4f}")
    return best_name, best_model, best_cv_res

# ---------------------------
# Evaluate model
# ---------------------------
def evaluate_on_test(model, X_test, y_test, use_log=True):
    if use_log:
        y_pred_tr = model.predict(X_test)
        y_pred = np.expm1(y_pred_tr).clip(min=0)
    else:
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    nse_val = nse(y_test, y_pred)
    thr = np.nanpercentile(y_test, 90)
    peak_mask = y_test >= thr
    r2_peak = r2_score(y_test[peak_mask], y_pred[peak_mask]) if peak_mask.sum() > 0 else float('nan')
    return {'mae':mae, 'rmse':rmse, 'r2':r2, 'nse':nse_val, 'r2_peak':r2_peak, 'y_pred':y_pred}

# ---------------------------
# Plotting
# ---------------------------
def plot_backtest(y_test, y_pred, filename=PLOT_FILENAME):
    plt.figure(figsize=(14,6))
    plt.plot(y_test.index, y_test.values, label='Observed', lw=1.5)
    plt.plot(y_test.index, y_pred, label='Predicted', lw=1, linestyle='--')
    plt.fill_between(y_test.index, y_test.values, y_pred, color='gray', alpha=0.15)
    plt.title('Backtest: Observed vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Discharge (m^3/s)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved backtest plot to {filename}")

# ---------------------------
# Main train pipeline
# ---------------------------
def main_train(lat, lon, start_date, end_date, lead_days, use_log_target, emphasize_floods, train_forecast_mode, archived_forecasts_csv):
    print("Fetching historical data...")
    df = fetch_historical_data(lat, lon, start_date, end_date)

    archived_forecasts = None
    if train_forecast_mode == 'archive_forecasts' and archived_forecasts_csv:
        print("Loading archived forecasts CSV for realistic forecast training...")
        af = pd.read_csv(archived_forecasts_csv, parse_dates=['date'])
        # expected columns: date (issue date), rainfall_forecast (forecasted rainfall for next day)
        archived_forecasts = {row['date'].date(): float(row['rainfall_forecast_24h']) for _, row in af.iterrows()}
        print(f"Loaded {len(archived_forecasts)} archived forecast records.")

    print("Engineering features...")
    df_feat = engineer_features(df, lead_days, archived_forecasts)

    feature_list = [
        'discharge_m3_s','discharge_lag_1','discharge_lag_2','discharge_rate_change','discharge_3_day_avg',
        'rainfall_lag_1','rainfall_3_day_sum','rainfall_7_day_avg','rainfall_forecast_Nd','ari_7','rain_x_soil',
        'soil_moisture','evapotranspiration','temperature','month_sin','month_cos'
    ]

    # ensure requested features exist (rainfall_3_day_sum may be NaN early in series)
    X, y, dfc = prepare_xy(df_feat, feature_list)

    if len(X) < 200:
        print("Warning: less than 200 rows after preprocessing. Model quality may be poor.")

    # time-based split 80/20
    split_i = int(len(X)*0.8)
    X_train, X_test = X.iloc[:split_i], X.iloc[split_i:]
    y_train, y_test = y.iloc[:split_i], y.iloc[split_i:]

    print(f"Training rows: {len(X_train)}  Test rows: {len(X_test)}")

    print("Training multiple candidate models (this may take time)...")
    name, model, cv_res = train_best_model(X_train, y_train, use_log=use_log_target, emphasize_floods=emphasize_floods, n_iter=40)

    print("Evaluating on held-out test set...")
    eval_res = evaluate_on_test(model, X_test, y_test, use_log=use_log_target)
    print("Test metrics:")
    print(f" MAE: {eval_res['mae']:.3f}  RMSE: {eval_res['rmse']:.3f}  R2: {eval_res['r2']:.3f}  NSE: {eval_res['nse']:.3f}  R2_peak(>90p): {eval_res['r2_peak']:.3f}")

    # Save model bundle
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    bundle = {
        'model': model,
        'features': feature_list,
        'use_log_target': use_log_target,
        'lead_days': lead_days,
        'trained_on': {'start':start_date, 'end':end_date},
        'model_name': name
    }
    out_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)
    joblib.dump(bundle, out_path)
    print(f"Saved model bundle to {out_path}")

    # save feature list separately (useful for inference script)
    feat_path = os.path.join(MODEL_OUTPUT_DIR, FEATURES_JSON)
    with open(feat_path, 'w') as f:
        json.dump({'features': feature_list, 'lead_days': lead_days}, f, indent=2)
    print(f"Saved feature metadata to {feat_path}")

    # save backtest plot
    plot_backtest(y_test, eval_res['y_pred'])

    return bundle, eval_res, dfc

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train'], default='train', help='Only training mode supported in this script (predicting handled by run_predictions.py).')
    p.add_argument('--lat', type=float, default=DEFAULT_LAT)
    p.add_argument('--lon', type=float, default=DEFAULT_LON)
    p.add_argument('--start', type=str, default=DEFAULT_START)
    p.add_argument('--end', type=str, default=DEFAULT_END)
    p.add_argument('--lead', type=int, default=DEFAULT_LEAD_DAYS, help='Lead time in days (1=24h)')
    p.add_argument('--use-log', action='store_true', default=USE_LOG_TARGET_DEFAULT)
    p.add_argument('--no-log', dest='use_log', action='store_false')
    p.add_argument('--no-emphasize', dest='emphasize', action='store_false')
    p.add_argument('--train-forecast-mode', choices=['pseudo_forecast','archive_forecasts'], default='pseudo_forecast')
    p.add_argument('--archived-forecast-csv', type=str, default='')
    return p.parse_args()

if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    bundle, metrics, df_used = main_train(
        lat=args.lat,
        lon=args.lon,
        start_date=args.start,
        end_date=args.end,
        lead_days=args.lead,
        use_log_target=args.use_log,
        emphasize_floods=args.emphasize,
        train_forecast_mode=args.train_forecast_mode,
        archived_forecasts_csv=args.archived_forecast_csv
    )
    print("Done.")
