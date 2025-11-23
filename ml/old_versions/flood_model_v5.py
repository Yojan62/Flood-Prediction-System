#!/usr/bin/env python3
"""
flood_model_v5.py

Refactored training pipeline for 1-day-ahead river discharge prediction
using Open-Meteo (flood + archive) inputs.

Features / improvements:
- Modular functions (fetch, preprocess, engineer, train, evaluate)
- Handles missing days and time alignment
- Adds Antecedent Rainfall Index (ARI), rainfall x soil interaction,
  discharge momentum, cyclical month encoding (sin/cos)
- Optional log1p target transform
- Sample weighting to emphasize high flows
- Time-aware GridSearchCV (simple expanding-window CV)
- Peak-focused evaluation metrics and diagnostic plots
- Saves model (joblib) and validation plot
"""

import requests
import pandas as pd
import numpy as np
import joblib
import sys
import os
from datetime import datetime, timedelta

# Model & evaluation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# -----------------------------
# Configuration (edit as needed)
# -----------------------------
LATITUDE = 23.81
LONGITUDE = 90.41
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
TIMEZONE = "auto"

MODEL_OUTPUT = "open_meteo_flood_model_v5.pkl"
PLOT_OUTPUT = "model_backtest_v5.png"
RANDOM_STATE = 42

# Toggle options
USE_LOG_TARGET = True   # Train on log1p(target) to stabilize variance
EMPHASIZE_FLOODS = True # Use sample weights to focus on high flows

# -----------------------------
# Utilities
# -----------------------------
def safe_request(url, params, timeout=30):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# -----------------------------
# 1) Data fetching
# -----------------------------
def fetch_open_meteo(lat, lon, start_date, end_date, timezone="auto"):
    """Fetch flood (daily discharge) and archive (hourly weather) from Open-Meteo."""
    # Flood endpoint
    flood_url = "https://flood-api.open-meteo.com/v1/flood"
    flood_params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "river_discharge",  # confirm via response keys if necessary
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone
    }
    print("Fetching flood data...")
    flood_json = safe_request(flood_url, flood_params)
    if "daily" not in flood_json:
        raise RuntimeError("Unexpected flood API response. Keys: {}".format(list(flood_json.keys())))
    df_flood = pd.DataFrame(flood_json['daily'])
    df_flood = df_flood.rename(columns={"time": "date", "river_discharge": "discharge_m3_s"})
    df_flood['date'] = pd.to_datetime(df_flood['date'])
    df_flood = df_flood.set_index('date').sort_index()
    print(f"Flood data rows: {len(df_flood)}")

    # Archive (hourly weather)
    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    archive_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["precipitation", "soil_moisture_0_1cm", "et0_fao_evapotranspiration", "temperature_2m"],
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone
    }
    print("Fetching hourly weather data...")
    weather_json = safe_request(archive_url, archive_params)
    if "hourly" not in weather_json:
        raise RuntimeError("Unexpected archive API response. Keys: {}".format(list(weather_json.keys())))
    df_weather_hourly = pd.DataFrame(weather_json['hourly'])
    df_weather_hourly['time'] = pd.to_datetime(df_weather_hourly['time'])
    df_weather_hourly = df_weather_hourly.set_index('time').sort_index()

    # Aggregate hourly -> daily
    df_weather_daily = df_weather_hourly.resample('D').agg({
        'precipitation': 'sum',
        'soil_moisture_0_1cm': 'mean',
        'et0_fao_evapotranspiration': 'mean',
        'temperature_2m': 'mean'
    }).rename(columns={
        'precipitation': 'rainfall_mm',
        'soil_moisture_0_1cm': 'soil_moisture',
        'et0_fao_evapotranspiration': 'evapotranspiration',
        'temperature_2m': 'temperature'
    })
    df_weather_daily.index.name = 'date'
    print(f"Aggregated hourly -> daily rows: {len(df_weather_daily)}")

    return df_flood, df_weather_daily

# -----------------------------
# 2) Preprocess & align
# -----------------------------
def align_and_fill(df_flood, df_weather_daily):
    """Merge flood and weather, ensure continuous daily index and fill small gaps."""
    # Create a full daily index covering both ranges (use the union)
    start = min(df_flood.index.min(), df_weather_daily.index.min())
    end = max(df_flood.index.max(), df_weather_daily.index.max())
    full_idx = pd.date_range(start=start, end=end, freq='D', name='date')

    df_flood = df_flood.reindex(full_idx)
    df_weather_daily = df_weather_daily.reindex(full_idx)

    # Merge
    df = pd.concat([df_flood, df_weather_daily], axis=1)

    # Interpolate short gaps in weather and discharge if missing (limit small gaps)
    df[['rainfall_mm', 'soil_moisture', 'evapotranspiration', 'temperature']] = df[
        ['rainfall_mm', 'soil_moisture', 'evapotranspiration', 'temperature']].interpolate(method='time', limit=3)

    # For discharge, do forward-fill for at most 2 days then leave NaN (don't invent long runs)
    df['discharge_m3_s'] = df['discharge_m3_s'].fillna(method='ffill', limit=2)

    # If remaining NaNs exist in key variables, they will be dropped later
    return df

# -----------------------------
# 3) Feature engineering
# -----------------------------
def engineer_features(df):
    """Add lags, rolling stats, ARI, interactions, cyclical month, and target."""
    df = df.copy()
    # base lags and rolling
    df['discharge_lag_1'] = df['discharge_m3_s'].shift(1)
    df['discharge_lag_2'] = df['discharge_m3_s'].shift(2)
    df['rainfall_lag_1'] = df['rainfall_mm'].shift(1)
    df['rainfall_3_day_sum'] = df['rainfall_mm'].shift(1).rolling(window=3).sum()
    df['rainfall_7_day_avg'] = df['rainfall_mm'].shift(1).rolling(window=7).mean()
    df['discharge_3_day_avg'] = df['discharge_m3_s'].shift(1).rolling(window=3).mean()
    df['discharge_rate_change'] = df['discharge_m3_s'] - df['discharge_lag_1']

    # Antecedent Rainfall Index (simple exponential decay)
    # ARI = sum_{i=1..7} rainfall_{t-i} * decay^{i-1}
    decay = 0.7
    ari = np.zeros(len(df))
    for i in range(1, 8):
        ari += (decay ** (i - 1)) * df['rainfall_mm'].shift(i).fillna(0).values
    df['ari_7'] = ari

    # Interaction: rainfall * soil moisture
    df['rain_x_soil'] = (df['rainfall_mm'] * df['soil_moisture']).fillna(0)

    # Cyclical encoding for month (better than integer month)
    month = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

    # "Perfect forecast" placeholder: tomorrow's rainfall (shift -1) - only for training if you have real forecast in production replace this
    df['rainfall_forecast_24h'] = df['rainfall_mm'].shift(-1)

    # Target: next day's discharge
    df['target_next_day_discharge'] = df['discharge_m3_s'].shift(-1)

    return df

# -----------------------------
# 4) Prepare dataset for training
# -----------------------------
def prepare_dataset(df, features, target, fill_na_strategy='zero'):
    df_proc = df.copy()
    # Drop rows where target is missing
    df_proc = df_proc.dropna(subset=[target])
    # Optionally fill NA for features
    if fill_na_strategy == 'zero':
        df_proc[features] = df_proc[features].fillna(0)
    elif fill_na_strategy == 'ffill':
        df_proc[features] = df_proc[features].fillna(method='ffill').fillna(0)
    else:
        df_proc[features] = df_proc[features].fillna(0)

    X = df_proc[features]
    y = df_proc[target]
    return X, y, df_proc

# -----------------------------
# 5) Train model (with optional GridSearch)
# -----------------------------
def train_and_tune(X_train, y_train, use_grid=True):
    if USE_LOG_TARGET:
        # If using log transform, expect caller to have provided y_train as original values.
        y_train_tr = np.log1p(y_train)
    else:
        y_train_tr = y_train.values

    # Basic RF estimator
    base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    if use_grid:
        param_grid = {
            'n_estimators': [200, 400],
            'max_depth': [12, 20],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 0.8]
        }
        # TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        grid = GridSearchCV(base, param_grid, cv=tscv, scoring='r2', n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train_tr)
        print("Best params:", grid.best_params_)
        model = grid.best_estimator_
    else:
        model = base.fit(X_train, y_train_tr)

    return model

# -----------------------------
# 6) Evaluate
# -----------------------------
def evaluate_model(model, X_test, y_test):
    if USE_LOG_TARGET:
        y_test_true = y_test.values
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log).clip(min=0)  # avoid negative due to numerical issues
    else:
        y_test_true = y_test.values
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test_true, y_pred)
    r2 = r2_score(y_test_true, y_pred)

    # Peak metrics
    threshold = np.nanpercentile(y_test_true, 90)  # top 10% flows
    is_peak = y_test_true >= threshold
    if is_peak.sum() > 0:
        r2_peak = r2_score(y_test_true[is_peak], y_pred[is_peak])
    else:
        r2_peak = np.nan

    print(f"MAE: {mae:.3f}   R2: {r2:.3f}   R2_peak(>90p): {r2_peak:.3f}")
    return y_pred, {'mae': mae, 'r2': r2, 'r2_peak': r2_peak, 'threshold_90p': threshold}

# -----------------------------
# 7) Plot diagnostics
# -----------------------------
def plot_results(y_test, y_pred, df_all_index, filename=PLOT_OUTPUT):
    plt.figure(figsize=(14,6))
    plt.plot(y_test.index, y_test, label='Actual', color='tab:blue', alpha=0.8)
    plt.plot(y_test.index, y_pred, label='Predicted', color='tab:red', linestyle='--', alpha=0.9)
    plt.fill_between(y_test.index, y_test, y_pred, color='gray', alpha=0.15)
    plt.title('Model Backtest: Actual vs Predicted (test period)')
    plt.ylabel('Discharge (m³/s)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

# -----------------------------
# main pipeline
# -----------------------------
def main():
    # 1) fetch
    df_flood, df_weather_daily = fetch_open_meteo(LATITUDE, LONGITUDE, START_DATE, END_DATE, TIMEZONE)

    # 2) align & merge
    df = align_and_fill(df_flood, df_weather_daily)

    # 3) features
    df = engineer_features(df)

    # 4) define features list
    features = [
        'discharge_m3_s', 'discharge_lag_1', 'discharge_lag_2',
        'discharge_rate_change', 'discharge_3_day_avg',
        'rainfall_lag_1', 'rainfall_3_day_sum', 'rainfall_7_day_avg',
        'rainfall_forecast_24h', 'ari_7', 'rain_x_soil',
        'soil_moisture', 'evapotranspiration', 'temperature',
        'month_sin', 'month_cos'
    ]
    target = 'target_next_day_discharge'

    # 5) prepare dataset
    X, y, df_proc = prepare_dataset(df, features, target, fill_na_strategy='zero')
    if len(X) < 100:
        raise RuntimeError("Not enough rows after preprocessing. Check data availability.")

    # 6) time-based split (80/20)
    df_proc = df_proc.sort_index()
    split_point = int(len(df_proc) * 0.8)
    split_date = df_proc.index[split_point]
    train_df = df_proc.loc[df_proc.index < split_date]
    test_df = df_proc.loc[df_proc.index >= split_date]
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}  Split date: {split_date.date()}")

    # 7) prepare sample weights to emphasize floods (optional)
    sample_weight = None
    if EMPHASIZE_FLOODS:
        thr = np.nanpercentile(y_train.values, 90)
        # weights = 1 + alpha for above-threshold rows
        alpha = 5.0
        sample_weight = np.where(y_train.values >= thr, 1.0 + alpha, 1.0)
        print(f"Sample weighting enabled. Emphasizing flows >= {thr:.2f} m3/s with alpha={alpha}.")

    # 8) training
    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    # Use grid search with TimeSeriesSplit; to keep runtime moderate we do a small grid
    use_grid = True
    if use_grid:
        param_grid = {
            'n_estimators': [200, 400],
            'max_depth': [12, 20],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt']
        }
        tscv = TimeSeriesSplit(n_splits=3)
        grid = GridSearchCV(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                            param_grid, cv=tscv, scoring='r2', n_jobs=-1, verbose=1)
        # apply log transform if selected
        if USE_LOG_TARGET:
            y_train_tr = np.log1p(y_train.values)
            grid.fit(X_train, y_train_tr, **({'sample_weight': sample_weight} if sample_weight is not None else {}))
        else:
            grid.fit(X_train, y_train.values, **({'sample_weight': sample_weight} if sample_weight is not None else {}))
        model = grid.best_estimator_
        print("GridSearch best params:", grid.best_params_)
    else:
        if USE_LOG_TARGET:
            y_train_tr = np.log1p(y_train.values)
            model.fit(X_train, y_train_tr, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train.values, sample_weight=sample_weight)

    # 9) Evaluate
    print("Evaluating on test set...")
    if USE_LOG_TARGET:
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log).clip(min=0)
    else:
        y_pred = model.predict(X_test)

    metrics = {
        'mae': mean_absolute_error(y_test.values, y_pred),
        'r2': r2_score(y_test.values, y_pred)
    }
    # Peak R2
    thr_90 = np.nanpercentile(y_test.values, 90)
    idx_peak = y_test.values >= thr_90
    if idx_peak.sum() > 0:
        metrics['r2_peak'] = r2_score(y_test.values[idx_peak], y_pred[idx_peak])
    else:
        metrics['r2_peak'] = np.nan

    print("Test MAE: {:.3f}, R2: {:.3f}, R2_peak(>90p): {:.3f}".format(metrics['mae'], metrics['r2'], metrics['r2_peak']))

    # 10) Save model
    joblib.dump({'model': model, 'features': features, 'use_log_target': USE_LOG_TARGET}, MODEL_OUTPUT)
    print(f"Saved model to {MODEL_OUTPUT}")

    # 11) Plot results (test period)
    y_test_series = pd.Series(y_test.values, index=y_test.index, name='Actual')
    y_pred_series = pd.Series(y_pred, index=y_test.index, name='Predicted')
    plot_results(y_test_series, y_pred_series, df_proc.index, filename=PLOT_OUTPUT)

if __name__ == "__main__":
    main()
