#!/usr/bin/env python3
"""
flood_model_v11.py - HydroFusion (v11)

Goals:
 - Improve flood-peak skill further using hybrid sequence+tabular stacking,
   time-series out-of-fold predictions (no leakage), and peak-weighted training.
 - Saves a bundle with base models, LSTM (if available), meta-learner, calibrator,
   feature list and metadata.

Usage:
    python flood_model_v11.py --mode train
    python flood_model_v11.py --mode train --quick
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

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

# optional libs
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False
try:
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# -----------------------
# CONFIG / defaults
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
MODEL_FILE = "open_meteo_flood_model_v11.pkl"
FEATURES_JSON = "open_meteo_flood_features_v11.json"
PLOT_FULL = "v11_backtest.png"
PLOT_PEAKS = "v11_backtest_peaks.png"
PRED_CSV = "v11_test_predictions.csv"

RANDOM_STATE = 42
FLOOD_Q = 0.90
PEAK_WEIGHT_ALPHA = 18.0  # stronger peak emphasis
TIME_SERIES_SPLITS = 4

SEQ_LENGTH = 7  # days used for LSTM

# ---------- utilities ----------
def safe_get_json(url, params, timeout=90):
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
# Fetching (split by year to avoid API 400)
# -----------------------
def fetch_archive_hourly(lat, lon, start_date, end_date):
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    today = date.today()
    if end > today:
        print(f"Warning: requested end_date {end} is in the future. Clipping to {today}.")
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
        time.sleep(0.6)
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
# Feature engineering (richer than v10)
# -----------------------
def build_daily_merged(df_hourly, df_flood):
    daily_precip = df_hourly['precipitation'].resample('D').sum().rename('rainfall_mm')
    daily_soil = df_hourly['soil_moisture_0_1cm'].resample('D').mean().rename('soil_moisture')
    daily_et = df_hourly['et0_fao_evapotranspiration'].resample('D').mean().rename('evapotranspiration')
    daily_temp = df_hourly['temperature_2m'].resample('D').mean().rename('temperature')
    daily_max_hour = df_hourly['precipitation'].resample('D').max().rename('max_hourly_rain_mm')
    df_daily = pd.concat([daily_precip, daily_soil, daily_et, daily_temp, daily_max_hour], axis=1)
    df_daily.index.name = 'date'
    df = pd.merge(df_flood, df_daily, left_index=True, right_index=True, how='inner')
    return df

def engineer_v11(df_daily, lead_days=1, api_k=0.85):
    d = df_daily.copy().sort_index()
    # discharge lags
    d['dis_lag1'] = d['discharge_m3_s'].shift(1)
    d['dis_lag2'] = d['discharge_m3_s'].shift(2)
    d['dis_lag3'] = d['discharge_m3_s'].shift(3)
    # rainfall lags + rolling sums
    for i in [1,2,3,7,14]:
        d[f'rain_lag_{i}'] = d['rainfall_mm'].shift(i)
    d['rain_roll_3'] = d['rainfall_mm'].shift(1).rolling(3).sum()
    d['rain_roll_7'] = d['rainfall_mm'].shift(1).rolling(7).sum()
    d['rain_roll_14'] = d['rainfall_mm'].shift(1).rolling(14).sum()
    # ET & ET lag
    d['et_lag_1'] = d['evapotranspiration'].shift(1)
    # API: recursive index for antecedent moisture
    api = []; prev=0.0
    for r in d['rainfall_mm'].fillna(0).values:
        val = r + api_k * prev; api.append(val); prev = val
    d['api'] = api
    # intensity & interactions
    d['max_hourly_rain_mm'] = d['max_hourly_rain_mm'].fillna(0)
    d['rain_x_soil'] = d['rainfall_mm'] * d['soil_moisture']
    d['dis_rate'] = d['discharge_m3_s'] - d['dis_lag1']
    # cyclical
    d['month'] = d.index.month
    d['month_sin'] = np.sin(2*np.pi*d['month']/12)
    d['month_cos'] = np.cos(2*np.pi*d['month']/12)
    # pseudo-forecast
    d['rainfall_forecast_Nd'] = d['rainfall_mm'].shift(-lead_days)
    # target
    d['target'] = d['discharge_m3_s'].shift(-lead_days)
    d = d.dropna(subset=['target'])
    return d

# -----------------------
# augmentation: synthetic surges (modern concat)
# -----------------------
def augment_with_surges(df, n_augment=500, max_extra_mm=160.0, prob_peak_day=0.75, runoff_coeff=0.55):
    rng = np.random.default_rng(RANDOM_STATE)
    df_aug = df.copy()
    indices = df.index.values
    for i in range(n_augment):
        base_idx = rng.choice(indices[len(indices)//4:])
        base_row = df.loc[base_idx].copy()
        day_col = 'rain_lag_1' if rng.random() < prob_peak_day else 'rain_lag_3'
        extra = rng.uniform(10.0, max_extra_mm)
        base_row[day_col] = (base_row.get(day_col, 0.0) or 0.0) + extra
        base_row['rain_roll_3'] = (base_row.get('rain_roll_3', 0) or 0) + extra
        base_row['api'] = (base_row.get('api', 0) or 0) + extra
        base_row['max_hourly_rain_mm'] = max(base_row.get('max_hourly_rain_mm', 0), extra)
        base_row['target'] = base_row['target'] + runoff_coeff * extra
        base_row['is_synthetic'] = 1
        df_aug = pd.concat([df_aug, pd.DataFrame([base_row])], ignore_index=False)
    if 'is_synthetic' not in df_aug.columns:
        df_aug['is_synthetic'] = 0
    df_aug['is_synthetic'] = df_aug['is_synthetic'].fillna(0).astype(int)
    df_aug = df_aug.sample(frac=1, random_state=RANDOM_STATE)
    return df_aug

# -----------------------
# features list
# -----------------------
FEATURE_LIST = [
 'discharge_m3_s','dis_lag1','dis_lag2','dis_lag3','dis_rate',
 'rain_lag_1','rain_lag_2','rain_lag_3','rain_lag_7','rain_lag_14',
 'rain_roll_3','rain_roll_7','rain_roll_14','api',
 'rainfall_forecast_Nd','rain_x_soil','max_hourly_rain_mm',
 'soil_moisture','evapotranspiration','et_lag_1','temperature',
 'month_sin','month_cos'
]

def prepare_tabular_Xy(df):
    X = df[FEATURE_LIST].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    y = df['target']
    return X, y

# -----------------------
# sequence utilities for LSTM
# -----------------------
def build_sequences(df, seq_len=SEQ_LENGTH, features_seq=['rainfall_mm','discharge_m3_s','soil_moisture','max_hourly_rain_mm']):
    d = df.copy().sort_index()
    for f in features_seq:
        if f not in d.columns:
            d[f] = 0.0
    arr = d[features_seq].values
    targets = d['target'].values
    Xs, ys, idxs = [], [], []
    for i in range(seq_len, len(d)):
        seq = arr[i-seq_len:i]
        Xs.append(seq)
        ys.append(targets[i])
        idxs.append(d.index[i])
    return np.array(Xs), np.array(ys), np.array(idxs)

def build_lstm_model(input_shape, units=48, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# -----------------------
# Out-of-fold helper for time series (produces oof preds for a model)
# -----------------------
def oof_preds_time_series(estimator_factory, X, y, n_splits=TIME_SERIES_SPLITS, fit_params=None, is_keras=False, seq_data=None):
    """
    estimator_factory: function that returns a fresh estimator instance
    X, y: pandas DataFrame/Series (tabular)
    fit_params: dict passed to fit() (e.g., sample_weight)
    is_keras: if True, estimator_factory should return a KerasRegressor-like wrapper
    seq_data: if is_keras True, seq_data is a tuple (Xs, ys, idxs) for sequences aligned to X
    Returns: oof_pred (np.array aligned to X.index), final_trained_estimator (trained on full X,y)
    """
    n = len(X)
    oof = np.full(n, np.nan, dtype=float)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    indices = np.arange(n)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(indices)):
        print(f"  OOF fold {fold+1}/{n_splits}: train {len(train_idx)} -> val {len(val_idx)}")
        est = estimator_factory()
        if is_keras and seq_data is not None:
            # seq_data: (Xs_all, ys_all, idxs_all) aligned to X.index
            Xs_all, ys_all, idxs_all = seq_data
            # find rows corresponding to train_idx and val_idx by index matching
            train_dates = X.index[train_idx]
            val_dates = X.index[val_idx]
            # build train and val sequences by matching idxs_all dates
            mask_train = np.isin(idxs_all, train_dates)
            mask_val = np.isin(idxs_all, val_dates)
            Xs_tr = Xs_all[mask_train]; ys_tr = ys_all[mask_train]
            Xs_val = Xs_all[mask_val]
            # fit keras model
            if fit_params:
                est.fit(Xs_tr, ys_tr, **fit_params)
            else:
                est.fit(Xs_tr, ys_tr)
            preds = est.predict(Xs_val).reshape(-1)
            # map preds into oof positions by matching val_dates order
            # create mapping from idxs_all[mask_val] -> preds
            dates_val_seq = idxs_all[mask_val]
            # for each val date, find its position in X.index and assign pred
            for d, p in zip(dates_val_seq, preds):
                pos = np.where(X.index == d)[0]
                if pos.size:
                    oof[pos[0]] = p
        else:
            # tabular estimator
            fit_kw = fit_params or {}
            est.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_kw) if fit_kw else est.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = est.predict(X.iloc[val_idx])
            oof[val_idx] = preds
    # retrain on full data
    final_est = estimator_factory()
    fit_kw = fit_params or {}
    try:
        final_est.fit(X, y, **fit_kw) if fit_kw else final_est.fit(X, y)
    except TypeError:
        final_est.fit(X, y)
    return oof, final_est

# -----------------------
# Training main (v11)
# -----------------------
def train_v11(lat, lon, start, end, lead_days=1, quick=False):
    print(f"Fetching data for {start} → {end}")
    df_hourly = fetch_archive_hourly(lat, lon, start, end)
    df_flood = fetch_flood_daily(lat, lon, start, end)
    print("Aggregating hourly -> daily and merging...")
    df_daily = build_daily_merged(df_hourly, df_flood)
    print(f"Total merged daily rows: {len(df_daily)}")
    df_feat = engineer_v11(df_daily, lead_days=lead_days)

    # time split 80/20
    split_idx = int(len(df_feat) * 0.8)
    train_df = df_feat.iloc[:split_idx].copy()
    test_df = df_feat.iloc[split_idx:].copy()
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")

    # augment training data with surges
    train_aug = augment_with_surges(train_df, n_augment=400 if not quick else 150)
    print(f"After augmentation: {len(train_aug)} training rows (incl synthetic)")

    # compute peak threshold on training augmented target
    thr = np.nanpercentile(train_aug['target'].values, FLOOD_Q*100)
    print(f"Peak threshold (train {int(FLOOD_Q*100)}th pct) = {thr:.2f}")

    # compute sample weights for training (strong emphasis on peaks)
    sw = np.where(train_aug['target'].values >= thr, 1.0 + PEAK_WEIGHT_ALPHA, 1.0)
    if 'is_synthetic' in train_aug.columns:
        sw = sw * np.where(train_aug['is_synthetic']==1, 0.6, 1.0)

    # prepare tabular X,y
    X_train_tab, y_train_tab = prepare_tabular_Xy(train_aug)
    X_test_tab, y_test_tab = prepare_tabular_Xy(test_df)

    # LSTM: build sequences and produce OOF predictions (no leakage)
    lstm_oof_train = None
    lstm_oof_test = None
    lstm_model_final = None
    seq_features = ['rainfall_mm','discharge_m3_s','soil_moisture','max_hourly_rain_mm']

    if TF_AVAILABLE:
        print("Preparing LSTM sequences and producing OOF predictions (time-safe)...")
        # sequences built on train_aug for OOF
        Xs_train_seq, ys_train_seq, idxs_train_seq = build_sequences(train_aug, seq_len=SEQ_LENGTH, features_seq=seq_features)
        # create KerasRegressor wrapper factory
        def make_keras_estimator():
            def build_fn():
                return build_lstm_model((Xs_train_seq.shape[1], Xs_train_seq.shape[2]), units=48, dropout=0.2)
            # Using simple wrapper for direct fit/predict
            return build_lstm_model((Xs_train_seq.shape[1], Xs_train_seq.shape[2]), units=48, dropout=0.2)
        # For OOF on sequences we'll implement a simpler time-split training loop
        n = len(train_aug)
        oof = np.full(n, np.nan)
        tscv = TimeSeriesSplit(n_splits=TIME_SERIES_SPLITS)
        # map sequence idxs to positions in train_aug by date
        seq_date_to_pos = {d: i for i,d in enumerate(train_aug.index)}
        # Build sequence index mapping: seq indices correspond to dates idxs_train_seq
        for fold, (train_idx, val_idx) in enumerate(tscv.split(np.arange(len(train_aug)))):
            print(f"LSTM OOF fold {fold+1}/{TIME_SERIES_SPLITS}")
            # get train dates and val dates
            train_dates = train_aug.index[train_idx]
            val_dates = train_aug.index[val_idx]
            # select sequences whose target date in train_dates or val_dates
            mask_train = np.isin(idxs_train_seq, train_dates)
            mask_val = np.isin(idxs_train_seq, val_dates)
            Xs_tr = Xs_train_seq[mask_train]; ys_tr = ys_train_seq[mask_train]
            Xs_val = Xs_train_seq[mask_val]
            if len(Xs_tr) < 10 or len(Xs_val) < 1:
                print("  skipping fold due to too few sequences")
                continue
            model = build_lstm_model((Xs_tr.shape[1], Xs_tr.shape[2]), units=48, dropout=0.2)
            # small epochs if quick, else more
            epochs = 8 if quick else 25
            model.fit(Xs_tr, ys_tr, epochs=epochs, batch_size=32, verbose=0)
            preds_val = model.predict(Xs_val).reshape(-1)
            # assign preds_val into oof where idxs_train_seq[mask_val] matches train_aug index
            val_dates_seq = idxs_train_seq[mask_val]
            for d,p in zip(val_dates_seq, preds_val):
                pos = np.where(train_aug.index == d)[0]
                if pos.size:
                    oof[pos[0]] = p
        # After OOF loops, for any remaining NaNs in oof, fill with median of oof (or simple model)
        nan_mask = np.isnan(oof)
        if nan_mask.any():
            fill_val = np.nanmedian(oof[~nan_mask]) if (~nan_mask).any() else np.nanmedian(ys_train_seq)
            oof[nan_mask] = fill_val
        # store oof as series aligned to train_aug.index
        lstm_oof_train = pd.Series(oof, index=train_aug.index)
        # train final LSTM on full train_aug sequences
        if len(Xs_train_seq) > 0:
            final_lstm = build_lstm_model((Xs_train_seq.shape[1], Xs_train_seq.shape[2]), units=48, dropout=0.2)
            final_epochs = 10 if quick else 40
            final_lstm.fit(Xs_train_seq, ys_train_seq, epochs=final_epochs, batch_size=32, verbose=0)
            lstm_model_final = final_lstm
            # create test sequences from train tail + test to predict test aligned sequences
            combined_for_seq = pd.concat([train_aug.tail(SEQ_LENGTH), test_df])
            Xs_test_seq, ys_test_seq, idxs_test_seq = build_sequences(combined_for_seq, seq_len=SEQ_LENGTH, features_seq=seq_features)
            # keep only those idxs that are in test_df.index
            # map idxs_test_seq to test_df positions
            preds_test_seq = final_lstm.predict(Xs_test_seq).reshape(-1)
            # align preds_test_seq to test_df index by matching dates
            test_preds_map = {}
            for d,p in zip(idxs_test_seq, preds_test_seq):
                if d in test_df.index:
                    test_preds_map[d] = p
            # build series aligned to X_test_tab.index
            lstm_oof_test = pd.Series([test_preds_map.get(d, np.nan) for d in X_test_tab.index], index=X_test_tab.index)
            # fill NaNs with median
            lstm_oof_test = lstm_oof_test.fillna(np.nanmedian(preds_test_seq))
        else:
            lstm_oof_train = None
            lstm_oof_test = None
            lstm_model_final = None
    else:
        print("TensorFlow not available; skipping LSTM (still proceed with tabular stack)")

    # incorporate lstm oof preds as feature if available
    if lstm_oof_train is not None:
        X_train_tab['lstm_oof'] = lstm_oof_train
    else:
        X_train_tab['lstm_oof'] = 0.0
    if lstm_oof_test is not None:
        X_test_tab['lstm_oof'] = lstm_oof_test
    else:
        X_test_tab['lstm_oof'] = 0.0

    # ---------- produce out-of-fold predictions for tree models (time-safe) ----------
    print("Producing OOF preds for RF and GB (time-aware)...")
    def rf_factory(): return RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE)
    def gb_factory(): return GradientBoostingRegressor(random_state=RANDOM_STATE)
    # fit_params include sample_weight for folds (scikit will pass it to fit)
    fit_params = {'sample_weight': sw}
    # OOF preds on augmented training set (aligned to train_aug.index)
    rf_oof, rf_full = oof_preds_time_series(rf_factory, X_train_tab, train_aug['target'], n_splits=TIME_SERIES_SPLITS, fit_params=fit_params)
    gb_oof, gb_full = oof_preds_time_series(gb_factory, X_train_tab, train_aug['target'], n_splits=TIME_SERIES_SPLITS, fit_params=fit_params)
    # optional CatBoost OOF
    cb_oof = None; cb_full = None
    if CATBOOST_AVAILABLE:
        def cb_factory(): return CatBoostRegressor(random_state=RANDOM_STATE, verbose=0)
        cb_oof, cb_full = oof_preds_time_series(cb_factory, X_train_tab, train_aug['target'], n_splits=TIME_SERIES_SPLITS, fit_params=fit_params)
    # Build DataFrame of OOF preds for meta training
    meta_train_df = pd.DataFrame({'rf_oof': rf_oof, 'gb_oof': gb_oof}, index=train_aug.index)
    if cb_oof is not None:
        meta_train_df['cb_oof'] = cb_oof
    # add lstm oof if present
    if lstm_oof_train is not None:
        meta_train_df['lstm_oof'] = lstm_oof_train
    # train full base models on entire augmented train (already returned as rf_full, gb_full)
    # predict on test using full models (no leakage)
    rf_test_pred = rf_full.predict(X_test_tab)
    gb_test_pred = gb_full.predict(X_test_tab)
    cb_test_pred = cb_full.predict(X_test_tab) if cb_full is not None else None
    # assemble base preds on test
    base_test_preds = pd.DataFrame({'rf': rf_test_pred, 'gb': gb_test_pred}, index=X_test_tab.index)
    if cb_test_pred is not None:
        base_test_preds['cb'] = cb_test_pred
    if lstm_oof_test is not None:
        base_test_preds['lstm'] = X_test_tab['lstm_oof'].values

    # ---------- meta learner: ridge with peak-weighted fitting ----------
    print("Training meta-learner (ridge) with peak-weighted fit...")
    meta_X = meta_train_df.fillna(0).values
    meta_y = train_aug['target'].values
    meta_sw = np.where(meta_y >= thr, 1.0 + PEAK_WEIGHT_ALPHA, 1.0)
    if 'is_synthetic' in train_aug.columns:
        meta_sw = meta_sw * np.where(train_aug['is_synthetic']==1, 0.6, 1.0)
    meta_lr = Ridge(alpha=1.0)
    meta_lr.fit(meta_X, meta_y, sample_weight=meta_sw)
    stack_test_pred = meta_lr.predict(base_test_preds.fillna(0).values)

    # peak calibrator (Ridge) trained on high flows from meta-val slice (last 10% of train_aug)
    val_slice = int(len(train_aug) * 0.9)
    meta_val_df = train_aug.iloc[val_slice:].copy()
    # build meta preds on val using base_full models
    base_val_df = pd.DataFrame({
        'rf': rf_full.predict(prepare_tabular_Xy(meta_val_df)[0]),
        'gb': gb_full.predict(prepare_tabular_Xy(meta_val_df)[0])
    }, index=meta_val_df.index)
    if cb_full is not None:
        base_val_df['cb'] = cb_full.predict(prepare_tabular_Xy(meta_val_df)[0])
    if lstm_oof_train is not None:
        base_val_df['lstm'] = meta_train_df.loc[meta_val_df.index, 'lstm_oof'].values
    ensemble_val_preds = meta_lr.predict(base_val_df.fillna(0).values)
    high_mask_val = meta_val_df['target'].values >= thr
    calibrator = None
    if high_mask_val.sum() >= 10:
        cal = Ridge(alpha=1.0)
        cal.fit(ensemble_val_preds[high_mask_val].reshape(-1,1), meta_val_df['target'].values[high_mask_val])
        calibrator = cal
        print("Trained peak calibrator on validation high flows.")
    else:
        print("Not enough high-flow samples for calibrator; skipping.")

    # apply calibrator to test predictions
    final_test_pred = stack_test_pred.copy()
    if calibrator is not None:
        high_mask_test = X_test_tab['discharge_m3_s'].values >= thr
        if high_mask_test.sum() > 0:
            final_test_pred[high_mask_test] = calibrator.predict(final_test_pred[high_mask_test].reshape(-1,1))

    # ---------- evaluation ----------
    y_test_vals = test_df['target'].values
    def metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        r2v = r2_score(y_true, y_pred)
        r2peak = r2_score(y_true[y_true>=thr], y_pred[y_true>=thr]) if np.sum(y_true>=thr) >= 2 else float('nan')
        return {'mae': mae, 'rmse': rmse, 'r2': r2v, 'r2_peak': r2peak}
    base_metrics = {
        'rf': metrics(y_test_vals, rf_test_pred),
        'gb': metrics(y_test_vals, gb_test_pred)
    }
    if cb_test_pred is not None:
        base_metrics['cb'] = metrics(y_test_vals, cb_test_pred)
    if lstm_oof_test is not None:
        base_metrics['lstm'] = metrics(y_test_vals, X_test_tab['lstm_oof'].values)
    final_metrics = metrics(y_test_vals, final_test_pred)
    print("Base metrics:", base_metrics)
    print("Final stacked+calibrated metrics:", final_metrics)

    # ---------- save bundle ----------
    bundle = {
        'rf_full': rf_full, 'gb_full': gb_full, 'cb_full': cb_full if CATBOOST_AVAILABLE else None,
        'lstm_model': lstm_model_final, 'meta_lr': meta_lr, 'calibrator': calibrator,
        'features': FEATURE_LIST, 'seq_features': seq_features, 'seq_length': SEQ_LENGTH,
        'use_lstm': TF_AVAILABLE, 'trained_range': {'start': start, 'end': end}
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, os.path.join(MODEL_DIR, MODEL_FILE))
    with open(os.path.join(MODEL_DIR, FEATURES_JSON), 'w') as f:
        json.dump({'features': FEATURE_LIST, 'seq_features': seq_features, 'lead_days': lead_days}, f, indent=2)
    print(f"Saved model bundle to {os.path.join(MODEL_DIR, MODEL_FILE)}")

    # save test predictions CSV
    out_df = pd.DataFrame({
        'date': test_df.index,
        'observed': y_test_vals,
        'rf_pred': rf_test_pred,
        'gb_pred': gb_test_pred,
        'stack_pred': stack_test_pred,
        'final_pred': final_test_pred
    }, index=test_df.index)
    out_df.to_csv(PRED_CSV, index=False)
    print(f"Saved test predictions to {PRED_CSV}")

    # plots
    idx = test_df.index
    plt.figure(figsize=(14,6))
    plt.plot(idx, y_test_vals, label='Observed')
    plt.plot(idx, final_test_pred, '--', label='Final Pred')
    plt.legend(); plt.grid(True); plt.title('v11 Backtest - Observed vs Final Pred'); plt.savefig(PLOT_FULL); plt.close()
    thr_mask = y_test_vals >= thr
    if thr_mask.sum() > 0:
        plt.figure(figsize=(12,5))
        plt.plot(idx[thr_mask], y_test_vals[thr_mask], 'o-', label='Observed peaks')
        plt.plot(idx[thr_mask], final_test_pred[thr_mask], 'x--', label='Predicted peaks')
        plt.legend(); plt.grid(True); plt.title('v11 Peaks'); plt.savefig(PLOT_PEAKS); plt.close()
    print(f"Saved {PLOT_FULL} and {PLOT_PEAKS}")

    return {'final_metrics': final_metrics, 'base_metrics': base_metrics}

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
    p.add_argument('--quick', action='store_true', help='smaller grids / fewer epochs for quick testing')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    res = train_v11(args.lat, args.lon, args.start, args.end, lead_days=args.lead, quick=args.quick)
    print("Training v11 complete. Results:", res)

