#!/usr/bin/env python3
"""
flood_model_v13.py - HydroFusion v13 (Merged & Fixed)

Drop-in replacement for your v13. Fixes:
 - XGBoost feature_name mismatch by recording per-model trained columns
 - Fill NaNs in OOF predictions before inverse-transform
 - Robust handling for lstm_oof being present / absent
 - Keeps LSTM + LGB/XGB/CatBoost + BayesianRidge meta + peak calibrator
 - Saves plots: backtest, peaks, calibration, obs vs pred, SHAP (if available)
"""

import os, time, math, json, argparse, warnings
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Optional model libs
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CAT_AVAILABLE = True
except Exception:
    CAT_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import tensorflow as tf
    from keras.models import Model
    from keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# -----------------------
# Config
# -----------------------
LAT = 23.81
LON = 90.41
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2025-12-31"
TIMEZONE = "auto"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FLOOD = "https://flood-api.open-meteo.com/v1/flood"

MODEL_DIR = "ml"
MODEL_FILE = "open_meteo_flood_model_v13.pkl"
FEATURES_JSON = "open_meteo_flood_features_v13.json"
PLOT_BACKTEST = "v13_backtest.png"
PLOT_PEAKS = "v13_backtest_peaks.png"
PLOT_SHAP = "v13_shap_summary.png"
PLOT_CAL = "v13_calibration.png"
PLOT_OBS_PRED = "v13_obs_vs_pred.png"
PLOT_CORR = "v13_model_corr.png"
PRED_CSV = "v13_test_predictions.csv"

RANDOM_STATE = 42
FLOOD_Q = 0.90
PEAK_WEIGHT_ALPHA = 18.0
TIME_SERIES_SPLITS = 4
SEQ_LENGTH = 14

# Feature lists (removed constant columns observed previously)
FEATURE_LIST = [
 'discharge_m3_s','dis_lag1','dis_lag2','dis_lag3','dis_rate',
 'rain_lag_1','rain_lag_2','rain_lag_3','rain_lag_7','rain_lag_14',
 'rain_roll_3','rain_roll_7','rain_roll_14','api',
 'rainfall_forecast_1d',
 'max_hourly_rain_mm',
 'evapotranspiration','et_lag_1','temperature',
 'month_sin','month_cos','rain_grad_1_2',
]
SEQ_FEATURES = ['rainfall_mm','discharge_m3_s','soil_moisture','max_hourly_rain_mm']

ARGS = None

# -----------------------
# Utilities
# -----------------------
def safe_get_json(url, params, timeout=90):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def metrics_summary(y_true, y_pred, thr_q=FLOOD_Q):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2v = r2_score(y_true, y_pred)
    thr = np.nanpercentile(y_true, thr_q * 100)
    mask = y_true >= thr
    r2_peak = r2_score(y_true[mask], y_pred[mask]) if mask.sum() >= 2 else float('nan')
    return {'mae': mae, 'rmse': rmse, 'r2': r2v, 'r2_peak': r2_peak, 'thr': thr}

def plot_obs_vs_pred(y_true, y_pred, title="Observed vs Predicted", out_file=PLOT_OBS_PRED):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()],[y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Observed"); plt.ylabel("Predicted"); plt.title(title); plt.grid(True)
    plt.tight_layout(); plt.savefig(out_file); plt.close()
    print(f"Saved observed vs predicted scatter: {out_file}")

def plot_model_correlation(base_df, out_file=PLOT_CORR):
    corr = base_df.corr()
    plt.figure(figsize=(6,5))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Base Model Correlations")
    plt.tight_layout(); plt.savefig(out_file); plt.close()
    print(f"Saved base model correlation heatmap: {out_file}")

# -----------------------
# Fetching helpers (v12 -> v13)
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
        print(f"→ Fetching archive {seg_start} → {seg_end}")
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "precipitation,soil_moisture_0_1cm,et0_fao_evapotranspiration,temperature_2m",
            "start_date": seg_start.isoformat(), "end_date": seg_end.isoformat(),
            "timezone": TIMEZONE
        }
        j = safe_get_json(OPEN_METEO_ARCHIVE, params)
        hourly = pd.DataFrame(j.get('hourly', {}))
        hourly['time'] = pd.to_datetime(hourly['time'])
        hourly = hourly.set_index('time').sort_index()
        dfs.append(hourly)
        year += 1
        time.sleep(0.6)
    if not dfs:
        return pd.DataFrame()
    df_hourly = pd.concat(dfs).sort_index()
    rename_map = {
        'precipitation': 'rainfall_mm',
        'soil_moisture_0_1cm': 'soil_moisture',
        'et0_fao_evapotranspiration': 'evapotranspiration',
        'temperature_2m': 'temperature'
    }
    df_hourly = df_hourly.rename(columns=rename_map)
    if 'rainfall_mm' in df_hourly.columns:
        df_hourly['max_hourly_rain_mm'] = df_hourly['rainfall_mm']
    else:
        df_hourly['max_hourly_rain_mm'] = 0.0
    return df_hourly

def fetch_flood_daily(lat, lon, start_date, end_date):
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    today = date.today()
    if end > today:
        end = today
    params = {"latitude": lat, "longitude": lon, "daily": "river_discharge",
              "start_date": start.isoformat(), "end_date": end.isoformat(), "timezone": TIMEZONE}
    j = safe_get_json(OPEN_METEO_FLOOD, params)
    df = pd.DataFrame(j.get("daily", {}))
    if 'time' in df.columns:
        df = df.rename(columns={"time": "date", "river_discharge": "discharge_m3_s"})
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
    return df

# -----------------------
# Feature engineering (v12 logic kept)
# -----------------------
def build_daily_merge(df_hourly, df_flood):
    print("Aggregating hourly -> daily and merging...")
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
    df_daily.index.name = 'date'
    df = pd.merge(df_flood, df_daily, left_index=True, right_index=True, how='inner')
    print(f"Total merged daily rows: {len(df)}")
    return df

def engineer_v12(df_daily, lead_days=1, api_k=0.85):
    print("Engineering features...")
    d = df_daily.copy().sort_index()
    required_cols = ['discharge_m3_s', 'rainfall_mm', 'soil_moisture', 'evapotranspiration', 'temperature']
    missing = [c for c in required_cols if c not in d.columns]
    if missing:
        print(f"Warning: Missing required columns, will be filled with 0: {missing}")
        for c in missing:
            d[c] = 0.0
    d['dis_lag1'] = d['discharge_m3_s'].shift(1)
    d['dis_lag2'] = d['discharge_m3_s'].shift(2)
    d['dis_lag3'] = d['discharge_m3_s'].shift(3)
    for i in [1,2,3,7,14]:
        d[f'rain_lag_{i}'] = d['rainfall_mm'].shift(i)
    d['rain_roll_3'] = d['rainfall_mm'].shift(1).rolling(3).sum()
    d['rain_roll_7'] = d['rainfall_mm'].shift(1).rolling(7).sum()
    d['rain_roll_14'] = d['rainfall_mm'].shift(1).rolling(14).sum()
    d['rain_grad_1_2'] = d['rain_lag_1'] - d['rain_lag_2']
    d['et_lag_1'] = d['evapotranspiration'].shift(1)
    WINDOW = 7 
    weights = np.power(api_k, np.arange(WINDOW))
    d['api'] = d['rainfall_mm'].shift(1).rolling(window=WINDOW).apply(
        lambda x: np.sum(x.values * weights[::-1]), raw=False
    )
    d['max_hourly_rain_mm'] = d['max_hourly_rain_mm'].fillna(0)
    d['rain_x_soil'] = d['rain_lag_1'] * d['soil_moisture'].shift(1)
    d['dis_rate'] = d['discharge_m3_s'] - d['dis_lag1']
    d['month'] = d.index.month
    d['month_sin'] = np.sin(2*np.pi*d['month']/12)
    d['month_cos'] = np.cos(2*np.pi*d['month']/12)
    d['rainfall_forecast_1d'] = d['rainfall_mm'].shift(-lead_days)
    d['target'] = d['discharge_m3_s'].shift(-lead_days)
    d = d.dropna(subset=['target', 'dis_lag3', 'rain_lag_14', 'api'])
    return d

# -----------------------
# Augmentation
# -----------------------
def augment_with_surges(df, n_augment=400, max_extra_mm=160.0, runoff_coeff=0.55):
    print("Augmenting training data with synthetic surges...")
    rng = np.random.default_rng(RANDOM_STATE)
    df_aug = df.copy()
    indices = df.index.values
    new_rows = []
    for i in range(n_augment):
        base_idx = rng.choice(indices[len(indices)//4:])
        base_row = df.loc[base_idx].copy()
        extra_total = rng.uniform(10.0, max_extra_mm)
        days = int(rng.integers(1,4))
        per_day = extra_total / days
        for d_i in range(1, days+1):
            col = f'rain_lag_{d_i}' if f'rain_lag_{d_i}' in base_row.index else 'rain_lag_1'
            base_row[col] = (base_row.get(col, 0.0) or 0.0) + per_day
        base_row['rain_roll_3'] = (base_row.get('rain_roll_3',0) or 0) + extra_total
        base_row['api'] = (base_row.get('api',0) or 0) + extra_total
        base_row['max_hourly_rain_mm'] = max(base_row.get('max_hourly_rain_mm',0), per_day)
        base_row['rain_x_soil'] = (base_row.get('rain_lag_1', 0.0) or 0.0) * (base_row.get('soil_moisture', 0.0) or 0.0)
        base_row['target'] = base_row['target'] + runoff_coeff * extra_total
        base_row['is_synthetic'] = 1
        new_index = base_idx + pd.Timedelta(nanoseconds=i+1)
        base_row.name = new_index
        new_rows.append(base_row)
    if new_rows:
        df_aug = pd.concat([df_aug, pd.DataFrame(new_rows)], ignore_index=False)
    df_aug['is_synthetic'] = df_aug.get('is_synthetic', 0).fillna(0).astype(int)
    df_aug = df_aug.sample(frac=1, random_state=RANDOM_STATE)
    print(f"After augmentation: {len(df_aug)} rows")
    return df_aug

def prepare_tabular_Xy(df, feature_list):
    X_cols = [f for f in feature_list if f in df.columns]
    X = df[X_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    y = df['target'].astype(np.float32)
    return X, y

# -----------------------
# Sequences (LSTM)
# -----------------------
def build_sequences(df, seq_len=SEQ_LENGTH, features_seq=SEQ_FEATURES):
    print("Building LSTM sequences (SEQ_LENGTH=%d)..." % seq_len)
    d = df.copy().sort_index()
    for f in features_seq:
        if f not in d.columns:
            print(f"Warning: LSTM sequence feature '{f}' not found. Filling with 0.")
            d[f] = 0.0
    arr = d[features_seq].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    targets = d['target'].astype(np.float32).values
    Xs, ys, idxs = [], [], []
    for i in range(seq_len, len(d)):
        seq = arr[i-seq_len:i]
        if not np.isfinite(seq).all():
            continue
        Xs.append(seq); ys.append(targets[i]); idxs.append(d.index[i])
    Xs = np.array(Xs, dtype=np.float32); ys = np.array(ys, dtype=np.float32); idxs = np.array(idxs)
    print("Sequence shapes:", Xs.shape, ys.shape)
    return Xs, ys, idxs

def build_bidirectional_lstm(input_shape, units=64, dropout=0.2):
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required to build this model.")
    inp = Input(shape=input_shape)
    x = Bidirectional(LSTM(units, return_sequences=True), merge_mode='concat')(inp)
    avgp = GlobalAveragePooling1D()(x)
    maxp = GlobalMaxPooling1D()(x)
    concat = Concatenate()([avgp, maxp])
    y = Dropout(dropout)(concat)
    out = Dense(1, activation='linear')(y)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

# -----------------------
# OOF helper with fixes
# -----------------------
def oof_preds_time_series(est_factory, X, y, n_splits=4, fit_params=None):
    """
    Time-aware OOF training and prediction.
    Returns: (oof_preds_array, final_trained_model)
    final model has attribute `_trained_columns` storing columns it was trained with.
    """
    print("\n🧩 Checking training data integrity...")
    # If X is numpy, require columns passed in externally; but typically we pass DataFrame
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        # keep index aligned if possible
        y = pd.Series(y, index=X.index if isinstance(X, pd.DataFrame) else None)
    print("Feature summary (top 5):"); print(X.describe().T.head())
    print("\nTarget summary:"); print(y.describe())
    if y.nunique() <= 1:
        raise ValueError("❌ Target has only one unique value — cannot train.")
    oof_preds = np.full(len(X), np.nan, dtype=float)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n📆 OOF fold {fold + 1}/{n_splits}: train {len(train_idx)} → val {len(val_idx)}")
        X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        if y_train.nunique() <= 1:
            print("⚠️ Skipping fold due to constant training labels.")
            oof_preds[val_idx] = y_train.mean()
            continue
        est = est_factory()
        fit_kw = (fit_params.copy() if fit_params else {})
        if fit_params and "sample_weight" in fit_params:
            sw = fit_params["sample_weight"]
            if len(sw) == len(X):
                fit_kw["sample_weight"] = np.array(sw)[train_idx]
            else:
                print(f"⚠️ sample_weight length mismatch (expected {len(X)}, got {len(sw)}); ignoring for this fold.")
                fit_kw.pop("sample_weight", None)
        const_cols = X_train.columns[X_train.std() == 0].tolist()
        if const_cols:
            print(f"⚠️ Removing constant features for this fold: {const_cols}")
            X_train = X_train.drop(columns=const_cols)
            X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
        # Fit
        est.fit(X_train, y_train, **fit_kw)
        # predict
        preds = est.predict(X_val)
        oof_preds[val_idx] = preds
    # Final fit on all data
    print("\n🚀 Training final model on all data...")
    final_est = est_factory()
    final_fit_kw = (fit_params.copy() if fit_params else {})
    if "sample_weight" in final_fit_kw:
        sw = final_fit_kw["sample_weight"]
        if len(sw) != len(X):
            print("⚠️ Final sample_weight length mismatch; ignoring.")
            final_fit_kw.pop("sample_weight", None)
    const_cols_full = X.columns[X.std() == 0].tolist()
    X_full_train = X.copy()
    if const_cols_full:
        print(f"⚠️ Removing constant features from final model: {const_cols_full}")
        X_full_train = X_full_train.drop(columns=const_cols_full)
    final_est.fit(X_full_train, y, **final_fit_kw)
    # record the columns the final model was trained on (used later during predict)
    try:
        final_est._trained_columns = list(X_full_train.columns)
    except Exception:
        final_est._trained_columns = list(X_full_train.columns)
    # Clean OOF: replace any NaNs with median of available OOF preds
    if np.isnan(oof_preds).any():
        mask = ~np.isnan(oof_preds)
        if mask.any():
            fill = float(np.nanmedian(oof_preds[mask]))
        else:
            fill = float(np.nanmedian(y))
        oof_preds[np.isnan(oof_preds)] = fill
    print("\n✅ OOF + Full model training completed successfully.")
    return oof_preds, final_est

# -----------------------
# Train v13 pipeline
# -----------------------
def train_v13(lat=LAT, lon=LON, start=DEFAULT_START, end=DEFAULT_END, lead_days=1, quick=False):
    print(f"HydroFusion v13 — Training {start} → {end} (lat={lat}, lon={lon})")
    df_hourly = fetch_archive_hourly(lat, lon, start, end)
    df_flood = fetch_flood_daily(lat, lon, start, end)
    if df_hourly.empty:
        print("❌ No hourly weather data found. Aborting."); return
    if df_flood.empty:
        print("❌ No daily flood data found. Aborting."); return
    df_daily = build_daily_merge(df_hourly, df_flood)
    if df_daily.empty:
        print("❌ No overlap between weather and flood data. Aborting."); return
    df = engineer_v12(df_daily, lead_days=lead_days)
    df = df.apply(pd.to_numeric, errors='coerce')
    split = int(len(df)*0.8)
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()
    print(f"Train {len(train_df)}, Test {len(test_df)}")
    train_aug = augment_with_surges(train_df, n_augment=400 if not quick else 120)
    train_aug['target'] = train_aug['target'].fillna(0)
    thr = float(np.nanpercentile(train_aug['target'], FLOOD_Q*100))
    sw = np.where(train_aug['target']>=thr, 1.0+PEAK_WEIGHT_ALPHA, 1.0)
    if 'is_synthetic' in train_aug.columns:
        sw *= np.where(train_aug['is_synthetic']==1, 0.6, 1.0)
    print(f"Peak threshold (train {int(FLOOD_Q*100)}th pct) = {thr:.2f}")
    X_train_full, y_train_full = prepare_tabular_Xy(train_aug, FEATURE_LIST)
    X_test_full, y_test_full = prepare_tabular_Xy(test_df, FEATURE_LIST)
    scaler = StandardScaler()
    X_train_real = X_train_full.loc[train_aug['is_synthetic']==0].fillna(0)
    scaler.fit(X_train_real)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train_full.fillna(0)), index=X_train_full.index, columns=X_train_full.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_full.fillna(0)), index=X_test_full.index, columns=X_test_full.columns)
    # Remove near-constant features
    constant_cols = X_train_scaled.columns[X_train_scaled.std() < 1e-6].tolist()
    if constant_cols:
        print(f"Warning: Found constant features, removing them: {constant_cols}")
        X_train_scaled = X_train_scaled.drop(columns=constant_cols)
        X_test_scaled = X_test_scaled.drop(columns=constant_cols, errors='ignore')
    else:
        print("No constant features found in training set.")
    FINAL_TABULAR_FEATURES = X_train_scaled.columns.tolist()
    print(f"Training on {len(FINAL_TABULAR_FEATURES)} features: {FINAL_TABULAR_FEATURES}")

    # LSTM branch
    lstm_oof_train = None; lstm_model_final = None
    if TF_AVAILABLE:
        Xs_train_seq, ys_train_seq, idxs_train_seq = build_sequences(train_aug, seq_len=SEQ_LENGTH, features_seq=SEQ_FEATURES)
        if len(Xs_train_seq) >= 50 and np.isfinite(Xs_train_seq).all() and np.isfinite(ys_train_seq).all():
            n = len(train_aug)
            lstm_oof = np.full(n, np.nan, dtype=float)
            tscv = TimeSeriesSplit(n_splits=TIME_SERIES_SPLITS)
            print("Preparing LSTM sequences and producing OOF predictions (time-safe)...")
            for fold, (train_idx, val_idx) in enumerate(tscv.split(np.arange(len(train_aug)))):
                print(f"LSTM OOF fold {fold+1}/{TIME_SERIES_SPLITS}")
                train_dates = train_aug.index[train_idx]; val_dates = train_aug.index[val_idx]
                mask_train = np.isin(idxs_train_seq, train_dates); mask_val = np.isin(idxs_train_seq, val_dates)
                Xs_tr = Xs_train_seq[mask_train]; ys_tr = ys_train_seq[mask_train]; Xs_val = Xs_train_seq[mask_val]; ys_val = ys_train_seq[mask_val]
                if len(Xs_tr) < 20 or len(Xs_val) < 1:
                    print("  skipping fold (too few sequences)"); continue
                model = build_bidirectional_lstm((Xs_tr.shape[1], Xs_tr.shape[2]), units=64, dropout=0.2)
                epochs = 6 if quick else 20
                callbacks = [EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                             ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5)]
                model.fit(Xs_tr, ys_tr.astype(np.float32), epochs=epochs, batch_size=32, verbose=0,
                          validation_data=(Xs_val, ys_val.astype(np.float32)), callbacks=callbacks)
                preds_val = model.predict(Xs_val).reshape(-1)
                val_dates_seq = idxs_train_seq[mask_val]
                for d,p in zip(val_dates_seq, preds_val):
                    pos = np.where(train_aug.index == d)[0]
                    if pos.size: lstm_oof[pos[0]] = p
            nan_mask = np.isnan(lstm_oof)
            if nan_mask.any():
                fill = np.nanmedian(lstm_oof[~nan_mask]) if (~nan_mask).any() else np.nanmedian(ys_train_seq)
                lstm_oof[nan_mask] = fill if np.isfinite(fill) else 0.0
            lstm_oof_train = pd.Series(lstm_oof, index=train_aug.index)
            final_lstm = build_bidirectional_lstm((Xs_train_seq.shape[1], Xs_train_seq.shape[2]), units=64, dropout=0.2)
            final_epochs = 8 if quick else 40
            print("Training final LSTM model...")
            final_lstm.fit(Xs_train_seq, ys_train_seq.astype(np.float32), epochs=final_epochs, batch_size=32, verbose=0)
            lstm_model_final = final_lstm
        else:
            print("Not enough sequence samples or bad data; skipping LSTM branch.")
    else:
        print("TensorFlow not available; skipping LSTM branch.")

    # Add LSTM oof as a feature if successful
    if lstm_oof_train is not None:
        print("Adding LSTM predictions as a feature.")
        X_train_scaled['lstm_oof'] = lstm_oof_train
        combined_df = pd.concat([train_aug.tail(SEQ_LENGTH), test_df])
        Xs_test_seq, _, idxs_test_seq = build_sequences(combined_df, seq_len=SEQ_LENGTH, features_seq=SEQ_FEATURES)
        if len(Xs_test_seq) > 0 and lstm_model_final is not None:
            preds_test_seq = lstm_model_final.predict(Xs_test_seq).reshape(-1)
            test_map = {d: p for d,p in zip(idxs_test_seq, preds_test_seq) if d in test_df.index}
            lstm_oof_test = pd.Series([test_map.get(d, np.nan) for d in X_test_scaled.index], index=X_test_scaled.index)
            lstm_oof_test = lstm_oof_test.fillna(np.nanmedian(preds_test_seq) if len(preds_test_seq) > 0 else 0.0)
            X_test_scaled['lstm_oof'] = lstm_oof_test
        else:
            print("No test sequences generated for LSTM.")
            X_test_scaled['lstm_oof'] = 0.0
        if 'lstm_oof' not in FINAL_TABULAR_FEATURES:
            FINAL_TABULAR_FEATURES.append('lstm_oof')
    else:
        print("Skipping LSTM feature in final models.")

    # Align test columns with train columns (fill missing columns with 0)
    X_test_scaled = X_test_scaled.reindex(columns=X_train_scaled.columns, fill_value=0)
    X_train_scaled = X_train_scaled[FINAL_TABULAR_FEATURES]
    X_test_scaled = X_test_scaled[FINAL_TABULAR_FEATURES]

    # ---- Base models OOF ----
    models_to_run = {}
    if LGB_AVAILABLE:
        print("LGBM model enabled.")
        models_to_run["lgb"] = lambda: lgb.LGBMRegressor(
            n_estimators=400 if not quick else 120, learning_rate=0.05, num_leaves=31,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
    if XGB_AVAILABLE:
        print("XGBoost model enabled.")
        models_to_run["xgb"] = lambda: xgb.XGBRegressor(
            n_estimators=400 if not quick else 120, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        )
    if CAT_AVAILABLE:
        print("CatBoost model enabled.")
        models_to_run["cat"] = lambda: CatBoostRegressor(
            iterations=400 if not quick else 150, learning_rate=0.05, depth=6, random_state=RANDOM_STATE, verbose=0, thread_count=-1
        )
    if not models_to_run:
        raise EnvironmentError("No GBDT models installed (LightGBM, XGBoost, CatBoost).")

    oof_preds = {}
    full_models = {}
    fit_params = {'sample_weight': sw}
    # train on log1p target for better numerical stability
    y_train_fit = np.log1p(y_train_full.values)

    for name, factory in models_to_run.items():
        print(f"Training base model: {name}")
        oof, full_model = oof_preds_time_series(factory, X_train_scaled, pd.Series(y_train_fit, index=X_train_scaled.index), n_splits=TIME_SERIES_SPLITS, fit_params=fit_params)
        # oof may be on log-scale; inverse-transform (ensure no NaNs)
        if np.isnan(oof).any():
            mask = ~np.isnan(oof)
            if mask.any():
                oof[np.isnan(oof)] = np.nanmedian(oof[mask])
            else:
                oof[:] = np.median(y_train_fit)
        oof_preds[name] = np.expm1(oof)
        # ensure _trained_columns is present on model (oof function sets it)
        if not hasattr(full_model, '_trained_columns'):
            full_model._trained_columns = X_train_scaled.columns.tolist()
        full_models[name] = full_model

    meta_train = pd.DataFrame(oof_preds, index=train_aug.index)

    # Test predictions from full models (use only columns each model was trained with)
    base_test = pd.DataFrame(index=X_test_scaled.index)
    for name, model in full_models.items():
        train_cols = getattr(model, '_trained_columns', X_train_scaled.columns.tolist())
        # safe reindex to the trained columns, fill missing with zeros
        X_for_pred = X_test_scaled.reindex(columns=train_cols, fill_value=0)
        preds_log = model.predict(X_for_pred)
        base_test[name] = np.expm1(preds_log)

    # add lstm if present
    if 'lstm_oof' in X_test_scaled.columns:
        base_test['lstm'] = X_test_scaled['lstm_oof']

    # ---- Meta learner ----
    print("Training meta-learner (BayesianRidge) with peak-weighted fit...")
    final_meta_cols = list(base_test.columns)
    # if some base model missing in meta_train, ensure columns align (fill missing cols with 0)
    meta_train = meta_train.reindex(columns=final_meta_cols, fill_value=0)
    meta_X = meta_train.values
    meta_y = y_train_full.values
    meta_sw = np.where(meta_y >= thr, 1.0 + PEAK_WEIGHT_ALPHA, 1.0)
    if 'is_synthetic' in train_aug.columns:
        meta_sw *= np.where(train_aug['is_synthetic']==1, 0.6, 1.0)
    meta_model = BayesianRidge()
    meta_model.fit(meta_X, meta_y, sample_weight=meta_sw)
    stack_test_pred = meta_model.predict(base_test[final_meta_cols].fillna(0).values)

    # ---- Peak calibrator (HuberRegressor) ----
    calibrator = None
    high_mask_train = y_train_full >= thr
    if high_mask_train.sum() >= 8:
        print("Training peak calibrator...")
        calibrator = HuberRegressor()
        calibrator_X = meta_train.loc[high_mask_train, final_meta_cols].mean(axis=1).values.reshape(-1,1)
        calibrator_y = meta_y[high_mask_train]
        valid_cal_mask = ~np.isnan(calibrator_X).flatten() & ~np.isnan(calibrator_y)
        if valid_cal_mask.sum() >= 2:
            calibrator.fit(calibrator_X[valid_cal_mask], calibrator_y[valid_cal_mask])
            stack_test_avg = base_test[final_meta_cols].mean(axis=1).values.reshape(-1,1)
            high_mask_test = stack_test_avg.flatten() >= thr
            if high_mask_test.sum() > 0:
                print(f"Applying calibration to {high_mask_test.sum()} high-flow test predictions...")
                cal_preds = calibrator.predict(stack_test_avg[high_mask_test])
                stack_test_pred[high_mask_test] = cal_preds
        else:
            print("⚠️ Not enough valid high-flow samples for calibration.")
    else:
        print("⚠️ Not enough high-flow samples for calibration.")

    # ---- Evaluation ----
    y_test_vals = y_test_full.values
    def eval_metrics(y_t, y_p, thr):
        mae = mean_absolute_error(y_t, y_p)
        rmse = math.sqrt(mean_squared_error(y_t, y_p))
        r2v = r2_score(y_t, y_p)
        peak_mask = y_t >= thr
        r2_peak = r2_score(y_t[peak_mask], y_p[peak_mask]) if peak_mask.sum() >= 2 else float('nan')
        return {'mae': mae, 'rmse': rmse, 'r2': r2v, 'r2_peak': r2_peak, 'thr': thr}

    base_metrics = {}
    if 'lgb' in full_models: base_metrics['lgb'] = eval_metrics(y_test_vals, base_test['lgb'].values, thr)
    if 'xgb' in full_models: base_metrics['xgb'] = eval_metrics(y_test_vals, base_test['xgb'].values, thr)
    if 'cat' in full_models: base_metrics['cat'] = eval_metrics(y_test_vals, base_test['cat'].values, thr)
    if 'lstm' in base_test.columns: base_metrics['lstm'] = eval_metrics(y_test_vals, base_test['lstm'].values, thr)

    final_metrics = eval_metrics(y_test_vals, stack_test_pred, thr)
    print("\n--- Base Model Metrics (Test Set) ---"); print(pd.DataFrame(base_metrics).T)
    print("\n--- Final Stacked Model Metrics (Test Set) ---"); print(pd.Series(final_metrics).to_frame('Score'))

    # ---- SHAP ----
    if SHAP_AVAILABLE and 'lgb' in full_models:
        try:
            print("\nComputing SHAP values for LightGBM...")
            lgb_model = full_models['lgb']
            shap_cols = getattr(lgb_model, '_trained_columns', X_train_scaled.columns.tolist())
            X_test_shap = X_test_scaled.reindex(columns=shap_cols, fill_value=0)
            explainer = shap.TreeExplainer(lgb_model)
            shap_values = explainer(X_test_shap)
            plt.figure(figsize=(10,8)); shap.summary_plot(shap_values, X_test_shap, show=False, plot_type="bar")
            plt.tight_layout(); plt.savefig(PLOT_SHAP); plt.close()
            print(f"Saved SHAP summary: {PLOT_SHAP}")
        except Exception as e:
            print(f"SHAP failed: {e}")

    # ---- Save bundle ----
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        'models': full_models,
        'meta_model': meta_model,
        'calibrator': calibrator,
        'scaler': scaler,
        'features': FINAL_TABULAR_FEATURES,
        'seq_features': SEQ_FEATURES,
        'seq_length': SEQ_LENGTH,
        'trained_range': {'start': start, 'end': end}
    }
    joblib.dump(bundle, os.path.join(MODEL_DIR, MODEL_FILE))
    print(f"\nSaved model bundle to {os.path.join(MODEL_DIR, MODEL_FILE)}")

    # ---- Predictions CSV & Plots ----
    idx = test_df.index
    out_df = pd.DataFrame({'date': idx, 'observed': y_test_full.values, 'predicted': stack_test_pred})
    for k in ['lgb','xgb','cat','lstm']:
        if k in base_test:
            out_df[f'{k}_pred'] = base_test[k].values
    out_df.to_csv(PRED_CSV, index=False)
    print(f"Saved predictions to {PRED_CSV}")

    plt.figure(figsize=(14,6))
    plt.plot(idx, y_test_full.values, label='Observed', linewidth=1.3)
    plt.plot(idx, stack_test_pred, '--', label='Final Pred', linewidth=1.3)
    plt.axhline(thr, color='orange', linestyle=':', label=f'{int(FLOOD_Q*100)}th Pct ({thr:.2f})')
    plt.legend(); plt.grid(True); plt.title('v13 Backtest - Observed vs Final Prediction'); plt.savefig(PLOT_BACKTEST); plt.close()
    print(f"Saved backtest plot to {PLOT_BACKTEST}")

    plot_obs_vs_pred(y_test_full.values, stack_test_pred)
    try:
        plot_model_correlation(base_test[final_meta_cols])
    except Exception as e:
        print(f"Could not plot model correlation: {e}")

    thr_mask = y_test_full.values >= thr
    if thr_mask.sum() > 0:
        plt.figure(figsize=(12,5))
        plt.plot(idx[thr_mask], y_test_full.values[thr_mask], 'o-', label='Observed peaks')
        plt.plot(idx[thr_mask], stack_test_pred[thr_mask], 'x--', label='Predicted peaks')
        plt.legend(); plt.grid(True); plt.title('v13 Peaks'); plt.savefig(PLOT_PEAKS); plt.close()
        print(f"Saved peaks plot to {PLOT_PEAKS}")

    if calibrator is not None:
        high_idx = np.where(y_test_full.values >= thr)[0]
        if high_idx.size:
            plt.figure(figsize=(6,6))
            plt.scatter(stack_test_pred[high_idx], y_test_full.values[high_idx], alpha=0.6)
            plt.plot([y_test_full.values[high_idx].min(), y_test_full.values[high_idx].max()],
                     [y_test_full.values[high_idx].min(), y_test_full.values[high_idx].max()], 'k--')
            plt.xlabel('Ensemble pred'); plt.ylabel('Observed'); plt.title('Peak Calibration (test)')
            plt.grid(True); plt.savefig(PLOT_CAL); plt.close()
            print(f"Saved calibration plot to {PLOT_CAL}")

    return {'metrics': final_metrics, 'base_metrics': base_metrics}

# -----------------------
# CLI
# -----------------------
def parse_args():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train'], default='train')
    p.add_argument('--lat', type=float, default=LAT)
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
    res = train_v13(args.lat, args.lon, args.start, args.end, lead_days=args.lead, quick=args.quick)
    print("✅ Training complete:", res)
