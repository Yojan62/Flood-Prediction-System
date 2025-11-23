#!/usr/bin/env python3
"""
flood_model_v15.py — HydroFusion v15 (Two-Stage Hybrid: Tree baseflow + LSTM residuals)

What’s new vs v14_1:
- Stage 1: LightGBM trained on log1p(target) for baseflow. Predictions are expm1.
- Stage 2: BiLSTM trained to predict residuals (observed - baseflow). Residual target is
  standardized for stable training and de-standardized before combining.
- Final pred = baseflow_pred + residual_pred.
- Robust sequence/date alignment, augmentation, SHAP (on LGBM), same plots/CSV/CLI.

Usage:
    python flood_model_v15.py --mode train [--quick]
"""

import os, time, math, json, argparse, warnings
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge  # kept if you want to test meta later

# Optional libs
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

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
OPEN_METEO_FLOOD   = "https://flood-api.open-meteo.com/v1/flood"

MODEL_DIR = "ml"
MODEL_FILE = "open_meteo_flood_model_v15.pkl"
FEATURES_JSON = "open_meteo_flood_features_v15.json"
PLOT_BACKTEST = "v15_backtest.png"
PLOT_PEAKS = "v15_backtest_peaks.png"
PLOT_SHAP = "v15_shap_summary.png"
PLOT_OBS_PRED = "v15_obs_vs_pred.png"
PRED_CSV = "v15_test_predictions.csv"
PLOT_RESID_TRAIN = "v15_residual_fit_train.png"

RANDOM_STATE = 42
FLOOD_Q = 0.90
PEAK_WEIGHT_ALPHA = 14.0
TIME_SERIES_SPLITS = 4
SEQ_LENGTH = 14

# Tabular features
FEATURE_LIST = [
    'discharge_m3_s','dis_lag1','dis_lag2','dis_lag3','dis_rate',
    'rain_lag_1','rain_lag_2','rain_lag_3','rain_lag_7','rain_lag_14',
    'rain_roll_3','rain_roll_7','rain_roll_14','api',
    'rainfall_forecast_1d',
    'max_hourly_rain_mm',
    'evapotranspiration','et_lag_1','temperature',
    'month_sin','month_cos','rain_grad_1_2'
]

# LSTM sequence features for residuals (expanded to capture short/medium responses)
SEQ_FEATURES = [
    'discharge_m3_s','dis_rate','rain_roll_3','rain_roll_7','api','rain_lag_1','rain_lag_3','max_hourly_rain_mm'
]

ARGS = None

# -----------------------
# Utils
# -----------------------
def safe_get_json(url, params, timeout=90):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def metrics_with_peak(y_true, y_pred, thr):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2v = r2_score(y_true, y_pred)
    peak_mask = y_true >= thr
    r2_peak = r2_score(y_true[peak_mask], y_pred[peak_mask]) if peak_mask.sum() >= 2 else float('nan')
    return {'mae': mae, 'rmse': rmse, 'r2': r2v, 'r2_peak': r2_peak, 'thr': thr}

def plot_obs_vs_pred(y_true, y_pred, title="Observed vs Predicted", out_file=PLOT_OBS_PRED):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot([lo, hi],[lo, hi], 'r--')
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Saved observed vs predicted scatter: {out_file}")

# -----------------------
# Fetch
# -----------------------
def fetch_archive_hourly(lat, lon, start_date, end_date):
    start = pd.to_datetime(start_date).date()
    end   = pd.to_datetime(end_date).date()
    today = date.today()
    if end > today:
        print(f"Warning: requested end_date {end} is in the future. Clipping to {today}.")
        end = today
    dfs = []
    year = start.year
    while year <= end.year:
        seg_start = max(start, date(year, 1, 1))
        seg_end   = min(end, date(year, 12, 31))
        print(f"→ Fetching archive {seg_start} → {seg_end}")
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
        df_hourly['rainfall_mm'] = 0.0
        df_hourly['max_hourly_rain_mm'] = 0.0
    return df_hourly

def fetch_flood_daily(lat, lon, start_date, end_date):
    start = pd.to_datetime(start_date).date()
    end   = pd.to_datetime(end_date).date()
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

# -----------------------
# Features
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
            df_hourly[col] = 0.0
    df_daily = df_hourly.resample('D').agg(agg_ops)
    df_daily.index.name = 'date'
    df = pd.merge(df_flood, df_daily, left_index=True, right_index=True, how='inner')
    print(f"Total merged daily rows: {len(df)}")
    return df

def engineer(df_daily, lead_days=1, api_k=0.85):
    print("Engineering features...")
    d = df_daily.copy().sort_index()
    for c in ['discharge_m3_s','rainfall_mm','soil_moisture','evapotranspiration','temperature','max_hourly_rain_mm']:
        if c not in d.columns:
            d[c] = 0.0

    d['dis_lag1'] = d['discharge_m3_s'].shift(1)
    d['dis_lag2'] = d['discharge_m3_s'].shift(2)
    d['dis_lag3'] = d['discharge_m3_s'].shift(3)
    d['dis_rate'] = d['discharge_m3_s'] - d['dis_lag1']

    for i in [1,2,3,7,14]:
        d[f'rain_lag_{i}'] = d['rainfall_mm'].shift(i)
    d['rain_roll_3']  = d['rainfall_mm'].shift(1).rolling(3).sum()
    d['rain_roll_7']  = d['rainfall_mm'].shift(1).rolling(7).sum()
    d['rain_roll_14'] = d['rainfall_mm'].shift(1).rolling(14).sum()
    d['rain_grad_1_2'] = d['rain_lag_1'] - d['rain_lag_2']
    d['et_lag_1'] = d['evapotranspiration'].shift(1)

    WINDOW = 7
    weights = np.power(api_k, np.arange(WINDOW))
    d['api'] = d['rainfall_mm'].shift(1).rolling(window=WINDOW).apply(
        lambda x: np.sum(np.asarray(x) * weights[::-1]), raw=False
    )

    d['month'] = d.index.month
    d['month_sin'] = np.sin(2*np.pi*d['month']/12)
    d['month_cos'] = np.cos(2*np.pi*d['month']/12)

    d['rainfall_forecast_1d'] = d['rainfall_mm'].shift(-lead_days)
    d['target'] = d['discharge_m3_s'].shift(-lead_days)

    d = d.dropna(subset=['target','dis_lag3','rain_lag_14','api'])
    return d

def augment_with_surges(df, n_augment=300, max_extra_mm=140.0, runoff_coeff=0.5):
    print("Augmenting training data with synthetic surges...")
    rng = np.random.default_rng(RANDOM_STATE)
    df_aug = df.copy()
    idxs = df.index.values
    new_rows = []
    for i in range(n_augment):
        base_idx = rng.choice(idxs[len(idxs)//4:])
        row = df.loc[base_idx].copy()
        extra_total = rng.uniform(8.0, max_extra_mm)
        days = rng.integers(1, 4)
        per_day = extra_total / days
        for d_i in range(1, days+1):
            col = f'rain_lag_{d_i}' if f'rain_lag_{d_i}' in row.index else 'rain_lag_1'
            row[col] = (row.get(col, 0.0) or 0.0) + per_day
        row['rain_roll_3'] = (row.get('rain_roll_3',0) or 0) + extra_total
        row['api'] = (row.get('api',0) or 0) + extra_total
        row['max_hourly_rain_mm'] = max(row.get('max_hourly_rain_mm',0), per_day)
        row['target'] = row['target'] + runoff_coeff * extra_total
        row['is_synthetic'] = 1
        row.name = base_idx + pd.Timedelta(nanoseconds=i+1)
        new_rows.append(row)
    if new_rows:
        df_aug = pd.concat([df_aug, pd.DataFrame(new_rows)], ignore_index=False)
    df_aug['is_synthetic'] = df_aug.get('is_synthetic', 0)
    df_aug['is_synthetic'] = df_aug['is_synthetic'].fillna(0).astype(int)
    df_aug = df_aug.sort_index()
    print(f"After augmentation: {len(df_aug)} rows")
    return df_aug

def prepare_tabular_Xy(df):
    X = df.reindex(columns=FEATURE_LIST).apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    y = df['target'].astype(np.float32)
    return X, y

# -----------------------
# Sequences
# -----------------------
def build_sequences(df, seq_len=SEQ_LENGTH, features_seq=SEQ_FEATURES, target_col='target'):
    """
    Returns sequences aligned by df.index for the specified target column.
    """
    print(f"Building sequences (SEQ_LENGTH={seq_len}) for target='{target_col}'...")
    d = df.copy().sort_index()
    for f in features_seq:
        if f not in d.columns:
            d[f] = 0.0
    arr = d[features_seq].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    targets = d[target_col].astype(np.float32).values
    Xs, ys, idxs = [], [], []
    for i in range(seq_len, len(d)):
        seq = arr[i-seq_len:i]
        if not np.isfinite(seq).all() or not np.isfinite(targets[i]):
            continue
        Xs.append(seq)
        ys.append(targets[i])
        idxs.append(d.index[i])
    Xs = np.array(Xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    idxs = np.array(idxs)
    print("Sequence shapes:", Xs.shape, ys.shape)
    return Xs, ys, idxs

def build_bilstm(input_shape, units=64, dropout=0.2):
    inp = Input(shape=input_shape)
    x = Bidirectional(LSTM(units, return_sequences=True))(inp)
    avgp = GlobalAveragePooling1D()(x)
    maxp = GlobalMaxPooling1D()(x)
    concat = Concatenate()([avgp, maxp])
    y = Dropout(dropout)(concat)
    out = Dense(1, activation='linear')(y)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

# -----------------------
# OOF helper (log-target for tree baseflow)
# -----------------------
def oof_time_series_tree(est_factory, X, y_log, n_splits=4, sample_weight=None):
    print("\nTree OOF (log-target) ...")
    oof_preds_log = np.full(len(X), np.nan, dtype=float)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold, (tr, va) in enumerate(tscv.split(X), 1):
        print(f"  fold {fold}/{n_splits}: train {len(tr)} → val {len(va)}")
        est = est_factory()
        fit_kw = {}
        if sample_weight is not None and len(sample_weight) == len(X):
            fit_kw['sample_weight'] = np.asarray(sample_weight)[tr]
        X_tr, y_tr = X.iloc[tr], y_log.iloc[tr]
        X_va = X.iloc[va]
        # remove constant columns in this fold
        const_cols = X_tr.columns[X_tr.std() == 0]
        if len(const_cols):
            X_tr = X_tr.drop(columns=const_cols)
            X_va = X_va[X_tr.columns]
        est.fit(X_tr, y_tr, **fit_kw)
        oof_preds_log[va] = est.predict(X_va)
    # final fit on all (use de-constantized)
    est_final = est_factory()
    const_full = X.columns[X.std() == 0]
    X_full = X.drop(columns=const_full) if len(const_full) else X
    if sample_weight is not None and len(sample_weight) == len(X):
        est_final.fit(X_full, y_log, sample_weight=sample_weight)
    else:
        est_final.fit(X_full, y_log)
    trained_cols = list(X_full.columns)
    return oof_preds_log, est_final, trained_cols

# -----------------------
# Train (Two-Stage)
# -----------------------
def train_v15(lat=LAT, lon=LON, start=DEFAULT_START, end=DEFAULT_END, lead_days=1, quick=False):
    print(f"HydroFusion v15 — Training {start} → {end} (lat={lat}, lon={lon})")
    # fetch
    df_hourly = fetch_archive_hourly(lat, lon, start, end)
    df_flood  = fetch_flood_daily(lat, lon, start, end)
    df_daily  = build_daily_merge(df_hourly, df_flood)

    # engineer
    df = engineer(df_daily, lead_days=lead_days).apply(pd.to_numeric, errors='coerce')
    split = int(len(df)*0.8)
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()
    print(f"Train {len(train_df)}, Test {len(test_df)}")

    # augment
    train_aug = augment_with_surges(train_df, n_augment=(180 if quick else 300))
    train_aug['target'] = train_aug['target'].fillna(0)
    thr = np.nanpercentile(train_aug['target'].values, int(FLOOD_Q*100))
    print(f"Peak threshold (train {int(FLOOD_Q*100)}th pct) = {thr:.2f}")

    # weights (on original scale)
    sw = np.where(train_aug['target'] >= thr, 1.0 + PEAK_WEIGHT_ALPHA, 1.0)
    if 'is_synthetic' in train_aug.columns:
        sw *= np.where(train_aug['is_synthetic']==1, 0.7, 1.0)

    # tabular + scale
    X_train_tab, y_train = prepare_tabular_Xy(train_aug)
    X_test_tab,  y_test  = prepare_tabular_Xy(test_df)

    scaler = StandardScaler()
    scaler.fit(X_train_tab.loc[train_aug['is_synthetic']==0])
    X_train = pd.DataFrame(scaler.transform(X_train_tab), index=X_train_tab.index, columns=X_train_tab.columns)
    X_test  = pd.DataFrame(scaler.transform(X_test_tab),  index=X_test_tab.index,  columns=X_test_tab.columns)

    # --------------- STAGE 1: LightGBM baseflow on log target ---------------
    if not LGB_AVAILABLE:
        raise RuntimeError("LightGBM is required for v15.")

    y_train_log = np.log1p(y_train.clip(lower=0))

    def lgb_factory():
        return lgb.LGBMRegressor(
            n_estimators=(260 if quick else 520),
            learning_rate=0.05,
            num_leaves=48,
            min_child_samples=12,
            subsample=0.85,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )

    oof_log, lgb_model, trained_cols = oof_time_series_tree(
        lgb_factory, X_train, y_train_log, n_splits=TIME_SERIES_SPLITS, sample_weight=sw
    )
    oof_base = np.expm1(oof_log)                          # baseflow OOF on train
    X_test_lgb = X_test[trained_cols]
    base_test_pred = np.expm1(lgb_model.predict(X_test_lgb))  # baseflow on test

    # --------------- Residuals ---------------
    # residual target for sequences (on train_aug): r = y - baseflow_oof
    residual_train = pd.Series(y_train.values - oof_base, index=train_aug.index, name='residual')
    # Standardize residual target for stable LSTM training
    resid_mu = float(residual_train.mean())
    resid_sd = float(residual_train.std() if residual_train.std() > 1e-6 else 1.0)
    residual_train_std = (residual_train - resid_mu) / resid_sd

    train_aug_resid = train_aug.copy()
    train_aug_resid['residual_std'] = residual_train_std

    # --------------- STAGE 2: BiLSTM on residuals (standardized) ---------------
    lstm_resid_test_std = None
    lstm_model = None
    if TF_AVAILABLE:
        # Build sequences to predict residual_std
        Xs_res, ys_res_std, idxs_res = build_sequences(
            train_aug_resid, SEQ_LENGTH, SEQ_FEATURES, target_col='residual_std'
        )
        if len(Xs_res) >= 80 and np.isfinite(Xs_res).all() and np.isfinite(ys_res_std).all():
            # Fit on all residual sequences with validation split from tail by dates
            # or use explicit TimeSeriesSplit for early stopping feedback:
            tscv = TimeSeriesSplit(n_splits=TIME_SERIES_SPLITS)
            best_model = None
            best_val = float('inf')
            for fold, (tr, va) in enumerate(tscv.split(idxs_res), 1):
                print(f"LSTM-residual CV fold {fold}/{TIME_SERIES_SPLITS}")
                tr_dates = idxs_res[tr]; va_dates = idxs_res[va]
                m_tr = np.isin(idxs_res, tr_dates)
                m_va = np.isin(idxs_res, va_dates)
                X_tr, y_tr = Xs_res[m_tr], ys_res_std[m_tr]
                X_va, y_va = Xs_res[m_va], ys_res_std[m_va]
                if len(X_tr) < 40 or len(X_va) < 5:
                    print("  skipping fold (too few sequences)")
                    continue
                model = build_bilstm((X_tr.shape[1], X_tr.shape[2]), units=(48 if quick else 64), dropout=0.2)
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=(4 if quick else 6), restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', patience=(2 if quick else 3), factor=0.5)
                ]
                model.fit(X_tr, y_tr, epochs=(10 if quick else 30), batch_size=32, verbose=0,
                          validation_data=(X_va, y_va), callbacks=callbacks)
                val_pred = model.predict(X_va, verbose=0).reshape(-1)
                val_rmse = math.sqrt(mean_squared_error(y_va, val_pred))
                if val_rmse < best_val:
                    best_val = val_rmse
                    best_model = model
            if best_model is None:
                # fall back: single fit on all data
                best_model = build_bilstm((Xs_res.shape[1], Xs_res.shape[2]), units=(48 if quick else 64), dropout=0.2)
                best_model.fit(Xs_res, ys_res_std, epochs=(12 if quick else 40), batch_size=32, verbose=0)
            lstm_model = best_model

            # Quick diagnostic: residual fit on train (with a held-out tail)
            try:
                tail_n = max(30, len(ys_res_std)//10)
                pred_tail_std = lstm_model.predict(Xs_res[-tail_n:], verbose=0).reshape(-1)
                pred_tail = pred_tail_std * resid_sd + resid_mu
                true_tail = ys_res_std[-tail_n:] * resid_sd + resid_mu
                plt.figure(figsize=(12,4))
                plt.plot(true_tail, label='True residual (tail)')
                plt.plot(pred_tail, '--', label='Pred residual (tail)')
                plt.legend(); plt.grid(True); plt.title('Residual fit (train tail)')
                plt.tight_layout(); plt.savefig(PLOT_RESID_TRAIN); plt.close()
                print(f"Saved residual training diagnostic: {PLOT_RESID_TRAIN}")
            except Exception as e:
                print(f"Residual diagnostic plot skipped: {e}")

            # Predict residuals for test:
            combo = pd.concat([train_aug.tail(SEQ_LENGTH), test_df])
            Xs_te, _, idxs_te = build_sequences(combo, SEQ_LENGTH, SEQ_FEATURES, target_col='target')
            if len(Xs_te) > 0:
                resid_te_std = lstm_model.predict(Xs_te, verbose=0).reshape(-1)  # standardized residuals
                # align by date to test index
                mp = {d: p for d, p in zip(idxs_te, resid_te_std)}
                lstm_resid_test_std = np.array([mp.get(d, np.nan) for d in X_test.index])
                # fill any NaNs with median
                if np.isnan(lstm_resid_test_std).any():
                    fill = np.nanmedian(lstm_resid_test_std[~np.isnan(lstm_resid_test_std)]) if np.isfinite(lstm_resid_test_std[~np.isnan(lstm_resid_test_std)]).any() else 0.0
                    lstm_resid_test_std = np.where(np.isnan(lstm_resid_test_std), fill, lstm_resid_test_std)
        else:
            print("Residual LSTM skipped: not enough valid sequences.")
    else:
        print("TensorFlow not available; skipping residual LSTM.")

    # --------------- Combine: baseflow + residual ---------------
    if lstm_resid_test_std is not None:
        resid_test = lstm_resid_test_std * resid_sd + resid_mu
        final_test_pred = base_test_pred + resid_test
    else:
        final_test_pred = base_test_pred

    # --------------- Evaluation ---------------
    y_test_vals = y_test.values
    base_metrics = {
        'lgb_base': metrics_with_peak(y_test_vals, base_test_pred, thr)
    }
    if lstm_resid_test_std is not None:
        base_metrics['lstm_residual'] = metrics_with_peak(y_test_vals, final_test_pred, thr)

    final_metrics = metrics_with_peak(y_test_vals, final_test_pred, thr)

    print("\n--- Base & Residual Metrics (Test) ---")
    print(pd.DataFrame(base_metrics).T)
    print("\n--- Final Hybrid Metrics (Test) ---")
    print(pd.Series(final_metrics).to_frame('Score'))

    # --------------- SHAP (on LGBM baseflow) ---------------
    if SHAP_AVAILABLE:
        try:
            print("Computing SHAP values for LightGBM (baseflow)...")
            X_test_shap = X_test_lgb
            explainer = shap.TreeExplainer(lgb_model)
            shap_values = explainer(X_test_shap)
            plt.figure(figsize=(10,6))
            shap.summary_plot(shap_values, X_test_shap, show=False, plot_type="bar")
            plt.tight_layout(); plt.savefig(PLOT_SHAP); plt.close()
            print(f"Saved SHAP summary: {PLOT_SHAP}")
        except Exception as e:
            print(f"SHAP failed: {e}")

    # --------------- Save artifacts ---------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        'lgb_model': lgb_model,
        'trained_cols': trained_cols,
        'scaler': scaler,
        'features_tab': FEATURE_LIST,
        'seq_features': SEQ_FEATURES,
        'seq_length': SEQ_LENGTH,
        'lstm_residual_model': lstm_model,
        'resid_mu': resid_mu,
        'resid_sd': resid_sd,
        'trained_range': {'start': start, 'end': end}
    }
    joblib.dump(bundle, os.path.join(MODEL_DIR, MODEL_FILE))
    with open(os.path.join(MODEL_DIR, FEATURES_JSON), 'w') as f:
        json.dump({'features': FEATURE_LIST, 'seq_features': SEQ_FEATURES, 'lead_days': lead_days}, f, indent=2)
    print(f"Saved model bundle to {os.path.join(MODEL_DIR, MODEL_FILE)}")

    # predictions CSV
    out = {
        'date': test_df.index,
        'observed': y_test_vals,
        'baseflow_pred': base_test_pred,
        'final_pred': final_test_pred
    }
    if lstm_resid_test_std is not None:
        out['residual_pred'] = resid_test
    out_df = pd.DataFrame(out, index=test_df.index)
    out_df.to_csv(PRED_CSV, index=False)
    print(f"Saved predictions to {PRED_CSV}")

    # plots
    idx = test_df.index
    plt.figure(figsize=(14,6))
    plt.plot(idx, y_test_vals, label='Observed', linewidth=1.2)
    plt.plot(idx, base_test_pred, ':', label='Baseflow (LGBM)', linewidth=1.2)
    plt.plot(idx, final_test_pred, '--', label='Final Hybrid', linewidth=1.4)
    plt.axhline(thr, color='orange', linestyle=':', label=f'{int(FLOOD_Q*100)}th pct ({thr:.2f})')
    plt.legend(); plt.grid(True); plt.title('v15 Backtest — Observed vs Final Hybrid Prediction')
    plt.tight_layout(); plt.savefig(PLOT_BACKTEST); plt.close()
    print(f"Saved backtest plot to {PLOT_BACKTEST}")

    plot_obs_vs_pred(y_test_vals, final_test_pred)

    thr_mask = y_test_vals >= thr
    if thr_mask.sum() > 0:
        plt.figure(figsize=(12,5))
        plt.plot(idx[thr_mask], y_test_vals[thr_mask], 'o-', label='Observed peaks')
        plt.plot(idx[thr_mask], final_test_pred[thr_mask], 'x--', label='Predicted peaks (hybrid)')
        plt.legend(); plt.grid(True); plt.title('v15 Peaks')
        plt.tight_layout(); plt.savefig(PLOT_PEAKS); plt.close()
        print(f"Saved peaks plot to {PLOT_PEAKS}")

    return {'final_metrics': final_metrics, 'base_metrics': base_metrics}

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
    args = parse_args()
    res = train_v15(args.lat, args.lon, args.start, args.end, lead_days=args.lead, quick=args.quick)
    print("✅ Training v15 complete:", res)
