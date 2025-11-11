#!/usr/bin/env python3
"""
HydroFusion v16 — Peak-Aware Hybrid (based on v15, with fixes)

What’s new vs v15
- Uses the real Open-Meteo + Flood API pipeline (no dummy loader)
- Peak-aware residual LSTM via sample weights (correctly tied to discharge, not residuals)
- Residual standardization (train mean/std), invert at inference
- Robust sequence building & safe Keras prediction to avoid empty-batch errors
- Cleaner plots & saved artifacts; clearer logs; constant-feature removal

Outputs:
  - ml/open_meteo_flood_model_v16.pkl         (model bundle)
  - v16_test_predictions.csv                  (CSV)
  - v16_backtest.png                          (line plot: observed vs final)
  - v16_obs_vs_pred.png                       (scatter: observed vs final)
  - v16_backtest_peaks.png                    (peaks-only lines)
  - v16_shap_summary.png                      (bar summary for LightGBM base)
"""

import os, time, math, json, argparse, warnings
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

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
OPEN_METEO_FLOOD = "https://flood-api.open-meteo.com/v1/flood"

MODEL_DIR = "ml"
MODEL_FILE = "open_meteo_flood_model_v16.pkl"
FEATURES_JSON = "open_meteo_flood_features_v16.json"
PLOT_BACKTEST = "v16_backtest.png"
PLOT_PEAKS = "v16_backtest_peaks.png"
PLOT_SHAP = "v16_shap_summary.png"
PLOT_OBS_PRED = "v16_obs_vs_pred.png"
PRED_CSV = "v16_test_predictions.csv"

RANDOM_STATE = 42
FLOOD_Q = 0.90
TIME_SERIES_SPLITS = 4
SEQ_LENGTH = 14
PEAK_ALPHA = 3.0     # how much extra weight to give to peaks in residual LSTM

# -----------------------
# Utilities
# -----------------------
def safe_get_json(url, params, timeout=90):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def metrics_summary(y_true, y_pred, thr_q=FLOOD_Q):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    thr = np.nanpercentile(y_true, thr_q * 100)
    mask = y_true >= thr
    out['r2_peak'] = r2_score(y_true[mask], y_pred[mask]) if mask.sum() >= 2 else float('nan')
    out['thr'] = float(thr)
    return out

def safe_predict(model, X, batch_size=32):
    """
    Robust TF/Keras predict that avoids empty-batch errors.
    Falls back to model(X, training=False) if needed.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available for safe_predict.")
    X = np.asarray(X)
    if X.shape[0] == 0:
        return np.array([], dtype=np.float32)
    bs = max(1, min(batch_size, X.shape[0]))
    try:
        return model.predict(X, verbose=0, batch_size=bs).reshape(-1)
    except Exception:
        # some TF versions can throw 'batch_outputs' unbound; fallback
        return model(X, training=False).numpy().reshape(-1)

# -----------------------
# Fetching helpers (year-split to avoid 400)
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
        hourly = pd.DataFrame(j['hourly'])
        hourly['time'] = pd.to_datetime(hourly['time'])
        hourly = hourly.set_index('time').sort_index()
        # rename to our schema
        hourly = hourly.rename(columns={
            'precipitation': 'rainfall_mm',
            'soil_moisture_0_1cm': 'soil_moisture',
            'et0_fao_evapotranspiration': 'evapotranspiration',
            'temperature_2m': 'temperature'
        })
        # keep a copy of hourly precip as max-hour proxy
        hourly['max_hourly_rain_mm'] = hourly['rainfall_mm']
        dfs.append(hourly)
        year += 1
        time.sleep(0.4)  # be nice to API
    df_hourly = pd.concat(dfs).sort_index() if dfs else pd.DataFrame()
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
# Feature engineering
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
    for col in list(agg_ops.keys()):
        if col not in df_hourly.columns:
            print(f"Warning: '{col}' missing in hourly; filling with 0.")
            df_hourly[col] = 0.0
    df_daily = df_hourly.resample('D').agg(agg_ops)
    df_daily.index.name = 'date'
    df = pd.merge(df_flood, df_daily, left_index=True, right_index=True, how='inner')
    print(f"Total merged daily rows: {len(df)}")
    return df

def engineer_v16(df_daily, lead_days=1, api_k=0.85):
    print("Engineering features...")
    d = df_daily.copy().sort_index()

    # ensure existence
    for c in ['discharge_m3_s','rainfall_mm','soil_moisture','evapotranspiration','temperature','max_hourly_rain_mm']:
        if c not in d.columns:
            d[c] = 0.0

    # discharge lags
    d['dis_lag1'] = d['discharge_m3_s'].shift(1)
    d['dis_lag2'] = d['discharge_m3_s'].shift(2)
    d['dis_lag3'] = d['discharge_m3_s'].shift(3)
    d['dis_rate'] = d['discharge_m3_s'] - d['dis_lag1']

    # rainfall lags/rolls
    for i in [1,2,3,7,14]:
        d[f'rain_lag_{i}'] = d['rainfall_mm'].shift(i)
    d['rain_roll_3']  = d['rainfall_mm'].shift(1).rolling(3).sum()
    d['rain_roll_7']  = d['rainfall_mm'].shift(1).rolling(7).sum()
    d['rain_roll_14'] = d['rainfall_mm'].shift(1).rolling(14).sum()
    d['rain_grad_1_2'] = d['rain_lag_1'] - d['rain_lag_2']

    # antecedent precipitation index
    WINDOW = 7
    weights = np.power(api_k, np.arange(WINDOW))
    d['api'] = d['rainfall_mm'].shift(1).rolling(window=WINDOW).apply(
        lambda x: np.sum(np.asarray(x) * weights[::-1]), raw=False
    )

    # ET lag
    d['et_lag_1'] = d['evapotranspiration'].shift(1)

    # cyclical
    d['month'] = d.index.month
    d['month_sin'] = np.sin(2*np.pi*d['month']/12)
    d['month_cos'] = np.cos(2*np.pi*d['month']/12)

    # pseudo forecast
    d['rainfall_forecast_1d'] = d['rainfall_mm'].shift(-lead_days)

    # target
    d['target'] = d['discharge_m3_s'].shift(-lead_days)

    # drop rows without needed features
    d = d.dropna(subset=['target','dis_lag3','rain_roll_14','api'])
    return d

FEATURE_LIST = [
 'discharge_m3_s','dis_lag1','dis_lag2','dis_lag3','dis_rate',
 'rain_lag_1','rain_lag_2','rain_lag_3','rain_lag_7','rain_lag_14',
 'rain_roll_3','rain_roll_7','rain_roll_14','api',
 'rainfall_forecast_1d',
 'max_hourly_rain_mm',
 'evapotranspiration','et_lag_1','temperature',
 'month_sin','month_cos','rain_grad_1_2'
]

SEQ_FEATURES = [
 'rainfall_mm', 'discharge_m3_s', 'max_hourly_rain_mm',
 'rain_roll_3','rain_roll_7','api','temperature','evapotranspiration'
]

# -----------------------
# Augmentation (gentle)
# -----------------------
def augment_with_surges(df, n_augment=250, max_extra_mm=120.0, runoff_coeff=0.5, seed=RANDOM_STATE):
    print("Augmenting training data with synthetic surges...")
    if n_augment <= 0:
        df['is_synthetic'] = 0
        return df
    rng = np.random.default_rng(seed)
    df_aug = df.copy()
    idx_vals = df.index.values
    new_rows = []
    for i in range(n_augment):
        base_idx = rng.choice(idx_vals[len(idx_vals)//4:])
        row = df.loc[base_idx].copy()
        extra = rng.uniform(15.0, max_extra_mm)
        days = rng.integers(1, 4)
        per_day = extra / days
        for di in range(1, min(14, days)+1):
            col = f'rain_lag_{di}' if f'rain_lag_{di}' in row.index else 'rain_lag_1'
            row[col] = (row.get(col, 0.0) or 0.0) + per_day
        row['rain_roll_3'] = (row.get('rain_roll_3',0.0) or 0.0) + extra
        row['api'] = (row.get('api',0.0) or 0.0) + extra
        row['max_hourly_rain_mm'] = max(row.get('max_hourly_rain_mm',0.0), per_day)
        row['target'] = row['target'] + runoff_coeff * extra
        row['is_synthetic'] = 1
        new_rows.append(row)
    if new_rows:
        df_aug = pd.concat([df_aug, pd.DataFrame(new_rows)], axis=0)
    df_aug['is_synthetic'] = df_aug['is_synthetic'].fillna(0).astype(int)
    df_aug = df_aug.sort_index()
    print(f"After augmentation: {len(df_aug)} rows")
    return df_aug

# -----------------------
# Sequences
# -----------------------
def make_sequences(df, seq_len=SEQ_LENGTH, feat_cols=SEQ_FEATURES, target_col='residual_std'):
    """
    Builds sequences aligned so that each y[i] corresponds to the value at the sequence end (i+seq_len-1).
    Returns X, y, idx (dates for y).
    """
    d = df.copy().sort_index()
    for c in feat_cols:
        if c not in d.columns:
            d[c] = 0.0
    arr = d[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    y_arr = d[target_col].astype(np.float32).values
    dates = d.index.values
    Xs, ys, idxs = [], [], []
    for i in range(seq_len-1, len(d)):
        Xs.append(arr[i-seq_len+1:i+1])
        ys.append(y_arr[i])
        idxs.append(dates[i])
    Xs = np.array(Xs, dtype=np.float32); ys = np.array(ys, dtype=np.float32); idxs = np.array(idxs)
    return Xs, ys, idxs

def build_bilstm(input_shape, units=64, dropout=0.2):
    inp = Input(shape=input_shape)
    x = Bidirectional(LSTM(units, return_sequences=True))(inp)
    avgp = GlobalAveragePooling1D()(x)
    maxp = GlobalMaxPooling1D()(x)
    cat = Concatenate()([avgp, maxp])
    y = Dropout(dropout)(cat)
    out = Dense(1, activation='linear')(y)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

# -----------------------
# Train v16
# -----------------------
def train_v16(lat=LAT, lon=LON, start=DEFAULT_START, end=DEFAULT_END, lead_days=1, quick=False):
    print(f"HydroFusion v16 — Training {start} → {end} (lat={lat}, lon={lon})")
    if not LGB_AVAILABLE:
        raise RuntimeError("LightGBM is required for v16.")

    # Fetch & features
    df_hourly = fetch_archive_hourly(lat, lon, start, end)
    df_flood = fetch_flood_daily(lat, lon, start, end)
    if df_hourly.empty or df_flood.empty:
        raise RuntimeError("No data returned from APIs.")
    df_daily = build_daily_merge(df_hourly, df_flood)
    df = engineer_v16(df_daily, lead_days=lead_days)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Split
    split = int(len(df)*0.8)
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()
    print(f"Train {len(train_df)}, Test {len(test_df)}")

    # Augment peaks gently
    train_df = augment_with_surges(train_df, n_augment=250 if not quick else 80)

    # Base features and scaling
    X_train = train_df[FEATURE_LIST].astype(np.float32)
    y_train = train_df['target'].astype(np.float32)
    X_test  = test_df[FEATURE_LIST].astype(np.float32)
    y_test  = test_df['target'].astype(np.float32)

    # remove constant features
    const_cols = X_train.columns[X_train.std() < 1e-8].tolist()
    if const_cols:
        print(f"Removing constant features: {const_cols}")
        X_train = X_train.drop(columns=const_cols)
        X_test  = X_test.drop(columns=const_cols, errors='ignore')

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    # ---- Base model (LightGBM) on log-target to stabilize variance ----
    print("\nTree OOF (log-target) ...")
    tscv = TimeSeriesSplit(n_splits=TIME_SERIES_SPLITS)
    y_train_log = np.log1p(y_train.clip(lower=0))
    oof_log = np.full(len(X_train_s), np.nan, dtype=float)
    models = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train_s)):
        print(f"  fold {fold+1}/{TIME_SERIES_SPLITS}: train {len(tr_idx)} → val {len(va_idx)}")
        Xtr, Xva = X_train_s.iloc[tr_idx], X_train_s.iloc[va_idx]
        ytr_log, yva_log = y_train_log.iloc[tr_idx], y_train_log.iloc[va_idx]

        m = lgb.LGBMRegressor(
            n_estimators=500 if not quick else 150,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
        m.fit(Xtr, ytr_log)
        oof_log[va_idx] = m.predict(Xva)
        models.append(m)

    # fit final model on all train
    lgb_final = lgb.LGBMRegressor(
        n_estimators=600 if not quick else 200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    lgb_final.fit(X_train_s, y_train_log)

    # base predictions
    base_train_pred = np.expm1(oof_log)
    base_test_pred  = np.expm1(lgb_final.predict(X_test_s))

    # residuals (observed - base)
    resid_train = (y_train.values - base_train_pred).astype(np.float32)
    resid_mean = float(np.nanmean(resid_train))
    resid_std  = float(np.nanstd(resid_train) + 1e-6)
    resid_train_std = (resid_train - resid_mean) / resid_std

    # ---- Build residual training frame for sequences ----
    seq_train = train_df.copy()
    seq_train['residual_std'] = resid_train_std
    # For positions where OOF is NaN (first fold gaps), fill with 0 residual_std
    seq_train['residual_std'] = seq_train['residual_std'].fillna(0.0).astype(np.float32)

    Xs_tr, ys_tr, idxs_tr = make_sequences(seq_train, seq_len=SEQ_LENGTH,
                                           feat_cols=SEQ_FEATURES, target_col='residual_std')

    # Sample weights for sequences: weight by TRUE discharge at target index
    # Map idxs_tr (dates) to discharge values (target ~ next-day shifted already)
    discharge_at_target = train_df.loc[idxs_tr, 'discharge_m3_s'].values.astype(np.float32)
    thr = np.nanpercentile(train_df['target'].values, FLOOD_Q*100)
    sw_seq = np.where(discharge_at_target >= thr, 1.0 + PEAK_ALPHA, 1.0).astype(np.float32)

    # ---- Train residual LSTM (peak-weighted) ----
    lstm_model = None
    if TF_AVAILABLE and len(Xs_tr) >= 32:
        print("Training residual LSTM (peak-weighted)...")
        lstm_model = build_bilstm((Xs_tr.shape[1], Xs_tr.shape[2]), units=64, dropout=0.2)
        cb = [
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)
        ]
        # validation split as 10% tail (time-safe-ish: we just avoid shuffling)
        # Keras will shuffle by default, so we disable shuffling
        lstm_model.fit(
            Xs_tr, ys_tr,
            epochs=20 if not quick else 6,
            batch_size=32,
            verbose=0,
            sample_weight=sw_seq,
            validation_split=0.1,
            shuffle=False,
            callbacks=cb
        )
    else:
        print("Skipping LSTM residual (TF unavailable or not enough sequences).")

    # ---- Build test sequences for residuals ----
    # We need sequences that end inside the test window. Concatenate last (SEQ_LENGTH-1) rows of train with test.
    tail = train_df.tail(SEQ_LENGTH-1)
    seq_test_src = pd.concat([tail, test_df], axis=0)
    # dummy column for shape; residual target is unknown in test
    seq_test_src['residual_std'] = 0.0
    Xs_te, _, idxs_te = make_sequences(seq_test_src, seq_len=SEQ_LENGTH,
                                       feat_cols=SEQ_FEATURES, target_col='residual_std')

    if lstm_model is not None and len(Xs_te) > 0:
        resid_std_pred_te = safe_predict(lstm_model, Xs_te, batch_size=32)
        resid_pred_te = resid_std_pred_te * resid_std + resid_mean
        # align with base_test_pred by index
        # idxs_te corresponds to the last date in each sequence
        pred_map = {d: p for d,p in zip(idxs_te, resid_pred_te)}
        aligned_resid_te = np.array([pred_map.get(d, 0.0) for d in X_test_s.index], dtype=np.float32)
    else:
        aligned_resid_te = np.zeros(len(X_test_s), dtype=np.float32)

    final_test_pred = base_test_pred + aligned_resid_te

    # ---- Metrics ----
    base_metrics = metrics_summary(y_test.values, base_test_pred)
    final_metrics = metrics_summary(y_test.values, final_test_pred)

    print("\n--- Base Model Metrics (Test) ---")
    for k,v in base_metrics.items():
        if k!='thr':
            print(f"{k}: {v:.6f}")
    print(f"thr: {base_metrics['thr']:.4f}")

    print("\n--- Final Hybrid Metrics (Test) ---")
    for k,v in final_metrics.items():
        if k!='thr':
            print(f"{k}: {v:.6f}")
    print(f"thr: {final_metrics['thr']:.4f}")

    # ---- SHAP on base model ----
    if SHAP_AVAILABLE:
        try:
            print("Computing SHAP values for LightGBM (base)...")
            # use a modest subset to avoid huge plots
            X_shap = X_test_s.sample(n=min(800, len(X_test_s)), random_state=RANDOM_STATE)
            explainer = shap.TreeExplainer(lgb_final)
            shap_values = explainer(X_shap)
            plt.figure(figsize=(10,6))
            shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
            plt.tight_layout(); plt.savefig(PLOT_SHAP); plt.close()
            print(f"Saved SHAP summary: {PLOT_SHAP}")
        except Exception as e:
            print(f"SHAP failed: {e}")

    # ---- Save artifacts ----
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        'lgb_model': lgb_final,
        'scaler': scaler,
        'resid_mean': resid_mean,
        'resid_std': resid_std,
        'thr': final_metrics['thr'],
        'seq_features': SEQ_FEATURES,
        'tab_features': X_train_s.columns.tolist(),
        'seq_length': SEQ_LENGTH,
        'lstm_model': lstm_model,          # may be None
        'trained_range': {'start': start, 'end': end},
        'lead_days': lead_days,
        'version': 16
    }
    joblib.dump(bundle, os.path.join(MODEL_DIR, MODEL_FILE))
    with open(os.path.join(MODEL_DIR, FEATURES_JSON), 'w') as f:
        json.dump({'features': X_train_s.columns.tolist(), 'seq_features': SEQ_FEATURES, 'lead_days': lead_days}, f, indent=2)
    print(f"Saved model bundle to {os.path.join(MODEL_DIR, MODEL_FILE)}")

    # predictions CSV
    out_df = pd.DataFrame({
        'date': X_test_s.index,
        'observed': y_test.values,
        'base_pred': base_test_pred,
        'residual_pred': aligned_resid_te,
        'final_pred': final_test_pred
    }, index=X_test_s.index)
    out_df.to_csv(PRED_CSV, index=False)
    print(f"Saved predictions to {PRED_CSV}")

    # plots
    idx = X_test_s.index
    plt.figure(figsize=(14,6))
    plt.plot(idx, y_test.values, label='Observed', linewidth=1.2)
    plt.plot(idx, final_test_pred, '--', label='Final Pred', linewidth=1.2)
    plt.axhline(final_metrics['thr'], color='orange', linestyle=':', label=f'{int(FLOOD_Q*100)}th pct ({final_metrics["thr"]:.2f})')
    plt.legend(); plt.grid(True); plt.title('v16 Backtest — Observed vs Final Prediction')
    plt.tight_layout(); plt.savefig(PLOT_BACKTEST); plt.close()
    print(f"Saved backtest plot to {PLOT_BACKTEST}")

    plt.figure(figsize=(6,6))
    plt.scatter(y_test.values, final_test_pred, alpha=0.6)
    lims = [min(y_test.min(), final_test_pred.min()), max(y_test.max(), final_test_pred.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel('Observed'); plt.ylabel('Predicted'); plt.title('Observed vs Predicted (v16)')
    plt.tight_layout(); plt.savefig(PLOT_OBS_PRED); plt.close()
    print(f"Saved observed vs predicted scatter: {PLOT_OBS_PRED}")

    peak_mask = y_test.values >= final_metrics['thr']
    if peak_mask.sum() > 0:
        plt.figure(figsize=(12,5))
        plt.plot(idx[peak_mask], y_test.values[peak_mask], 'o-', label='Observed peaks')
        plt.plot(idx[peak_mask], final_test_pred[peak_mask], 'x--', label='Predicted peaks')
        plt.legend(); plt.grid(True); plt.title('v16 Peaks')
        plt.tight_layout(); plt.savefig(PLOT_PEAKS); plt.close()
        print(f"Saved peaks plot to {PLOT_PEAKS}")

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
    p.add_argument('--lead', type=int, default=1)
    p.add_argument('--quick', action='store_true', help='smaller models / fewer epochs for quick iteration')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    res = train_v16(args.lat, args.lon, args.start, args.end, lead_days=args.lead, quick=args.quick)
    print("✅ Training v16 complete:", res)
