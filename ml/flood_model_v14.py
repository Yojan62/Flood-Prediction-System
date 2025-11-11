#!/usr/bin/env python3
"""
flood_model_v14.py — HydroFusion v14

Goal:
- Keep v13's stable data pipeline, diagnostics and plots
- Simplify ensemble: LightGBM (primary) + LSTM (aux) only
- Train in original discharge units (no log), reduce over-smoothing
- High-flow weighting via sample_weight (no separate calibrator)
- Huber loss for LightGBM (robust peaks)
- Adaptive blending weight tuned on validation RMSE
- Same CLI & outputs style as v13

Usage:
    python flood_model_v14.py --mode train --lat 23.81 --lon 90.41 --start 2020-01-01 --end 2025-12-31 [--lead 1] [--quick]

Outputs:
    ml/open_meteo_flood_model_v14.pkl         (model bundle)
    v14_test_predictions.csv                  (per-day predictions)
    v14_backtest.png                          (observed vs final pred)
    v14_obs_vs_pred.png                       (scatter)
    v14_backtest_peaks.png                    (peaks only)
    v14_shap_summary.png                      (if LightGBM + shap installed)
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
MODEL_FILE = "open_meteo_flood_model_v14.pkl"
FEATURES_JSON = "open_meteo_flood_features_v14.json"
PLOT_BACKTEST = "v14_backtest.png"
PLOT_PEAKS = "v14_backtest_peaks.png"
PLOT_SHAP = "v14_shap_summary.png"
PLOT_OBS_PRED = "v14_obs_vs_pred.png"
PRED_CSV = "v14_test_predictions.csv"

RANDOM_STATE = 42
FLOOD_Q = 0.90
PEAK_WEIGHT_ALPHA = 18.0
TIME_SERIES_SPLITS = 4
SEQ_LENGTH = 14

# Tabular features (kept from v13, sans any constant columns)
FEATURE_LIST = [
    'discharge_m3_s','dis_lag1','dis_lag2','dis_lag3','dis_rate',
    'rain_lag_1','rain_lag_2','rain_lag_3','rain_lag_7','rain_lag_14',
    'rain_roll_3','rain_roll_7','rain_roll_14','api',
    'rainfall_forecast_1d', 'max_hourly_rain_mm',
    'evapotranspiration','et_lag_1','temperature',
    'month_sin','month_cos','rain_grad_1_2',
]
# LSTM features (a bit richer than v13 to capture rises)
SEQ_FEATURES = ['discharge_m3_s', 'dis_rate', 'rain_roll_3', 'rain_roll_7', 'api']

ARGS = None

# -----------------------
# Utils
# -----------------------
def safe_get_json(url, params, timeout=90):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def nse(y_true, y_pred):
    """Nash–Sutcliffe efficiency."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return float('nan')
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom

def metrics_block(y_true, y_pred, thr):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2v = r2_score(y_true, y_pred)
    nsev = nse(y_true, y_pred)
    peak_mask = y_true >= thr
    r2_peak = r2_score(y_true[peak_mask], y_pred[peak_mask]) if peak_mask.sum() >= 2 else float('nan')
    return {'mae': mae, 'rmse': rmse, 'r2': r2v, 'nse': nsev, 'r2_peak': r2_peak, 'thr': float(thr)}

def plot_obs_vs_pred(y_true, y_pred, title="Observed vs Predicted", out_file=PLOT_OBS_PRED):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("Observed"); plt.ylabel("Predicted")
    plt.title(title); plt.grid(True); plt.tight_layout()
    plt.savefig(out_file); plt.close()
    print(f"Saved observed vs predicted scatter: {out_file}")

# -----------------------
# Fetch
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
        dfs.append(hourly)
        year += 1
        time.sleep(0.6)
    if not dfs:
        return pd.DataFrame()

    df_hourly = pd.concat(dfs).sort_index()

    # Rename to internal names
    df_hourly = df_hourly.rename(columns={
        'precipitation': 'rainfall_mm',
        'soil_moisture_0_1cm': 'soil_moisture',
        'et0_fao_evapotranspiration': 'evapotranspiration',
        'temperature_2m': 'temperature'
    })
    # Max hourly intensity proxy
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
            print(f"Warning: Column '{col}' missing in hourly; filling with 0.")
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

    # Lags on discharge
    d['dis_lag1'] = d['discharge_m3_s'].shift(1)
    d['dis_lag2'] = d['discharge_m3_s'].shift(2)
    d['dis_lag3'] = d['discharge_m3_s'].shift(3)
    d['dis_rate'] = d['discharge_m3_s'] - d['dis_lag1']

    # Rain lags & rolls
    for i in [1,2,3,7,14]:
        d[f'rain_lag_{i}'] = d['rainfall_mm'].shift(i)
    d['rain_roll_3']  = d['rainfall_mm'].shift(1).rolling(3).sum()
    d['rain_roll_7']  = d['rainfall_mm'].shift(1).rolling(7).sum()
    d['rain_roll_14'] = d['rainfall_mm'].shift(1).rolling(14).sum()
    d['rain_grad_1_2'] = d['rain_lag_1'] - d['rain_lag_2']

    # ET lag
    d['et_lag_1'] = d['evapotranspiration'].shift(1)

    # API (weighted antecedent rainfall)
    WINDOW = 7
    weights = np.power(api_k, np.arange(WINDOW))
    d['api'] = d['rainfall_mm'].shift(1).rolling(window=WINDOW).apply(
        lambda x: np.sum(x.values * weights[::-1]), raw=False
    )

    # Cyclical month
    d['month'] = d.index.month
    d['month_sin'] = np.sin(2*np.pi*d['month']/12.)
    d['month_cos'] = np.cos(2*np.pi*d['month']/12.)

    # “Perfect” 1-day forecast proxy
    d['rainfall_forecast_1d'] = d['rainfall_mm'].shift(-lead_days)

    # Target
    d['target'] = d['discharge_m3_s'].shift(-lead_days)

    d = d.dropna(subset=['target','dis_lag3','rain_lag_14','api'])
    return d

def augment_with_surges(df, n_augment=400, max_extra_mm=160.0, runoff_coeff=0.55):
    print("Augmenting training data with synthetic surges...")
    rng = np.random.default_rng(RANDOM_STATE)
    df_aug = df.copy()
    idx_vals = df.index.values
    new_rows = []

    for i in range(n_augment):
        base_idx = rng.choice(idx_vals[len(idx_vals)//4:])
        row = df.loc[base_idx].copy()

        extra_total = rng.uniform(10.0, max_extra_mm)
        days = rng.integers(1,4)
        per_day = extra_total / days

        for d_i in range(1, days+1):
            col = f'rain_lag_{d_i}'
            row[col] = (row.get(col, 0.0) or 0.0) + per_day

        row['rain_roll_3'] = (row.get('rain_roll_3',0) or 0) + extra_total
        row['api'] = (row.get('api',0) or 0) + extra_total
        row['max_hourly_rain_mm'] = max(row.get('max_hourly_rain_mm',0), per_day)

        row['target'] = row['target'] + runoff_coeff * extra_total
        row['is_synthetic'] = 1

        new_index = base_idx + pd.Timedelta(nanoseconds=i+1)
        row.name = new_index
        new_rows.append(row)

    if new_rows:
        df_aug = pd.concat([df_aug, pd.DataFrame(new_rows)], ignore_index=False)

    df_aug['is_synthetic'] = df_aug.get('is_synthetic', 0)
    df_aug['is_synthetic'] = df_aug['is_synthetic'].fillna(0).astype(int)
    df_aug = df_aug.sample(frac=1, random_state=RANDOM_STATE)
    print(f"After augmentation: {len(df_aug)} rows")
    return df_aug

def prepare_tabular_Xy(df, feature_list):
    X_cols = [c for c in feature_list if c in df.columns]
    X = df[X_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    y = df['target'].astype(np.float32)
    return X, y

# -----------------------
# LSTM
# -----------------------
def build_sequences(df, seq_len=SEQ_LENGTH, features_seq=SEQ_FEATURES):
    print(f"Building LSTM sequences (SEQ_LENGTH={seq_len})...")
    d = df.copy().sort_index()
    for f in features_seq:
        if f not in d.columns:
            d[f] = 0.0
    arr = d[features_seq].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    targets = d['target'].astype(np.float32).values
    Xs, ys, idxs = [], [], []
    for i in range(seq_len, len(d)):
        seq = arr[i-seq_len:i]
        if not np.isfinite(seq).all():
            continue
        Xs.append(seq); ys.append(targets[i]); idxs.append(d.index[i])
    Xs = np.array(Xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    idxs = np.array(idxs)
    print("Sequence shapes:", Xs.shape, ys.shape)
    return Xs, ys, idxs

def build_bilstm(input_shape, units=64, dropout=0.2):
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required for the LSTM branch.")
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
# Train
# -----------------------
def train_v14(lat=LAT, lon=LON, start=DEFAULT_START, end=DEFAULT_END, lead_days=1, quick=False):
    print(f"HydroFusion v14 — Training {start} → {end} (lat={lat}, lon={lon})")

    df_hourly = fetch_archive_hourly(lat, lon, start, end)
    df_flood = fetch_flood_daily(lat, lon, start, end)
    if df_hourly.empty or df_flood.empty:
        print("❌ Missing data from APIs; aborting.")
        return

    df_daily = build_daily_merge(df_hourly, df_flood)
    if df_daily.empty:
        print("❌ No overlap between hourly & flood datasets.")
        return

    df = engineer(df_daily, lead_days=lead_days)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Time split
    split = int(len(df)*0.8)
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()
    print(f"Train {len(train_df)}, Test {len(test_df)}")

    # Augment
    train_aug = augment_with_surges(train_df, n_augment=400 if not quick else 120)
    train_aug['target'] = train_aug['target'].fillna(0)
    thr = np.nanpercentile(train_aug['target'], FLOOD_Q*100)
    print(f"Peak threshold (train {int(FLOOD_Q*100)}th pct) = {thr:.2f}")

    # High-flow weights (used for LGBM)
    sw = np.where(train_aug['target'] >= thr, 1.0 + PEAK_WEIGHT_ALPHA, 1.0)
    sw *= np.where(train_aug['is_synthetic']==1, 0.6, 1.0)

    # Tabular
    X_train_full, y_train_full = prepare_tabular_Xy(train_aug, FEATURE_LIST)
    X_test_full,  y_test_full  = prepare_tabular_Xy(test_df,  FEATURE_LIST)

    scaler = StandardScaler()
    scaler.fit(X_train_full.loc[train_aug['is_synthetic']==0].fillna(0))
    X_train = pd.DataFrame(scaler.transform(X_train_full.fillna(0)), index=X_train_full.index, columns=X_train_full.columns)
    X_test  = pd.DataFrame(scaler.transform(X_test_full.fillna(0)),  index=X_test_full.index,  columns=X_test_full.columns)

    # Remove constant columns (safety)
    const_cols = X_train.columns[X_train.std() < 1e-6].tolist()
    if const_cols:
        print(f"Removing constant features: {const_cols}")
        X_train.drop(columns=const_cols, inplace=True)
        X_test.drop(columns=const_cols, errors='ignore', inplace=True)

    # ---------------- LightGBM (primary) ----------------
    if not LGB_AVAILABLE:
        raise EnvironmentError("LightGBM not installed. Please install lightgbm to run v14.")

    lgb_params = dict(
        n_estimators=450 if not quick else 160,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='huber',      # robust to peaks; behaves like L2 near center
        alpha=0.85,             # Huber delta; higher → more L2-like
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model = lgb.LGBMRegressor(**lgb_params)

    # Use a small val slice from the tail of train_aug to early-stop/tune blending
    val_frac = 0.15
    val_idx_start = int(len(X_train) * (1 - val_frac))
    X_tr, X_val = X_train.iloc[:val_idx_start], X_train.iloc[val_idx_start:]
    y_tr, y_val = y_train_full.iloc[:val_idx_start], y_train_full.iloc[val_idx_start:]
    sw_tr, sw_val = sw[:val_idx_start], sw[val_idx_start:]

    lgb_model.fit(X_tr, y_tr, sample_weight=sw_tr,
                  eval_set=[(X_val, y_val)],
                  eval_metric='l2',
                  callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)] if hasattr(lgb, "early_stopping") else None)

    pred_lgb_val = lgb_model.predict(X_val)
    pred_lgb_test = lgb_model.predict(X_test)

    # ---------------- LSTM (aux) ----------------
    pred_lstm_val = np.zeros_like(y_val.values, dtype=float)
    pred_lstm_test = np.zeros_like(y_test_full.values, dtype=float)
    lstm_model = None

    if TF_AVAILABLE:
        # Build sequences on training+validation to avoid leakage into test blending
        Xs_train_seq, ys_train_seq, idxs_train_seq = build_sequences(train_aug, seq_len=SEQ_LENGTH, features_seq=SEQ_FEATURES)
        # Map those to val/train slices by date
        train_dates = X_train.index
        val_dates = X_val.index

        mask_val = np.isin(idxs_train_seq, val_dates)
        mask_trn = np.isin(idxs_train_seq, train_dates)

        Xs_tr = Xs_train_seq[mask_trn]; ys_tr_seq = ys_train_seq[mask_trn]
        Xs_vl = Xs_train_seq[mask_val]; ys_vl_seq = ys_train_seq[mask_val]

        if len(Xs_tr) >= 20 and len(Xs_vl) >= 5:
            lstm_model = build_bilstm((Xs_tr.shape[1], Xs_tr.shape[2]), units=64, dropout=0.2)
            epochs = 10 if quick else 35
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5)
            ]
            lstm_model.fit(Xs_tr, ys_tr_seq, epochs=epochs, batch_size=32, verbose=0,
                           validation_data=(Xs_vl, ys_vl_seq), callbacks=callbacks)

            # Validation preds (align by date)
            lstm_val_raw = lstm_model.predict(Xs_vl).reshape(-1)
            # map date->pred
            val_map = {d: p for d, p in zip(idxs_train_seq[mask_val], lstm_val_raw)}
            pred_lstm_val = np.array([val_map.get(d, np.nan) for d in X_val.index])
            if np.isnan(pred_lstm_val).any():
                fill = np.nanmedian(pred_lstm_val[~np.isnan(pred_lstm_val)]) if np.isfinite(pred_lstm_val[~np.isnan(pred_lstm_val)]).any() else np.nanmedian(ys_vl_seq)
                pred_lstm_val = np.where(np.isnan(pred_lstm_val), fill, pred_lstm_val)

            # Test sequences: build from tail of full train+test
            combined = pd.concat([train_aug.tail(SEQ_LENGTH), test_df])
            Xs_test_seq, _, idxs_test_seq = build_sequences(combined, seq_len=SEQ_LENGTH, features_seq=SEQ_FEATURES)
            if len(Xs_test_seq) > 0:
                lstm_test_raw = lstm_model.predict(Xs_test_seq).reshape(-1)
                test_map = {d: p for d, p in zip(idxs_test_seq, lstm_test_raw)}
                pred_lstm_test = np.array([test_map.get(d, np.nan) for d in X_test.index])
                if np.isnan(pred_lstm_test).any():
                    fillt = np.nanmedian(pred_lstm_test[~np.isnan(pred_lstm_test)]) if np.isfinite(pred_lstm_test[~np.isnan(pred_lstm_test)]).any() else np.nanmedian(ys_train_seq)
                    pred_lstm_test = np.where(np.isnan(pred_lstm_test), fillt, pred_lstm_test)
        else:
            print("LSTM skipped: not enough valid sequences.")
    else:
        print("TensorFlow not available; skipping LSTM branch.")

    # ---------------- Adaptive blending ----------------
    # Find alpha in [0,1] that minimizes RMSE on validation: y_blend = (1-alpha)*lgb + alpha*lstm
    alphas = np.linspace(0.0, 0.6, 13)  # allow up to 60% LSTM weight (usually much lower works best)
    best_alpha, best_rmse = 0.0, 1e18
    for a in alphas:
        yb = (1.0 - a) * pred_lgb_val + a * pred_lstm_val
        rm = math.sqrt(mean_squared_error(y_val, yb))
        if rm < best_rmse:
            best_rmse, best_alpha = rm, a
    print(f"Adaptive blend weight: alpha={best_alpha:.2f} (val RMSE={best_rmse:.3f})")

    final_test_pred = (1.0 - best_alpha) * pred_lgb_test + best_alpha * pred_lstm_test

    # ---------------- Evaluation & Plots ----------------
    y_test_vals = y_test_full.values
    base_metrics_lgb  = metrics_block(y_test_vals, pred_lgb_test, thr)
    base_metrics_lstm = metrics_block(y_test_vals, pred_lstm_test, thr)
    final_metrics     = metrics_block(y_test_vals, final_test_pred, thr)

    print("\n--- Base Model Metrics (Test) ---")
    print(pd.DataFrame({'lgb': base_metrics_lgb, 'lstm': base_metrics_lstm}).T)
    print("\n--- Final Blended Metrics (Test) ---")
    print(pd.Series(final_metrics).to_frame('Score'))

    # SHAP (optional) only for LightGBM
    if SHAP_AVAILABLE:
        try:
            print("\nComputing SHAP values for LightGBM...")
            explainer = shap.TreeExplainer(lgb_model)
            shap_values = explainer(X_test)
            plt.figure(figsize=(10,6))
            shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")
            plt.tight_layout(); plt.savefig(PLOT_SHAP); plt.close()
            print(f"Saved SHAP summary: {PLOT_SHAP}")
        except Exception as e:
            print(f"SHAP failed: {e}")

    # Save bundle
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        'lgb_model': lgb_model,
        'lstm_model': lstm_model,
        'blend_alpha': best_alpha,
        'scaler': scaler,
        'features': X_train.columns.tolist(),
        'seq_features': SEQ_FEATURES,
        'seq_length': SEQ_LENGTH,
        'trained_range': {'start': start, 'end': end},
        'threshold_train_q': float(thr)
    }
    joblib.dump(bundle, os.path.join(MODEL_DIR, MODEL_FILE))
    with open(os.path.join(MODEL_DIR, FEATURES_JSON), 'w') as f:
        json.dump({'features': X_train.columns.tolist(), 'seq_features': SEQ_FEATURES, 'lead_days': lead_days}, f, indent=2)
    print(f"Saved model bundle to {os.path.join(MODEL_DIR, MODEL_FILE)}")

    # Save predictions CSV
    out_df = pd.DataFrame({
        'date': test_df.index,
        'observed': y_test_vals,
        'lgb_pred': pred_lgb_test,
        'lstm_pred': pred_lstm_test,
        'final_pred': final_test_pred
    }, index=test_df.index)
    out_df.to_csv(PRED_CSV, index=False)
    print(f"Saved predictions to {PRED_CSV}")

    # Backtest plot
    idx = test_df.index
    plt.figure(figsize=(14,6))
    plt.plot(idx, y_test_vals, label='Observed', linewidth=1.3)
    plt.plot(idx, final_test_pred, '--', label='Final Pred', linewidth=1.3)
    plt.axhline(thr, color='orange', linestyle=':', label=f'{int(FLOOD_Q*100)}th Pct ({thr:.2f})')
    plt.legend(); plt.grid(True); plt.title('v14 Backtest - Observed vs Final Prediction')
    plt.tight_layout(); plt.savefig(PLOT_BACKTEST); plt.close()
    print(f"Saved backtest plot to {PLOT_BACKTEST}")

    # Peaks plot
    thr_mask = y_test_vals >= thr
    if thr_mask.sum() > 0:
        plt.figure(figsize=(12,5))
        plt.plot(idx[thr_mask], y_test_vals[thr_mask], 'o-', label='Observed peaks')
        plt.plot(idx[thr_mask], final_test_pred[thr_mask], 'x--', label='Predicted peaks')
        plt.legend(); plt.grid(True); plt.title('v14 Peaks')
        plt.tight_layout(); plt.savefig(PLOT_PEAKS); plt.close()
        print(f"Saved peaks plot to {PLOT_PEAKS}")

    # Scatter
    plot_obs_vs_pred(y_test_vals, final_test_pred, title="v14 Observed vs Final Prediction", out_file=PLOT_OBS_PRED)

    # Return brief summary
    return {
        'final_metrics': final_metrics,
        'base_metrics': {'lgb': base_metrics_lgb, 'lstm': base_metrics_lstm},
        'blend_alpha': float(best_alpha)
    }

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
    res = train_v14(args.lat, args.lon, args.start, args.end, lead_days=args.lead, quick=args.quick)
    print("✅ Training v14 complete:", res)
