#!/usr/bin/env python3
"""
flood_model_v18.py — HydroFusion v18 (Peak-First Hybrid)

Goal: Max accuracy, especially on flood peaks, while keeping train/test leakage out.
- Base learner: LightGBM on engineered daily features (log-target training).
- Residual learner: Peak-weighted BiLSTM on short sequences to fix timing/shape errors near peaks.
- Peak calibration head: Huber regressor corrects the ensemble on predicted high flows.
- Time-aware CV for both blocks, strict feature alignment, and full plotting (backtest/peaks/SHAP).
- Safe & explicit LightGBM callbacks (no 'verbose' keyword).
- NaN-safe peak augmentation.

CLI:
    python flood_model_v18.py --lat 23.81 --lon 90.41 --start 2020-01-01 --end 2025-12-31 [--lead 1] [--quick]

"""

import os, math, json, time, argparse, warnings
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional libs
try:
    import lightgbm as lgb
    LGB_OK = True
except Exception:
    LGB_OK = False

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

try:
    import tensorflow as tf
    from keras.models import Model
    from keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate, LayerNormalization
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_OK = True
except Exception:
    TF_OK = False

import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPUs detected:", tf.config.list_physical_devices('GPU'))
print("Build info:", tf.sysconfig.get_build_info())


# -------------------
# Config
# -------------------
LAT = 23.81
LON = 90.41
DEFAULT_START = "2020-01-01"
DEFAULT_END   = "2025-12-31"
TIMEZONE = "auto"

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FLOOD   = "https://flood-api.open-meteo.com/v1/flood"

MODEL_DIR  = "ml"
MODEL_FILE = "open_meteo_flood_model_v18.pkl"

# Plots
PLOT_BACKTEST = "v18_backtest.png"
PLOT_PEAKS    = "v18_backtest_peaks.png"
PLOT_OBS_PRED = "v18_obs_vs_pred.png"
PLOT_SHAP     = "v18_shap_summary.png"
PRED_CSV      = "v18_test_predictions.csv"

# General
RANDOM_STATE = 42
FLOOD_Q      = 0.90
TIME_SPLITS  = 4
SEQ_LENGTH   = 14

# Peak aug
AUG_MULTIPLIER     = 3      # duplicate peak windows k times
AUG_RUNOFF_K       = 0.45   # how much added rainfall converts to discharge in synthetic rows
AUG_MAX_BURST_MM   = 140.0  # cap of extra mm per augmentation
AUG_MAX_SPREAD_D   = 3      # spread the burst across 1..k days

# -------------------
# Utilities
# -------------------
def safe_get_json(url, params, timeout=60):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def rmse(a, b): return math.sqrt(mean_squared_error(a, b))

def metric_pack(y_true, y_pred, thr=None):
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    d = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }
    if thr is None:
        thr = np.nanpercentile(y_true, FLOOD_Q*100)
    peak_mask = y_true >= thr
    d["r2_peak"] = r2_score(y_true[peak_mask], y_pred[peak_mask]) if peak_mask.sum() >= 2 else float("nan")
    d["thr"] = float(thr)
    return d

def log(msg): print(msg, flush=True)

# -------------------
# Data fetch
# -------------------
def fetch_archive_hourly(lat, lon, start_date, end_date):
    start = pd.to_datetime(start_date).date()
    end   = pd.to_datetime(end_date).date()
    today = date.today()
    if end > today:
        log(f"Warning: requested end_date {end} is in the future. Clipping to {today}.")
        end = today

    dfs = []
    y = start.year
    while y <= end.year:
        seg_s = max(start, date(y,1,1))
        seg_e = min(end,   date(y,12,31))
        log(f"→ Fetching archive {seg_s} → {seg_e}")
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation,soil_moisture_0_1cm,et0_fao_evapotranspiration,temperature_2m",
            "start_date": seg_s.isoformat(),
            "end_date": seg_e.isoformat(),
            "timezone": TIMEZONE
        }
        j = safe_get_json(OPEN_METEO_ARCHIVE, params)
        h = pd.DataFrame(j["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        h = h.set_index("time").sort_index()
        # Rename to our canonical names
        h = h.rename(columns={
            "precipitation":"rainfall_mm",
            "soil_moisture_0_1cm":"soil_moisture",
            "et0_fao_evapotranspiration":"evapotranspiration",
            "temperature_2m":"temperature"
        })
        # we’ll later take daily max of hourly rain as 'max_hourly_rain_mm'
        dfs.append(h)
        y += 1
        time.sleep(0.3)
    if not dfs: 
        return pd.DataFrame()
    return pd.concat(dfs).sort_index()

def fetch_flood_daily(lat, lon, start_date, end_date):
    start = pd.to_datetime(start_date).date()
    end   = pd.to_datetime(end_date).date()
    today = date.today()
    if end > today:
        end = today
    params = {
        "latitude": lat, "longitude": lon, "daily": "river_discharge",
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "timezone": TIMEZONE
    }
    j = safe_get_json(OPEN_METEO_FLOOD, params)
    df = pd.DataFrame(j.get("daily", {}))
    if "time" not in df.columns: return pd.DataFrame()
    df = df.rename(columns={"time":"date", "river_discharge":"discharge_m3_s"})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()

def build_daily_merge(df_hourly, df_flood):
    if df_hourly.empty or df_flood.empty:
        return pd.DataFrame()
    agg = {
        "rainfall_mm":"sum",
        "soil_moisture":"mean",
        "evapotranspiration":"mean",
        "temperature":"mean"
    }
    # ensure all exist
    for c in list(agg.keys()):
        if c not in df_hourly.columns:
            df_hourly[c] = 0.0
    # compute daily max hour rain
    df_hourly["max_hourly_rain_mm"] = df_hourly["rainfall_mm"]
    agg["max_hourly_rain_mm"] = "max"

    daily = df_hourly.resample("D").agg(agg)
    daily.index.name = "date"
    merged = pd.merge(df_flood, daily, left_index=True, right_index=True, how="inner")
    return merged

# -------------------
# Feature engineering
# -------------------
BASE_FEATURES = [
    "discharge_m3_s","rainfall_mm","soil_moisture","evapotranspiration","temperature","max_hourly_rain_mm"
]

TAB_FEATURES = [
    "discharge_m3_s","dis_lag1","dis_lag2","dis_lag3","dis_rate",
    "rain_lag_1","rain_lag_2","rain_lag_3","rain_lag_7","rain_lag_14",
    "rain_roll_3","rain_roll_7","rain_roll_14","api",
    "rainfall_forecast_1d",
    "max_hourly_rain_mm","evapotranspiration","et_lag_1","temperature",
    "month_sin","month_cos","rain_grad_1_2"
]

SEQ_FEATURES = [
    "rainfall_mm","discharge_m3_s","soil_moisture","max_hourly_rain_mm",
    "evapotranspiration","temperature","rain_roll_3","api"
]

def engineer(df_daily, lead_days=1, api_k=0.85):
    d = df_daily.copy().sort_index()
    # ensure all base columns exist
    for c in BASE_FEATURES:
        if c not in d.columns:
            d[c] = 0.0

    # lags & rolls
    d["dis_lag1"] = d["discharge_m3_s"].shift(1)
    d["dis_lag2"] = d["discharge_m3_s"].shift(2)
    d["dis_lag3"] = d["discharge_m3_s"].shift(3)
    d["dis_rate"] = d["discharge_m3_s"] - d["dis_lag1"]

    for i in [1,2,3,7,14]:
        d[f"rain_lag_{i}"] = d["rainfall_mm"].shift(i)
    d["rain_roll_3"]  = d["rainfall_mm"].shift(1).rolling(3).sum()
    d["rain_roll_7"]  = d["rainfall_mm"].shift(1).rolling(7).sum()
    d["rain_roll_14"] = d["rainfall_mm"].shift(1).rolling(14).sum()
    d["rain_grad_1_2"] = d["rain_lag_1"] - d["rain_lag_2"]

    d["et_lag_1"] = d["evapotranspiration"].shift(1)

    # API weighted sum of antecedent rain
    W = 7
    weights = np.power(api_k, np.arange(W))
    d["api"] = d["rainfall_mm"].shift(1).rolling(W).apply(lambda x: np.sum(x.values*weights[::-1]), raw=False)

    # time
    d["month"]     = d.index.month
    d["month_sin"] = np.sin(2*np.pi*d["month"]/12)
    d["month_cos"] = np.cos(2*np.pi*d["month"]/12)

    # perfect forecast for training target at +lead
    d["rainfall_forecast_1d"] = d["rainfall_mm"].shift(-lead_days)

    # target
    d["target"] = d["discharge_m3_s"].shift(-lead_days)

    # drop rows without target/required lags
    d = d.dropna(subset=["target","dis_lag3","rain_lag_14","api"])
    d = d.replace([np.inf,-np.inf], np.nan).dropna()
    return d

# -------------------
# Peak augmentation (NaN-safe)
# -------------------
def augment_peaks(df, multiplier=AUG_MULTIPLIER, runoff=AUG_RUNOFF_K, max_burst=AUG_MAX_BURST_MM, max_spread=AUG_MAX_SPREAD_D):
    df = df.copy()
    df["__aug__"] = df.get("__aug__", 0)
    df["__aug__"] = df["__aug__"].fillna(0)  # NaN-safe
    df["__aug__"] = df["__aug__"].astype(int)

    thr = np.nanpercentile(df["target"].values, FLOOD_Q*100)
    peak_idx = df.index[df["target"] >= thr]
    if len(peak_idx) == 0 or multiplier <= 0:
        return df, thr

    rng = np.random.default_rng(RANDOM_STATE)
    new_rows = []
    for ts in peak_idx:
        base = df.loc[ts]
        for _ in range(multiplier):
            extra = rng.uniform(10.0, max_burst)
            span  = int(rng.integers(1, max_spread+1))
            per_d = extra / span

            row = base.copy()
            # bump recent rain proxies
            for di in range(1, span+1):
                col = f"rain_lag_{di}"
                if col in row.index:
                    row[col] = float(row[col]) + per_d
            row["rain_roll_3"]  = float(row.get("rain_roll_3",0))  + extra
            row["rain_roll_7"]  = float(row.get("rain_roll_7",0))  + extra
            row["rain_roll_14"] = float(row.get("rain_roll_14",0)) + extra
            row["api"]          = float(row.get("api",0))          + extra
            row["max_hourly_rain_mm"] = max(float(row.get("max_hourly_rain_mm",0)), per_d)

            # outcome bump modestly
            row["target"] = float(row["target"]) + runoff*extra
            row["__aug__"] = 1
            # Unique timestamp (keep order)
            row.name = ts + pd.Timedelta(microseconds=rng.integers(1, 1000000))
            new_rows.append(row)

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], axis=0).sort_index()
    return df, thr

# -------------------
# Tabular prep
# -------------------
def prepare_tabular(df):
    X = df[TAB_FEATURES].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    y = df["target"].astype(np.float32)
    return X, y

# -------------------
# Sequence prep
# -------------------
def build_sequences(df, targets, seq_features=SEQ_FEATURES, seq_len=SEQ_LENGTH):
    d = df.copy().sort_index()
    for c in seq_features:
        if c not in d.columns:
            d[c] = 0.0
    arr = d[seq_features].astype(np.float32).values
    y   = targets.values.astype(np.float32)

    Xs, ys, idxs = [], [], []
    for i in range(seq_len, len(d)):
        Xs.append(arr[i-seq_len:i])
        ys.append(y[i])
        idxs.append(d.index[i])
    if len(Xs) == 0:
        return np.zeros((0,seq_len,len(seq_features)),dtype=np.float32), np.zeros((0,),dtype=np.float32), np.array([],dtype="datetime64[ns]")
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(idxs)

def make_res_lstm(input_shape, units=64, dropout=0.25):
    inp = Input(shape=input_shape)
    x = LayerNormalization()(inp)
    x = Bidirectional(LSTM(units, return_sequences=True))(x)
    avgp = GlobalAveragePooling1D()(x)
    maxp = GlobalMaxPooling1D()(x)
    x = Concatenate()([avgp, maxp])
    x = Dropout(dropout)(x)
    out = Dense(1, activation="linear")(x)
    m = Model(inputs=inp, outputs=out)
    m.compile(optimizer="adam", loss="mse")
    return m

# -------------------
# Training helpers
# -------------------
def lgb_callbacks(early_rounds=60, log_every=50):
    cbs = []
    try:
        cbs.append(lgb.early_stopping(stopping_rounds=early_rounds))
    except Exception:
        pass
    try:
        cbs.append(lgb.log_evaluation(period=log_every))
    except Exception:
        pass
    return cbs

def train_lgb_oof_log(X, y, weight=None, quick=False):
    """Train LightGBM on log1p(target) with time-aware OOF."""
    assert LGB_OK, "LightGBM is not installed."

    # log target
    y_log = np.log1p(np.clip(y, a_min=0, a_max=None))

    params = dict(
        n_estimators=(600 if not quick else 200),
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model_full = lgb.LGBMRegressor(**params)

    tscv = TimeSeriesSplit(n_splits=TIME_SPLITS)
    oof = np.full(len(X), np.nan, dtype=float)

    for k, (tr, va) in enumerate(tscv.split(X), 1):
        log(f"  fold {k}/{TIME_SPLITS}: train {len(tr)} → val {len(va)}")
        Xm_tr, Xm_va = X.iloc[tr], X.iloc[va]
        ym_tr, ym_va = y_log[tr], y_log[va]
        w_tr = None
        if weight is not None:
            w = np.asarray(weight, dtype=float)
            w_tr = w[tr]

        m = lgb.LGBMRegressor(**params)
        m.fit(
            Xm_tr, ym_tr,
            sample_weight=w_tr,
            eval_set=[(Xm_va, ym_va)],
            eval_metric="l2",
            callbacks=lgb_callbacks(early_rounds=(30 if quick else 60), log_every=0)
        )
        pred = m.predict(Xm_va)
        oof[va] = pred

    # final fit on all
    model_full.fit(
        X, y_log,
        sample_weight=weight,
        callbacks=lgb_callbacks(early_rounds=(30 if quick else 60), log_every=0)
    )
    return np.expm1(oof), model_full

def train_lstm_residual(train_df, base_oof, seq_len=SEQ_LENGTH, quick=False, thr=None):
    """Train LSTM on residuals (obs - base_oof) to fix timing/shape near peaks."""
    if not TF_OK:
        log("TensorFlow not available; skipping residual LSTM.")
        return None, None, None

    # align
    base_series = pd.Series(base_oof, index=train_df.index)
    resid = (train_df["target"] - base_series).astype(np.float32)
    resid_std = (resid - resid.mean()) / (resid.std() + 1e-6)

    # peaks weighting
    if thr is None:
        thr = np.nanpercentile(train_df["target"].values, FLOOD_Q*100)
    w = np.where(train_df["target"].values >= thr, 1.0 + 10.0, 1.0).astype(np.float32)

    Xs, ys, idxs = build_sequences(train_df, resid_std, SEQ_FEATURES, seq_len)
    if len(Xs) < 40:
        log("Not enough sequences for residual LSTM; skipping.")
        return None, None, None

    # build weights for sequence points (use the end index weight)
    w_seq = np.array([w[list(train_df.index).index(ix)] for ix in idxs], dtype=np.float32)

    # CV
    tscv = TimeSeriesSplit(n_splits=TIME_SPLITS)
    oof_resid_std = np.full(len(train_df), np.nan, dtype=float)
    input_shape = (Xs.shape[1], Xs.shape[2])

    for k, (tr, va) in enumerate(tscv.split(np.arange(len(idxs))), 1):
        log(f"  LSTM residual fold {k}/{TIME_SPLITS}")
        Xtr, Xva = Xs[tr], Xs[va]
        ytr, yva = ys[tr], ys[va]
        wtr = w_seq[tr]

        m = make_res_lstm(input_shape, units=(64 if not quick else 32), dropout=(0.25 if not quick else 0.2))
        EPOCHS = (24 if not quick else 10)
        cbs = [
            EarlyStopping(monitor="val_loss", patience=(6 if not quick else 4), restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", patience=(3 if not quick else 2), factor=0.5)
        ]
        m.fit(Xtr, ytr,
              sample_weight=wtr,
              epochs=EPOCHS, batch_size=32, verbose=0,
              validation_split=0.1, shuffle=False, callbacks=cbs)
        pred_va = m.predict(Xva, verbose=0).reshape(-1)

        # map back to train_df index for only the validation indices
        for ix, p in zip(idxs[va], pred_va):
            oof_resid_std[ list(train_df.index).index(ix) ] = p

    # final residual model on ALL sequences
    m_final = make_res_lstm(input_shape, units=(64 if not quick else 32), dropout=(0.25 if not quick else 0.2))
    EPOCHS = (30 if not quick else 12)
    cbs = [
        EarlyStopping(monitor="val_loss", patience=(6 if not quick else 4), restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=(3 if not quick else 2), factor=0.5)
    ]
    m_final.fit(Xs, ys, sample_weight=w_seq, epochs=EPOCHS, batch_size=32, verbose=0,
                validation_split=0.1, shuffle=False, callbacks=cbs)

    return oof_resid_std, m_final, idxs

# -------------------
# Plotting
# -------------------
def plot_backtest(idx, y_true, y_pred, thr, fname=PLOT_BACKTEST, title="v18 Backtest — Observed vs Final"):
    plt.figure(figsize=(14,6))
    plt.plot(idx, y_true, label="Observed", linewidth=1.2)
    plt.plot(idx, y_pred, "--", label="Final Pred", linewidth=1.2)
    plt.axhline(thr, color="orange", linestyle=":", label=f"{int(FLOOD_Q*100)}th pct ({thr:.2f})")
    plt.legend(); plt.grid(True); plt.title(title); plt.tight_layout()
    plt.savefig(fname); plt.close()
    log(f"Saved backtest plot to {fname}")

def plot_peaks(idx, y_true, y_pred, thr, fname=PLOT_PEAKS):
    mask = y_true >= thr
    if mask.sum() == 0:
        return
    plt.figure(figsize=(12,5))
    plt.plot(idx[mask], y_true[mask], "o-", label="Observed peaks")
    plt.plot(idx[mask], y_pred[mask], "x--", label="Predicted peaks")
    plt.legend(); plt.grid(True); plt.title("v18 Peaks"); plt.tight_layout()
    plt.savefig(fname); plt.close()
    log(f"Saved peaks plot to {fname}")

def plot_obs_vs_pred(y_true, y_pred, fname=PLOT_OBS_PRED):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Observed"); plt.ylabel("Predicted"); plt.title("Observed vs Predicted")
    plt.grid(True); plt.tight_layout(); plt.savefig(fname); plt.close()
    log(f"Saved observed vs predicted scatter: {fname}")

# -------------------
# Train v18
# -------------------
def train_v18(lat=LAT, lon=LON, start=DEFAULT_START, end=DEFAULT_END, lead_days=1, quick=False):
    if not LGB_OK:
        raise RuntimeError("LightGBM is required for v18.")

    log(f"HydroFusion v18 — Training {start} → {end} (lat={lat}, lon={lon})")

    # 1) FETCH
    df_hourly = fetch_archive_hourly(lat, lon, start, end)
    df_flood  = fetch_flood_daily(lat, lon, start, end)
    daily = build_daily_merge(df_hourly, df_flood)
    if daily.empty:
        raise RuntimeError("No overlapping weather/flood data; cannot train.")

    # 2) FEATURES
    df = engineer(daily, lead_days=lead_days)
    df = df.replace([np.inf,-np.inf], np.nan).dropna()
    if len(df) < 400:
        log("Very few rows after feature engineering; results may be unstable.")

    # 3) TIME SPLIT
    split = int(len(df)*0.8)
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()
    log(f"Train {len(train_df)}, Test {len(test_df)}")

    # 4) SAFE PEAK AUGMENTATION (train only)
    train_aug, thr = augment_peaks(train_df, multiplier=(1 if quick else AUG_MULTIPLIER))
    log(f"Augmented peaks (thr={thr:.2f}): total rows after aug = {len(train_aug)}")

    # 5) TABULAR PREP + SCALER
    Xtr, ytr = prepare_tabular(train_aug)
    Xte, yte = prepare_tabular(test_df)

    scaler = StandardScaler()
    scaler.fit(Xtr)  # include aug in scale; aug is part of train distribution
    Xtr_s = pd.DataFrame(scaler.transform(Xtr), index=Xtr.index, columns=Xtr.columns)
    Xte_s = pd.DataFrame(scaler.transform(Xte), index=Xte.index, columns=Xte.columns)

    # 6) BASE LGB OOF (log-target), with peak weights
    w_base = np.where(train_aug["target"].values >= thr, 1.0 + 12.0, 1.0).astype(np.float32)
    log("\nBase LightGBM OOF (log-target) ...")
    base_oof, lgb_full = train_lgb_oof_log(Xtr_s, ytr.values, weight=w_base, quick=quick)
    base_oof_series = pd.Series(base_oof, index=Xtr.index)

    # 7) RESIDUAL LSTM on train_aug (peak-weighted)
    log("\nTraining residual LSTM (peak-weighted)...")
    lstm_oof_std, lstm_model, seq_train_idxs = train_lstm_residual(train_aug, base_oof_series.values,
                                                                   seq_len=SEQ_LENGTH, quick=quick, thr=thr)

    # 8) FIT PEAK CALIBRATION HEAD on training end-slice (time-safe)
    # meta features for test-time: [base_pred, resid_pred]
    # On train: build aligned arrays; for samples without LSTM pred, set resid=0
    meta_train = pd.DataFrame({
        "base": base_oof_series.values,
        "resid": np.zeros(len(train_aug), dtype=float)
    }, index=train_aug.index)

    if lstm_oof_std is not None:
        # convert standardized residual preds back to real residual scale
        # empirically, multiplying by train residual std tends to stabilize
        resid = (train_aug["target"].values - base_oof_series.values)
        rstd  = resid.std() + 1e-6
        resid_recon = np.where(np.isnan(lstm_oof_std), 0.0, lstm_oof_std * rstd)
        meta_train["resid"] = resid_recon

    # Validation slice from last 10% of augmented training for calibration
    cal_slice = int(len(train_aug)*0.9)
    cal_df  = train_aug.iloc[cal_slice:]
    X_cal   = meta_train.iloc[cal_slice:].values
    y_cal   = cal_df["target"].values
    high_mask_cal = y_cal >= thr
    calibrator = None
    if high_mask_cal.sum() >= 8:
        calibrator = HuberRegressor()
        calibrator.fit(X_cal[high_mask_cal], y_cal[high_mask_cal])
    else:
        log("Not enough high samples for calibrator; skipping.")

    # 9) TEST PREDICTIONS
    # base
    base_log_pred_test = lgb_full.predict(Xte_s)
    base_pred_test = np.expm1(base_log_pred_test)

    # residual seq for test
    resid_seq_pred = np.zeros(len(test_df), dtype=float)
    if TF_OK and lstm_model is not None:
        # Build sequences for (train tail + test) to have warm-up context
        combine = pd.concat([train_aug.tail(SEQ_LENGTH), test_df], axis=0)
        Xseq_all, yseq_all, idxs_all = build_sequences(combine, combine["target"], SEQ_FEATURES, SEQ_LENGTH)
        if len(Xseq_all) > 0:
            pred_resid_std = lstm_model.predict(Xseq_all, verbose=0).reshape(-1)
            # map only test indices
            test_idxs = idxs_all[idxs_all >= test_df.index.min()]
            # scale back to real residuals using train residual std
            resid = (train_aug["target"].values - base_oof_series.values)
            rstd  = resid.std() + 1e-6
            pred_resid_real = pred_resid_std * rstd
            # assign to test rows by date alignment
            m = {d:p for d,p in zip(test_idxs, pred_resid_real[-len(test_idxs):])}
            for i, d in enumerate(test_df.index):
                resid_seq_pred[i] = m.get(d, 0.0)

    # blend base + residual
    final_test_pred = base_pred_test + resid_seq_pred

    # apply calibrator on predicted highs (meta with 2 features: base, resid)
    if calibrator is not None:
        meta_test = np.vstack([base_pred_test, resid_seq_pred]).T
        high_mask_test = final_test_pred >= thr
        if high_mask_test.sum() > 0:
            final_test_pred[high_mask_test] = calibrator.predict(meta_test[high_mask_test])

    # 10) METRICS & PLOTS
    y_true = test_df["target"].values
    base_metrics  = metric_pack(y_true, base_pred_test, thr)
    final_metrics = metric_pack(y_true, final_test_pred, thr)

    log("\n--- Base Model Metrics (Test) ---")
    for k,v in base_metrics.items(): 
        log(f"{k}: {v:.6f}" if isinstance(v,(int,float,np.floating)) else f"{k}: {v}")
    log("\n--- Final Hybrid Metrics (Test) ---")
    for k,v in final_metrics.items():
        log(f"{k}: {v:.6f}" if isinstance(v,(int,float,np.floating)) else f"{k}: {v}")

    # Save predictions CSV
    out = pd.DataFrame({
        "date": test_df.index,
        "observed": y_true,
        "base_pred": base_pred_test,
        "resid_pred": resid_seq_pred,
        "final_pred": final_test_pred
    }, index=test_df.index)
    out.to_csv(PRED_CSV, index=False)
    log(f"Saved predictions to {PRED_CSV}")

    # Plots
    plot_backtest(test_df.index, y_true, final_test_pred, thr, PLOT_BACKTEST)
    plot_peaks(test_df.index, y_true, final_test_pred, thr, PLOT_PEAKS)
    plot_obs_vs_pred(y_true, final_test_pred, PLOT_OBS_PRED)

    # SHAP on base model (tabular)
    if SHAP_OK:
        try:
            log("Computing SHAP values for LightGBM (base)...")
            explainer = shap.TreeExplainer(lgb_full)
            shap_values = explainer(Xte_s, check_additivity=False)
            plt.figure(figsize=(10,7))
            shap.summary_plot(shap_values, Xte_s, show=False, plot_type="bar")
            plt.tight_layout(); plt.savefig(PLOT_SHAP); plt.close()
            log(f"Saved SHAP summary to {PLOT_SHAP}")
        except Exception as e:
            log(f"SHAP failed: {e}")

    # Save bundle
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        "lgb_base": lgb_full,
        "scaler": scaler,
        "calibrator": calibrator,
        "thr": float(thr),
        "tab_features": TAB_FEATURES,
        "seq_features": SEQ_FEATURES,
        "seq_length": SEQ_LENGTH,
        "residual_lstm": lstm_model,
        "trained_range": {"start": start, "end": end},
        "final_metrics": final_metrics,
        "base_metrics": base_metrics
    }
    joblib.dump(bundle, os.path.join(MODEL_DIR, MODEL_FILE))
    log(f"Saved model bundle to {os.path.join(MODEL_DIR, MODEL_FILE)}")

    return {"final_metrics": final_metrics, "base_metrics": base_metrics}

# -------------------
# CLI
# -------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float, default=LAT)
    p.add_argument("--lon", type=float, default=LON)
    p.add_argument("--start", type=str, default=DEFAULT_START)
    p.add_argument("--end", type=str, default=DEFAULT_END)
    p.add_argument("--lead", type=int, default=1)
    p.add_argument("--quick", action="store_true", help="smaller models / fewer epochs / lighter augmentation")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    res = train_v18(args.lat, args.lon, args.start, args.end, lead_days=args.lead, quick=args.quick)
    log(f"✅ Training complete: {res}")



