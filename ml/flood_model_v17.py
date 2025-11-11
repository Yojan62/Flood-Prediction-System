#!/usr/bin/env python3
"""
HydroFusion v17 — Flood-Focused Hybrid Model
✅ Stable + Flood-sensitive version
---------------------------------------------------
- Larger LSTM (96 units)
- Stronger flood weighting (alpha=5.0)
- Adds discharge lags to sequence inputs
- More synthetic floods (n_augment=400)
- Safe feature engineering (no data loss)
- Extra data integrity checks
"""

import os, time, json, math, argparse, warnings
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, requests

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=FutureWarning)

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
    from keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# ---------------------- CONFIG ----------------------
LAT, LON = 23.81, 90.41
DEFAULT_START, DEFAULT_END = "2020-01-01", "2025-12-31"
TIMEZONE = "auto"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FLOOD_API = "https://flood-api.open-meteo.com/v1/flood"
MODEL_DIR = "ml"
MODEL_FILE = "open_meteo_flood_model_v17.pkl"

RANDOM_STATE = 42
SEQ_LENGTH = 14
FLOOD_Q = 0.90
PEAK_ALPHA = 5.0
N_AUGMENT = 400
LSTM_UNITS = 96
DROPOUT = 0.25
EPOCHS = 35


# ---------------------- UTILITIES ----------------------
def rmse(y_true, y_pred): 
    return math.sqrt(mean_squared_error(y_true, y_pred))

def metrics_summary(y_true, y_pred, thr_q=FLOOD_Q):
    thr = np.nanpercentile(y_true, thr_q * 100)
    mask = y_true >= thr
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "r2_peak": r2_score(y_true[mask], y_pred[mask]) if mask.sum() >= 2 else float("nan"),
        "thr": float(thr)
    }

def safe_predict(model, X, batch_size=32):
    X = np.asarray(X)
    if X.shape[0] == 0:
        return np.zeros(0)
    bs = max(1, min(batch_size, X.shape[0]))
    try:
        return model.predict(X, verbose=0, batch_size=bs).reshape(-1)
    except Exception:
        return model(X, training=False).numpy().reshape(-1)

def get_json(url, params):
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    return r.json()


# ---------------------- FETCHERS ----------------------
def fetch_archive(lat, lon, start, end):
    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end).date()
    today = date.today()
    if end > today: end = today
    dfs = []
    for y in range(start.year, end.year + 1):
        s = max(start, date(y, 1, 1))
        e = min(end, date(y, 12, 31))
        print(f"→ Fetching archive {s} → {e}")
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "precipitation,soil_moisture_0_1cm,et0_fao_evapotranspiration,temperature_2m",
            "start_date": s.isoformat(), "end_date": e.isoformat(), "timezone": TIMEZONE
        }
        j = get_json(ARCHIVE, params)
        if "hourly" not in j: continue
        h = pd.DataFrame(j["hourly"])
        if h.empty: continue
        h["time"] = pd.to_datetime(h["time"])
        h = h.set_index("time").rename(columns={
            "precipitation": "rainfall_mm",
            "soil_moisture_0_1cm": "soil_moisture",
            "et0_fao_evapotranspiration": "evapotranspiration",
            "temperature_2m": "temperature"
        })
        h["max_hourly_rain_mm"] = h["rainfall_mm"]
        dfs.append(h)
        time.sleep(0.4)
    return pd.concat(dfs).sort_index() if dfs else pd.DataFrame()


def fetch_flood(lat, lon, start, end):
    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end).date()
    today = date.today()
    if end > today: end = today
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "river_discharge",
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "timezone": TIMEZONE
    }
    j = get_json(FLOOD_API, params)
    d = pd.DataFrame(j.get("daily", {}))
    if d.empty: return d
    d = d.rename(columns={"time": "date", "river_discharge": "discharge_m3_s"})
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date").sort_index()


# ---------------------- FEATURES ----------------------
def merge_daily(dfh, dff):
    if dfh.empty or dff.empty:
        return pd.DataFrame()
    agg = {
        "rainfall_mm": "sum",
        "soil_moisture": "mean",
        "evapotranspiration": "mean",
        "temperature": "mean",
        "max_hourly_rain_mm": "max"
    }
    d = dfh.resample("D").agg(agg)
    return pd.merge(dff, d, left_index=True, right_index=True, how="inner")

def engineer(df, lead_days=1):
    d = df.copy().sort_index()
    d["dis_lag1"] = d["discharge_m3_s"].shift(1)
    d["dis_lag2"] = d["discharge_m3_s"].shift(2)
    d["dis_lag3"] = d["discharge_m3_s"].shift(3)
    d["rain_roll_3"] = d["rainfall_mm"].rolling(3, min_periods=1).sum().shift(1)
    d["rain_roll_7"] = d["rainfall_mm"].rolling(7, min_periods=1).sum().shift(1)
    d["api"] = d["rainfall_mm"].shift(1).rolling(7, min_periods=1).apply(
        lambda x: (x * np.power(0.85, np.arange(len(x))[::-1])).sum(), raw=False
    )
    d["month_sin"] = np.sin(2 * np.pi * d.index.month / 12)
    d["month_cos"] = np.cos(2 * np.pi * d.index.month / 12)
    d["target"] = d["discharge_m3_s"].shift(-lead_days)
    d = d.fillna(method="bfill").fillna(method="ffill")
    d = d.dropna(subset=["target"])
    return d


FEATURES = [
    "discharge_m3_s", "dis_lag1", "dis_lag2", "dis_lag3",
    "rainfall_mm", "rain_roll_3", "rain_roll_7", "api",
    "soil_moisture", "evapotranspiration", "temperature",
    "month_sin", "month_cos"
]

SEQ_FEATURES = [
    "rainfall_mm", "discharge_m3_s", "dis_lag1", "dis_lag2",
    "rain_roll_3", "rain_roll_7", "api", "temperature", "evapotranspiration"
]


def augment_peaks(df, n_augment=N_AUGMENT, factor_range=(1.1, 1.4)):
    if df.empty: return df
    thr = df["discharge_m3_s"].quantile(0.9)
    peaks = df[df["discharge_m3_s"] > thr]
    if peaks.empty: return df
    out = [df]
    for _ in range(n_augment):
        s = np.random.uniform(*factor_range)
        cp = peaks.copy()
        cp["discharge_m3_s"] *= s
        out.append(cp)
    return pd.concat(out)


def make_seq(df, seq_len=SEQ_LENGTH, feat_cols=SEQ_FEATURES, target="resid_std"):
    arr = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df[target].astype(np.float32).values
    Xs, ys = [], []
    for i in range(seq_len - 1, len(df)):
        Xs.append(arr[i - seq_len + 1:i + 1])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def build_bilstm(input_shape, units=LSTM_UNITS, dropout=DROPOUT):
    i = Input(shape=input_shape)
    x = Bidirectional(LSTM(units, return_sequences=True))(i)
    avg, mx = GlobalAveragePooling1D()(x), GlobalMaxPooling1D()(x)
    c = Concatenate()([avg, mx])
    c = Dropout(dropout)(c)
    o = Dense(1)(c)
    m = Model(i, o)
    m.compile(optimizer="adam", loss="mse")
    return m


# ---------------------- TRAIN ----------------------
def train_v17(lat=LAT, lon=LON, start=DEFAULT_START, end=DEFAULT_END, lead=1, quick=False):
    print(f"HydroFusion v17 — Training {start}→{end} (lat={lat},lon={lon})")

    dfh = fetch_archive(lat, lon, start, end)
    dff = fetch_flood(lat, lon, start, end)
    if dfh.empty or dff.empty:
        raise RuntimeError("⚠️ No data returned from the APIs.")
    d = merge_daily(dfh, dff)
    print(f"→ Merged daily shape: {d.shape}")
    d = engineer(d, lead)

    if len(d) < 100:
        raise RuntimeError(f"⚠️ Not enough valid data after feature engineering ({len(d)} rows).")

    split = int(len(d) * 0.8)
    tr, te = d.iloc[:split], d.iloc[split:]
    print(f"Train {len(tr)}, Test {len(te)}")

    tr = augment_peaks(tr)
    Xtr, ytr = tr[FEATURES], tr["target"]
    Xte, yte = te[FEATURES], te["target"]

    scaler = StandardScaler()
    Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr), columns=Xtr.columns)
    Xte_s = pd.DataFrame(scaler.transform(Xte), columns=Xte.columns)

    # ---- LightGBM base ----
    tscv = TimeSeriesSplit(4)
    ytr_log = np.log1p(ytr.clip(lower=0))
    oof = np.zeros(len(Xtr))
    for f, (a, b) in enumerate(tscv.split(Xtr_s)):
        print(f"Fold {f+1}")
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8)
        m.fit(Xtr_s.iloc[a], ytr_log.iloc[a])
        oof[b] = m.predict(Xtr_s.iloc[b])

    lgb_final = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03)
    lgb_final.fit(Xtr_s, ytr_log)
    base_pred = np.expm1(lgb_final.predict(Xte_s))

    # ---- Residual learning ----
    resid = ytr.values - np.expm1(oof)
    mu, std = resid.mean(), resid.std() + 1e-6
    tr["resid_std"] = (resid - mu) / std

    Xs, ys = make_seq(tr)
    discharge_at_end = tr["discharge_m3_s"].iloc[-len(ys):]
    thr = ytr.quantile(0.9)
    w = np.where(discharge_at_end > thr, 1 + PEAK_ALPHA, 1)

    lstm = build_bilstm((Xs.shape[1], Xs.shape[2]))
    lstm.fit(Xs, ys, sample_weight=w, epochs=EPOCHS, batch_size=32, verbose=0,
             validation_split=0.1, shuffle=False,
             callbacks=[EarlyStopping(patience=6, restore_best_weights=True),
                        ReduceLROnPlateau(patience=3, factor=0.5)])

    tail = tr.tail(SEQ_LENGTH - 1)
    te_seq = pd.concat([tail, te])
    te_seq["resid_std"] = 0.0
    Xs_te, _ = make_seq(te_seq)
    r_pred = safe_predict(lstm, Xs_te) * std + mu
    final_pred = base_pred[-len(r_pred):] + r_pred

    # ---- Metrics ----
    base_m = metrics_summary(yte.values, base_pred)
    fin_m = metrics_summary(yte.values, final_pred)
    print("\n--- Base ---"); [print(k, ":", round(v, 3)) for k, v in base_m.items()]
    print("\n--- Final ---"); [print(k, ":", round(v, 3)) for k, v in fin_m.items()]

    # ---- Plots ----
    idx = Xte_s.index
    plt.figure(figsize=(12, 5))
    plt.plot(idx, yte.values, label="Observed")
    plt.plot(idx, final_pred, "--", label="Predicted")
    plt.legend(); plt.title("v17 Observed vs Predicted")
    plt.tight_layout(); plt.savefig("v17_backtest.png"); plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(yte, final_pred, alpha=0.6)
    lims = [min(yte.min(), final_pred.min()), max(yte.max(), final_pred.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Observed"); plt.ylabel("Predicted")
    plt.tight_layout(); plt.savefig("v17_obs_vs_pred.png"); plt.close()

    joblib.dump({"lgb": lgb_final, "scaler": scaler}, os.path.join(MODEL_DIR, MODEL_FILE))
    print("✅ Saved model to", MODEL_FILE)
    return {"base": base_m, "final": fin_m}


if __name__ == "__main__":
    res = train_v17()
    print("✅ Training complete:", res)
