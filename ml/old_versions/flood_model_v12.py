#!/usr/bin/env python3
"""
flood_model_v12.py - HydroFusion v12

Major upgrades vs v11:
 - Bidirectional LSTM with temporal pooling
 - LightGBM + XGBoost added to ensemble (optional)
 - SHAP interpretability (optional)
 - BayesianRidge peak calibrator
 - Better training callbacks, reduced leakage, robust dtype handling
 - Automatic plots saved: backtest, peaks, shap, calibration

Usage:
    python flood_model_v12.py --mode train [--quick]

Notes:
 - TensorFlow, lightgbm, xgboost, shap are optional but recommended.
 - Training may take time for full dataset; use --quick for fast iteration.
"""

import os, time, math, json, argparse
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler

# Optional libraries guarded
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
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import tensorflow as tf
    # I've updated these imports to the modern standard TF 2.x locations
    from keras.models import Model, Sequential
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
MODEL_FILE = "open_meteo_flood_model_v12.pkl"
FEATURES_JSON = "open_meteo_flood_features_v12.json"
PLOT_BACKTEST = "v12_backtest.png"
PLOT_PEAKS = "v12_backtest_peaks.png"
PLOT_SHAP = "v12_shap_summary.png"
PLOT_CAL = "v12_calibration.png"
PRED_CSV = "v12_test_predictions.csv"

RANDOM_STATE = 42
FLOOD_Q = 0.90
PEAK_WEIGHT_ALPHA = 18.0
TIME_SERIES_SPLITS = 4
SEQ_LENGTH = 14  # longer sequence for longer antecedent memory

# -----------------------
# Utilities
# -----------------------
def safe_get_json(url, params, timeout=90):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def metrics_summary(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}

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
            # I'm asking for the correct hourly variables here
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
        time.sleep(0.6) # I'm being polite to the API
    df_hourly = pd.concat(dfs).sort_index()
    return df_hourly

def fetch_flood_daily(lat, lon, start_date, end_date):
    """Fetches historical river discharge (daily) from the Flood API."""
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
# Feature engineering (v12)
# -----------------------
def build_daily_merge(df_hourly, df_flood):
    # I'll aggregate all hourly data into daily data
    daily_precip = df_hourly['precipitation'].resample('D').sum().rename('rainfall_mm')
    daily_soil = df_hourly['soil_moisture_0_1cm'].resample('D').mean().rename('soil_moisture')
    daily_et = df_hourly['et0_fao_evapotranspiration'].resample('D').mean().rename('evapotranspiration')
    daily_temp = df_hourly['temperature_2m'].resample('D').mean().rename('temperature')
    daily_max_hour = df_hourly['precipitation'].resample('D').max().rename('max_hourly_rain_mm')
    
    df_daily = pd.concat([daily_precip, daily_soil, daily_et, daily_temp, daily_max_hour], axis=1)
    df_daily.index.name = 'date'
    
    # I'm merging the river data with the new daily weather data
    df = pd.merge(df_flood, df_daily, left_index=True, right_index=True, how='inner')
    return df

def engineer_v12(df_daily, lead_days=1, api_k=0.85):
    d = df_daily.copy().sort_index()
    
    # I check for required columns
    required_cols = ['discharge_m3_s', 'rainfall_mm', 'soil_moisture', 'evapotranspiration', 'temperature']
    missing = [c for c in required_cols if c not in d.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for feature engineering: {missing}\n"
            f"Available columns: {list(d.columns)}\n"
        )

    # Discharge lags
    d['dis_lag1'] = d['discharge_m3_s'].shift(1)
    d['dis_lag2'] = d['discharge_m3_s'].shift(2)
    d['dis_lag3'] = d['discharge_m3_s'].shift(3)

    # Rainfall lags and rolling sums
    for i in [1,2,3,7,14]:
        d[f'rain_lag_{i}'] = d['rainfall_mm'].shift(i)
    d['rain_roll_3'] = d['rainfall_mm'].shift(1).rolling(3).sum()
    d['rain_roll_7'] = d['rainfall_mm'].shift(1).rolling(7).sum()
    d['rain_roll_14'] = d['rainfall_mm'].shift(1).rolling(14).sum()
    d['rain_grad_1_2'] = d['rain_lag_1'] - d['rain_lag_2']
    
    # ET & ET lag
    d['et_lag_1'] = d['evapotranspiration'].shift(1)
    
    # Antecedent Rainfall Index (Weighted Sum)
    WINDOW = 7 
    weights = np.power(api_k, np.arange(WINDOW))
    d['api'] = d['rainfall_mm'].shift(1).rolling(window=WINDOW).apply(
        lambda x: np.sum(x.values * weights[::-1]), raw=False
    )

    # intensity & interactions
    d['max_hourly_rain_mm'] = d['max_hourly_rain_mm'].fillna(0)
    d['rain_x_soil'] = d['rain_lag_1'] * d['soil_moisture'].shift(1)
    d['dis_rate'] = d['discharge_m3_s'] - d['dis_lag1']
    
    # cyclical time encodings
    d['month'] = d.index.month
    d['month_sin'] = np.sin(2*np.pi*d['month']/12)
    d['month_cos'] = np.cos(2*np.pi*d['month']/12)
    
    # pseudo-forecast (for training use future observed as perfect forecast)
    d['rainfall_forecast_1d'] = d['rainfall_mm'].shift(-lead_days)
    
    # target
    d['target'] = d['discharge_m3_s'].shift(-lead_days)
    
    # drop rows without target or lags
    d = d.dropna(subset=['target', 'dis_lag3', 'rain_lag_14', 'api'])
    return d

# -----------------------
# Augmentation (multi-day surges)
# -----------------------
def augment_with_surges(df, n_augment=400, max_extra_mm=160.0, runoff_coeff=0.55):
    rng = np.random.default_rng(RANDOM_STATE)
    df_aug = df.copy()
    indices = df.index.values
    
    new_rows = []
    
    for i in range(n_augment):
        base_idx = rng.choice(indices[len(indices)//4:])
        base_row = df.loc[base_idx].copy()
        
        extra_total = rng.uniform(10.0, max_extra_mm)
        days = rng.integers(1,4)
        per_day = extra_total / days
        
        for d_i in range(1, days+1):
            col = f'rain_lag_{d_i}' if f'rain_lag_{d_i}' in base_row.index else 'rain_lag_1'
            base_row[col] = (base_row.get(col, 0.0) or 0.0) + per_day
        
        base_row['rain_roll_3'] = (base_row.get('rain_roll_3',0) or 0) + extra_total
        base_row['api'] = (base_row.get('api',0) or 0) + extra_total
        base_row['max_hourly_rain_mm'] = max(base_row.get('max_hourly_rain_mm',0), per_day)
        
        # Apply interaction feature
        base_row['rain_x_soil'] = (base_row.get('rain_lag_1', 0.0) or 0.0) * (base_row.get('soil_moisture', 0.0) or 0.0)

        base_row['target'] = base_row['target'] + runoff_coeff * extra_total
        base_row['is_synthetic'] = 1
        
        # Use a unique timestamp to avoid index collision
        new_index = base_idx + pd.Timedelta(nanoseconds=i+1)
        base_row.name = new_index
        new_rows.append(base_row)

    if new_rows:
        df_aug = pd.concat([df_aug, pd.DataFrame(new_rows)], ignore_index=False)
        
    df_aug['is_synthetic'] = df_aug['is_synthetic'].fillna(0).astype(int)
    df_aug = df_aug.sample(frac=1, random_state=RANDOM_STATE)
    return df_aug

# -----------------------
# Features
# -----------------------
# --- FIX: REMOVED THE CONSTANT FEATURES ---
FEATURE_LIST = [
 'discharge_m3_s','dis_lag1','dis_lag2','dis_lag3','dis_rate',
 'rain_lag_1','rain_lag_2','rain_lag_3','rain_lag_7','rain_lag_14',
 'rain_roll_3','rain_roll_7','rain_roll_14','api',
 'rainfall_forecast_1d', #'rain_x_soil', <-- This was constant
 'max_hourly_rain_mm',
 'evapotranspiration','et_lag_1','temperature', #'soil_moisture', <-- This was constant
 'month_sin','month_cos','rain_grad_1_2',
 #'lstm_oof' <-- This was also constant
]

def prepare_tabular_Xy(df):
    # I added a check here for missing features from the list
    missing_in_df = [f for f in FEATURE_LIST if f not in df.columns]
    if missing_in_df:
        print(f"Warning: Features missing from DataFrame, will be filled with 0: {missing_in_df}")
        for f in missing_in_df:
            df[f] = 0.0
            
    X = df[FEATURE_LIST].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    y = df['target'].astype(np.float32)
    return X, y

# -----------------------
# Sequence utilities (for LSTM)
# -----------------------
def build_sequences(df, seq_len=SEQ_LENGTH, features_seq=['rainfall_mm','discharge_m3_s','soil_moisture','max_hourly_rain_mm']):
    d = df.copy().sort_index()
    
    # I check and fill missing seq features
    for f in features_seq:
        if f not in d.columns:
            d[f] = 0.0
            
    arr = d[features_seq].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    targets = d['target'].astype(np.float32).values
    Xs, ys, idxs = [], [], []
    
    for i in range(seq_len, len(d)):
        seq = arr[i-seq_len:i]
        # I ensure no NaNs/Infs are in the sequence
        if not np.isfinite(seq).all():
            continue 
        Xs.append(seq)
        ys.append(targets[i])
        idxs.append(d.index[i])
        
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(idxs)

def build_bidirectional_lstm(input_shape, units=64, dropout=0.2):
    # input_shape = (timesteps, features)
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

# Factory to create a new model instance
def create_lgbm_model():
    return lgb.LGBMRegressor(
        n_estimators=400 if not ARGS.quick else 100,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_data_in_leaf=5,
        min_gain_to_split=0.001,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1
    )


# -----------------------
# OOF helper for time series
# -----------------------
def oof_preds_time_series(est_factory, X, y, n_splits=4, fit_params=None):
    """
    Time-aware Out-Of-Fold (OOF) training and prediction.
    """

    print("\n🧩 Checking training data integrity...")
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, index=y.index)
    if isinstance(y, np.ndarray):
        y = pd.Series(y, index=y.index)

    print("Feature summary:")
    print(X.describe().T)
    print("\nTarget summary:")
    print(y.describe())

    if y.nunique() <= 1:
        raise ValueError("❌ Target has only one unique value — cannot train.")

    # Initialize OOF preds
    oof_preds = np.full(len(X), np.nan, dtype=float)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n📆 OOF fold {fold + 1}/{n_splits}: train {len(train_idx)} → val {len(val_idx)}")

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]

        if y_train.nunique() <= 1:
            print("⚠️ Skipping fold due to constant training labels.")
            continue

        est = est_factory()
        fit_kw = fit_params.copy() if fit_params else {}

        if "sample_weight" in fit_kw:
            sw = fit_kw["sample_weight"]
            if len(sw) == len(X):
                fit_kw["sample_weight"] = sw[train_idx]
            else:
                print("⚠️ sample_weight length mismatch; ignoring for this fold.")
                fit_kw.pop("sample_weight", None)

        # 🚧 Check for constant features in this specific fold
        const_cols = X_train.columns[X_train.std() == 0].tolist()
        if const_cols:
            print(f"⚠️ Removing constant features: {const_cols}")
            X_train = X_train.loc[:, X_train.std() > 0]
            X_val = X_val[X_train.columns] # Keep only non-constant cols

        est.fit(X_train, y_train, **fit_kw)
        
        preds = est.predict(X_val)
        oof_preds[val_idx] = preds

    # Final fit on all data
    print("\n🚀 Training final model on all data...")
    final_est = est_factory()
    final_fit_kw = fit_params.copy() if fit_params else {}
    if "sample_weight" in final_fit_kw:
        sw = final_fit_kw["sample_weight"]
        if len(sw) != len(X):
            print("⚠️ Final sample_weight length mismatch; ignoring.")
            final_fit_kw.pop("sample_weight", None)

    # Check for constant features in the *full* training set
    const_cols_full = X.columns[X.std() == 0].tolist()
    if const_cols_full:
        print(f"⚠️ Removing constant features from final model: {const_cols_full}")
        X_full_train = X.loc[:, X.std() > 0]
    else:
        X_full_train = X
        
    final_est.fit(X_full_train, y, **final_fit_kw)

    print("\n✅ OOF + Full model training completed successfully.")
    # --- FIX 1: Return the OOF predictions AND the final trained model ---
    return oof_preds, final_est

# -----------------------
# Train v12 pipeline
# -----------------------
ARGS = None # Global to store CLI args

def train_v12(lat=LAT, lon=LON, start=DEFAULT_START, end=DEFAULT_END, lead_days=1, quick=False):
    print(f"Fetching data for {start} → {end} (lat={lat}, lon={lon})")
    df_hourly = fetch_archive_hourly(lat, lon, start, end)
    df_flood = fetch_flood_daily(lat, lon, start, end)
    print("Aggregating hourly -> daily and merging...")
    df_daily = build_daily_merge(df_hourly, df_flood)
    print(f"Total merged daily rows: {len(df_daily)}")
    
    print("Engineering features...")
    df = engineer_v12(df_daily, lead_days=lead_days)

    # numeric safety
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # train/test split (time-based)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")

    # augmentation
    print("Augmenting training data with synthetic surges...")
    train_aug = augment_with_surges(train_df, n_augment=400 if not quick else 120)
    print(f"After augmentation: {len(train_aug)} rows")

    # threshold & sample weights
    thr = np.nanpercentile(train_aug['target'].values, FLOOD_Q*100)
    print(f"Peak threshold (train {int(FLOOD_Q*100)}th pct) = {thr:.2f}")
    sw = np.where(train_aug['target'].values >= thr, 1.0 + PEAK_WEIGHT_ALPHA, 1.0)
    if 'is_synthetic' in train_aug.columns:
        sw = sw * np.where(train_aug['is_synthetic']==1, 0.6, 1.0)

    # prepare tabular features
    X_train_tab, y_train_tab = prepare_tabular_Xy(train_aug)
    X_test_tab, y_test_tab = prepare_tabular_Xy(test_df)
    
    # scale numeric features
    # I scale based *only* on the non-synthetic part of the training data
    scaler = StandardScaler()
    scaler.fit(X_train_tab.loc[train_aug['is_synthetic']==0])
    X_train_tab_scaled = pd.DataFrame(scaler.transform(X_train_tab), index=X_train_tab.index, columns=X_train_tab.columns)
    X_test_tab_scaled = pd.DataFrame(scaler.transform(X_test_tab), index=X_test_tab.index, columns=X_test_tab.columns)

    # sequence (LSTM) OOF
    lstm_oof_train = None; lstm_oof_test = None; lstm_model_final = None
    seq_features = ['rainfall_mm','discharge_m3_s','soil_moisture','max_hourly_rain_mm']

    if TF_AVAILABLE:
        print("Building LSTM sequences (SEQ_LENGTH=%d)..." % SEQ_LENGTH)
        # I'm using the full augmented data for sequences
        Xs_train_seq, ys_train_seq, idxs_train_seq = build_sequences(train_aug, seq_len=SEQ_LENGTH, features_seq=seq_features)
        print("Sequence shapes:", Xs_train_seq.shape, ys_train_seq.shape)
        
        # --- FIX 2: Ensure data is valid for LSTM ---
        if len(Xs_train_seq) >= 50 and np.isfinite(Xs_train_seq).all() and np.isfinite(ys_train_seq).all():
            n = len(train_aug)
            lstm_oof = np.full(n, np.nan, dtype=float)
            tscv = TimeSeriesSplit(n_splits=TIME_SERIES_SPLITS)
            
            print("Preparing LSTM sequences and producing OOF predictions (time-safe)...")
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(np.arange(len(train_aug)))):
                print(f"LSTM OOF fold {fold+1}/{TIME_SERIES_SPLITS}")
                train_dates = train_aug.index[train_idx]
                val_dates = train_aug.index[val_idx]
                mask_train = np.isin(idxs_train_seq, train_dates)
                mask_val = np.isin(idxs_train_seq, val_dates)
                Xs_tr = Xs_train_seq[mask_train]; ys_tr = ys_train_seq[mask_train]
                Xs_val = Xs_train_seq[mask_val]
                
                if len(Xs_tr) < 20 or len(Xs_val) < 1:
                    print("  skipping fold (too few sequences)")
                    continue
                    
                model = build_bidirectional_lstm((Xs_tr.shape[1], Xs_tr.shape[2]), units=64, dropout=0.2)
                epochs = 6 if quick else 20
                
                # --- FIX 3: Add validation_data to callbacks ---
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True), 
                    ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5)
                ]
                model.fit(Xs_tr, ys_tr, epochs=epochs, batch_size=32, verbose=0, 
                          validation_data=(Xs_val, ys_train_seq[mask_val]), # <-- Pass validation data
                          callbacks=callbacks)
                
                preds_val = model.predict(Xs_val).reshape(-1)
                val_dates_seq = idxs_train_seq[mask_val]
                
                for d,p in zip(val_dates_seq, preds_val):
                    pos = np.where(train_aug.index == d)[0]
                    if pos.size:
                        lstm_oof[pos[0]] = p
            
            # fill nan
            nan_mask = np.isnan(lstm_oof)
            if nan_mask.any():
                fill = np.nanmedian(lstm_oof[~nan_mask]) if (~nan_mask).any() else np.nanmedian(ys_train_seq)
                lstm_oof[nan_mask] = fill if np.isfinite(fill) else 0.0 # Ensure fill is finite
            
            lstm_oof_train = pd.Series(lstm_oof, index=train_aug.index)
            
            # final train on all sequence data
            final_lstm = build_bidirectional_lstm((Xs_train_seq.shape[1], Xs_train_seq.shape[2]), units=64, dropout=0.2)
            final_epochs = 8 if quick else 40
            
            # We don't use callbacks for the final fit, just train for the full epochs
            final_lstm.fit(Xs_train_seq, ys_train_seq, epochs=final_epochs, batch_size=32, verbose=0)
            lstm_model_final = final_lstm
            
            # build test sequences from train tail + test for alignment
            combined = pd.concat([train_aug.tail(SEQ_LENGTH), test_df])
            Xs_test_seq, ys_test_seq, idxs_test_seq = build_sequences(combined, seq_len=SEQ_LENGTH, features_seq=seq_features)
            
            if len(Xs_test_seq) > 0:
                preds_test_seq = lstm_model_final.predict(Xs_test_seq).reshape(-1)
                test_map = {d: p for d,p in zip(idxs_test_seq, preds_test_seq) if d in test_df.index}
                lstm_oof_test = pd.Series([test_map.get(d, np.nan) for d in X_test_tab.index], index=X_test_tab.index)
                lstm_oof_test = lstm_oof_test.fillna(np.nanmedian(preds_test_seq) if len(preds_test_seq) > 0 else 0.0)
            else:
                print("No test sequences generated for LSTM.")
                lstm_oof_test = pd.Series(0.0, index=X_test_tab.index)
                
        else:
            print("Not enough sequence samples to train LSTM; skipping LSTM branch.")
    else:
        print("TensorFlow not available; skipping LSTM branch.")

    # attach LSTM oof preds as a tabular column
    if lstm_oof_train is not None:
        X_train_tab_scaled['lstm_oof'] = lstm_oof_train
    else:
        # Create a constant 0 feature if LSTM failed
        X_train_tab_scaled['lstm_oof'] = 0.0 
    if lstm_oof_test is not None:
        X_test_tab_scaled['lstm_oof'] = lstm_oof_test
    else:
        X_test_tab_scaled['lstm_oof'] = 0.0

    # ---------- train base models with OOF (time-aware) ----------
    print("Training base models with time-aware OOF...")

    # I've updated the factories to use the newer, more advanced LightGBM
    def lgb_factory(): return (lgb.LGBMRegressor(
        n_estimators=400 if not quick else 100,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1
    ) if LGB_AVAILABLE else None)
    
    def xgb_factory(): return (xgb.XGBRegressor(
        n_estimators=300 if not quick else 100,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1
    ) if XGB_AVAILABLE else None)

    # I've removed RF and GB from the OOF stack for simplicity and to focus on better models
    # You can add them back by creating factories and adding them to the 'models_to_run' list
    
    models_to_run = {
        'lgb': lgb_factory,
        'xgb': xgb_factory,
    }
    
    fit_params = {'sample_weight': sw}

    oof_preds = {}
    full_models = {}
    
    for name, factory in models_to_run.items():
        if factory() is not None:
            # --- I'm fixing the 'y' variable in this function call ---
            # I'm changing train_df['target'] (1710 rows) to y_train_tab (2112 rows)
            oof, full_model = oof_preds_time_series(factory, X_train_tab_scaled, y_train_tab, n_splits=TIME_SERIES_SPLITS, fit_params=fit_params)
            oof_preds[name] = oof
            full_models[name] = full_model
        else:
            print(f"{name} not available, skipping.")

    # meta training DataFrame
    meta_train = pd.DataFrame(oof_preds, index=train_aug.index)
    if 'lstm_oof' in X_train_tab_scaled:
       meta_train['lstm'] = X_train_tab_scaled['lstm_oof']

    # --- FIX 4: Remove Constant Features *before* prediction ---
    # The log showed these were constant. I am removing them from the feature list.
    const_cols = ['rain_x_soil', 'soil_moisture', 'lstm_oof']
    X_test_tab_scaled_final = X_test_tab_scaled.drop(columns=const_cols, errors='ignore')

    # full-model predictions on test
    base_test = pd.DataFrame(index=X_test_tab_scaled_final.index)
    if 'lgb' in full_models:
        base_test['lgb'] = full_models['lgb'].predict(X_test_tab_scaled_final)
    if 'xgb' in full_models:
        base_test['xgb'] = full_models['xgb'].predict(X_test_tab_scaled_final)
    if 'lstm' in meta_train.columns: # Check if LSTM training was successful
        base_test['lstm'] = X_test_tab_scaled['lstm_oof'].values

    # ---------- meta-learner (bayesian ridge or ridge) with peak-weighted fit ----------
    print("Training meta-learner (BayesianRidge) with peak-weighted fit...")
    
    # --- FIX 5: Use non-constant features for meta-learner ---
    meta_X = meta_train.drop(columns=const_cols, errors='ignore').fillna(0).values
    meta_y = train_aug['target'].values
    meta_sw = np.where(meta_y >= thr, 1.0 + PEAK_WEIGHT_ALPHA, 1.0)
    if 'is_synthetic' in train_aug.columns:
        meta_sw = meta_sw * np.where(train_aug['is_synthetic']==1, 0.6, 1.0)

    meta_model = BayesianRidge()
    meta_model.fit(meta_X, meta_y, sample_weight=meta_sw)
    
    # Ensure base_test has the same columns as meta_X
    base_test_final = base_test.drop(columns=const_cols, errors='ignore').fillna(0).values
    stack_test_pred = meta_model.predict(base_test_final)

    # ... (Peak calibrator logic remains the same)
    val_slice = int(len(train_aug) * 0.9)
    meta_val_df = train_aug.iloc[val_slice:].copy()
    
    # --- FIX 6: Ensure calibrator X data is correct ---
    X_val_tab_scaled = pd.DataFrame(scaler.transform(prepare_tabular_Xy(meta_val_df)[0]), index=meta_val_df.index, columns=X_train_tab.columns)
    if lstm_oof_train is not None:
        X_val_tab_scaled['lstm_oof'] = lstm_oof_train.loc[meta_val_df.index]
    else:
        X_val_tab_scaled['lstm_oof'] = 0.0

    X_val_tab_scaled_final = X_val_tab_scaled.drop(columns=const_cols, errors='ignore')

    base_val = pd.DataFrame(index=X_val_tab_scaled_final.index)
    if 'lgb' in full_models:
        base_val['lgb'] = full_models['lgb'].predict(X_val_tab_scaled_final)
    if 'xgb' in full_models:
        base_val['xgb'] = full_models['xgb'].predict(X_val_tab_scaled_final)
    if 'lstm' in meta_train.columns:
        base_val['lstm'] = X_val_tab_scaled['lstm_oof'].values

    base_val_final = base_val.drop(columns=const_cols, errors='ignore').fillna(0).values
    ensemble_val_preds = meta_model.predict(base_val_final)
    
    high_mask_val = meta_val_df['target'].values >= thr
    calibrator = None
    if high_mask_val.sum() >= 8:
        cal = Ridge(alpha=1.0)
        cal.fit(ensemble_val_preds[high_mask_val].reshape(-1,1), meta_val_df['target'].values[high_mask_val])
        calibrator = cal
        print("Trained peak calibrator.")
    else:
        print("Not enough high-flow samples for calibrator; skipping.")

    final_test_pred = stack_test_pred.copy()
    if calibrator is not None:
        # We need to find the high_mask_test based on unscaled data
        unscaled_thr = scaler.inverse_transform(X_test_tab.fillna(0))[X_test_tab.columns.get_loc('discharge_m3_s')].mean() # This is tricky, using a proxy
        # Let's use the original 'thr' as a proxy, assuming 'discharge_m3_s' is similar
        
        # A simpler way: predict on all, then find which ones to apply calibration to
        high_mask_test = final_test_pred >= thr # Predicts high
        if high_mask_test.sum() > 0:
            print(f"Applying calibration to {high_mask_test.sum()} high-flow test predictions...")
            final_test_pred[high_mask_test] = calibrator.predict(final_test_pred[high_mask_test].reshape(-1,1))
        else:
            print("No high-flow test predictions to calibrate.")


    # ---------- evaluation ----------
    y_test_vals = test_df['target'].values
    def eval_metrics(y_t, y_p):
        mae = mean_absolute_error(y_t, y_p)
        rmse = math.sqrt(mean_squared_error(y_t, y_p))
        r2v = r2_score(y_t, y_p)
        peak_mask = y_t >= thr
        r2_peak = r2_score(y_t[peak_mask], y_p[peak_mask]) if peak_mask.sum() >= 2 else float('nan')
        return {'mae': mae, 'rmse': rmse, 'r2': r2v, 'r2_peak': r2_peak}

    base_metrics = {}
    if 'lgb' in full_models: base_metrics['lgb'] = eval_metrics(y_test_vals, base_test['lgb'].values)
    if 'xgb' in full_models: base_metrics['xgb'] = eval_metrics(y_test_vals, base_test['xgb'].values)
    if 'lstm' in meta_train.columns: base_metrics['lstm'] = eval_metrics(y_test_vals, base_test['lstm'].values)

    final_metrics = eval_metrics(y_test_vals, final_test_pred)
    print("Base metrics:", base_metrics)
    print("Final stacked+calibrated metrics:", final_metrics)

    # ---------- SHAP explainability ----------
    if SHAP_AVAILABLE and 'lgb' in full_models: # Use LightGBM for SHAP
        try:
            print("Computing SHAP values for LightGBM (may take time)...")
            # We must use the non-constant features
            X_train_final_shap = X_train_tab_scaled.drop(columns=const_cols, errors='ignore')
            X_test_final_shap = X_test_tab_scaled.drop(columns=const_cols, errors='ignore')

            explainer = shap.TreeExplainer(full_models['lgb'])
            shap_values = explainer(X_test_final_shap)
            
            plt.figure(figsize=(10,6))
            shap.summary_plot(shap_values, X_test_final_shap, show=False, plot_type="bar")
            plt.tight_layout(); plt.savefig(PLOT_SHAP); plt.close()
            print(f"Saved SHAP summary to {PLOT_SHAP}")
        except Exception as e:
            print(f"SHAP failed: {e}")
    else:
        print("SHAP or LightGBM not available; skipping SHAP plots.")

    # ---------- Save model bundle ----------
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        'lgb_full': full_models.get('lgb'), 
        'xgb_full': full_models.get('xgb'),
        'lstm_model': lstm_model_final, 'meta_model': meta_model, 'calibrator': calibrator,
        'features': FEATURE_LIST, # Full list, including constant ones
        'features_used_by_model': X_train_final_shap.columns.tolist(), # Actual list used
        'seq_features': seq_features, 'seq_length': SEQ_LENGTH,
        'scaler': scaler, 'trained_range': {'start': start, 'end': end}
    }
    joblib.dump(bundle, os.path.join(MODEL_DIR, MODEL_FILE))
    with open(os.path.join(MODEL_DIR, FEATURES_JSON), 'w') as f:
        json.dump({'features': FEATURE_LIST, 'seq_features': seq_features, 'lead_days': lead_days}, f, indent=2)
    print(f"Saved model bundle to {os.path.join(MODEL_DIR, MODEL_FILE)}")

    # ---------- save predictions and plots ----------
    out_df = pd.DataFrame({
        'date': test_df.index,
        'observed': y_test_vals,
        'lgb_pred': base_test['lgb'] if 'lgb' in base_test else np.nan,
        'xgb_pred': base_test['xgb'] if 'xgb' in base_test else np.nan,
        'lstm_pred': base_test['lstm'] if 'lstm' in base_test else np.nan,
        'stack_pred': stack_test_pred,
        'final_pred': final_test_pred
    }, index=test_df.index)
    out_df.to_csv(PRED_CSV, index=False)
    print(f"Saved test predictions to {PRED_CSV}")

    # backtest plot
    idx = test_df.index
    plt.figure(figsize=(14,6))
    plt.plot(idx, y_test_vals, label='Observed', linewidth=1)
    plt.plot(idx, final_test_pred, '--', label='Final Pred', linewidth=1)
    plt.legend(); plt.grid(True); plt.title('v12 Backtest - Observed vs Final Pred'); plt.savefig(PLOT_BACKTEST); plt.close()
    print(f"Saved backtest plot to {PLOT_BACKTEST}")

    # peaks plot
    thr_mask = y_test_vals >= thr
    if thr_mask.sum() > 0:
        plt.figure(figsize=(12,5))
        plt.plot(idx[thr_mask], y_test_vals[thr_mask], 'o-', label='Observed peaks')
        plt.plot(idx[thr_mask], final_test_pred[thr_mask], 'x--', label='Predicted peaks')
        plt.legend(); plt.grid(True); plt.title('v12 Peaks'); plt.savefig(PLOT_PEAKS); plt.close()
        print(f"Saved peaks plot to {PLOT_PEAKS}")

    # calibration plot if calibrator exists
    if calibrator is not None:
        high_idx = np.where(y_test_vals >= thr)[0]
        if high_idx.size:
            plt.figure(figsize=(6,6))
            plt.scatter(stack_test_pred[high_idx], y_test_vals[high_idx], alpha=0.6)
            plt.plot([y_test_vals.min(), y_test_vals.max()],[y_test_vals.min(), y_test_vals.max()], 'k--')
            plt.xlabel('Ensemble pred'); plt.ylabel('Observed'); plt.title('Peak Calibration (test)')
            plt.grid(True); plt.savefig(PLOT_CAL); plt.close()
            print(f"Saved calibration plot to {PLOT_CAL}")

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
    res = train_v12(args.lat, args.lon, args.start, args.end, lead_days=args.lead, quick=args.quick)
    print("Training v12 complete. Results:", res)