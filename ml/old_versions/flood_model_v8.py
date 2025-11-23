#!/usr/bin/env python3
import argparse
import os
import joblib
import json
import math
from datetime import timedelta
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from dotenv import load_dotenv
import time 

# -------------------------
# Configuration / Defaults
# -------------------------
LAT = 23.81
LON = 90.41
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_LEAD = 1  # days ahead

MODEL_DIR = "ml"
MODEL_FILENAME = "open_meteo_flood_model_v8.pkl"
FEATURES_JSON = "open_meteo_flood_features_v8.json"
PLOT_FULL = "v8_backtest.png"
PLOT_PEAKS = "v8_backtest_peaks.png"

FLOOD_QUANTILE = 0.90
FLOOD_WEIGHT_ALPHA = 12.0  # strong emphasis for peaks (tunable)
TIMEZONE = "auto"
OPEN_METEO_FLOOD = "https://flood-api.open-meteo.com/v1/flood"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

RANDOM_STATE = 42

# -------------------------
# Utilities
# -------------------------
def safe_get_json(url, params, timeout=60):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def nse(obs, sim):
    obs = np.array(obs)
    sim = np.array(sim)
    denom = np.sum((obs - obs.mean())**2)
    if denom == 0:
        return float("nan")
    return 1 - np.sum((obs - sim)**2) / denom

# Combined scorer: 70% overall R2 + 30% peak R2
def combined_r2_peak(y_true, y_pred, alpha=0.7, peak_q=FLOOD_QUANTILE):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    overall = r2_score(y_true, y_pred)
    thr = np.nanpercentile(y_true, peak_q * 100)
    mask = y_true >= thr
    if mask.sum() >= 2:
        peak = r2_score(y_true[mask], y_pred[mask])
    else:
        peak = overall
    return alpha * overall + (1 - alpha) * peak

COMBINED_SCORER = make_scorer(combined_r2_peak, greater_is_better=True)

# -------------------------
# Data fetching & aggregation
# -------------------------
def fetch_weather_data(lat, lon, start_date, end_date):
    print(f"Fetching data for {start_date} → {end_date}")

    # normalize and clip to today — archive API will reject future dates
    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()
    today = pd.Timestamp.utcnow().date()
    if end_dt > today:
        print(f"Warning: requested end_date {end_dt} is in the future. Clipping to today {today}.")
        end_dt = today
    if start_dt > end_dt:
        raise ValueError(f"start_date ({start_dt}) is after end_date ({end_dt}) after clipping to today.")

    # build year-segmented requests (one segment per year span)
    dates = pd.date_range(start=start_dt, end=end_dt, freq="YS")
    dfs = []
    for i in range(len(dates)):
        seg_start = dates[i].date()
        seg_end = (dates[i + 1] - pd.Timedelta(days=1)).date() if i + 1 < len(dates) else end_dt
        print(f"→ Fetching segment {seg_start} → {seg_end}")
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation,soil_moisture_0_1cm,et0_fao_evapotranspiration,temperature_2m",
            "start_date": str(seg_start),
            "end_date": str(seg_end),
            "timezone": TIMEZONE
        }
        try:
            jw = safe_get_json(OPEN_METEO_ARCHIVE, params)
        except requests.HTTPError as e:
            # provide clearer diagnostics for 4xx/5xx from the API
            resp_text = ""
            try:
                resp_text = e.response.text
            except Exception:
                pass
            print(f"Error fetching {seg_start}→{seg_end}: {e}. Response body: {resp_text}")
            raise

        temp_df = pd.DataFrame(jw.get("hourly", {}))

        # ensure datetime index
        if 'time' in temp_df.columns:
            temp_df['time'] = pd.to_datetime(temp_df['time'])
            temp_df = temp_df.set_index('time')
        else:
            # if no explicit time column, try to interpret index or raise later
            pass

        # normalize common column names from Open-Meteo -> script expected names
        rename_map = {}
        cols = set(temp_df.columns)
        if 'precipitation' in cols:
            rename_map['precipitation'] = 'rainfall_mm'
        elif 'rainfall' in cols:
            rename_map['rainfall'] = 'rainfall_mm'

        if 'soil_moisture_0_1cm' in cols:
            rename_map['soil_moisture_0_1cm'] = 'soil_moisture'
        if 'et0_fao_evapotranspiration' in cols:
            rename_map['et0_fao_evapotranspiration'] = 'evapotranspiration'
        if 'temperature_2m' in cols:
            rename_map['temperature_2m'] = 'temperature'

        temp_df = temp_df.rename(columns=rename_map)

        # ensure max_hourly_rain_mm exists (placeholder: equal to hourly rainfall if not provided)
        if 'max_hourly_rain_mm' not in temp_df.columns:
            if 'rainfall_mm' in temp_df.columns:
                temp_df['max_hourly_rain_mm'] = temp_df['rainfall_mm']
            else:
                temp_df['max_hourly_rain_mm'] = 0.0

        dfs.append(temp_df)
        time.sleep(1.5)  # polite delay to avoid throttling
    if len(dfs) == 0:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=False).sort_index()
    return df

def fetch_discharge_data(lat, lon, start_date, end_date):
    """Fetches historical river discharge (daily) from the Flood API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "river_discharge",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TIMEZONE
    }
    
    # Use the Flood API URL
    jw = safe_get_json(OPEN_METEO_FLOOD, params)
    
    df = pd.DataFrame(jw.get("daily", {}))
    if 'time' in df.columns:
        df = df.rename(columns={"time": "date", "river_discharge": "discharge_m3_s"})
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    return df

# -------------------------
# Feature engineering (v8)
# -------------------------
def engineer_features_v8(df, lead_days=1, api_k=0.85):
    df = df.copy()

    # ensure datetime index (some fetches may not have set it)
    if not np.issubdtype(df.index.dtype, np.datetime64):
        if 'time' in df.columns:
            df.index = pd.to_datetime(df['time'])
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                raise ValueError("Dataframe index is not datetime and no 'time' column found. "
                                 "Ensure fetch_data produced a 'time' index/column.")

    df = df.sort_index()

    # check required input columns are present
    required_cols = ['discharge_m3_s', 'rainfall_mm', 'soil_moisture', 'evapotranspiration', 'temperature']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for feature engineering: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            "Tip: the archive hourly result often includes 'precipitation' (mapped to 'rainfall_mm'), "
            "'soil_moisture_0_1cm' (mapped to 'soil_moisture'), 'et0_fao_evapotranspiration' and 'temperature_2m'. "
            "However river discharge is NOT returned by the archive endpoint in many setups — "
            "you must supply a column named 'discharge_m3_s' (e.g. from the flood API or an external discharge dataset)."
        )

    # Discharge lags
    df['discharge_lag_1'] = df['discharge_m3_s'].shift(1)
    df['discharge_lag_2'] = df['discharge_m3_s'].shift(2)
    df['discharge_lag_3'] = df['discharge_m3_s'].shift(3)

    # Rainfall lags and rolling sums
    df['rain_lag_1'] = df['rainfall_mm'].shift(1)
    df['rain_lag_2'] = df['rainfall_mm'].shift(2)
    df['rain_lag_3'] = df['rainfall_mm'].shift(3)
    df['rain_lag_7'] = df['rainfall_mm'].shift(7)

    df['rain_roll_3'] = df['rainfall_mm'].shift(1).rolling(window=3).sum()
    df['rain_roll_7'] = df['rainfall_mm'].shift(1).rolling(window=7).sum()

    # # Antecedent Precipitation Index (recursive)
    # api = []
    # prev = 0.0
    # for t, r in enumerate(df['rainfall_mm'].fillna(0).values):
    #     val = r + api_k * prev
    #     api.append(val)
    #     prev = val
    # df['api_recursive'] = np.array(api).reshape(-1)

    # Antecedent Rainfall Index (Weighted Sum - decays over 7 days)
    # We use np.power(api_k, i) to generate the decaying weights (0.85^1, 0.85^2, etc.)
    WINDOW = 7 # Look back 7 days
    weights = np.power(api_k, np.arange(WINDOW)) # e.g., [1, 0.85, 0.72, ...]

    # Apply the weighted sum. We shift by 1 day so we use past rain only.
    df['ari_weighted_7d'] = df['rainfall_mm'].shift(1).rolling(window=WINDOW).apply(
        lambda x: np.sum(x.values * weights[::-1]), raw=False
    )

    # storm intensity proxy
    df['max_hourly_rain_mm'] = df['max_hourly_rain_mm'].fillna(0)

    # interaction & momentum
    df['rain_x_soil'] = df['rain_lag_1'] * df['soil_moisture'].shift(1)
    df['discharge_rate'] = df['discharge_m3_s'] - df['discharge_lag_1']

    # cyclical month
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # forecast-style feature: pseudo-forecast = observed rainfall on target day (optimistic)
    df['rainfall_forecast_Nd'] = df['rainfall_mm'].shift(-lead_days)

    # target: discharge at t + lead_days
    df['target'] = df['discharge_m3_s'].shift(-lead_days)

    # drop rows without target
    df = df.dropna(subset=['target'])
    return df

# -------------------------
# Prepare X, y and split
# -------------------------
def prepare_xy(df_feat, feature_list):
    X = df_feat[feature_list].fillna(0)
    y = df_feat['target']
    return X, y

# -------------------------
# Training & evaluation helpers
# -------------------------
def compute_sample_weights(y_train, quantile=FLOOD_QUANTILE, alpha=FLOOD_WEIGHT_ALPHA):
    thr = np.nanpercentile(y_train, quantile * 100)
    weights = np.where(y_train >= thr, 1.0 + alpha, 1.0)
    return weights, thr

def evaluate_predictions(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    nse_val = nse(y_test, y_pred)
    thr = np.nanpercentile(y_test, FLOOD_QUANTILE * 100)
    mask = y_test >= thr
    r2_peak = r2_score(y_test[mask], y_pred[mask]) if mask.sum() >= 2 else float('nan')
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'nse': nse_val, 'r2_peak': r2_peak, 'thr': thr}

# -------------------------
# Main training routine
# -------------------------
# Assuming engineer_features_v8, fetch_weather_data, fetch_discharge_data, prepare_xy,
# compute_sample_weights, evaluate_predictions, predict_and_unlog, 
# score_for_selection are defined elsewhere in your file.

def main_train(lat, lon, start_date, end_date, lead_days=1, quick=False):
    """Orchestrates fetching, feature engineering, training, and evaluation."""
    print(f"Fetching data for lat={lat}, lon={lon}, {start_date} → {end_date}")
    
    # 1. Fetch HOURLY weather data (from the Archive API)
    # This function should handle the yearly segmentation and column renaming.
    df_weather_hourly = fetch_weather_data(lat, lon, start_date, end_date)

    # 2. Fetch DAILY discharge data (from the Flood API - NEW)
    print("Fetching historical river discharge...")
    df_discharge_daily = fetch_discharge_data(lat, lon, start_date, end_date)
    
    # 3. Aggregate hourly weather data into daily averages/sums
    print("Aggregating hourly weather data...")
    if df_weather_hourly.empty:
        raise ValueError("No hourly weather data returned.")
        
    # Standard column aggregation logic used in engineer_features_v8:
    df_weather_hourly_agg = df_weather_hourly.resample('D').agg({
        'rainfall_mm': 'sum',
        'soil_moisture': 'mean',
        'evapotranspiration': 'mean',
        'temperature': 'mean',
        'max_hourly_rain_mm': 'max',
    })
    
    # 4. Merge Discharge and Aggregated Weather (CRITICAL STEP)
    # The merge must be left_index (Discharge Date) and right_index (Weather Date)
    df = df_discharge_daily.merge(
        df_weather_hourly_agg, 
        left_index=True, 
        right_index=True, 
        how='inner'
    ).dropna(subset=['discharge_m3_s']) # Drop rows where we have weather but no discharge
    
    print(f"Total merged daily records for feature engineering: {len(df)}")
    
    # 5. Feature Engineering
    print("Engineering features...")
    df_feat = engineer_features_v8(df, lead_days=lead_days) 

    # --- Define Feature List ---
    # This needs to be defined in main_train to ensure it's saved correctly
    feature_list = [
        'discharge_m3_s', 'discharge_lag_1', 'discharge_lag_2', 'discharge_lag_3',
        'discharge_rate',
        'rain_lag_1', 'rain_lag_2', 'rain_lag_3', 'rain_lag_7',
        'rain_roll_3', 'rain_roll_7', 'ari_weighted_7d',
        'rainfall_forecast_Nd', 'rain_x_soil', 'max_hourly_rain_mm',
        'soil_moisture', 'evapotranspiration', 'temperature',
        'month_sin', 'month_cos'
    ]
    # --- End Feature List ---

    # 6. Prepare X, y and time-based split
    X, y = prepare_xy(df_feat, feature_list)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}")

    # 7. Model Training and Selection (GridSearch)
    # (Rest of the model selection logic remains the same)
    
    # ... (code for sample_weights, log-transform, GridSearchCV setup) ...
    # ... (code for fitting RF and GB models) ...
    # ... (code for prediction and evaluation) ...
    # ... (code for choosing best model and saving results) ...

    # Final execution is here:
    # (The rest of your existing main_train logic for fitting, evaluating, and saving)
    
    # ... (The training, evaluation, and saving code block that starts around line 330) ...
    
    # Assuming you still want to run the full training process:
    # ----------------------------------------------------
    # sample weights to emphasize floods
    sample_weights, thr = compute_sample_weights(y_train.values)
    print(f"Flood threshold (train 90th pct) = {thr:.2f} m3/s  -> weights: high flows get +{FLOOD_WEIGHT_ALPHA}")

    # optionally log-transform the target to stabilize variance
    use_log = True
    y_train_fit = np.log1p(y_train) if use_log else y_train.values

    # model candidates: RandomForest and GradientBoosting
    # tune small grid (increase if you have time)
    rf_param_grid = {'n_estimators': [200, 400], 'max_depth': [12, 20], 'min_samples_leaf': [1, 2]}
    gb_param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
    cv_splits = 3 if quick else 4
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    print(f"Training models directly with sample_weight (alpha={FLOOD_WEIGHT_ALPHA})...")
    
    # 1) RandomForest
    # We use the best params we just found in the last run
    print("Training RandomForest...")
    rf_best = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    # Fit DIRECTLY with sample_weight
    rf_best.fit(X_train, y_train_fit, sample_weight=sample_weights)

    # 2) GradientBoosting
    # We use the best params we just found in the last run
    print("Training GradientBoosting...")
    gb_best = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    )
    # Fit DIRECTLY with sample_weight
    gb_best.fit(X_train, y_train_fit, sample_weight=sample_weights)
    
    print("Direct training complete.")

    # Evaluate both on test set (inverse-transform if log used)
    def predict_and_unlog(model, X):
        pred = model.predict(X)
        return np.expm1(pred) if use_log else pred

    rf_pred = predict_and_unlog(rf_best, X_test)
    gb_pred = predict_and_unlog(gb_best, X_test)

    rf_metrics = evaluate_predictions(y_test.values, rf_pred)
    gb_metrics = evaluate_predictions(y_test.values, gb_pred)

    print("\nRandomForest test metrics:", rf_metrics)
    print("GradientBoosting test metrics:", gb_metrics)

    # choose best by R2_peak first, falling back to R2
    def score_for_selection(m):
        # prefer higher peak R2, then higher overall R2
        return (m['r2_peak'] if not math.isnan(m['r2_peak']) else -999), m['r2']

    rf_sel = score_for_selection(rf_metrics)
    gb_sel = score_for_selection(gb_metrics)
    chosen = 'rf' if rf_sel > gb_sel else 'gb'
    best_model = rf_best if chosen == 'rf' else gb_best
    best_metrics = rf_metrics if chosen == 'rf' else gb_metrics
    print(f"\nSelected best model: {chosen} (rf_sel={rf_sel}, gb_sel={gb_sel})")

    # Save model bundle and features
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {'model': best_model, 'features': feature_list, 'use_log_target': use_log, 'lead_days': lead_days}
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    joblib.dump(bundle, model_path)
    with open(os.path.join(MODEL_DIR, FEATURES_JSON), 'w') as f:
        json.dump({'features': feature_list, 'lead_days': lead_days}, f, indent=2)
    print(f"Saved model bundle to {model_path}")
    
    # 1. Prepare data for plotting
    y_test_series = pd.Series(y_test.values, index=y_test.index)

    # rf_pred and gb_pred are defined from evaluation, so best_pred is chosen based on logic
    best_pred = rf_pred if chosen == 'rf' else gb_pred 

    # 2. Plot Full Backtest (Overall Performance)
    plt.figure(figsize=(14,6))
    plt.plot(y_test_series.index, y_test_series.values, label='Observed Discharge', color='blue', alpha=0.8)
    plt.plot(y_test_series.index, best_pred, label='Predicted', color='red', linestyle='--')
    plt.fill_between(y_test_series.index, y_test_series.values, best_pred, color='gray', alpha=0.15)

    # Optional: Highlight the 90th percentile threshold where the model is struggling
    plt.axhline(best_metrics['thr'], color='orange', linestyle=':', label='90th Percentile Threshold')

    plt.title('v8 Backtest: Observed vs Predicted (General Flow)')
    plt.xlabel('Date'); plt.ylabel('Discharge (m^3/s)')
    plt.legend(); plt.grid(True)
    plt.tight_layout(); 
    plt.savefig(PLOT_FULL); # Saves as v8_backtest.png
    plt.close()
    print(f"Saved full backtest to {PLOT_FULL}")

    # 3. Plot Peaks Backtest (Flood Performance)
    # Get the 90th percentile threshold for filtering the flood peaks
    thr = best_metrics['thr'] if 'thr' in best_metrics else np.nanpercentile(y_test.values, FLOOD_QUANTILE*100)
    mask = y_test.values >= thr # Mask selects only the values above the flood threshold

    if mask.sum() > 0:
        plt.figure(figsize=(12,5))
        
        # Plot only the observed data points above the threshold
        plt.plot(y_test_series.index[mask], y_test_series.values[mask], 'o-', label='Observed Peaks', color='blue')
        
        # Plot only the predicted data points corresponding to the observed peaks
        plt.plot(y_test_series.index[mask], best_pred[mask], 'x--', label='Predicted Peaks', color='red')
        
        plt.title(f'v8 Peaks Backtest (>= {int(FLOOD_QUANTILE*100)}th percentile) - R2 Peak: {best_metrics["r2_peak"]:.3f}')
        plt.legend(); plt.grid(True); 
        plt.tight_layout(); 
        plt.savefig(PLOT_PEAKS); # Saves as v8_backtest_peaks.png
        plt.close()
        print(f"Saved peaks backtest to {PLOT_PEAKS}")
    else:
        print("No peaks to plot in test period.")
        
    print("\nFinal selected model metrics:", best_metrics)
    return best_metrics

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train'], default='train')
    p.add_argument('--lat', type=float, default=LAT)
    p.add_argument('--lon', type=float, default=LON)
    p.add_argument('--start', type=str, default=DEFAULT_START)
    p.add_argument('--end', type=str, default=DEFAULT_END)
    p.add_argument('--lead', type=int, default=DEFAULT_LEAD)
    p.add_argument('--quick', action='store_true', help='Use smaller hyperparameter grids for quicker runs')
    return p.parse_args()

if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    metrics = main_train(args.lat, args.lon, args.start, args.end, lead_days=args.lead, quick=args.quick)
    print("Done. Metrics:", metrics)
