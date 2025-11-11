#!/usr/bin/env python3
"""
flood_model_v7.py

v7 training pipeline: explicitly optimized for flood peak skill (top 10% flows).
Improvements over v6:
 - multiple antecedent rainfall windows (3/7/14/30 day sums)
 - multiple ARI (exponential) windows
 - daily max hourly precipitation (storm intensity)
 - optional upstream rainfall averaging hook
 - stronger flood weighting and a custom combined R2+peak-R2 scorer for hyperparameter search
 - RandomizedSearchCV over RF/XGB/LGB (if available), TimeSeriesSplit CV
 - saves model bundle + feature metadata and diagnostic plots
"""

import os
import argparse
import joblib
import json
from datetime import timedelta
import numpy as np
import pandas as pd
import requests
import math
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, make_scorer, mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import randint, uniform
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# optional model libs
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------
# CONFIG
# -----------------------
OPEN_METEO_FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "auto"

DEFAULT_LAT = 23.81
DEFAULT_LON = 90.41
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_LEAD_DAYS = 1

MODEL_OUTPUT_DIR = "ml"
MODEL_FILENAME = "open_meteo_flood_model_v7.pkl"
FEATURES_JSON = "open_meteo_flood_features_v7.json"
PLOT_FILENAME = "v7_backtest.png"
FLOOD_PLOT_FILENAME = "v7_backtest_peaks.png"

RANDOM_STATE = 42

# hyper
RANDOM_SEARCH_ITERS = 60
CV_SPLITS = 4

# flood emphasis
FLOOD_QUANTILE = 0.90
FLOOD_WEIGHT_ALPHA = 10.0  # stronger than v6

# -----------------------
# UTILITIES
# -----------------------
def safe_get_json(url, params, timeout=60):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def nse(observed, simulated):
    obs = np.array(observed)
    sim = np.array(simulated)
    denom = np.sum((obs - obs.mean())**2)
    if denom == 0:
        return float("nan")
    return 1 - np.sum((obs - sim)**2)/denom

# -----------------------
# DATA FETCHING
# -----------------------
def fetch_historical(lat, lon, start_date, end_date, include_hourly_max=True, upstream_coords=None):
    """
    Fetch flood daily discharge and hourly weather archive, aggregate to daily.
    If upstream_coords is list of (lat, lon) tuples, compute upstream average rainfall daily and add 'rainfall_upstream_mm'.
    """
    # flood daily
    flood_params = {"latitude": lat, "longitude": lon, "daily":"river_discharge", "start_date":start_date, "end_date":end_date, "timezone":TIMEZONE}
    jf = safe_get_json(OPEN_METEO_FLOOD_URL, flood_params)
    if 'daily' not in jf:
        raise RuntimeError("Flood API returned unexpected structure.")
    df_flood = pd.DataFrame(jf['daily']).rename(columns={"time":"date", "river_discharge":"discharge_m3_s"})
    df_flood['date'] = pd.to_datetime(df_flood['date'])
    df_flood = df_flood.set_index('date').sort_index()

    # archive hourly -> daily aggregates
    arc_params = {"latitude": lat, "longitude": lon, "hourly":["precipitation","soil_moisture_0_1cm","et0_fao_evapotranspiration","temperature_2m"], "start_date":start_date, "end_date":end_date, "timezone":TIMEZONE}
    jw = safe_get_json(OPEN_METEO_ARCHIVE_URL, arc_params)
    if 'hourly' not in jw:
        raise RuntimeError("Archive API returned unexpected structure.")
    df_hour = pd.DataFrame(jw['hourly'])
    df_hour['time'] = pd.to_datetime(df_hour['time'])
    df_hour = df_hour.set_index('time').sort_index()
    # daily sum/means
    daily_precip = df_hour['precipitation'].resample('D').sum()
    daily_soil = df_hour['soil_moisture_0_1cm'].resample('D').mean()
    daily_et = df_hour['et0_fao_evapotranspiration'].resample('D').mean()
    daily_temp = df_hour['temperature_2m'].resample('D').mean()
    df_daily = pd.concat([daily_precip, daily_soil, daily_et, daily_temp], axis=1)
    df_daily.columns = ['rainfall_mm','soil_moisture','evapotranspiration','temperature']

    # compute daily max hourly precip (intensity)
    if include_hourly_max:
        hourly_max = df_hour['precipitation'].resample('D').max().rename('max_hourly_rain_mm')
        df_daily = df_daily.join(hourly_max)

    # upstream rainfall average (optional) - simple approach: average daily precip across upstream coords
    if upstream_coords:
        upstream_means = []
        for (ulat, ulon) in upstream_coords:
            params = {"latitude": ulat, "longitude": ulon, "hourly":["precipitation"], "start_date":start_date, "end_date":end_date, "timezone":TIMEZONE}
            jk = safe_get_json(OPEN_METEO_ARCHIVE_URL, params)
            dfh = pd.DataFrame(jk['hourly'])
            dfh['time'] = pd.to_datetime(dfh['time'])
            dfh = dfh.set_index('time')
            upstream_means.append(dfh['precipitation'].resample('D').sum())
        if upstream_means:
            df_up = pd.concat(upstream_means, axis=1).mean(axis=1).rename('rainfall_upstream_mm')
            df_daily = df_daily.join(df_up)

    df_daily.index.name = 'date'
    # merge with flood daily (inner)
    df = pd.merge(df_flood, df_daily, left_index=True, right_index=True, how='inner')
    return df

# -----------------------
# FEATURE ENGINEERING (richer)
# -----------------------
def engineer_rich(df, lead_days=1):
    d = df.copy().sort_index()

    # lags & rolling sums
    d['discharge_lag_1'] = d['discharge_m3_s'].shift(1)
    d['discharge_lag_2'] = d['discharge_m3_s'].shift(2)
    d['rainfall_lag_1'] = d['rainfall_mm'].shift(1)
    d['rainfall_3_day_sum'] = d['rainfall_mm'].shift(1).rolling(window=3).sum()
    d['rainfall_7_day_sum'] = d['rainfall_mm'].shift(1).rolling(window=7).sum()
    d['rainfall_14_day_sum'] = d['rainfall_mm'].shift(1).rolling(window=14).sum()
    d['rainfall_30_day_sum'] = d['rainfall_mm'].shift(1).rolling(window=30).sum()
    d['rainfall_7_day_avg'] = d['rainfall_mm'].shift(1).rolling(window=7).mean()
    d['discharge_3_day_avg'] = d['discharge_m3_s'].shift(1).rolling(window=3).mean()
    d['discharge_rate_change'] = d['discharge_m3_s'] - d['discharge_lag_1']

    # ARIs with different decays & windows
    def ari_series(series, days=7, decay=0.7):
        res = np.zeros(len(series))
        for i in range(1, days+1):
            res += (decay ** (i-1)) * series.shift(i).fillna(0).values
        return res

    d['ari_7_d0.7'] = ari_series(d['rainfall_mm'], days=7, decay=0.7)
    d['ari_14_d0.85'] = ari_series(d['rainfall_mm'], days=14, decay=0.85)
    d['ari_30_d0.9'] = ari_series(d['rainfall_mm'], days=30, decay=0.9)

    # storm intensity proxies
    if 'max_hourly_rain_mm' in d.columns:
        d['max_hourly_rain_mm'] = d['max_hourly_rain_mm'].fillna(0)
    else:
        d['max_hourly_rain_mm'] = 0.0

    # interaction terms
    d['rain_x_soil'] = d['rainfall_mm'] * d['soil_moisture']
    if 'rainfall_upstream_mm' in d.columns:
        d['local_vs_upstream'] = d['rainfall_mm'] - d['rainfall_upstream_mm']
    else:
        d['local_vs_upstream'] = 0.0

    # cyclical month
    d['month'] = d.index.month
    d['month_sin'] = np.sin(2*np.pi*d['month']/12)
    d['month_cos'] = np.cos(2*np.pi*d['month']/12)

    # rainfall_forecast feature - training uses pseudo-forecast (next-day observed) as default
    d['rainfall_forecast_Nd'] = d['rainfall_mm'].shift(-lead_days)

    # target: discharge at t + lead_days
    d['target_discharge'] = d['discharge_m3_s'].shift(-lead_days)

    return d

# -----------------------
# Custom scorer (combined overall R2 + peak R2)
# -----------------------
def combined_peak_scorer(y_true, y_pred, alpha=0.6, peak_q=FLOOD_QUANTILE):
    """
    alpha: weight for overall R2 (0..1). 1-alpha for peak R2.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    overall = r2_score(y_true, y_pred)
    thr = np.nanpercentile(y_true, peak_q*100)
    peak_mask = y_true >= thr
    if peak_mask.sum() >= 2:
        peak = r2_score(y_true[peak_mask], y_pred[peak_mask])
    else:
        peak = overall  # fallback
    return alpha * overall + (1 - alpha) * peak

# sklearn scorer factory
from sklearn.metrics import make_scorer
COMBINED_SCORER = make_scorer(combined_peak_scorer, greater_is_better=True)

# -----------------------
# Model candidate builder
# -----------------------
def build_candidates():
    cands = []
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    rf_params = {'n_estimators': randint(150, 700),
                 'max_depth': randint(6, 30),
                 'min_samples_leaf': randint(1, 5),
                 'max_features': ['sqrt', 'log2', None]}
    cands.append(('rf', rf, rf_params))

    if XGB_AVAILABLE:
        xg = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1)
        xg_params = {'n_estimators': randint(100, 1000),
                     'max_depth': randint(3, 12),
                     'learning_rate': uniform(0.01, 0.3),
                     'subsample': uniform(0.6, 0.4),
                     'colsample_bytree': uniform(0.5, 0.5)}
        cands.append(('xgb', xg, xg_params))
    if LGB_AVAILABLE:
        lg = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        lg_params = {'n_estimators': randint(100, 1000),
                     'num_leaves': randint(16, 128),
                     'learning_rate': uniform(0.01, 0.3),
                     'subsample': uniform(0.6, 0.4)}
        cands.append(('lgb', lg, lg_params))
    return cands

# -----------------------
# Train & tune with RandomizedSearchCV (time-aware)
# -----------------------
def train_with_random_search(X_train, y_train, use_log=True, emphasize_floods=True, n_iter=RANDOM_SEARCH_ITERS):
    y_fit = np.log1p(y_train) if use_log else y_train.values
    candidates = build_candidates()
    best_model = None
    best_name = None
    best_score = -1e9

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    for name, estimator, param_dist in candidates:
        print(f"\nTuning candidate: {name}")
        search = RandomizedSearchCV(estimator, param_distributions=param_dist, n_iter=n_iter, cv=tscv,
                                    scoring=COMBINED_SCORER, n_jobs=-1, random_state=RANDOM_STATE, verbose=2)
        # sample weights for emphasis
        fit_kwargs = {}
        if emphasize_floods:
            thr = np.nanpercentile(y_train.values, FLOOD_QUANTILE*100)
            sample_weight = np.where(y_train.values >= thr, 1.0 + FLOOD_WEIGHT_ALPHA, 1.0)
            fit_kwargs['sample_weight'] = sample_weight
            print(f"  applying sample_weight emphasizing flows >= {thr:.2f} m3/s by alpha={FLOOD_WEIGHT_ALPHA}")

        try:
            search.fit(X_train, y_fit, **fit_kwargs)
        except TypeError:
            # some estimators or sklearn versions don't accept sample_weight in RandomizedSearchCV.fit
            search.fit(X_train, y_fit)
        print(f"  Best combined score (cv): {search.best_score_:.4f}")
        # Evaluate on held-out folds: use search.best_estimator_
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = search.best_estimator_
            best_name = name

    print(f"\nSelected best model: {best_name} (cv score {best_score:.4f})")
    return best_name, best_model

# -----------------------
# Evaluate helper
# -----------------------
def evaluate_model(model, X_test, y_test, use_log=True):
    if use_log:
        y_pred_tr = model.predict(X_test)
        y_pred = np.expm1(y_pred_tr).clip(min=0)
    else:
        y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    nse_val = nse(y_test, y_pred)
    thr = np.nanpercentile(y_test, FLOOD_QUANTILE*100)
    peak_mask = y_test >= thr
    r2_peak = r2_score(y_test[peak_mask], y_pred[peak_mask]) if peak_mask.sum()>=2 else float('nan')
    return {'mae':mae,'rmse':rmse,'r2':r2,'nse':nse_val,'r2_peak':r2_peak,'y_pred':y_pred}

# -----------------------
# Plotting helpers
# -----------------------
def plot_backtest(y_test, y_pred, out_full=PLOT_FILENAME, out_peaks=FLOOD_PLOT_FILENAME):
    plt.figure(figsize=(14,6))
    plt.plot(y_test.index, y_test.values, label='Observed', lw=1.5)
    plt.plot(y_test.index, y_pred, label='Predicted', lw=1.0, linestyle='--')
    plt.fill_between(y_test.index, y_test.values, y_pred, color='gray', alpha=0.15)
    plt.title('Backtest: Observed vs Predicted (full)')
    plt.xlabel('Date'); plt.ylabel('Discharge (m3/s)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_full); plt.close()
    print(f"Saved full backtest to {out_full}")

    # peaks only plot
    thr = np.nanpercentile(y_test.values, FLOOD_QUANTILE*100)
    peak_idx = y_test >= thr
    if peak_idx.sum() > 0:
        plt.figure(figsize=(12,5))
        plt.plot(y_test.index[peak_idx], y_test.values[peak_idx], 'o-', label='Observed peaks')
        plt.plot(y_test.index[peak_idx], y_pred[peak_idx], 'x--', label='Predicted peaks')
        plt.title(f'Peaks (>= {int(FLOOD_QUANTILE*100)}th percentile)')
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(out_peaks); plt.close()
        print(f"Saved peaks backtest to {out_peaks}")
    else:
        print("No peaks to plot.")

# -----------------------
# MAIN TRAIN FLOW
# -----------------------
def main_train(lat, lon, start, end, lead_days, use_log, emphasize_floods, upstream_coords):
    print("Fetching historical data...")
    df = fetch_historical(lat, lon, start, end, include_hourly_max=True, upstream_coords=upstream_coords)
    print(f"Fetched {len(df)} daily rows. Engineering features...")
    df_feat = engineer_rich(df, lead_days)

    feature_list = [
        'discharge_m3_s','discharge_lag_1','discharge_lag_2','discharge_rate_change','discharge_3_day_avg',
        'rainfall_lag_1','rainfall_3_day_sum','rainfall_7_day_sum','rainfall_14_day_sum','rainfall_30_day_sum',
        'rainfall_7_day_avg','rainfall_forecast_Nd',
        'ari_7_d0.7','ari_14_d0.85','ari_30_d0.9',
        'max_hourly_rain_mm','rain_x_soil','local_vs_upstream',
        'soil_moisture','evapotranspiration','temperature',
        'month_sin','month_cos'
    ]

    # trim rows with NaN target
    X, y, dfc = prepare_xy_df(df_feat, feature_list)
    if len(X) < 200:
        print("WARNING: <200 rows after preprocessing — results may be noisy.")

    # time split
    split_i = int(len(X)*0.8)
    X_train, X_test = X.iloc[:split_i], X.iloc[split_i:]
    y_train, y_test = y.iloc[:split_i], y.iloc[split_i:]

    print(f"Training rows: {len(X_train)}  Test rows: {len(X_test)}")
    name, model = train_and_select(X_train, y_train, use_log, emphasize_floods)
    print("Evaluating...")
    res = evaluate_model(model, X_test, y_test, use_log)
    print(f"Test MAE: {res['mae']:.3f} RMSE: {res['rmse']:.3f} R2: {res['r2']:.3f} NSE: {res['nse']:.3f} R2_peak: {res['r2_peak']:.3f}")

    # save model bundle
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    bundle = {'model':model, 'features':feature_list, 'use_log_target':use_log, 'lead_days':lead_days}
    outp = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)
    joblib.dump(bundle, outp)
    with open(os.path.join(MODEL_OUTPUT_DIR, FEATURES_JSON), 'w') as f:
        json.dump({'features':feature_list,'lead_days':lead_days}, f, indent=2)
    print(f"Saved model bundle to {outp}")

    # plots
    plot_backtest(y_test, res['y_pred'])
    return bundle, res, dfc

# small wrappers to keep main flow readable (split out to aid unit testing)
def prepare_xy_df(df_feat, feature_list):
    dfc = df_feat.copy().dropna(subset=['target_discharge'])
    X = dfc[feature_list].fillna(0)
    y = dfc['target_discharge']
    return X, y, dfc

def train_and_select(X_train, y_train, use_log, emphasize_floods):
    # wrapper to call randomized search and return name, model
    name, model = None, None
    # We'll reuse earlier functions but adapt them to smaller n_iter if compute limited
    # Use reduced iterations for heavier models if they are available
    global RANDOM_SEARCH_ITERS
    n_iter = RANDOM_SEARCH_ITERS
    # If XGB/LGB not available reduce iterations
    if not (XGB_AVAILABLE or LGB_AVAILABLE):
        n_iter = min(n_iter, 40)
    name, model = train_with_random_search_v7(X_train, y_train, use_log, emphasize_floods, n_iter=n_iter)
    return name, model

# we define train_with_random_search_v7 as alias to the implemented routine above
def train_with_random_search_v7(X_train, y_train, use_log, emphasize_floods, n_iter=RANDOM_SEARCH_ITERS):
    # adapt earlier function for v7 naming
    return _train_with_random_search_internal(X_train, y_train, use_log, emphasize_floods, n_iter)

def _train_with_random_search_internal(X_train, y_train, use_log, emphasize_floods, n_iter):
    # reuse train_with_random_search logic but adjusted for v7 candidates
    y_fit = np.log1p(y_train) if use_log else y_train.values
    candidates = build_candidates()
    best_score = -1e9
    best_model = None
    best_name = None
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    for name, estimator, param_dist in candidates:
        print(f"\nTuning {name} (iters={n_iter})")
        search = RandomizedSearchCV(estimator, param_distributions=param_dist, n_iter=n_iter, cv=tscv,
                                    scoring=COMBINED_SCORER, n_jobs=-1, random_state=RANDOM_STATE, verbose=2)
        fit_kwargs = {}
        if emphasize_floods:
            thr = np.nanpercentile(y_train.values, FLOOD_QUANTILE*100)
            sample_weight = np.where(y_train.values >= thr, 1.0 + FLOOD_WEIGHT_ALPHA, 1.0)
            fit_kwargs['sample_weight'] = sample_weight
            print(f"  applying sample weights: threshold {thr:.2f}, alpha {FLOOD_WEIGHT_ALPHA}")
        try:
            search.fit(X_train, y_fit, **fit_kwargs)
        except TypeError:
            search.fit(X_train, y_fit)
        print(f"  {name} best cv combined score: {search.best_score_:.4f}")
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = search.best_estimator_
            best_name = name
    print(f"Selected {best_name} with cv score {best_score:.4f}")
    return best_name, best_model

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train'], default='train')
    p.add_argument('--lat', type=float, default=DEFAULT_LAT)
    p.add_argument('--lon', type=float, default=DEFAULT_LON)
    p.add_argument('--start', type=str, default=DEFAULT_START)
    p.add_argument('--end', type=str, default=DEFAULT_END)
    p.add_argument('--lead', type=int, default=DEFAULT_LEAD_DAYS)
    p.add_argument('--no-log', dest='use_log', action='store_false', help='Disable log1p target transform')
    p.add_argument('--no-emphasize', dest='emphasize', action='store_false', help='Disable flood emphasis sample weights')
    p.add_argument('--upstream', nargs='*', help='Optional upstream coords as lat,lon pairs (e.g. --upstream "24.0,90.1" "24.2,90.5")')
    return p.parse_args()

if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    # parse upstream coords
    upstream_coords = None
    if args.upstream:
        upstream_coords = []
        for s in args.upstream:
            lat_s, lon_s = s.split(',')
            upstream_coords.append((float(lat_s), float(lon_s)))

    bundle, metrics, dfc = main_train(
    lat=args.lat, lon=args.lon,
    start=args.start, end=args.end,
    lead_days=args.lead,
    use_log=args.use_log,
    emphasize_floods=args.emphasize,
    upstream_coords=upstream_coords
    )

    print("Training complete. Metrics:", metrics)

