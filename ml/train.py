"""
Training script for FLOW v23 (Stabilized Super Model).
"""
import os # <-- FIX: Added this import
import pandas as pd
import numpy as np
import joblib
import json
import argparse
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Import our modules
from .config import (
    DEFAULT_LAT, DEFAULT_LON, DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_LEAD_DAYS,
    FEATURE_LIST, FLOOD_Q, PEAK_WEIGHT_ALPHA, RANDOM_STATE, RF_PARAMS,
    MODELS_DIR, MODEL_FILE, FEATURES_JSON, PLOT_FULL, PRED_CSV,
    PLOT_OBS_PRED, PLOT_PEAKS, PLOT_SHAP, 
    PLOT_IMPORTANCE, PLOT_RESIDUALS, PLOT_ZOOM, PLOT_HYDROGRAPH, PLOT_SEASONAL, PLOT_ERROR_DIST
)
from .utils import (
    metrics_with_peak, plot_obs_vs_pred, save_backtest_plot, redirect_lgb_stderr,
    plot_feature_importance, plot_residuals, plot_zoom_flood, plot_hydrograph, plot_seasonal_accuracy, plot_error_distribution
)
from .data_fetch import fetch_archive_hourly, fetch_flood_daily
from .feature_engineering import build_daily_merge, engineer_features, prepare_tabular_Xy

# Optional Models
try:
    from catboost import CatBoostRegressor
    CAT_AVAILABLE = True
except:
    CAT_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False
    
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False


def train_v23(lat=DEFAULT_LAT, lon=DEFAULT_LON, start=DEFAULT_START_DATE, end=DEFAULT_END_DATE, lead_days=DEFAULT_LEAD_DAYS, quick=False):
    print(f"HydroFusion v23 — Training {start} -> {end}")

    # 1. Fetch & Merge
    df_hourly = fetch_archive_hourly(lat, lon, start, end)
    df_flood = fetch_flood_daily(lat, lon, start, end)
    if df_hourly.empty or df_flood.empty: return
    df_daily = build_daily_merge(df_hourly, df_flood)
    
    # 2. Engineer
    df = engineer_features(df_daily, lead_days=lead_days)
    
    # 3. Split
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()
    
    # 4. Prepare
    thr = np.nanpercentile(train_df['target'].fillna(0), int(FLOOD_Q*100))
    sw = np.where(train_df['target'] >= thr, 1.0 + PEAK_WEIGHT_ALPHA, 1.0)

    X_train_full, y_train, features_used = prepare_tabular_Xy(train_df, FEATURE_LIST)
    X_test_full, y_test, _ = prepare_tabular_Xy(test_df, FEATURE_LIST)
    
    FINAL_TABULAR_FEATURES = features_used
    print(f"Training on {len(FINAL_TABULAR_FEATURES)} features")
    
    scaler = StandardScaler()
    scaler.fit(X_train_full.fillna(0))
    
    X_train = pd.DataFrame(scaler.transform(X_train_full.fillna(0)), index=X_train_full.index, columns=features_used)
    X_test = pd.DataFrame(scaler.transform(X_test_full.fillna(0)), index=X_test_full.index, columns=features_used)

    y_train_log = np.log1p(y_train.clip(lower=0))

    # 5. Train Loop
    models = {}
    print(f"Training models (Peak Weight: {PEAK_WEIGHT_ALPHA})...")
    
    # Random Forest
    rf = RandomForestRegressor(**RF_PARAMS)
    if quick: rf.set_params(n_estimators=50)
    rf.fit(X_train, y_train_log, sample_weight=sw)
    models['rf'] = rf
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=200 if not quick else 50, max_depth=3, random_state=RANDOM_STATE)
    gb.fit(X_train, y_train_log, sample_weight=sw)
    models['gb'] = gb
    
    # LightGBM
    if LGB_AVAILABLE:
        lgb_model = lgb.LGBMRegressor(n_estimators=400 if not quick else 50, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        with redirect_lgb_stderr():
            lgb_model.fit(X_train, y_train_log, sample_weight=sw)
        models['lgb'] = lgb_model
        
    # CatBoost
    if CAT_AVAILABLE:
        cb = CatBoostRegressor(iterations=400 if not quick else 50, depth=6, random_state=RANDOM_STATE, verbose=0, thread_count=-1)
        cb.fit(X_train, y_train_log, sample_weight=sw)
        models['cat'] = cb

    # 6. Evaluate
    print("Evaluating models...")
    best_score = -999
    best_model_name = ""
    test_preds = {}
    
    for name, model in models.items():
        p_log = model.predict(X_test)
        p_val = np.expm1(p_log)
        test_preds[name] = p_val
        
        m = metrics_with_peak(y_test, p_val, thr)
        print(f"  {name.upper()} -> R2: {m['r2']:.4f}, Peak R2: {m['r2_peak']:.4f}")
        
        if m['r2_peak'] > best_score:
            best_score = m['r2_peak']
            best_model_name = name
            
    print(f"\n🏆 CHAMPION: {best_model_name.upper()} (Peak R2: {best_score:.4f})")
    champion_model = models[best_model_name]
    final_pred = test_preds[best_model_name]

    # 7. Save
    bundle = {
        'model': champion_model,
        'name': best_model_name,
        'scaler': scaler,
        'features': features_used,
        'thr': thr
    }
    joblib.dump(bundle, MODELS_DIR / MODEL_FILE)
    with open(MODELS_DIR / FEATURES_JSON, 'w') as f:
        json.dump({'features': features_used, 'lead_days': lead_days}, f, indent=2)
    print(f"Saved model bundle to {MODELS_DIR / MODEL_FILE}")
    
    # Save CSV
    out_df = pd.DataFrame({'observed': y_test, 'predicted': final_pred}, index=test_df.index)
    out_df.to_csv(PRED_CSV)
    print(f"Saved predictions to {PRED_CSV}")
    
    # 8. Generate Plots (Now includes 6 types)
    print("Generating plots...")
    save_backtest_plot(test_df.index, y_test, final_pred, thr, PLOT_FULL)
    plot_obs_vs_pred(y_test, final_pred, out_file=PLOT_OBS_PRED)
    plot_feature_importance(champion_model, features_used, out_file=PLOT_IMPORTANCE)
    plot_residuals(y_test, final_pred, out_file=PLOT_RESIDUALS)
    plot_zoom_flood(test_df.index, y_test, final_pred, thr, out_file=PLOT_ZOOM)
    
    # Hydrograph & Seasonal (Need raw rain data)
    plot_df = test_df.copy()
    plot_df['observed'] = y_test
    plot_df['predicted'] = final_pred
    if 'rainfall_mm' not in plot_df.columns and 'rain_lag_1' in plot_df.columns:
        plot_df['rainfall_mm'] = plot_df['rain_lag_1'].shift(-1).fillna(0)
    
    plot_hydrograph(plot_df, out_file=PLOT_HYDROGRAPH)
    plot_seasonal_accuracy(plot_df, out_file=PLOT_SEASONAL)
    plot_error_distribution(y_test, final_pred, out_file=PLOT_ERROR_DIST)

    # SHAP
    if SHAP_AVAILABLE and best_model_name in ['lgb', 'cat', 'rf', 'gb']:
        try:
            print("Generating SHAP plot...")
            explainer = shap.TreeExplainer(champion_model)
            X_samp = X_test.iloc[:500]
            shap_vals = explainer(X_samp)
            plt.figure(figsize=(10,8))
            shap.summary_plot(shap_vals, X_samp, show=False)
            plt.tight_layout()
            plt.savefig(PLOT_SHAP)
            plt.close()
            print(f"Saved plot: {PLOT_SHAP}")
        except Exception as e:
            print(f"SHAP failed: {e}")

    return {'final_metrics': metrics_with_peak(y_test, final_pred, thr)}

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    train_v23(quick=args.quick)