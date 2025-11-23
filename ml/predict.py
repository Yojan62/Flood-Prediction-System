"""
Prediction script for FLOW v23.
"""
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta

# --- HYBRID IMPORT FIX ---
try:
    from ml.config import MODELS_DIR, MODEL_FILE, DEFAULT_LAT, DEFAULT_LON
    from ml.flood_api import fetch_flood_discharge
    from ml.data_fetch import fetch_archive_hourly
    from ml.feature_engineering import build_daily_merge, engineer_features
except ImportError:
    from config import MODELS_DIR, MODEL_FILE, DEFAULT_LAT, DEFAULT_LON
    from flood_api import fetch_flood_discharge
    from data_fetch import fetch_archive_hourly
    from feature_engineering import build_daily_merge, engineer_features
# -------------------------

def _load_model():
    path = MODELS_DIR / MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run train.py first.")
    return joblib.load(path)

def predict_next_day(lat=DEFAULT_LAT, lon=DEFAULT_LON):
    """
    Predicts discharge for tomorrow only.
    Returns a dictionary: {date, predicted_discharge, input_date}
    """
    bundle = _load_model()
    model = bundle['model']
    scaler = bundle['scaler']
    features = bundle['features']
    
    # 1. Fetch Data (~30 days)
    end = pd.Timestamp.now().date()
    start = end - timedelta(days=30)
    
    df_h = fetch_archive_hourly(lat, lon, start.isoformat(), end.isoformat())
    df_f = fetch_flood_discharge(lat, lon, start.isoformat(), end.isoformat())
    
    if df_h.empty or df_f.empty:
        return {"error": "Insufficient data for prediction"}

    df = build_daily_merge(df_h, df_f)
    
    # 2. Engineer (lead_days=0 so we use today's data to predict tomorrow)
    df = engineer_features(df, lead_days=0)
    
    if df.empty:
        return {"error": "Feature engineering resulted in empty data"}

    # 3. Predict
    last_row = df.iloc[[-1]][features]
    X_scaled = scaler.transform(last_row.fillna(0))
    pred_log = model.predict(X_scaled)
    pred_val = np.expm1(pred_log)[0]
    
    return {
        "date": (df.index[-1] + timedelta(days=1)).date().isoformat(),
        "predicted_discharge": float(max(0, pred_val)),
        "input_date": df.index[-1].date().isoformat()
    }

def predict_recent_days(lat=DEFAULT_LAT, lon=DEFAULT_LON, days=3, danger_threshold=None):
    """
    Predicts discharge for the last N days.
    
    Args:
        danger_threshold (float, optional): The specific danger level for this station from the DB.
                                            If provided, risk is calculated relative to this.
    """
    bundle = _load_model()
    model = bundle['model']
    scaler = bundle['scaler']
    features = bundle['features']
    
    # Fetch enough history
    end_date = pd.Timestamp.now().date() + timedelta(days=2) 
    start_date = end_date - timedelta(days=45) 
    
    # Fetch Data
    df_h = fetch_archive_hourly(lat, lon, start_date.isoformat(), end_date.isoformat())
    df_f = fetch_flood_discharge(lat, lon, start_date.isoformat(), end_date.isoformat())
    
    if df_h.empty:
        print("Warning: No weather data found.")
        return pd.DataFrame()
        
    df = build_daily_merge(df_h, df_f)
    
    if df.empty:
        return pd.DataFrame()

    # Engineer Features
    df_eng = engineer_features(df, lead_days=0)
    
    if df_eng.empty:
        return pd.DataFrame()

    # Filter to the requested 'days' window
    df_subset = df_eng.iloc[-days:].copy()
    
    # Predict
    X = df_subset[features].fillna(0)
    X_scaled = scaler.transform(X)
    preds_log = model.predict(X_scaled)
    preds_val = np.expm1(preds_log)
    
    # Format Output
    df_subset['pred_final'] = np.maximum(0, preds_val)
    
    # --- DYNAMIC RISK LOGIC ---
    def get_risk(val):
        # 1. Use specific station threshold if available (THIS IS WHAT YOU WANTED)
        if danger_threshold is not None and danger_threshold > 0:
            if val >= danger_threshold: 
                return "HIGH"
            if val >= (danger_threshold * 0.8): 
                return "MEDIUM" # Warning zone (80% of danger level)
            return "LOW"
            
        # 2. Fallback to generic thresholds only if DB has no data
        if val > 5000: return "HIGH"
        if val > 2000: return "MEDIUM"
        return "LOW"
        
    df_subset['risk_combined_level'] = df_subset['pred_final'].apply(get_risk)
    
    return df_subset[['pred_final', 'risk_combined_level']]