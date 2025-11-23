"""
Global configuration for the FLOW flood model (v23 - Optimized).
"""
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# API Endpoints
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FLOOD = "https://flood-api.open-meteo.com/v1/flood"

FLOOD_API_URL = OPEN_METEO_FLOOD
FLOOD_DAILY_VARS = ["river_discharge"]
# -----------------------------------------

TIMEZONE = "auto"

# Location Defaults (Dhaka)
DEFAULT_LAT = 23.81
DEFAULT_LON = 90.41
DEFAULT_START_DATE = "2020-01-01" # High quality recent data only
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_LEAD_DAYS = 1

# Training Settings
RANDOM_STATE = 42
FLOOD_Q = 0.90
PEAK_WEIGHT_ALPHA = 25.0 # The proven sweet spot
TIME_SERIES_SPLITS = 4

# --- FEATURE LIST (Enhanced River Momentum) ---
FEATURE_LIST = [
    # River Momentum (Full week history)
    'discharge_m3_s',
    'dis_lag1', 'dis_lag2', 'dis_lag3', 'dis_lag4', 'dis_lag5', 'dis_lag6', 'dis_lag7',
    'dis_rate', 'dis_roll_3', 'dis_roll_7',
    
    # Weather Context
    'rain_lag_1','rain_lag_2','rain_lag_3','rain_roll_3','rain_roll_7',
    'api', 'rainfall_forecast_1d',
    'relative_humidity', 'humidity_lag_1', 'rain_x_humidity',
    
    # Seasonality
    'month_sin','month_cos', 'day_of_year'
]

# Sequence features for LSTM (if we revert to hybrid later)
SEQ_FEATURES = [
    'discharge_m3_s','dis_rate','rain_roll_3','rain_roll_7','api',
    'rain_lag_1','rain_lag_3','max_hourly_rain_mm', 'relative_humidity'
]

# Model Params
RF_PARAMS = dict(n_estimators=400, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)

# Output Files
MODEL_FILE = "open_meteo_flood_model_v23.pkl"
FEATURES_JSON = "open_meteo_flood_features_v23.json"
PLOT_BACKTEST = "v23_backtest.png"
PLOT_PEAKS = "v23_backtest_peaks.png"
PLOT_SHAP = "v23_shap_summary.png"
PLOT_OBS_PRED = "v23_obs_vs_pred.png"
PRED_CSV = "v23_test_predictions.csv"
PLOT_RESID_TRAIN = "v23_residual_fit_train.png"

# Aliases
PLOT_FULL = PLOT_BACKTEST
PLOT_IMPORTANCE = "v23_feature_importance.png"
PLOT_RESIDUALS = "v23_residuals.png"
PLOT_ZOOM = "v23_zoom_flood.png"
PLOT_HYDROGRAPH = "v23_hydrograph.png"
PLOT_SEASONAL = "v23_seasonal_accuracy.png"
PLOT_ERROR_DIST = "v23_error_distribution.png"