"""
Feature engineering for v23 (River + Weather Context).
"""
import numpy as np
import pandas as pd
from .config import FEATURE_LIST

def build_daily_merge(df_hourly, df_flood):
    print("Aggregating hourly -> daily and merging...")
    agg_ops = {
        'rainfall_mm': 'sum',
        'relative_humidity': 'mean',
        'evapotranspiration': 'mean',
        'temperature': 'mean',
        'max_hourly_rain_mm': 'max'
    }
    for col in agg_ops.keys():
        if col not in df_hourly.columns:
            df_hourly[col] = 0.0
            
    df_daily = df_hourly.resample('D').agg(agg_ops)
    df_daily.index.name = 'date'
    df = pd.merge(df_flood, df_daily, left_index=True, right_index=True, how='inner')
    print(f"Total merged daily rows: {len(df)}")
    return df

def engineer_features(df_daily, lead_days=1, api_k=0.85):
    print("Engineering features...")
    d = df_daily.copy().sort_index()
    for c in ['discharge_m3_s','rainfall_mm','relative_humidity','evapotranspiration','temperature','max_hourly_rain_mm']:
        if c not in d.columns: d[c] = 0.0

    # 1. Discharge Lags (Full Week History)
    for i in range(1, 8): # Lags 1 to 7
        d[f'dis_lag{i}'] = d['discharge_m3_s'].shift(i)
        
    d['dis_rate'] = d['discharge_m3_s'] - d['dis_lag1']
    d['dis_roll_3'] = d['discharge_m3_s'].shift(1).rolling(3).mean()
    d['dis_roll_7'] = d['discharge_m3_s'].shift(1).rolling(7).mean()

    # 2. Weather Features
    for i in [1,2,3,7,14]:
        d[f'rain_lag_{i}'] = d['rainfall_mm'].shift(i)
    d['rain_roll_3']  = d['rainfall_mm'].shift(1).rolling(3).sum()
    d['rain_roll_7']  = d['rainfall_mm'].shift(1).rolling(7).sum()
    d['rain_roll_14'] = d['rainfall_mm'].shift(1).rolling(14).sum()
    d['rain_grad_1_2'] = d['rain_lag_1'] - d['rain_lag_2']
    d['et_lag_1'] = d['evapotranspiration'].shift(1)
    
    d['humidity_lag_1'] = d['relative_humidity'].shift(1)
    d['rain_x_humidity'] = d['rain_lag_1'] * d['humidity_lag_1']

    # 3. API & Seasonality
    weights = np.power(api_k, np.arange(7))
    d['api'] = d['rainfall_mm'].shift(1).rolling(7).apply(
        lambda x: np.sum(np.asarray(x) * weights[::-1]), raw=False
    )
    d['month'] = d.index.month
    d['day_of_year'] = d.index.dayofyear
    d['month_sin'] = np.sin(2*np.pi*d['month']/12); d['month_cos'] = np.cos(2*np.pi*d['month']/12)
    
    d['rainfall_forecast_1d'] = d['rainfall_mm'].shift(-lead_days)
    d['target'] = d['discharge_m3_s'].shift(-lead_days)
    
    # We drop rows where lag 7 is NaN (the first week of data)
    return d.dropna(subset=['target','dis_lag7','rain_roll_14'])

def augment_with_surges(df, n_augment=300, max_extra_mm=140.0, runoff_coeff=0.5):
    # (Keep existing augmentation logic if you want, but v23 usually skips it for stability)
    # For this run, we will rely on the natural river momentum + weights
    return df 

def prepare_tabular_Xy(df, feature_list):
    constant_cols = []
    for col in feature_list:
        if col in df.columns and df[col].std() < 1e-6:
            constant_cols.append(col)
    
    if constant_cols: print(f"Warning: Removing constant features: {constant_cols}")
    
    features_to_use = [f for f in feature_list if f not in constant_cols]
    X = df[features_to_use].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    y = df['target'].astype(np.float32)
    return X, y, features_to_use

def build_sequences(df, seq_len, features_seq, target_col='target'):
    # (Keep for compatibility, though v23 uses tree models)
    return np.array([]), np.array([]), np.array([]), []