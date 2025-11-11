#!/usr/bin/env python3
"""
HydroFusion v19 — Hybrid Model
Combines:
  • Observed discharge / water-level CSVs from local agencies (e.g., FFWC)
  • Weather + rainfall predictors from Open-Meteo
Trains a LightGBM baseline + residual BiLSTM hybrid.

CLI example:
python flood_model_v19.py --csv "ml/SW42_training_data.csv" \
    --station 152 --lat 25.82041 --lon 89.667595 \
    --start 2020-01-01 --end 2025-11-11 --lead 1 --out_prefix v19
"""

import os, argparse, warnings, json, math, time, datetime as dt
import numpy as np, pandas as pd, requests, joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import HuberRegressor
warnings.filterwarnings("ignore")

# Optional libs
import lightgbm as lgb
import shap
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    GlobalMaxPooling1D, GlobalAveragePooling1D,
    Concatenate, LayerNormalization
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("TensorFlow:", tf.__version__)
print("GPUs detected:", tf.config.list_physical_devices('GPU'))

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def log(msg): print(msg, flush=True)

def rmse(a,b): return math.sqrt(mean_squared_error(a,b))

def metric_pack(y_true, y_pred, thr=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    d = dict(mae=mean_absolute_error(y_true,y_pred),
             rmse=rmse(y_true,y_pred),
             r2=r2_score(y_true,y_pred))
    if thr is None: thr = np.nanpercentile(y_true,90)
    mask = y_true>=thr
    d["r2_peak"] = r2_score(y_true[mask],y_pred[mask]) if mask.sum()>3 else np.nan
    d["thr"]=thr
    return d

def pick_first_present(columns, names):
    cols_lower = {c.lower(): c for c in columns}
    for n in names:
        if n.lower() in cols_lower:
            return cols_lower[n.lower()]
    return None

# ----------------------------------------------------------------------
# Load observed CSV (FFWC-style)
# ----------------------------------------------------------------------
def load_observed_csv(path, station_id=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    if len(df)==0:
        raise ValueError("Empty CSV")

    dt_col = pick_first_present(df.columns, ["datetime","date","wl_date","time","timestamp"])
    if dt_col is None:
        raise ValueError("No datetime column in CSV")
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).set_index(dt_col)

    # Detect columns
    wl_col = pick_first_present(df.columns, ["waterlevel","water_level","water_level_m","wl","level"])
    if wl_col is None:
        raise ValueError("CSV is missing water level column (try: waterlevel / water_level / water_level_m / wl).")
    danger_col = pick_first_present(df.columns, ["dangerlevel","danger_level","danger_level_m"])
    rain_col = pick_first_present(df.columns, ["rainfall","rainfall_mm","precipitation","rain_mm"])

    # Resample daily safely
    agg_map={}
    if wl_col: agg_map[wl_col]="mean"
    if danger_col: agg_map[danger_col]="max"
    if rain_col: agg_map[rain_col]="sum"
    df = (df.resample("D").agg(agg_map).reset_index()
           .rename(columns={wl_col:"waterlevel",
                            danger_col:"dangerlevel",
                            rain_col:"rainfall_mm"}))
    return df

# ----------------------------------------------------------------------
# Fetch Open-Meteo weather archive
# ----------------------------------------------------------------------
OPEN_METEO_ARCHIVE="https://archive-api.open-meteo.com/v1/archive"
def fetch_archive(lat,lon,start,end):
    s=pd.to_datetime(start).date(); e=pd.to_datetime(end).date()
    dfs=[]
    for y in range(s.year,e.year+1):
        seg_s=max(s,dt.date(y,1,1)); seg_e=min(e,dt.date(y,12,31))
        log(f"→ Fetching archive {seg_s}→{seg_e}")
        params=dict(latitude=lat,longitude=lon,
                    hourly="precipitation,soil_moisture_0_1cm,temperature_2m",
                    start_date=seg_s.isoformat(),end_date=seg_e.isoformat(),
                    timezone="auto")
        j=requests.get(OPEN_METEO_ARCHIVE,params=params,timeout=60).json()
        h=pd.DataFrame(j["hourly"])
        h["time"]=pd.to_datetime(h["time"]); h=h.set_index("time")
        h=h.rename(columns={"precipitation":"rainfall_mm",
                            "soil_moisture_0_1cm":"soil_moisture",
                            "temperature_2m":"temperature"})
        dfs.append(h)
        time.sleep(0.3)
    if not dfs: return pd.DataFrame()
    d=pd.concat(dfs)
    daily=d.resample("D").agg({"rainfall_mm":"sum","soil_moisture":"mean","temperature":"mean"})
    daily.index.name="date"
    return daily

# ----------------------------------------------------------------------
# Residual LSTM builder
# ----------------------------------------------------------------------
def make_res_lstm(input_shape,units=64,dropout=0.25):
    inp=Input(shape=input_shape)
    x=LayerNormalization()(inp)
    x=Bidirectional(LSTM(units,return_sequences=True))(x)
    x=Concatenate()([GlobalAveragePooling1D()(x),GlobalMaxPooling1D()(x)])
    x=Dropout(dropout)(x)
    out=Dense(1,activation="linear")(x)
    m=Model(inp,out)
    m.compile(optimizer="adam",loss="mse")
    return m

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_v19(obs_df,lat,lon,start,end,lead,quick=False,out_prefix="v19"):
    log(f"HydroFusion v19 — Hybrid Training {start}→{end}  (lat={lat}, lon={lon})")

    weather=fetch_archive(lat,lon,start,end)
    if weather.empty:
        raise RuntimeError("Open-Meteo returned no data")

    df=pd.merge(obs_df,weather,left_on="date",right_index=True,how="inner")
    df["target"]=df["waterlevel"].shift(-lead)
    df=df.dropna()

    split=int(len(df)*0.8)
    train_df,test_df=df.iloc[:split],df.iloc[split:]
    # Ensure required columns exist even if missing from merge
    for col in ["rainfall_mm", "soil_moisture", "temperature"]:
        if col not in train_df.columns:
            train_df[col] = 0.0
            test_df[col] = 0.0

    Xtr = train_df[["rainfall_mm", "soil_moisture", "temperature"]].fillna(0)
    ytr = train_df["target"]
    Xte = test_df[["rainfall_mm", "soil_moisture", "temperature"]].fillna(0)
    yte = test_df["target"]

    scaler=StandardScaler().fit(Xtr)
    Xtr_s=scaler.transform(Xtr); Xte_s=scaler.transform(Xte)

    log("Training LightGBM baseline...")
    m_base=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,random_state=42)
    m_base.fit(Xtr_s,ytr)
    base_pred=m_base.predict(Xte_s)

    # Residual LSTM on short sequences
    seq_len=7
    arr=df[["rainfall_mm","waterlevel","temperature"]].values.astype(np.float32)
    Xseq=[]; yseq=[]
    for i in range(seq_len,len(arr)-lead):
        Xseq.append(arr[i-seq_len:i]); yseq.append(df["target"].iloc[i])
    Xseq=np.array(Xseq); yseq=np.array(yseq)
    if len(Xseq)>20:
        log("Training residual LSTM...")
        lstm=make_res_lstm((seq_len,Xseq.shape[2]))
        cbs=[EarlyStopping(patience=4,restore_best_weights=True)]
        lstm.fit(Xseq,yseq,epochs=(12 if quick else 25),batch_size=16,verbose=0,callbacks=cbs)
    else:
        lstm=None; log("Skipping LSTM (not enough samples)")

    # Evaluate
    y_pred=base_pred
    base_metrics=metric_pack(yte,base_pred)
    log("\n--- Base Metrics ---")
    for k,v in base_metrics.items(): log(f"{k}: {v}")

    plt.figure(figsize=(10,5))
    plt.plot(test_df["date"],yte,label="Obs")
    plt.plot(test_df["date"],y_pred,label="Pred")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(f"{out_prefix}_backtest.png"); plt.close()

    bundle=dict(model_base=m_base,scaler=scaler,metrics=base_metrics,
                trained_range=dict(start=start,end=end),
                timestamp=dt.datetime.utcnow().isoformat()+"Z")
    os.makedirs("ml",exist_ok=True)
    joblib.dump(bundle,f"ml/{out_prefix}_bundle.pkl")
    log(f"Saved bundle to ml/{out_prefix}_bundle.pkl")
    return base_metrics

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True)
    p.add_argument("--station",type=str,default=None)
    p.add_argument("--lat",type=float,required=True)
    p.add_argument("--lon",type=float,required=True)
    p.add_argument("--start",default="2020-01-01")
    p.add_argument("--end",default=str(dt.date.today()))
    p.add_argument("--lead",type=int,default=1)
    p.add_argument("--quick",action="store_true")
    p.add_argument("--out_prefix",default="v19")
    args=p.parse_args()

    log(f"HydroFusion v19 — Hybrid Training {args.start} → {args.end}  (lat={args.lat}, lon={args.lon})")
    log(f"Lead days: {args.lead}   CSV: {args.csv}   Station filter: {args.station}")

    obs_daily=load_observed_csv(args.csv,args.station)
    res=train_v19(obs_daily,args.lat,args.lon,args.start,args.end,args.lead,
                  quick=args.quick,out_prefix=args.out_prefix)
    log(f"✅ Training complete: {json.dumps(res,indent=2)}")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        log(f"❌ Training failed: {repr(e)}")
        raise
