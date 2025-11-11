#!/usr/bin/env python3
"""
run_predictions.py (improved)

- Loads saved model (accepts either raw sklearn model or dict with keys: model, features, use_log_target)
- Fetches flood + forecast weather from Open-Meteo API
- Assembles live feature vector including real rainfall forecast for TOMORROW (calendar day)
- Adapts to expected feature ordering and supports month_sin/month_cos if required
- Saves prediction to DB and sends alerts
"""

import os
import joblib
import numpy as np
import requests
import datetime as dt
import pandas as pd
from dotenv import load_dotenv

# DB / email imports (your project modules)
import database
import models
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Config
DEFAULT_LAT = 23.81
DEFAULT_LON = 90.41
TIMEOUT = 15
MODEL_REL_PATHS = [
    os.path.join(os.path.dirname(__file__), '..', 'ml', 'open_meteo_flood_model_v5.pkl'),
    os.path.join(os.path.dirname(__file__), '..', 'ml', 'open_meteo_flood_model.pkl'),
]
OPEN_METEO_FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


# -------------------------
# Model loader (robust)
# -------------------------
def load_model():
    model_obj = None
    for p in MODEL_REL_PATHS:
        try:
            p_abs = os.path.abspath(p)
            if os.path.exists(p_abs):
                print(f"Loading model from: {p_abs}")
                loaded = joblib.load(p_abs)
                # Accept either raw model or saved dict
                if isinstance(loaded, dict):
                    model = loaded.get('model')
                    features = loaded.get('features')
                    use_log_target = loaded.get('use_log_target', False)
                else:
                    model = loaded
                    features = None
                    use_log_target = False
                return {'model': model, 'features': features, 'use_log_target': use_log_target}
        except Exception as e:
            print(f"Error loading {p}: {e}")
    print("No model found in expected locations.")
    return None


# -------------------------
# Helper: sum precipitation for a calendar date
# -------------------------
def daily_sum_for_date(df_hourly, date):
    """
    df_hourly: DataFrame index = timezone-aware datetimes, column 'precipitation'
    date: datetime.date object (local date)
    Returns sum of precipitation for that calendar date
    """
    day_start = pd.Timestamp(date).tz_localize(df_hourly.index.tz) if df_hourly.index.tzinfo is None else pd.Timestamp(date).tz_localize(None)
    # Filter by date (use .date() to be safe)
    sel = df_hourly.index.date == date
    if sel.sum() == 0:
        return np.nan
    return float(df_hourly.loc[sel, 'precipitation'].sum())


# -------------------------
# Fetch and assemble live features (robust)
# -------------------------
def get_live_features_for_model(lat=DEFAULT_LAT, lon=DEFAULT_LON, expected_features=None):
    """
    Returns: (feature_vector_df, meta) or (None, error_message)
    feature_vector_df: single-row DataFrame with columns matching expected_features (if provided), otherwise common defaults
    """

    try:
        # 1) Fetch flood daily (use past 5 days to be safe)
        flood_params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "river_discharge",
            "past_days": 5,
            "forecast_days": 0,
            "timezone": "auto"
        }
        r_flood = requests.get(OPEN_METEO_FLOOD_URL, params=flood_params, timeout=TIMEOUT)
        r_flood.raise_for_status()
        flood_json = r_flood.json()
        if 'daily' not in flood_json or 'time' not in flood_json['daily']:
            return None, "Flood API did not return expected daily/time keys."

        df_flood = pd.DataFrame(flood_json['daily'])
        df_flood['date'] = pd.to_datetime(df_flood['time'])
        if 'river_discharge' not in df_flood.columns:
            # try alternate name(s)
            if 'river_discharge_mean' in df_flood.columns:
                df_flood.rename(columns={'river_discharge_mean': 'river_discharge'}, inplace=True)
            else:
                return None, "river_discharge not found in flood API response."
        df_flood = df_flood.set_index('date').sort_index()

        # 2) Fetch forecast hourly weather (past + forecast)
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["precipitation", "soil_moisture_0_1cm", "et0_fao_evapotranspiration", "temperature_2m"],
            "past_days": 7,
            "forecast_days": 2,  # get at least next day
            "timezone": "auto"
        }
        r_weather = requests.get(OPEN_METEO_FORECAST_URL, params=weather_params, timeout=TIMEOUT)
        r_weather.raise_for_status()
        weather_json = r_weather.json()
        if 'hourly' not in weather_json:
            return None, "Forecast API did not return hourly data."

        # Convert hourly to DataFrame
        hourly = weather_json['hourly']
        df_hourly = pd.DataFrame(hourly)
        # 'time' may be present as list; ensure parsed
        df_hourly['time'] = pd.to_datetime(df_hourly['time'])
        df_hourly = df_hourly.set_index('time').sort_index()

        # Get local "today" date according to API timezone (use index last timestamp to determine tz)
        # We'll take the earliest future date available and call that 'tomorrow'
        now = pd.Timestamp(df_hourly.index[-1])
        # We want the next calendar date after 'today' in the hourly index
        last_date_in_hourly = df_hourly.index[-1].date()
        # Candidate tomorrow is (today + 1)
        local_today = pd.Timestamp(df_hourly.index[0]).date()
        # Use actual current date from system if index spans present moment
        today_date = dt.date.today()
        # safest: set tomorrow_date = min(date in hourly that's greater than today_date) if possible
        future_dates = sorted({ts.date() for ts in df_hourly.index if ts.date() > today_date})
        if future_dates:
            tomorrow_date = future_dates[0]
        else:
            # fallback: today + 1
            tomorrow_date = today_date + dt.timedelta(days=1)

        # Sum precipitation for yesterday (to compute rainfall_lag_1 and 7-day avg) and tomorrow
        # Build a daily-summed DataFrame from hourly (for past N days)
        df_hourly['precipitation'] = df_hourly['precipitation'].astype(float)
        daily_precip = df_hourly['precipitation'].resample('D').sum()
        daily_soil = df_hourly['soil_moisture_0_1cm'].resample('D').mean()
        daily_et0 = df_hourly['et0_fao_evapotranspiration'].resample('D').mean()
        daily_temp = df_hourly['temperature_2m'].resample('D').mean()
        df_daily = pd.concat([daily_precip, daily_soil, daily_et0, daily_temp], axis=1)
        df_daily.columns = ['rainfall_mm', 'soil_moisture', 'evapotranspiration', 'temperature']

        # Determine the last fully-observed past day to use as "yesterday"
        # Use today's date based on server/system - pick most recent date that is <= today_date
        available_dates = [d for d in df_daily.index.date if d <= today_date]
        if not available_dates:
            # fallback to using earliest available
            available_dates = sorted({d for d in df_daily.index.date})
        last_past_date = sorted(available_dates)[-1]

        # yesterday (lag 1) = last_past_date
        yesterday_date = last_past_date
        rainfall_lag_1 = float(df_daily.loc[str(yesterday_date), 'rainfall_mm']) if str(yesterday_date) in df_daily.index.astype(str) else float(df_daily.loc[df_daily.index.date == yesterday_date, 'rainfall_mm'].values[-1])
        soil_moisture = float(df_daily.loc[str(yesterday_date), 'soil_moisture'])
        evapotranspiration = float(df_daily.loc[str(yesterday_date), 'evapotranspiration'])
        temperature = float(df_daily.loc[str(yesterday_date), 'temperature'])

        # rainfall_7_day_avg: average of last 7 days ending yesterday
        # get the 7-day window
        window_end = pd.Timestamp(yesterday_date)
        window_start = window_end - pd.Timedelta(days=6)
        rainfall_7_day_avg = float(df_daily.loc[str(window_start.date()):str(window_end.date()), 'rainfall_mm'].mean())

        # rainfall_forecast_24h: total precipitation on TOMORROW_DATE (calendar day)
        rainfall_forecast_24h = float(df_daily.loc[str(tomorrow_date), 'rainfall_mm']) if str(tomorrow_date) in df_daily.index.astype(str) else float(df_daily[df_daily.index.date == tomorrow_date]['rainfall_mm'].sum())

        # Discharge lags: use the flood API daily output and pick latest known days
        # Build flood df (time already parsed)
        df_flood = df_flood.sort_index()
        # we expect df_flood to include last days; take last 3 valid values
        last_discharge_vals = df_flood['river_discharge'].dropna()
        if len(last_discharge_vals) < 3:
            # try forward/backfill a bit or fail gracefully
            last_discharge_vals = last_discharge_vals.reindex(pd.date_range(last_discharge_vals.index.min(), periods=3, freq='D')).ffill().bfill()
        # select last available as today, previous as lag1, lag2
        discharge_today = float(last_discharge_vals.iloc[-1])
        discharge_lag_1 = float(last_discharge_vals.iloc[-2])
        discharge_lag_2 = float(last_discharge_vals.iloc[-3])
        discharge_3_day_avg = float(np.mean([discharge_lag_1, discharge_lag_2, discharge_today]))

        # month (or month_sin/month_cos depending on expected features)
        now_local = dt.date.today()
        month_int = now_local.month
        month_sin = np.sin(2 * np.pi * month_int / 12)
        month_cos = np.cos(2 * np.pi * month_int / 12)

        # Assemble dictionary of potential features (all names that v5 expects)
        feat_dict = {
            'discharge_m3_s': discharge_today,
            'discharge_lag_1': discharge_lag_1,
            'discharge_lag_2': discharge_lag_2,
            'rainfall_lag_1': rainfall_lag_1,
            'month': month_int,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'rainfall_7_day_avg': rainfall_7_day_avg,
            'discharge_3_day_avg': discharge_3_day_avg,
            'soil_moisture': soil_moisture,
            'rainfall_forecast_24h': rainfall_forecast_24h,
            'evapotranspiration': evapotranspiration,
            'temperature': temperature,
            # extras (ari / rain_x_soil) — compute simple versions
            'ari_7': float(df_daily.loc[str(window_start.date()):str(window_end.date()), 'rainfall_mm'].multiply([0.7**i for i in range(7)][::-1]).sum()) if True else 0.0,
            'rain_x_soil': rainfall_lag_1 * soil_moisture
        }

        # If expected_features provided, preserve that order, otherwise use a sensible default list
        if expected_features:
            # Build a one-row DataFrame in the exact order expected
            row = {}
            for f in expected_features:
                if f in feat_dict:
                    row[f] = feat_dict[f]
                else:
                    # fallback to 0 with a warning
                    row[f] = 0.0
            feature_df = pd.DataFrame([row], index=[pd.Timestamp.now()])
        else:
            # default ordering similar to v5 training
            default_order = [
                'discharge_m3_s', 'discharge_lag_1', 'discharge_lag_2',
                'discharge_3_day_avg', 'rainfall_lag_1', 'rainfall_3_day_sum' if False else 'rainfall_7_day_avg',
                'rainfall_7_day_avg', 'rainfall_forecast_24h', 'ari_7', 'rain_x_soil',
                'soil_moisture', 'evapotranspiration', 'temperature', 'month_sin', 'month_cos'
            ]
            row = {k: feat_dict.get(k, 0.0) for k in default_order}
            feature_df = pd.DataFrame([row], index=[pd.Timestamp.now()])

        return feature_df, {'tomorrow_date': tomorrow_date, 'yesterday_date': yesterday_date}

    except Exception as e:
        return None, f"Exception while fetching live features: {e}"


# -------------------------
# SendGrid helper (unchanged)
# -------------------------
def send_flood_alert(user_email, location_name, risk_level, predicted_discharge):
    sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
    from_email = os.getenv("SENDGRID_FROM_EMAIL")
    if not sendgrid_api_key or not from_email:
        print("SendGrid not configured.")
        return
    message = Mail(
        from_email=from_email,
        to_emails=user_email,
        subject=f"FLOOD ALERT: {risk_level} Risk Detected for {location_name}",
        html_content=f"<strong>Automated flood alert for {location_name}</strong><p>Risk: {risk_level}</p><p>Predicted discharge: {predicted_discharge:.2f} m³/s</p>"
    )
    try:
        sg = SendGridAPIClient(sendgrid_api_key)
        sg.send(message)
    except Exception as e:
        print("SendGrid error:", e)


# -------------------------
# Main prediction cycle
# -------------------------
def run_prediction_cycle():
    model_bundle = load_model()
    if not model_bundle:
        print("No model loaded. Exiting.")
        return
    model = model_bundle['model']
    features_expected = model_bundle['features']
    use_log_target = model_bundle['use_log_target']

    db = database.SessionLocal()
    try:
        locations = db.query(models.Location).all()
        for location in locations:
            print(f"Processing location {location.location_id} - {location.name}")
            feature_df, meta = get_live_features_for_model(lat=location.latitude, lon=location.longitude, expected_features=features_expected)
            if feature_df is None:
                print("Failed to get features:", meta)
                continue

            # Ensure correct ordering of columns to match model
            if features_expected:
                X = feature_df[features_expected]
            else:
                X = feature_df

            # Predict
            try:
                y_pred_raw = model.predict(X.values)[0]
                # If model was trained on log1p, the saved metadata indicates this. We'll assume model returns transformed if so.
                if use_log_target:
                    predicted_discharge = float(np.expm1(y_pred_raw))
                else:
                    predicted_discharge = float(y_pred_raw)
            except Exception as e:
                print("Prediction failed:", e)
                continue

            # Determine risk
            threshold = location.danger_threshold if location.danger_threshold is not None else 999999.0
            risk = "LOW"
            if predicted_discharge >= threshold:
                risk = "HIGH"
            elif predicted_discharge >= 0.8 * threshold:
                risk = "MEDIUM"

            # Save to DB
            pred = models.Prediction(location_id=location.location_id, predicted_discharge=predicted_discharge, risk_level=risk)
            db.add(pred)
            db.commit()
            print(f"Saved prediction id={pred.prediction_id} discharge={predicted_discharge:.2f} risk={risk}")

            # Send alerts
            if risk == "HIGH":
                users = db.query(models.User).filter(models.User.subscribed_location_id == location.location_id).all()
                for u in users:
                    send_flood_alert(u.email, location.name, risk, predicted_discharge)

    finally:
        db.close()


if __name__ == "__main__":
    load_dotenv()
    run_prediction_cycle()
