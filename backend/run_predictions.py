import os
import joblib
import numpy as np
import pandas as pd
import requests
import datetime as dt
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# DB / email imports (my project modules)
import database
import models
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# --- 1. Configuration & Model Loading ---

# Defines the paths to check for the model file.
# This variable is defined at the top, so it will be
# available for the load_model() function below.
MODEL_REL_PATHS = [
    # This path is correct for your 'ml/ml' structure 
    os.path.join(os.path.dirname(__file__), '..', 'ml', 'ml', 'open_meteo_flood_model_v18.pkl'),
    # Add any other backup paths here if needed
]
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_LAT = 23.81
DEFAULT_LON = 90.41
TIMEOUT = 15

# -------------------------
# Model loader (robust)
# -------------------------
def load_model(): # <-- The function is named 'load_model'
    """
    I'll load the v18 model bundle from the .pkl file.
    """
    model_obj = None
    # The 'MODEL_REL_PATHS' variable is now defined and can be used here.
    for p in MODEL_REL_PATHS:
        try:
            p_abs = os.path.abspath(p)
            if os.path.exists(p_abs):
                print(f"Loading model from: {p_abs}")
                loaded = joblib.load(p_abs)
                
                if isinstance(loaded, dict):
                    # Unpacks the v18 bundle
                    model = loaded.get('lgb_model')
                    features = loaded.get('feature_names')
                    scaler = loaded.get('scaler')
                    threshold = loaded.get('thr')
                    use_log_target = True # v18 model uses a log target
                    
                    if not all([model, features, scaler, threshold is not None]):
                        print(f"Error: Model bundle at {p_abs} is missing keys (lgb_model, feature_names, scaler, thr).")
                        continue
                        
                    print("Model bundle loaded successfully.")
                    return {'model': model, 'features': features, 'use_log_target': use_log_target, 'scaler': scaler, 'thr': threshold}
                else:
                    # Fallback for older, raw models
                    print(f"Warning: Loaded a raw model file from {p_abs}. This is not the v18 bundle.")
                    model = loaded
                    return {'model': model, 'features': None, 'use_log_target': False, 'scaler': None, 'thr': 10000.0} # Fallback
        except Exception as e:
            print(f"Error loading {p}: {e}")
    print("No valid model found in expected locations.")
    return None

# -------------------------
# Feature Engineering (from v18)
# -------------------------
def engineer_features(daily: pd.DataFrame, lead_days: int) -> pd.DataFrame:
    """Create features (exactly as in flood_model_v18.py) ."""
    df = daily.copy()
    # Basic derived vars
    df["tmean"] = (df["tmax"] + df["tmin"]) / 2.0
    df["diurnal_range"] = df["tmax"] - df["tmin"]
    df["precip_roll3"] = df["precip"].rolling(3, min_periods=1).sum()
    df["precip_roll7"] = df["precip"].rolling(7, min_periods=1).sum()
    df["rain_roll7"] = df["rain"].rolling(7, min_periods=1).sum()
    df["et0_roll7"] = df["et0"].rolling(7, min_periods=1).sum()
    df["tmean_roll7"] = df["tmean"].rolling(7, min_periods=1).mean()
    # Lags
    for l in [1, 2, 3, 7, 14]:
        df[f"precip_lag{l}"] = df["precip"].shift(l)
        df[f"rain_lag{l}"] = df["rain"].shift(l)
        df[f"tmean_lag{l}"] = df["tmean"].shift(l)
    # Drop rows with NaNs created by the 14-day lag
    df = df.iloc[14:].copy() 
    return df

# -------------------------
# Fetch and assemble live features (robust)
# -------------------------
def get_live_features_for_model(lat=DEFAULT_LAT, lon=DEFAULT_LON, expected_features=None):
    """
    Returns: (feature_vector_df, meta) or (None, error_message)
    feature_vector_df: single-row DataFrame with columns matching expected_features
    """
    if expected_features is None:
        return None, "Error: 'expected_features' list not provided by model bundle."

    try:
        # 1) Fetch live weather data
        # We must fetch the last 30 days for all lags and rolling averages
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": [
                "precipitation_sum",
                "rain_sum",
                "temperature_2m_max",
                "temperature_2m_min",
                "windspeed_10m_max",
                "shortwave_radiation_sum",
                "et0_fao_evapotranspiration"
            ],
            "past_days": 30, # Must be >= 14 for the lags in engineer_features
            "forecast_days": 1, # We only need today's data
            "timezone": "UTC",
        }
        
        r_weather = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=TIMEOUT)
        r_weather.raise_for_status()
        weather_json = r_weather.json()

        if 'daily' not in weather_json:
            return None, "Open-Meteo API did not return 'daily' data."

        # 2) Convert to DataFrame and rename
        daily = pd.DataFrame(weather_json["daily"])
        daily["time"] = pd.to_datetime(daily["time"])
        daily = daily.set_index("time").sort_index()

        # I'll rename columns to the short names my 'engineer_features' function expects
        daily = daily.rename(columns={
            "precipitation_sum": "precip",
            "rain_sum": "rain",
            "temperature_2m_max": "tmax",
            "temperature_2m_min": "tmin",
            "windspeed_10m_max": "wind",
            "shortwave_radiation_sum": "swrad",
            "et0_fao_evapotranspiration": "et0",
        })

        # 3) Engineer features (using the *exact* same function from v18 script)
        df_features = engineer_features(daily, lead_days=0)
        
        # 4) Get the *very last row* of features (this is "today")
        #    and ensure it only has the columns the model expects
        latest_features_row = df_features[expected_features].iloc[-1:]
        
        if latest_features_row.empty:
            return None, "Feature engineering resulted in empty data."
            
        print("Successfully fetched and engineered live features.")
        # Return the single-row DataFrame
        return latest_features_row, None

    except Exception as e:
        return None, f"Exception while fetching live features: {e}"


# -------------------------
# SendGrid helper (unchanged)
# -------------------------
def send_flood_alert(user_email, location_name, risk_level, predicted_discharge):
    """
    Sends a single flood alert email using SendGrid.
    """
    sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
    from_email = os.getenv("SENDGRID_FROM_EMAIL")
    if not sendgrid_api_key or not from_email:
        print("Error: SendGrid credentials not configured.")
        return

    # Creates an email message with the discharge data
    message = Mail(
        from_email=from_email,
        to_emails=user_email,
        subject=f"FLOOD ALERT: {risk_level} Risk Detected for {location_name}",
        html_content=f"""
            <strong>This is an automated flood alert for {location_name}.</strong>
            <p>A new prediction has detected a <strong>{risk_level}</strong> risk of flooding.</p>
            <p>The predicted flood metric (discharge proxy) for tomorrow is <strong>{predicted_discharge:.2f}</strong>.</p>
            <p>Please take necessary precautions.</p>
        """
    )
    
    try:
        sg = SendGridAPIClient(sendgrid_api_key)
        sg.send(message)
        print(f"Successfully sent alert to {user_email}")
    except Exception as e:
        print(f"Error sending email to {user_email}: {e}")
        
# -------------------------
# Main prediction cycle
# -------------------------
def run_prediction_cycle():
    """
    Main function to run the prediction, save it, and trigger alerts.
    """
    model_bundle = load_model()
    if not model_bundle:
        print("Model bundle is not loaded. Aborting prediction cycle.")
        return
        
    # I'll unpack the v18 model bundle
    model = model_bundle['model']
    features_expected = model_bundle['features']
    use_log_target = model_bundle['use_log_target']
    scaler = model_bundle['scaler']
    DANGER_DISCHARGE = model_bundle['thr'] # The threshold calculated during training
    MEDIUM_DISCHARGE = DANGER_DISCHARGE * 0.8 # 80% of danger level

    print(f"--- Starting new prediction cycle (Model v18) ---")
    print(f"Flood Threshold set to: {DANGER_DISCHARGE:.2f}")
    
    db: Session = database.SessionLocal()
    
    try:
        # I'll process all locations in my database
        locations = db.query(models.Location).all()
        if not locations:
            print("No locations found in database. Add locations via 'batch_upload_all.py' or API.")
            return

        for location in locations:
            print(f"\nProcessing location: {location.name} (ID: {location.location_id})")
            
            # --- 1. Get Live Data ---
            # This returns a single-row DataFrame
            feature_df, error = get_live_features_for_model(
                lat=location.latitude, 
                lon=location.longitude, 
                expected_features=features_expected
            )
            
            if feature_df is None:
                print(f"Could not get live data for {location.name}: {error}")
                continue # Skip to the next location

            # --- 2. Run the REAL Model ---
            
            # --- THIS IS THE FIX ---
            # 1. Get the feature values as an array (no names) for the scaler
            X_unscaled_values = feature_df[features_expected].values
            
            # 2. Scale the data (Array -> Array)
            X_scaled_values = scaler.transform(X_unscaled_values)
            
            # 3. Convert the scaled array *back* to a DataFrame with names for the model
            X_scaled_df = pd.DataFrame(X_scaled_values, columns=features_expected)
            # --- END FIX ---
            
            # Predict using the DataFrame
            try:
                y_pred_raw = model.predict(X_scaled_df)[0]
                
                # I must invert the log (np.expm1) if the model was trained on a log target
                if use_log_target:
                    predicted_discharge = float(np.expm1(y_pred_raw))
                else:
                    predicted_discharge = float(y_pred_raw)

            # ... (rest of the try/except block) ...
            except Exception as e:
                print("Prediction failed:", e)
                continue

            # --- 3. Determine Risk Level ---
            risk_level = "LOW"
            if predicted_discharge >= DANGER_DISCHARGE:
                risk_level = "HIGH"
            elif predicted_discharge >= MEDIUM_DISCHARGE:
                risk_level = "MEDIUM"
            
            print(f"Prediction for {location.name}: {risk_level} (Predicted Discharge: {predicted_discharge:.2f})")

            # --- 4. SAVE THE PREDICTION TO THE DATABASE ---
            # (This part is correct)
            db_prediction = models.Prediction(
                location_id=location.location_id,
                predicted_discharge=predicted_discharge, 
                risk_level=risk_level
            )
            db.add(db_prediction)
            db.commit()
            print(f"Successfully saved prediction {db_prediction.prediction_id} to database.")

            # --- 5. TRIGGER ALERTS IF RISK IS HIGH ---
            # (This part is correct)
            if db_prediction.risk_level == "HIGH":
                print(f"Risk is HIGH for {location.name}. Checking for subscribed users...")
                
                users_to_alert = db.query(models.User).filter(
                    models.User.subscribed_location_id == location.location_id
                ).all()

                if not users_to_alert:
                    print("No users are subscribed to this location. No alerts sent.")
                else:
                    print(f"Found {len(users_to_alert)} user(s) to alert.")
                    for user in users_to_alert:
                        send_flood_alert(user.email, location.name, db_prediction.risk_level, predicted_discharge)
            else:
                print("Risk is not HIGH. No alerts will be sent.")

    except Exception as e:
        print(f"An error occurred during the prediction cycle: {e}")
        db.rollback()
    finally:
        db.close()
        print("\n--- Prediction cycle finished ---")

if __name__ == "__main__":
    load_dotenv() # Loads my .env file
    run_prediction_cycle()