import os
import joblib # Used to load the .pkl model
import numpy as np # Used to shape the data for the model
import requests # Used to call the Open-Meteo API
import datetime
import pandas as pd
from sqlalchemy.orm import Session
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv

import database
import models

# --- 1. Load the REAL Model ---
try:
    # Loads the 'open_meteo_flood_model.pkl' file from the 'ml' folder
    model_path = os.path.join(os.path.dirname(__file__), '..', 'ml', 'open_meteo_flood_model.pkl')
    model = joblib.load(model_path) 
    print(f"Successfully loaded '{model_path}'.")
except FileNotFoundError:
    print("Error: 'open_meteo_flood_model.pkl' not found. Make sure it's in the 'ml' folder.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def send_flood_alert(user_email, location_name, risk_level, predicted_discharge):
    """
    Sends a single flood alert email using SendGrid.
    """
    sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
    from_email = os.getenv("SENDGRID_FROM_EMAIL")
    if not sendgrid_api_key or not from_email:
        print("Error: SendGrid credentials not configured.")
        return

    # Creates an email message with the new discharge data.
    message = Mail(
        from_email=from_email,
        to_emails=user_email,
        subject=f"FLOOD ALERT: {risk_level} Risk Detected for {location_name}",
        html_content=f"""
            <strong>This is an automated flood alert for {location_name}.</strong>
            <p>A new prediction has detected a <strong>{risk_level}</strong> risk of flooding.</p>
            <p>The predicted river discharge for tomorrow is <strong>{predicted_discharge:.2f} m³/s</strong>.</p>
            <p>Please take necessary precautions.</p>
        """
    )
    
    try:
        sg = SendGridAPIClient(sendgrid_api_key)
        sg.send(message)
        print(f"Successfully sent alert to {user_email}")
    except Exception as e:
        print(f"Error sending email to {user_email}: {e}")

def get_live_features_for_model(lat=23.81, lon=90.41):
    """
    Fetches and assembles the live data features needed by the Open-Meteo model.
    Model Features: 9 features including forecast
    """
    print("Fetching live data from Open-Meteo...")
    try:
        # --- 1. Fetch Live Flood Data (All discharge data) ---
        flood_api_url = "https://flood-api.open-meteo.com/v1/flood"
        flood_params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "river_discharge", 
            "forecast_days": 1, 
            "past_days": 2, 
            "timezone": "auto"
        }
        response_flood = requests.get(flood_api_url, params=flood_params, timeout=10)
        response_flood.raise_for_status()
        data_flood = response_flood.json()
        
        # --- 2. Fetch Past & Future HOURLY Weather Data ---
        weather_api_url = "https://api.open-meteo.com/v1/forecast"
        
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["precipitation", "soil_moisture_0_1cm", "et0_fao_evapotranspiration", "temperature_2m"], # Ask for hourly vars
            "past_days": 7,  # Get 7 days of hourly data
            "forecast_days": 1, # Get 1 day of future data
            "timezone": "auto"
        }
        response_weather = requests.get(weather_api_url, params=weather_params, timeout=10)
        response_weather.raise_for_status()
        data_weather = response_weather.json()

        # --- 2b. Aggregate HOURLY data into DAILY data ---
        
        # --- THIS IS THE MISSING LINE THAT FIXES THE ERROR ---
        df_weather_hourly = pd.DataFrame(data_weather['hourly'])
        
        # The 'time' array now has past AND future, so we get it from the root
        df_weather_hourly['time'] = pd.to_datetime(data_weather['hourly']['time'])
        df_weather_hourly = df_weather_hourly.set_index('time')
        
        # Resample the hourly data into daily data
        df_weather_daily = df_weather_hourly.resample('D').agg({
            'precipitation': 'sum',
            'soil_moisture_0_1cm': 'mean',
            'et0_fao_evapotranspiration': 'mean',
            'temperature_2m': 'mean'
        })
        
        # Rename the columns to match our training script
        df_weather_daily = df_weather_daily.rename(columns={
            'precipitation': 'rainfall_mm',
            'soil_moisture_0_1cm': 'soil_moisture',
            'et0_fao_evapotranspiration': 'evapotranspiration',
            'temperature_2m': 'temperature'
        })
        
        # --- 3. Assemble Features ---
        
        # Get Discharge data (this logic is still correct)
        discharge_lag_2 = data_flood['daily']['river_discharge'][0] # 2 days ago
        discharge_lag_1 = data_flood['daily']['river_discharge'][1] # 1 day ago
        live_discharge_today = data_flood['daily']['river_discharge'][2] # Today
        
        # --- NEW: Get features from our aggregated DataFrame ---
        
        # Split the 8-day dataframe into past and future
        past_weather_df = df_weather_daily.iloc[:-1] # The first 7 rows are the past
        future_weather_df = df_weather_daily.iloc[-1:] # The last row is the forecast
        
        # Get features from the PAST data
        rainfall_lag_1 = past_weather_df['rainfall_mm'].iloc[-1] # Yesterday's rain
        soil_moisture = past_weather_df['soil_moisture'].iloc[-1] # Yesterday's soil moisture
        evapotranspiration = past_weather_df['evapotranspiration'].iloc[-1] # Yesterday's evapotranspiration
        temperature = past_weather_df['temperature'].iloc[-1] # Yesterday's temperature
        rainfall_7_day_avg = past_weather_df['rainfall_mm'].mean() # 7-day rain average
        
        # Get our new "forecast" feature from the FUTURE data
        rainfall_forecast_24h = future_weather_df['rainfall_mm'].iloc[0] # Sum of next 24h rain
        
        # --- This logic is still correct ---
        all_past_discharge = np.array([discharge_lag_2, discharge_lag_1, live_discharge_today])
        discharge_3_day_avg = np.mean(all_past_discharge)
        
        live_month = datetime.datetime.now().month
        
        # Creates the input array for the model
        # The order MUST match your training script's feature list
        model_input = np.array([[
            live_discharge_today,
            discharge_lag_1,
            discharge_lag_2,
            rainfall_lag_1,
            live_month,
            rainfall_7_day_avg,
            discharge_3_day_avg,
            soil_moisture,
            evapotranspiration,
            temperature,
            rainfall_forecast_24h  
        ]])
        
        print("Successfully fetched and assembled live features.")
        return model_input

    except Exception as e:
        print(f"Error fetching live data from Open-Meteo: {e}")
        return None

def run_prediction_cycle():
    """
    Main function to run the prediction, save it, and trigger alerts.
    """
    if model is None:
        print("Model is not loaded. Aborting prediction cycle.")
        return
        
    print("--- Starting new prediction cycle ---")
    db: Session = database.SessionLocal()
    
    # --- 1. Get Live Data ---
    # I'll run the prediction for Dhaka (location_id=1)
    model_input = get_live_features_for_model(lat=23.81, lon=90.41)
    
    if model_input is None:
        print("Could not get live data. Aborting cycle.")
        db.close()
        return

    # --- 2. Run the REAL Model ---
    # Predicts tomorrow's river discharge
    predicted_discharge_numpy = model.predict(model_input)[0]
    predicted_discharge = float(predicted_discharge_numpy) # Converts from NumPy to Python float

    # --- 3. Determine Risk Level ---
    # This is a new threshold based on river discharge (m³/s)
    # This value MUST be adjusted based on real data for Dhaka.
    DANGER_DISCHARGE = 10000.0 # EXAMPLE THRESHOLD (10,000 m³/s)
    
    risk_level = "LOW"
    if predicted_discharge >= DANGER_DISCHARGE:
        risk_level = "HIGH"
    elif (DANGER_DISCHARGE - predicted_discharge) < 2000: # If within 2000 m³/s of danger
        risk_level = "MEDIUM"
    
    print(f"Prediction for location 1: {risk_level} (Predicted Discharge: {predicted_discharge:.2f} m³/s)")

    try:
        # --- 4. SAVE THE PREDICTION TO THE DATABASE ---
        db_prediction = models.Prediction(
            location_id=1, # Hard-coded to Dhaka (location_id 1)
            predicted_discharge=predicted_discharge, # Saves the new predicted discharge
            risk_level=risk_level
        )
        db.add(db_prediction)
        db.commit()
        print(f"Successfully saved prediction {db_prediction.prediction_id} to database.")

        # --- 5. TRIGGER ALERTS IF RISK IS HIGH ---
        if db_prediction.risk_level == "HIGH":
            print("Risk is HIGH. Checking for subscribed users...")
            
            location = db.query(models.Location).filter(models.Location.location_id == 1).first()
            location_name = location.name if location else "your subscribed area"

            users_to_alert = db.query(models.User).filter(
                models.User.subscribed_location_id == 1
            ).all()

            if not users_to_alert:
                print("No users are subscribed to this location. No alerts sent.")
            else:
                print(f"Found {len(users_to_alert)} user(s) to alert.")
                for user in users_to_alert:
                    send_flood_alert(user.email, location_name, db_prediction.risk_level, predicted_discharge)
        else:
            print("Risk is not HIGH. No alerts will be sent.")

    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()
    finally:
        db.close()
        print("--- Prediction cycle finished ---")

if __name__ == "__main__":
    load_dotenv()
    run_prediction_cycle()