# ---
# Flood Prediction Model (v3) - Training on Open-Meteo API Data
#
# This script trains a realistic model using two separate
# Open-Meteo APIs: one for flood (river discharge) and one for weather (rainfall).
# ---

# --- Step 1: Import Libraries ---
import requests 
import pandas as pd
import numpy as np 
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import sys 

# My libraries imported successfully.
print("Libraries imported successfully.")

# --- Step 2: Fetch Historical Data from Open-Meteo API ---

# I'll define the coordinates for Dhaka.
LATITUDE = 23.81
LONGITUDE = 90.41
START_DATE = "2020-01-01" # Start date for historical data
END_DATE = "2024-12-31"   # End date for historical data

print(f"Preparing to fetch real data for Dhaka (Lat: {LATITUDE}, Lon: {LONGITUDE})")

try:
    # --- 2a. Fetch Historical Flood Data (River Discharge) ---
    # This call uses the 'flood-api.open-meteo.com' domain.
    print("Fetching historical river discharge data...")
    # As per the docs, the endpoint for historical data is /v1/historical
    flood_api_url = "https://flood-api.open-meteo.com/v1/flood"
    flood_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "daily": "river_discharge", # Only request river discharge
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": "auto"
    }
    response_flood = requests.get(flood_api_url, params=flood_params, timeout=10)
    response_flood.raise_for_status() # This raises an error if the request fails
    data_flood = response_flood.json()
    
    # Creates the first DataFrame for discharge.
    df_flood = pd.DataFrame(data_flood['daily'])
    df_flood = df_flood.rename(columns={
        "time": "date",
        "river_discharge": "discharge_m3_s"
    })
    print(f"Found {len(df_flood)} river discharge records.")

    # --- 2b. Fetch Historical Weather Data (Rainfall) ---
    # This call uses the 'api.open-meteo.com' domain.
    print("Fetching historical rainfall data...")
    # As per Open-Meteo docs, the 'archive' endpoint is for past weather.
    weather_api_url = "https://archive-api.open-meteo.com/v1/archive" 
    weather_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ["precipitation", "soil_moisture_0_1cm", "et0_fao_evapotranspiration", "temperature_2m"], # Request hourly data
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": "auto"
    }
    response_weather = requests.get(weather_api_url, params=weather_params, timeout=10)
    response_weather.raise_for_status()
    data_weather = response_weather.json()

    # --- 2c. Prepare and Merge the Data ---
    # Converts 'date' columns to datetime objects for a clean merge.
    
    # First, create the flood DataFrame (this is still daily)
    df_flood = pd.DataFrame(data_flood['daily'])
    df_flood = df_flood.rename(columns={"time": "date", "river_discharge": "discharge_m3_s"})
    df_flood['date'] = pd.to_datetime(df_flood['date'])
    print(f"Found {len(df_flood)} daily river discharge records.")
    
    # Now, process the new HOURLY weather data
    # This line now correctly looks for the 'hourly' key
    df_weather_hourly = pd.DataFrame(data_weather['hourly']) 
    df_weather_hourly['time'] = pd.to_datetime(df_weather_hourly['time'])
    df_weather_hourly = df_weather_hourly.set_index('time')
    
    # Resample the hourly data into daily data
    # We SUM the rain and take the MEAN (average) of soil moisture
    df_weather_daily = df_weather_hourly.resample('D').agg({
        'precipitation': 'sum',
        'soil_moisture_0_1cm': 'mean',
        'et0_fao_evapotranspiration': 'mean',
        'temperature_2m': 'mean'
    })
    
    # Rename the new daily columns
    df_weather_daily = df_weather_daily.rename(columns={
        'precipitation': 'rainfall_mm',
        'soil_moisture_0_1cm': 'soil_moisture',
        'et0_fao_evapotranspiration': 'evapotranspiration',
        'temperature_2m': 'temperature'
    })
    
    # Reset the index so we can merge on the 'date' column
    df_weather_daily = df_weather_daily.reset_index().rename(columns={'time': 'date'})
    print(f"Aggregated {len(df_weather_hourly)} hourly records into {len(df_weather_daily)} daily weather records.")

    # Merges the two data sources on the 'date' column.
    df = pd.merge(df_flood, df_weather_daily, on='date', how='inner')
    
    print("Successfully merged river discharge and aggregated weather data.")


except requests.exceptions.RequestException as e:
    print(f"--- API CALL FAILED ---")
    print(f"Error: {e}")
    print("Could not fetch data from Open-Meteo. Please check your connection.")
    sys.exit() # Exit the script
    
except KeyError as e:
    print(f"--- DATA PARSING FAILED ---")
    print(f"Error: A column name was not found: {e}")
    print("The API response may have changed. Please check the JSON output.")
    sys.exit()

# --- Step 3: Prepare Data for Time-Series Prediction (Feature Engineering) ---

print("Starting feature engineering...")
# Sets 'date' as the DataFrame index for easier time-series manipulation.
df = df.set_index(pd.to_datetime(df['date']))

# 1. Creates the "clues" (Features, X)
#    .shift(1) gets yesterday's value.
df['discharge_lag_1'] = df['discharge_m3_s'].shift(1) # Yesterday's discharge
df['discharge_lag_2'] = df['discharge_m3_s'].shift(2) # 2 days ago
df['rainfall_lag_1'] = df['rainfall_mm'].shift(1) # Yesterday's rainfall
df['month'] = df.index.month # The month (to capture seasonality)
df['rainfall_7_day_avg'] = df['rainfall_mm'].shift(1).rolling(window=7).mean()
df['discharge_3_day_avg'] = df['discharge_m3_s'].shift(1).rolling(window=3).mean()
df['rainfall_forecast_24h'] = df['rainfall_mm'].shift(-1) # Tomorrow's rainfall (forecast)

# 2. Creates the "answer key" (Target, y)
#    I will predict tomorrow's river discharge.
df['target_next_day_discharge'] = df['discharge_m3_s'].shift(-1) # Tomorrow's discharge

# 3. Cleans the data (drop rows with no history or no future)
features = [
            'discharge_m3_s',
            'discharge_lag_1',
            'discharge_lag_2', 
            'rainfall_lag_1', 
            'month',
            'rainfall_7_day_avg',
            'discharge_3_day_avg',
            'soil_moisture',
            'rainfall_forecast_24h',
            'evapotranspiration',
            'temperature'
           ]
target = 'target_next_day_discharge'
# Fills any remaining NaNs in features with 0 (e.g., if rainfall was 0)
df_clean = df.fillna(0)
df_clean = df_clean.dropna(subset=[target]) # Only drop rows where the target is missing

print("Feature engineering complete.")

# --- Step 4: Split Data and Train Model ---

X = df_clean[features]
y = df_clean[target]

# Splits the data into training (80%) and testing (20%) sets.
print("Splitting data by date (80% train, 20% test)...")
# Sorts by date just to be 100% sure
df_clean = df_clean.sort_index()

# Finds the date that splits the data 80/20
split_point = int(len(df_clean) * 0.8)
split_date = df_clean.index[split_point]

# Splits into "past" (train) and "future" (test)
train_df = df_clean.loc[df_clean.index < split_date]
test_df = df_clean.loc[df_clean.index >= split_date]

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"Training on data BEFORE {split_date.date()}")
print(f"Testing on data AFTER {split_date.date()}")

print(f"Data split into {len(X_train)} training rows and {len(X_test)} test rows.")

# Creates and trains the RandomForestRegressor.
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Model trained successfully on Open-Meteo data.")

# --- Step 5: Evaluate the New Model ---
y_pred = model.predict(X_test)
# Checks the Mean Absolute Error (how 'off' the prediction is on average).
mae = mean_absolute_error(y_test, y_pred)
print(f"\n--- Model Evaluation ---")
print(f"Mean Absolute Error: {mae:.3f} m³/s") # Units are cubic meters per second

# Runs one example prediction using a row from the test set.
example_input = X_test.iloc[-1].values.reshape(1, -1)
example_prediction = model.predict(example_input)[0]
print(f"\n--- Example Prediction ---")
print(f"Input (today's discharge, lags, etc.): {X_test.iloc[-1].values}")
print(f"Predicted Discharge for Tomorrow: {example_prediction:.2f} m³/s")

# --- Step 6: Save the New, Realistic Model ---
# This is the file the backend will load and use.
model_filename = 'open_meteo_flood_model.pkl'
joblib.dump(model, model_filename)

print(f"\nNew, realistic model saved as '{model_filename}'")