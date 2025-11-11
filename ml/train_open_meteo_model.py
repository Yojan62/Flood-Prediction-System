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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
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
        "hourly": ["precipitation", "soil_moisture_0_1cm", "et0_fao_evapotranspiration", "temperature_2m"], 
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

print("Defining focused hyperparameter grid...")
# --- REPLACE YOUR OLD param_grid WITH THIS ---
param_grid = {
    'n_estimators': [100, 150, 200],       # Test default (100) and slightly higher
    'min_samples_split': [2, 5],         # Test default (2) and a slightly more robust (5)
    'min_samples_leaf': [1, 2],           # Test default (1) and a more robust (2)
}

print("Creating and training the model with default settings...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Model trained successfully on Open-Meteo data.")

# --- Step 5: Evaluate the New Model ---
y_pred = model.predict(X_test)
# Checks the Mean Absolute Error (how 'off' the prediction is on average).
mae = mean_absolute_error(y_test, y_pred)
print(f"\n--- Model Evaluation ---")
print(f"Mean Absolute Error: {mae:.3f} m³/s") # Units are cubic meters per second

r2 = r2_score(y_test,y_pred)
print(f"R2 Score: {r2:.3f}")

# Runs one example prediction using a row from the test set.
example_input = X_test.iloc[-1].values.reshape(1, -1)
example_prediction = model.predict(example_input)[0]# ---
# Flood Prediction Model (v4) - Tuned & Validated
#
# This script trains a realistic model on 11 features,
# tunes it with GridSearchCV, and validates the result.
# ---

# --- Step 1: Import Libraries ---
import requests 
import pandas as pd
import numpy as np 
import joblib 
import sys 

# Model building
from sklearn.model_selection import GridSearchCV # We'll use this for tuning
from sklearn.ensemble import RandomForestRegressor

# Model evaluation
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt # For plotting

# My libraries imported successfully.
print("Libraries imported successfully.")

# --- Step 2: Fetch Historical Data from Open-Meteo API ---

# I'll define the coordinates for Dhaka.
LATITUDE = 23.81
LONGITUDE = 90.41
START_DATE = "2020-01-01" # Start date for historical data
END_DATE = "2024-12-31" # End date for historical data

print(f"Preparing to fetch real data for Dhaka (Lat: {LATITUDE}, Lon: {LONGITUDE})")

try:
    # --- 2a. Fetch Historical Flood Data (River Discharge) ---
    print("Fetching historical river discharge data...")
    flood_api_url = "https://flood-api.open-meteo.com/v1/flood"
    flood_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "daily": "river_discharge", 
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": "auto"
    }
    response_flood = requests.get(flood_api_url, params=flood_params, timeout=10)
    response_flood.raise_for_status() 
    data_flood = response_flood.json()
    
    # --- 2b. Fetch Historical Weather Data (Hourly) ---
    print("Fetching historical rainfall data...")
    weather_api_url = "https://archive-api.open-meteo.com/v1/archive" 
    weather_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ["precipitation", "soil_moisture_0_1cm", "et0_fao_evapotranspiration", "temperature_2m"], 
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": "auto"
    }
    response_weather = requests.get(weather_api_url, params=weather_params, timeout=10)
    response_weather.raise_for_status()
    data_weather = response_weather.json()

    # --- 2c. Prepare and Merge the Data ---
    
    # First, create the flood DataFrame (this is still daily)
    df_flood = pd.DataFrame(data_flood['daily'])
    df_flood = df_flood.rename(columns={"time": "date", "river_discharge": "discharge_m3_s"})
    df_flood['date'] = pd.to_datetime(df_flood['date'])
    print(f"Found {len(df_flood)} daily river discharge records.")
    
    # Now, process the new HOURLY weather data
    df_weather_hourly = pd.DataFrame(data_weather['hourly']) 
    df_weather_hourly['time'] = pd.to_datetime(df_weather_hourly['time'])
    df_weather_hourly = df_weather_hourly.set_index('time')
    
    # Resample the hourly data into daily data
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
df = df.set_index(pd.to_datetime(df['date']))

# 1. Creates the "clues" (Features, X)
df['discharge_lag_1'] = df['discharge_m3_s'].shift(1) # Yesterday's discharge
df['discharge_lag_2'] = df['discharge_m3_s'].shift(2) # 2 days ago
df['rainfall_lag_1'] = df['rainfall_mm'].shift(1) # Yesterday's rainfall
df['month'] = df.index.month # The month (to capture seasonality)
df['rainfall_7_day_avg'] = df['rainfall_mm'].shift(1).rolling(window=7).mean()
df['discharge_3_day_avg'] = df['discharge_m3_s'].shift(1).rolling(window=3).mean()
df['rainfall_forecast_24h'] = df['rainfall_mm'].shift(-1) # "Perfect forecast"

# 2. Creates the "answer key" (Target, y)
df['target_next_day_discharge'] = df['discharge_m3_s'].shift(-1) # Tomorrow's discharge

# 3. Cleans the data
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

# Fills NaNs from rolling/shifting, then drops rows where target is unknown
df_clean = df.fillna(0)
df_clean = df_clean.dropna(subset=[target]) 

print("Feature engineering complete.")

# --- Step 4: Split Data and Train Model ---

X = df_clean[features]
y = df_clean[target]

# Splits the data into training (80%) and testing (20%) sets.
print("Splitting data by date (80% train, 20% test)...")
df_clean = df_clean.sort_index()

split_point = int(len(df_clean) * 0.8)
split_date = df_clean.index[split_point]

train_df = df_clean.loc[df_clean.index < split_date]
test_df = df_clean.loc[df_clean.index >= split_date]

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"Training on data BEFORE {split_date.date()}")
print(f"Testing on data AFTER {split_date.date()}")
print(f"Data split into {len(X_train)} training rows and {len(X_test)} test rows.")


print("Creating and training the (champion) RandomForest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)


print("Model trained successfully on Open-Meteo data.")

# --- Step 5: Evaluate the New Model ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) # This calculates the R2 score

print(f"\n--- Model Evaluation ---")
print(f"Mean Absolute Error: {mae:.3f} m³/s")
print(f"R2 Score: {r2:.3f}")


# --- Step 6: Save the New, Realistic Model ---
model_filename = 'open_meteo_flood_model.pkl'
joblib.dump(model, model_filename)

print(f"\nNew, realistic model saved as '{model_filename}'")


# --- Step 7: VISUALIZE PREDICTIONS ---

try:
    print("\n--- Visualizing Model Performance ---")

    # We need to test on the FULL dataset to see all flood events
    full_dataset_pred = model.predict(X)

    # Create a new DataFrame for plotting
    plot_df = pd.DataFrame({
        'Actual Discharge': y,
        'Predicted Discharge': full_dataset_pred
    }, index=y.index) # Use the original dates as the index

    # Filter down to our 2024 test set to see the flood
    plot_df_2024 = plot_df.loc['2024-01-01':]

    print("Generating plot...")
    plt.figure(figsize=(15, 7))
    plt.plot(plot_df_2024.index, plot_df_2024['Actual Discharge'], label='Actual River Discharge', color='blue', alpha=0.7)
    plt.plot(plot_df_2024.index, plot_df_2024['Predicted Discharge'], label='Model Prediction', color='red', linestyle='--')
    
    # Highlight the known August/September 2024 flood period
    plt.axvspan('2024-08-15', '2024-09-15', color='orange', alpha=0.2, label='Known Flood Period (Aug-Sep 2024)')
    
    plt.title('Model Backtest: Actual vs. Predicted (2024)')
    plt.ylabel('River Discharge (m³/s)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as an image in your 'ml' folder
    plot_filename = 'model_backtest_2024.png'
    plt.savefig(plot_filename)
    print(f"Successfully saved validation plot as '{plot_filename}'")
    plt.show() # This will also open the plot window

except ImportError:
    print("\nTo visualize the results, please install matplotlib: pip install matplotlib")
print(f"\n--- Example Prediction ---")
print(f"Input (today's discharge, lags, etc.): {X_test.iloc[-1].values}")
print(f"Predicted Discharge for Tomorrow: {example_prediction:.2f} m³/s")

# --- Step 6: Save the New, Realistic Model ---
# This is the file the backend will load and use.
model_filename = 'open_meteo_flood_model.pkl'
joblib.dump(model, model_filename)

print(f"\nNew, realistic model saved as '{model_filename}'")