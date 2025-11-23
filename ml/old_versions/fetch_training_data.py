import requests
import pandas as pd
import sys
import time

# --- 1. Configuration ---
# We use the string '140' for the water data, and the number 140 for rainfall
STATION_ID_STR = '171' # For the water level dictionary key
STATION_ID_NUM = 171  # For the rainfall list filtering 
API_URL = 'https://api3.ffwc.gov.bd/data_load/'
YEARS_TO_FETCH = [2020, 2021, 2022, 2023, 2024]
HOUR = '12' # The hour the API requires
TIMEOUT_SECONDS = 30 

print(f"--- Starting data fetch for Station ID: {STATION_ID_NUM} ---")

all_water_data = {}
all_rain_data = []

try:
    # --- 2. Loop Through Each Year and Fetch Data ---
    for year in YEARS_TO_FETCH:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        print(f"\nFetching data for {year}...")

        # We remove 'station_id' from params since the API ignores it
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'hour': HOUR
        }

        # --- 2a. Fetch Historical Water Level ---
        print(f"Fetching historical water levels for {year}...")
        water_level_url = f"{API_URL}historical-waterlevel/"
        response_water = requests.get(water_level_url, params=params, timeout=TIMEOUT_SECONDS)
        response_water.raise_for_status()
        
        # This is the big dictionary: {'6': {...}, '10': {...}, '140': {...}}
        water_data_all_stations = response_water.json()
        
        # We need to find our specific station's data
        if STATION_ID_STR in water_data_all_stations:
            station_water_data = water_data_all_stations[STATION_ID_STR]
            all_water_data.update(station_water_data) # Add this year's data
            print(f"Found {len(station_water_data)} water level records for {year}.")
        else:
            print(f"Warning: No water level data found for station {STATION_ID_STR} in {year}.")

        # --- 2b. Fetch Historical Rainfall ---
        print(f"Fetching historical rainfall for {year}...")
        rainfall_url = f"{API_URL}rainfall-observations/"
        response_rain = requests.get(rainfall_url, params=params, timeout=TIMEOUT_SECONDS)
        response_rain.raise_for_status()
        
        # This is the big list: [{'station_id': 6, ...}, {'station_id': 140, ...}]
        rain_data_all_stations = response_rain.json()
        
        # We must filter this list to get only our station's data
        station_rain_data = [row for row in rain_data_all_stations if row['station_id'] == STATION_ID_NUM]
        
        if station_rain_data:
            all_rain_data.extend(station_rain_data)
        print(f"Found {len(station_rain_data)} rainfall records for {year}.")
        
        time.sleep(1) # Be nice to the API

    # --- 3. Prepare and Merge the Data ---
    print("\n--- All data fetched. Merging now... ---")
    
    if not all_water_data or not all_rain_data:
        print("No data found for the specified range. Exiting.")
        sys.exit()

    # --- THIS IS THE FIX ---
    # Convert the water data dictionary {'date': 'value', ...} to a DataFrame
    df_water = pd.DataFrame(all_water_data.items(), columns=['observation_date', 'water_level'])
    df_water['date'] = pd.to_datetime(df_water['observation_date']).dt.date
    df_water['water_level_m'] = pd.to_numeric(df_water['water_level'])
    # We'll have to get the danger_level from the 'stations' endpoint later
    df_water['danger_level_m'] = 7.0 # Using a placeholder danger level for now
    
    # Convert the rainfall list of dictionaries [{'observation_date': ..., 'rainfall': ...}] to a DataFrame
    df_rain = pd.DataFrame(all_rain_data)
    df_rain['date'] = pd.to_datetime(df_rain['observation_date']).dt.date
    df_rain['rainfall_mm'] = pd.to_numeric(df_rain['rainfall'])
    
    # Group rainfall by date and sum it
    df_rain = df_rain.groupby('date')['rainfall_mm'].sum().reset_index()
    # --- END FIX ---

    # Merge the two data sources on the 'date' column
    df_training_data = pd.merge(df_water, df_rain, on='date', how='inner')
    
    # Clean up the final table and rename for clarity
    df_training_data = df_training_data[['date', 'water_level_m', 'danger_level_m', 'rainfall_mm']]
    
    print("\n--- Data Merged Successfully ---")
    print(df_training_data.head())
    
    # --- 4. Save to CSV ---
    output_filename = f"ml/SW{STATION_ID_NUM}_training_data.csv"
    df_training_data.to_csv(output_filename, index=False)
    print(f"\nSuccessfully saved training data to '{output_filename}'")


except requests.exceptions.RequestException as e:
    print(f"\n--- API CALL FAILED ---")
    print(f"Error: {e}")
    print("Could not fetch data. Please check the API status or your parameters.")
    sys.exit()
    
except KeyError as e:
    print(f"\n--- DATA PARSING FAILED ---")
    print(f"Error: A column name was not found: {e}")
    print("The API response may have changed. I've updated the script, but double-check the JSON.")
    sys.exit()