import requests
import json
import sys
import datetime

# --- 1. Configuration ---
# We'll check for Dhaka, Station ID 42 
STATION_ID_STR = '42' 
STATION_ID_NUM = 42
API_URL = 'https://api3.ffwc.gov.bd/data_load/'
HOUR = '12' # The hour the API requires
TIMEOUT_SECONDS = 20

print(f"--- Starting live data check for Station ID: {STATION_ID_NUM} (Dhaka) ---")

try:
    # --- 2. Calculate Date Range ---
    today = datetime.date.today()
    seven_days_ago = today - datetime.timedelta(days=7)
    
    start_date = seven_days_ago.isoformat()
    end_date = today.isoformat()

    print(f"Fetching data from {start_date} to {end_date}...")

    params = {
        'station_id': STATION_ID_NUM, # API seems to accept number here
        'start_date': start_date,
        'end_date': end_date,
        'hour': HOUR
    }

    # --- 3. Fetch Historical Water Level ---
    print(f"\nFetching water levels...")
    water_level_url = f"{API_URL}historical-waterlevel/"
    response_water = requests.get(water_level_url, params=params, timeout=TIMEOUT_SECONDS)
    response_water.raise_for_status()
    
    water_data = response_water.json()
    
    # --- 4. Process and Print Water Data ---
    if not water_data or STATION_ID_STR not in water_data:
        print("No water level data found for this station in the last 7 days.")
    else:
        station_water_data = water_data[STATION_ID_STR]
        print("--- Water Levels (Last 7 Days) ---")
        print(json.dumps(station_water_data, indent=2))


    # --- 5. Fetch Historical Rainfall ---
    print(f"\nFetching rainfall...")
    rainfall_url = f"{API_URL}rainfall-observations/"
    response_rain = requests.get(rainfall_url, params=params, timeout=TIMEOUT_SECONDS)
    response_rain.raise_for_status()
    
    rain_data_all_stations = response_rain.json()
    
    # We must filter this list to get only our station's data
    station_rain_data = [row for row in rain_data_all_stations if row['station_id'] == STATION_ID_NUM]
    
    # --- 6. Process and Print Rain Data ---
    if not station_rain_data:
        print("\nNo rainfall data found for this station in the last 7 days.")
    else:
        print("\n--- Rainfall (Last 7 Days) ---")
        print(json.dumps(station_rain_data, indent=2))


except requests.exceptions.RequestException as e:
    print(f"\n--- API CALL FAILED ---")
    print(f"Error: {e}")
    print("Could not fetch data. Please check the API status or your parameters.")
    
except KeyError as e:
    print(f"\n--- DATA PARSING FAILED ---")
    print(f"Error: A column name was not found: {e}")
    
print("\n--- Live data check finished ---")