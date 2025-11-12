import requests
import json

# This script will help us debug the FFWC API response

STATION_ID = 'SW140' # Panchagarh
API_URL = 'https://api3.ffwc.gov.bd/data_load/'

# We'll just ask for one day of data
params = {
    'station_id': STATION_ID,
    'start_date': '2024-01-01',
    'end_date': '2024-01-02',
    'hour': '12'
}

print(f"--- Debugging FFWC API for Station: {STATION_ID} ---")

try:
    # --- 1. Test the 'historical-waterlevel' endpoint ---
    print("\n--- Testing 'historical-waterlevel' ---")
    water_url = f"{API_URL}historical-waterlevel/"
    response_water = requests.get(water_url, params=params, timeout=10)
    response_water.raise_for_status()
    water_data = response_water.json()
    
    print("Success. JSON response:")
    # Pretty-print the JSON response so we can read it
    print(json.dumps(water_data, indent=2))

    # --- 2. Test the 'rainfall-observations' endpoint ---
    print("\n--- Testing 'rainfall-observations' ---")
    rain_url = f"{API_URL}rainfall-observations/"
    response_rain = requests.get(rain_url, params=params, timeout=10)
    response_rain.raise_for_status()
    rain_data = response_rain.json()
    
    print("Success. JSON response:")
    print(json.dumps(rain_data, indent=2))

except requests.exceptions.RequestException as e:
    print(f"\n--- API CALL FAILED ---")
    print(f"Error: {e}")

except Exception as e:
    print(f"\n--- SCRIPT FAILED ---")
    print(f"An error occurred: {e}")

print("\n--- Debug script finished ---")