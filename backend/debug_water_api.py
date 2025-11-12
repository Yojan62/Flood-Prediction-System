import requests
import json
import sys

# This script will help us debug the 'historical-waterlevel' endpoint

STATION_ID = 140 # Panchagarh (as a number)
API_URL = 'https://api3.ffwc.gov.bd/data_load/'

# We'll just ask for a small, recent date range
params = {
    'station_id': STATION_ID,
    'start_date': '2024-01-01',
    'end_date': '2024-01-05', # Just 5 days
    'hour': '12'
}

print(f"--- Debugging FFWC 'historical-waterlevel' for Station: {STATION_ID} ---")

try:
    water_url = f"{API_URL}historical-waterlevel/"
    response_water = requests.get(water_url, params=params, timeout=10)
    response_water.raise_for_status()
    water_data = response_water.json()
    
    if not water_data:
        print("API returned no data for this date range.")
    else:
        print("\n--- API Call Successful! ---")
        print("Here is the first record from the JSON response:")
        # Pretty-print the first item in the list
        print(json.dumps(water_data[0], indent=2))

except requests.exceptions.RequestException as e:
    print(f"\n--- API CALL FAILED ---")
    print(f"Error: {e}")

except Exception as e:
    print(f"\n--- SCRIPT FAILED ---")
    print(f"An error occurred: {e}")

print("\n--- Debug script finished ---")