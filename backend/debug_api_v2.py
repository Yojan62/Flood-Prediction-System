import requests
import json
import sys

# This script will help us debug the 'historical-waterlevel' endpoint
# with better error checking.

STATION_ID = 12 # Panchagarh
API_URL = 'https://api3.ffwc.gov.bd/data_load/'

params = {
    'station_id': STATION_ID,
    'start_date': '2025-10-01',
    'end_date': '2024--10-30', # Just 5 days
    'hour': '12'
}

print(f"--- Debugging FFWC 'historical-waterlevel' for Station: {STATION_ID} ---")

try:
    water_url = f"{API_URL}historical-waterlevel/"
    response_water = requests.get(water_url, params=params, timeout=10)
    response_water.raise_for_status()
    water_data = response_water.json()
    
    print("\n--- API Call Successful! ---")
    
    # --- NEW CHECK ---
    # First, let's see the *entire* raw response
    print(f"Full raw JSON response: {water_data}")
    
    # Now, let's check if it's an empty list
    if not water_data:
        print("\nAPI returned an empty list []. No records found for this date range.")
    else:
        # If it's not empty, print the first record
        print("\nHere is the first record from the JSON response:")
        print(json.dumps(water_data[0], indent=2))
    # --- END NEW CHECK ---

except requests.exceptions.RequestException as e:
    print(f"\n--- API CALL FAILED ---")
    print(f"Error: {e}")

except Exception as e:
    # --- NEW ERROR PRINTING ---
    print(f"\n--- SCRIPT FAILED ---")
    print(f"An error of type {type(e)} occurred.")
    print(f"Error details: {e}")
    # --- END NEW ERROR PRINTING ---

print("\n--- Debug script finished ---")