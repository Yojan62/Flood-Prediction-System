"""
fetch_ffwc_live.py
──────────────────
Fetches the latest real observed water level data from Bangladesh Flood Forecasting & Warning Centre (FFWC).
Filters selected stations and stores them as a clean CSV file for ML training or monitoring.

Usage:
    python fetch_ffwc_live.py --stations 77 80 109 --out ffwc_latest.csv
"""

import requests
import pandas as pd
import argparse
import datetime as dt
from pathlib import Path

FFWC_API = "https://api3.ffwc.gov.bd/data_load/observed"

def fetch_ffwc_data():
    """Fetch raw observed JSON from FFWC API."""
    print("🌊 Fetching observed water level data from FFWC...")
    resp = requests.get(FFWC_API, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    print(f"✅ Received {len(data)} station entries.")
    return data

def process_ffwc_data(data, stations=None):
    """Convert JSON to DataFrame and filter selected stations."""
    df = pd.DataFrame(data)
    
    # Clean and rename relevant columns
    df = df.rename(columns={
        "st_id": "station_id",
        "name": "station_name",
        "lat": "latitude",
        "long": "longitude",
        "wl_date": "datetime",
        "waterlevel": "water_level_m",
        "dangerlevel": "danger_level_m",
        "riverhighestwaterlevel": "highest_recorded_m"
    })

    # Convert numeric columns
    for col in ["water_level_m", "danger_level_m", "highest_recorded_m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Filter stations
    if stations:
        df = df[df["station_id"].isin(stations)]
        print(f"📍 Filtered to {len(df)} rows for stations {stations}")

    # Add fetch timestamp
    df["fetched_utc"] = dt.datetime.utcnow().isoformat() + "Z"

    # Sort and reset
    df = df.sort_values(["station_id"]).reset_index(drop=True)

    return df[[
        "station_id", "station_name", "river", "district",
        "datetime", "water_level_m", "danger_level_m",
        "highest_recorded_m", "latitude", "longitude", "fetched_utc"
    ]]

def main():
    parser = argparse.ArgumentParser(description="Fetch and save FFWC observed data.")
    parser.add_argument("--stations", nargs="*", type=int, default=None, help="Station IDs to include (e.g., 77 80 109)")
    parser.add_argument("--out", default="ffwc_latest.csv", help="Output CSV path")
    args = parser.parse_args()

    # Fetch and process
    try:
        data = fetch_ffwc_data()
        df = process_ffwc_data(data, args.stations)

        out_path = Path(args.out)
        df.to_csv(out_path, index=False)
        print(f"💾 Saved {len(df)} rows → {out_path.resolve()}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
