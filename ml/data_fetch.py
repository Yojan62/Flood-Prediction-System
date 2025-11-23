"""
Data fetching module for FLOW v23.
"""
import time
import pandas as pd
import requests
from datetime import date
from .config import OPEN_METEO_ARCHIVE, OPEN_METEO_FLOOD, TIMEZONE

def safe_get_json(url, params, timeout=90):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fetch_archive_hourly(lat, lon, start_date, end_date):
    """Fetches hourly weather data."""
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    today = date.today()
    if end > today: end = today

    dfs = []
    year = start.year
    while year <= end.year:
        seg_start = max(start, date(year, 1, 1))
        seg_end   = min(end, date(year, 12, 31))
        print(f"→ Fetching weather segment {seg_start} → {seg_end}")
        
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "precipitation,relative_humidity_2m,et0_fao_evapotranspiration,temperature_2m",
            "start_date": seg_start.isoformat(), "end_date": seg_end.isoformat(),
            "timezone": TIMEZONE
        }
        try:
            j = safe_get_json(OPEN_METEO_ARCHIVE, params)
            hourly = pd.DataFrame(j['hourly'])
            hourly['time'] = pd.to_datetime(hourly['time'])
            hourly = hourly.set_index('time').sort_index()
            dfs.append(hourly)
        except Exception as e:
            print(f"Error fetching weather: {e}")
            
        year += 1
        time.sleep(0.6)
        
    if not dfs: return pd.DataFrame()
    
    df = pd.concat(dfs).sort_index()
    rename_map = {
        'precipitation': 'rainfall_mm',
        'relative_humidity_2m': 'relative_humidity',
        'et0_fao_evapotranspiration': 'evapotranspiration',
        'temperature_2m': 'temperature'
    }
    df = df.rename(columns=rename_map)
    if 'rainfall_mm' in df.columns:
        df['max_hourly_rain_mm'] = df['rainfall_mm']
    else:
        df['rainfall_mm'] = 0.0
        df['max_hourly_rain_mm'] = 0.0
    return df

def fetch_flood_daily(lat, lon, start_date, end_date):
    """Fetches REAL historical river discharge."""
    print(f"Fetching river discharge {start_date} -> {end_date}")
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    today = date.today()
    if end > today: end = today
    
    dfs = []
    year = start.year
    while year <= end.year:
        seg_start = max(start, date(year, 1, 1))
        seg_end   = min(end, date(year + 4, 12, 31))
        print(f"→ Fetching discharge segment {seg_start} → {seg_end}")
        params = {
            "latitude": lat, "longitude": lon, "daily": "river_discharge",
            "start_date": seg_start.isoformat(), "end_date": seg_end.isoformat(), "timezone": TIMEZONE
        }
        try:
            j = safe_get_json(OPEN_METEO_FLOOD, params)
            df_chunk = pd.DataFrame(j.get("daily", {}))
            if 'time' in df_chunk.columns:
                df_chunk = df_chunk.rename(columns={"time": "date", "river_discharge": "discharge_m3_s"})
                df_chunk['date'] = pd.to_datetime(df_chunk['date'])
                df_chunk = df_chunk.set_index('date').sort_index()
                dfs.append(df_chunk)
        except Exception as e:
            print(f"Error fetching discharge: {e}")
            
        year += 5 
        time.sleep(0.5)

    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df