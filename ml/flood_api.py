"""
Flood API integration for the FLOW model.

This module talks to the Open-Meteo Global Flood API
and returns daily river discharge data from GloFAS.
"""

from __future__ import annotations

from typing import List, Optional

import requests
import pandas as pd

try:
    from ml.config import FLOOD_API_URL, FLOOD_DAILY_VARS
except ImportError:
    from config import FLOOD_API_URL, FLOOD_DAILY_VARS


def fetch_flood_discharge(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    daily_vars: Optional[List[str]] = None,
    cell_selection: str = "land",
    timeout: int = 40,
) -> pd.DataFrame:
    """
    Fetch daily river discharge data from the Open-Meteo Global Flood API.

    Parameters
    ----------
    lat : float
        Latitude of the location.
    lon : float
        Longitude of the location.
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
    daily_vars : list of str, optional
        Which daily flood variables to request. Defaults to FLOOD_DAILY_VARS.
    cell_selection : str
        How the underlying grid cell is chosen. One of {"land", "sea", "nearest"}.
        "land" (default) tries to pick a representative river grid cell on land.
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with columns prefixed with "gf_".
        Example columns: gf_river_discharge, gf_river_discharge_max, ...
    """
    if daily_vars is None:
        daily_vars = FLOOD_DAILY_VARS

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": daily_vars,
        "cell_selection": cell_selection,
        "timeformat": "iso8601",
    }

    response = requests.get(FLOOD_API_URL, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    if "daily" not in data or "time" not in data["daily"]:
        # API returned no usable daily data
        return pd.DataFrame()

    daily = pd.DataFrame(data["daily"])
    daily["time"] = pd.to_datetime(daily["time"])
    daily = daily.set_index("time").sort_index()

    # Prefix columns to avoid collisions (gf_ = GloFAS)
    rename_map = {
        col: f"gf_{col}"
        for col in daily.columns
        if col != "time"
    }
    daily = daily.rename(columns=rename_map)

    return daily
