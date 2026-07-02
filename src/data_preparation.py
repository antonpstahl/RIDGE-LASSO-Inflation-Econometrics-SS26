"""Stage 1: Data acquisition from ECB SDW + Eurostat, caching to CSV."""
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests

from .config import (
    DATA_RAW, ECB_BASE, ESTAT_BASE,
    PROD_SECTORS, BS_INDICATORS, PPI_SECTORS, UNEMP_GROUPS, LCI_SERIES,
)


def _fetch_ecb_series(flow_key, start="2000-01"):
    """Loads a monthly time series from the ECB SDW (CSV format)."""
    r = requests.get(
        f"{ECB_BASE}/{flow_key}",
        params={"startPeriod": start, "format": "csvdata"},
        timeout=30,
    )
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m")
    df = df.set_index("TIME_PERIOD").sort_index()
    return df["OBS_VALUE"].astype(float)


def _eurostat_json_to_series(data, freq="M"):
    """Converts a Eurostat JSON response into a pd.Series."""
    time_dim    = data["dimension"]["time"]
    pos_to_label = {v: k for k, v in time_dim["category"]["index"].items()}
    records = []
    for pos_str, val in data["value"].items():
        label = pos_to_label[int(pos_str)]
        if freq == "Q":
            year, q = label.split("-Q")
            month   = (int(q) - 1) * 3 + 1
            date    = pd.Timestamp(f"{year}-{month:02d}-01")
        else:
            date = pd.to_datetime(label, format="%Y-%m")
        records.append((date, float(val)))
    if not records:
        return pd.Series(dtype=float)
    s = pd.Series(dict(records)).sort_index()
    s.index = pd.DatetimeIndex(s.index)
    return s


def _fetch_eurostat(dataset, params, start_year=2000, freq="M"):
    """Loads a time series from Eurostat and filters from start_year."""
    r = requests.get(
        f"{ESTAT_BASE}/{dataset}",
        params={**params, "format": "JSON"},
        timeout=30,
    )
    r.raise_for_status()
    s = _eurostat_json_to_series(r.json(), freq=freq)
    return s[s.index.year >= start_year]


def load_all_data(verbose=True):
    """Loads all time series from ECB + Eurostat and returns a DataFrame."""
    series_dict = {}

    def _log(name, s):
        if verbose and len(s) > 0:
            print(f"  + {name:35s} [{len(s):3d} obs, "
                  f"{s.index[0]:%Y-%m} - {s.index[-1]:%Y-%m}]")

    if verbose:
        print("\n-- Target variable ----------------------------------------------")
    s = _fetch_ecb_series("ICP/M.DE.N.000000.4.INX")
    series_dict["HVPI"] = s
    _log("HICP (overall index 2015=100)", s)
    time.sleep(0.3)

    blocks = [
        ("Industrial production", "sts_inpr_m",
         lambda nace: {"geo": "DE", "s_adj": "NSA", "unit": "I21", "nace_r2": nace},
         PROD_SECTORS, "M"),
        ("Business Surveys", "ei_bsin_m_r2",
         lambda indic: {"geo": "DE", "s_adj": "SA", "indic": indic},
         BS_INDICATORS, "M"),
        ("Producer prices", "sts_inppd_m",
         lambda nace: {"geo": "DE", "s_adj": "NSA", "unit": "I21", "nace_r2": nace},
         PPI_SECTORS, "M"),
        ("Labour market", "une_rt_m",
         lambda grp: {"geo": "DE", "s_adj": "SA", "unit": "PC_ACT", **grp},
         UNEMP_GROUPS, "M"),
    ]
    for title, dataset, mk_params, defs, freq in blocks:
        if verbose:
            print(f"\n-- {title} " + "-" * max(0, 50 - len(title)))
        for name, val in defs.items():
            try:
                s = _fetch_eurostat(dataset, mk_params(val), freq=freq)
                if len(s) > 0:
                    series_dict[name] = s
                    _log(name, s)
            except Exception as e:
                if verbose:
                    print(f"  x {name}: {e}")
            time.sleep(0.2)

    # Labour costs (quarterly -> monthly, step function via ffill - no look-ahead)
    if verbose:
        print("\n-- Labour costs (LCI, quarterly -> monthly) ---------------------")
    for name, flt in LCI_SERIES.items():
        try:
            s_q = _fetch_eurostat(
                "lc_lci_r2_q",
                {"geo": "DE", "s_adj": "NSA", "unit": "I20", **flt},
                freq="Q",
            )
            if len(s_q) > 0:
                s_m = s_q.resample("MS").ffill()
                series_dict[name] = s_m
                _log(name, s_m)
        except Exception as e:
            if verbose:
                print(f"  x {name}: {e}")
        time.sleep(0.2)

    df = pd.DataFrame(series_dict)
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()
    if verbose:
        print(f"\n+ Raw data: {df.shape[0]} periods x {df.shape[1]} variables "
              f"({df.index[0]:%Y-%m} - {df.index[-1]:%Y-%m})")
    return df


def _fix_lci_in_place(df):
    """Ensures that LCI quarterly series are filled via ffill (not interpolation)."""
    for col in [c for c in df.columns if c.startswith("LCI_")]:
        s = df[col].copy()
        s[~s.index.month.isin([1, 4, 7, 10])] = np.nan
        df[col] = s.ffill()


def get_raw_data(use_cache=True, save=True, verbose=True):
    """Returns raw data - from the CSV cache or freshly from the API.

    use_cache=True  -> reads data/raw/data_raw.csv if present.
    use_cache=False -> forces a new API download (~1-2 minutes).
    save=True       -> saves a fresh download as cache.
    """
    if use_cache and DATA_RAW.exists():
        if verbose:
            print(f"Raw data loaded from cache: {DATA_RAW}")
        df = pd.read_csv(DATA_RAW, index_col=0, parse_dates=True)
        # Ensure that LCI columns are ffill-corrected (idempotent)
        _fix_lci_in_place(df)
        df.to_csv(DATA_RAW)
        return df

    df = load_all_data(verbose=verbose)
    if save:
        df.to_csv(DATA_RAW)
        if verbose:
            print(f"Raw data saved as cache: {DATA_RAW}")
    return df


def print_truncation_info(df_raw, y):
    """Diagnoses which series truncate the feature matrix over time."""
    _last_obs = df_raw.apply(lambda s: s.last_valid_index())
    print("Last valid observation per series (shortest first):")
    print(_last_obs.sort_values().head(8).to_string())
    _gap = (
        (df_raw.index[-1].year - y.index[-1].year) * 12
        + df_raw.index[-1].month - y.index[-1].month
    )
    print(f"\nRaw data up to:      {df_raw.index[-1]:%Y-%m}")
    print(f"Feature matrix up to: {y.index[-1]:%Y-%m}")
    print(f"Unused months:       {_gap}")
    print("Reason: shortest series in the list above determines the end date of the feature matrix.")
