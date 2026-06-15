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
    """Laedt eine monatliche Zeitreihe vom ECB SDW (CSV-Format)."""
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
    """Konvertiert eine Eurostat-JSON-Antwort in eine pd.Series."""
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
    """Laedt eine Zeitreihe von Eurostat und filtert ab start_year."""
    r = requests.get(
        f"{ESTAT_BASE}/{dataset}",
        params={**params, "format": "JSON"},
        timeout=30,
    )
    r.raise_for_status()
    s = _eurostat_json_to_series(r.json(), freq=freq)
    return s[s.index.year >= start_year]


def load_all_data(verbose=True):
    """Laedt alle Zeitreihen von ECB + Eurostat; gibt einen DataFrame zurueck."""
    series_dict = {}

    def _log(name, s):
        if verbose and len(s) > 0:
            print(f"  + {name:35s} [{len(s):3d} obs, "
                  f"{s.index[0]:%Y-%m} - {s.index[-1]:%Y-%m}]")

    if verbose:
        print("\n-- Zielvariable -------------------------------------------------")
    s = _fetch_ecb_series("ICP/M.DE.N.000000.4.INX")
    series_dict["HVPI"] = s
    _log("HVPI (Gesamtindex 2015=100)", s)
    time.sleep(0.3)

    blocks = [
        ("Industrieproduktion", "sts_inpr_m",
         lambda nace: {"geo": "DE", "s_adj": "NSA", "unit": "I21", "nace_r2": nace},
         PROD_SECTORS, "M"),
        ("Business Surveys", "ei_bsin_m_r2",
         lambda indic: {"geo": "DE", "s_adj": "SA", "indic": indic},
         BS_INDICATORS, "M"),
        ("Produzentenpreise", "sts_inppd_m",
         lambda nace: {"geo": "DE", "s_adj": "NSA", "unit": "I21", "nace_r2": nace},
         PPI_SECTORS, "M"),
        ("Arbeitsmarkt", "une_rt_m",
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

    # Lohnkosten (Quartal -> Monat, Treppenfunktion via ffill – kein Look-ahead)
    if verbose:
        print("\n-- Lohnkosten (LCI, Quartal -> Monat) ---------------------------")
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
        print(f"\n+ Rohdaten: {df.shape[0]} Perioden x {df.shape[1]} Variablen "
              f"({df.index[0]:%Y-%m} - {df.index[-1]:%Y-%m})")
    return df


def _fix_lci_in_place(df):
    """Stellt sicher, dass LCI-Quartalsreihen per ffill (nicht Interpolation) gefüllt sind."""
    for col in [c for c in df.columns if c.startswith("LCI_")]:
        s = df[col].copy()
        s[~s.index.month.isin([1, 4, 7, 10])] = np.nan
        df[col] = s.ffill()


def get_raw_data(use_cache=True, save=True, verbose=True):
    """Liefert Rohdaten – aus dem CSV-Cache oder frisch von der API.

    use_cache=True  -> liest data/raw/data_raw.csv, falls vorhanden.
    use_cache=False -> erzwingt neuen API-Download (~1-2 Minuten).
    save=True       -> speichert einen frischen Download als Cache.
    """
    if use_cache and DATA_RAW.exists():
        if verbose:
            print(f"Rohdaten aus Cache geladen: {DATA_RAW}")
        df = pd.read_csv(DATA_RAW, index_col=0, parse_dates=True)
        # Sicherstellen, dass LCI-Spalten ffill-korrigiert sind (idempotent)
        _fix_lci_in_place(df)
        df.to_csv(DATA_RAW)
        return df

    df = load_all_data(verbose=verbose)
    if save:
        df.to_csv(DATA_RAW)
        if verbose:
            print(f"Rohdaten als Cache gespeichert: {DATA_RAW}")
    return df


def print_truncation_info(df_raw, y):
    """Diagnostiziert, welche Reihen die Feature-Matrix zeitlich beschneiden."""
    _last_obs = df_raw.apply(lambda s: s.last_valid_index())
    print("Letzte gültige Beobachtung je Reihe (kürzeste zuerst):")
    print(_last_obs.sort_values().head(8).to_string())
    _gap = (
        (df_raw.index[-1].year - y.index[-1].year) * 12
        + df_raw.index[-1].month - y.index[-1].month
    )
    print(f"\nRohdaten bis:        {df_raw.index[-1]:%Y-%m}")
    print(f"Feature-Matrix bis:  {y.index[-1]:%Y-%m}")
    print(f"Ungenutzte Monate:   {_gap}")
    print("Grund: kürzeste Reihe in obiger Liste bestimmt das Enddatum der Feature-Matrix.")
