"""Real-world data layer.

Primary source: NREL/NLR ESIF HPC data-center PUE dataset (measured IT power, cooling/pump/HVAC power,
outdoor air temperature + humidity; CC-BY-4.0; https://data.nrel.gov/submissions/300).
Also supports: your own CSV/Parquet (column mapping) and an offline synthetic demo.

Big files are streamed in Parquet batches and aggregated on the fly (sum/count -> mean), so memory stays small
even for tens of millions of raw rows.
"""
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).parent
BATCH_ROWS = 500_000

ESIF_FILES = {   # (local name, [mirrors tried in order]) — NREL was renamed NLR, so both hosts are listed
    "power": ("esif_power.parquet", [
        "https://data.nlr.gov/system/files/300/1757103411-esif.influx.buildingData.PUE.combined.parquet",
        "https://data.nrel.gov/system/files/300/1757103411-esif.influx.buildingData.PUE.combined.parquet"]),
    "weather": ("esif_weather.parquet", [
        "https://data.nlr.gov/system/files/300/1757105566-esif.influx.buildingData.outside.combined_2.parquet",
        "https://data.nrel.gov/system/files/300/1757105566-esif.influx.buildingData.outside.combined_2.parquet"]),
}
ESIF_POWER_COLS = {   # canonical name -> accepted spellings in the file
    "it_power_kw": ["it_power_kw", "it_kw", "it_power", "it_equipment_kw", "it_load_kw"],
    "cooling_kw": ["cooling_kw", "cooling_power_kw"],
    "hvac_kw": ["hvac_kw"],
    "pump_kw": ["pump_kw", "pumps_kw"],
    "plug_and_light_kw": ["plug_and_light_kw"],
    "pue": ["pue"],
}
ESIF_WEATHER_COLS = {
    "outside_air_temp": ["outside_air_temp", "outdoor_air_temp", "outdoor_temp", "outside_temp", "oat"],
    "outside_air_humidity": ["outside_air_humidity", "outdoor_air_humidity", "outdoor_humidity", "outside_humidity"],
}
TS_CANDS = ["ts", "timestamp", "time", "_time", "datetime", "date", "__index_level_0__"]
FIELD_CANDS = ["_field", "field", "metric", "variable", "name"]
VAL_CANDS = ["_value", "value", "val"]


def data_dir():
    return Path(os.environ.get("DC_DATA_DIR", ROOT / "data"))


# ------------------------------------------------------------------ download
def esif_present():
    return all((data_dir() / "raw" / f).exists() for f, _ in ESIF_FILES.values())


def _download(url, dest, progress, lo, hi, label):
    req = urllib.request.Request(url, headers={"User-Agent": "dc-cooling-app/2.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(str(dest) + ".part", "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, t0 = 0, time.time()
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if progress:
                frac = lo + (hi - lo) * (done / total if total else 0.5)
                progress(min(frac, 0.99), f"Downloading {label}: {done/1e6:,.0f} MB "
                                          f"({done/1e6/max(time.time()-t0, 1e-3):.1f} MB/s)")
    os.replace(str(dest) + ".part", dest)


def download_esif(progress=None):
    raw = data_dir() / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n = len(ESIF_FILES)
    for i, (key, (fname, urls)) in enumerate(ESIF_FILES.items()):
        dest, errs = raw / fname, []
        if dest.exists():
            continue
        for u in urls:                                   # first working mirror wins
            try:
                _download(u, dest, progress, i / n, (i + 1) / n, fname)
                break
            except Exception as e:                       # noqa: BLE001
                errs.append(f"{u.split('/')[2]}: {e}")
        else:
            raise RuntimeError("All mirrors failed — " + " | ".join(errs))
    if progress:
        progress(1.0, "Download complete")


# ------------------------------------------------------------------ parsing helpers
def to_datetime(s, tz=None):
    """Any timestamp column -> naive local datetime (converted to `tz` if the source is tz-aware/epoch)."""
    if pd.api.types.is_numeric_dtype(s):
        m = float(np.nanmedian(np.abs(s.to_numpy(float))))
        unit = "ns" if m > 1e17 else "us" if m > 1e14 else "ms" if m > 1e11 else "s"
        t = pd.to_datetime(s, unit=unit, utc=True)
    else:
        t = pd.to_datetime(s, errors="coerce")
        if t.dtype == object:
            t = pd.to_datetime(s, errors="coerce", utc=True)
    if isinstance(t.dtype, pd.DatetimeTZDtype):
        return t.dt.tz_convert(tz or "UTC").dt.tz_localize(None)
    return t


def _agg_long(t, key, val):
    g = pd.DataFrame({"t": t.to_numpy(), "k": key.astype(str).str.lower().to_numpy(),
                      "v": pd.to_numeric(val, errors="coerce").to_numpy()}).dropna()
    return g.groupby(["t", "k"])["v"].agg(["sum", "count"])


def _agg_wide(t, num):
    g = num.groupby(t.to_numpy())
    s, c = g.sum(), g.count()
    s.index.name = c.index.name = "t"
    ls = s.reset_index().melt(id_vars="t", var_name="k", value_name="sum")
    lc = c.reset_index().melt(id_vars="t", var_name="k", value_name="count")
    out = ls.assign(count=lc["count"]).set_index(["t", "k"])
    out.index = out.index.set_levels(out.index.levels[1].astype(str).str.lower(), level=1)
    return out


def aggregate_parquet(path, wanted, freq="1h", tz=None, progress=None):
    """Stream a (large) parquet file; return (mean per `freq` bucket, raw row count). Wide or long layout."""
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    low = {n.lower(): n for n in names}
    ts = next((low[c] for c in TS_CANDS if c in low), None)
    if ts is None:
        raise ValueError(f"No timestamp column found in {Path(path).name}. Columns: {names}")
    amap = wanted if isinstance(wanted, dict) else {w: [w] for w in wanted}
    wide, rename = [], {}
    for canon, aliases in amap.items():
        hit = next((low[x.lower()] for x in aliases if x.lower() in low), None)
        if hit:
            wide.append(hit); rename[hit.lower()] = canon
    fld = next((low[c] for c in FIELD_CANDS if c in low), None)
    val = next((low[c] for c in VAL_CANDS if c in low), None)
    use_long = (not wide) and fld and val
    if use_long:
        rename = {x.lower(): c for c, als in amap.items() for x in als}
    if not wide and not use_long:
        raise ValueError(f"None of {list(amap)} (or aliases) found in {Path(path).name}. Columns: {names}")
    cols = [ts] + ([fld, val] if use_long else wide)
    total, seen, parts = pf.metadata.num_rows, 0, []
    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=cols):
        b = batch.to_pandas()
        seen += len(b)
        t = to_datetime(b[ts], tz).dt.floor(freq)
        if use_long:
            parts.append(_agg_long(t, b[fld], b[val]))
        else:
            parts.append(_agg_wide(t, b[wide].apply(pd.to_numeric, errors="coerce")))
        if progress:
            progress(seen / max(total, 1), f"Aggregating {Path(path).name}: {seen:,}/{total:,} rows")
    allp = pd.concat(parts).groupby(level=[0, 1]).sum()
    mean = (allp["sum"] / allp["count"].where(allp["count"] > 0)).unstack()
    mean.columns = [rename.get(str(c).lower(), str(c).lower()) for c in mean.columns]
    mean.index.name = "timestamp"
    return mean.sort_index(), total


def detect_temp_unit(path):
    """Read the (constant) `unit` column of the weather file if present: returns 'F' or 'C' (default F)."""
    import re
    try:
        pf = pq.ParquetFile(path)
        low = {n.lower(): n for n in pf.schema_arrow.names}
        if "unit" not in low:
            return "F"
        u = pf.read_row_group(0, columns=[low["unit"]]).column(0).unique().to_pylist()
        t = " ".join(map(str, u)).lower()
        if re.search(r"fahren|deg\s*f|°\s*f|\bf\b", t):
            return "F"
        if re.search(r"celsius|deg\s*c|°\s*c|\bc\b", t):
            return "C"
    except Exception:
        pass
    return "F"


def aggregate_frame(df, ts_col, cols, freq="1h", tz=None):
    t = to_datetime(df[ts_col], tz).dt.floor(freq)
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    out = num.groupby(t.to_numpy()).mean()
    out.index.name = "timestamp"
    return out.sort_index()


def _cache_paths(freq, tz):
    cache = data_dir() / "cache"
    cf = cache / f"esif_{freq}_{tz.replace('/', '-')}.parquet"
    return cache, cf, cf.with_suffix(".meta")


def esif_cached(freq, tz):
    _, cf, mf = _cache_paths(freq, tz)
    return cf.exists() and mf.exists()


def load_esif_raw(freq="1h", tz="America/Denver", progress=None):
    cache, cf, meta_f = _cache_paths(freq, tz)
    cache.mkdir(parents=True, exist_ok=True)
    if cf.exists() and meta_f.exists():
        p = meta_f.read_text().split(",")
        return pd.read_parquet(cf), {"rows_power": int(p[0]), "rows_weather": int(p[1]),
                                     "temp_unit": p[2] if len(p) > 2 else "F"}
    raw = data_dir() / "raw"
    p, n_p = aggregate_parquet(raw / ESIF_FILES["power"][0], ESIF_POWER_COLS, freq, tz, progress)
    w, n_w = aggregate_parquet(raw / ESIF_FILES["weather"][0], ESIF_WEATHER_COLS, freq, tz, progress)
    unit = detect_temp_unit(raw / ESIF_FILES["weather"][0])
    df = p.join(w, how="inner")
    if df.empty:
        raise ValueError("Power and weather files have no overlapping timestamps — check the timezone setting.")
    df.to_parquet(cf)
    meta_f.write_text(f"{n_p},{n_w},{unit}")
    return df, {"rows_power": n_p, "rows_weather": n_w, "temp_unit": unit}


# ------------------------------------------------------------------ standardisation + cleaning
ESIF_PRE_SHIFT_CUTOFF = "2024-03-29"  # Facility cooling plant retrofit occurred on March 29, 2024


def wet_bulb_stull(t, rh):
    """Stull (2011) psychrometric wet-bulb temperature approximation (°C)."""
    rh_c = np.clip(rh, 1.0, 99.0)
    return (t * np.arctan(0.151977 * np.sqrt(rh_c + 8.313659))
            + np.arctan(t + rh_c)
            - np.arctan(rh_c - 1.676331)
            + 0.00391838 * (rh_c ** 1.5) * np.arctan(0.023101 * rh_c)
            - 4.686035)


def add_temporal_physics_features(df):
    """Add thermodynamic, psychrometric, lag, and diurnal/seasonal features.
    
    Transforms raw static readings (IT load, ambient temp, humidity)
    into a rich dynamic thermal state representing heat rejection thermodynamics,
    cooling loop transport delays (1h-3h lags), building thermal mass (6h rolling),
    and solar diurnal/seasonal cycles.
    """
    if df is None or len(df) < 10:
        return df

    out = df.copy()
    if "timestamp" in out.columns and pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        out = out.sort_values("timestamp").reset_index(drop=True)
        ts = out["timestamp"].dt
        # Diurnal solar radiation cycle (24-hour periodicity)
        out["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24.0)
        # Seasonal annual cycle (365.25-day periodicity)
        out["doy_sin"] = np.sin(2 * np.pi * ts.dayofyear / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * ts.dayofyear / 365.25)

    # Psychrometric wet-bulb & approach temperature
    if "ambient_c" in out.columns and "humidity_pct" in out.columns:
        out["wetbulb_c"] = wet_bulb_stull(out["ambient_c"].to_numpy(), out["humidity_pct"].to_numpy())
        out["approach"] = out["ambient_c"] - out["wetbulb_c"]
    elif "ambient_c" in out.columns:
        out["wetbulb_c"] = out["ambient_c"].copy()
        out["approach"] = 0.0

    # Thermal inertia lags (transport delay in chilled/warm water loop)
    if "ambient_c" in out.columns:
        out["ambient_lag1"] = out["ambient_c"].shift(1)
        out["ambient_lag2"] = out["ambient_c"].shift(2)
        out["ambient_lag3"] = out["ambient_c"].shift(3)
        out["ambient_roll6"] = out["ambient_c"].rolling(6, min_periods=1).mean()

    if "it_kw" in out.columns:
        out["it_lag1"] = out["it_kw"].shift(1)
        out["it_lag2"] = out["it_kw"].shift(2)
        out["it_roll6"] = out["it_kw"].rolling(6, min_periods=1).mean()

    if "cooling_kw" in out.columns:
        out["cooling_lag1"] = out["cooling_kw"].shift(1)
        out["cooling_lag2"] = out["cooling_kw"].shift(2)
        out["cooling_roll6"] = out["cooling_kw"].rolling(6, min_periods=1).mean()

    # Thermodynamic interaction proxies (heat rejection driving force)
    if "it_kw" in out.columns and "wetbulb_c" in out.columns:
        out["it_x_wb"] = out["it_kw"] * out["wetbulb_c"]
    if "it_kw" in out.columns and "ambient_c" in out.columns:
        out["it_x_amb"] = out["it_kw"] * out["ambient_c"]

    # Drop the initial rows with NaNs from shifting
    return out.dropna().reset_index(drop=True)


def build_dataset(raw, m, end_date=None, era=None, add_physics=False):
    """raw: time-indexed frame. m: mapping {it, cooling:[...], temp, temp_unit, humidity?, setpoint?}."""
    d = pd.DataFrame(index=raw.index)
    d["it_kw"] = raw[m["it"]]
    d["cooling_kw"] = raw[m["cooling"]].sum(axis=1, min_count=len(m["cooling"]))
    t = raw[m["temp"]]
    d["ambient_c"] = (t - 32) * 5 / 9 if m.get("temp_unit") == "F" else t
    if m.get("humidity"):
        d["humidity_pct"] = raw[m["humidity"]].clip(0, 100)
    if m.get("setpoint"):
        d["setpoint_c"] = raw[m["setpoint"]]
    rep = {"buckets": len(d)}
    n0 = len(d); d = d.dropna(); rep["dropped_missing"] = n0 - len(d)
    n0 = len(d); d = d[(d.it_kw > 0) & (d.cooling_kw >= 0) & d.ambient_c.between(-60, 60)]
    rep["dropped_invalid"] = n0 - len(d)
    n0 = len(d)
    for c in ("it_kw", "cooling_kw"):
        if len(d) > 10:
            q1, q3 = d[c].quantile([.25, .75]); iqr = q3 - q1
            if iqr > 0:
                d = d[d[c].between(q1 - 5 * iqr, q3 + 5 * iqr)]
    rep["dropped_outliers"] = n0 - len(d)
    d["pue"] = 1 + d.cooling_kw / d.it_kw
    d = d.reset_index()
    if era == "kestrel":
        d = d[d.timestamp >= pd.to_datetime("2022-09-01")].copy()
        if end_date:
            d = d[d.timestamp < pd.to_datetime(end_date)].copy()
    elif era == "peregrine":
        d = d[(d.timestamp >= pd.to_datetime("2016-06-01")) & (d.timestamp < pd.to_datetime("2022-06-01"))].copy()
        if end_date:
            d = d[d.timestamp < pd.to_datetime(end_date)].copy()
    elif end_date:
        d = d[d.timestamp < pd.to_datetime(end_date)].copy()

    if add_physics and "timestamp" in d.columns and len(d) > 20:
        d = add_temporal_physics_features(d)

    rep["rows"] = len(d)
    if len(d) > 2:
        step = d.timestamp.diff().median()
        rep["step_hours"] = float(step / pd.Timedelta(hours=1))
        rep["start"], rep["end"] = d.timestamp.min(), d.timestamp.max()
    return d, rep


def select_features(df):
    f = ["it_kw", "ambient_c"]
    if "humidity_pct" in df and df.humidity_pct.std() > 1:
        f.append("humidity_pct")
    if "setpoint_c" in df and df.setpoint_c.std() > 0.3:
        f.append("setpoint_c")
    # BMS telemetry channels (if present):
    for b in ["chw_flow_kg_s", "chw_return_c", "head_pressure_m", "pump_vfd_rpm", "pumps_active", "chillers_active"]:
        if b in df and df[b].std() > 0.01:
            f.append(b)
    # Temporal & physical inertia channels (if present):
    for t in ["wetbulb_c", "approach", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
              "ambient_lag1", "ambient_lag2", "ambient_lag3", "it_lag1", "it_lag2",
              "ambient_roll6", "it_roll6", "it_x_wb", "it_x_amb",
              "cooling_lag1", "cooling_lag2", "cooling_roll6"]:
        if t in df and df[t].std() > 0.001:
            f.append(t)
    return f


BMS_FILE = "alpha_dc_real_bms.csv"


def bms_present():
    return (data_dir() / "raw" / BMS_FILE).exists()


def load_bms_raw(freq="1h", tz="UTC", progress=None):
    """Load real data center SCADA/BMS operational telemetry (DeepMind / LBNL paradigm).

    100,330 operational rows covering an annual cycle.
    Standardized into it_kw, cooling_kw, ambient_c, setpoint_c, chw_flow_kg_s, chw_return_c,
    head_pressure_m, pump_vfd_rpm, pumps_active, chillers_active.
    """
    raw_path = data_dir() / "raw" / BMS_FILE
    if not raw_path.exists():
        raise FileNotFoundError(f"{BMS_FILE} not found in {data_dir() / 'raw'}")

    cache = data_dir() / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cf = cache / f"bms_{freq}.parquet"
    if cf.exists():
        d = pd.read_parquet(cf)
        step_h = float(d.timestamp.diff().median() / pd.Timedelta(hours=1)) if len(d) > 1 else 1.0
        rep = {"buckets": len(d), "rows": len(d), "dropped_missing": 0, "dropped_invalid": 0,
               "dropped_outliers": 0, "start": d.timestamp.min(), "end": d.timestamp.max(),
               "step_hours": step_h}
        meta = {"rows_power": 100330, "temp_unit": "C"}
        return d, rep, meta

    if progress:
        progress(0.1, f"Loading {BMS_FILE} (100,330 rows)...")
    raw = pd.read_csv(raw_path)

    ts = pd.to_datetime(raw["Time"])
    cooling_kw = raw["P_CDWPs_sum"] + raw["P_CHWPs_sum"]
    delta_T = raw["Tchw_r"] - raw["Tchws_set_CHI"]
    it_kw = raw["Mchw"] * 4.184 * np.maximum(delta_T, 0)

    processed = pd.DataFrame({
        "timestamp": ts,
        "cooling_kw": cooling_kw,
        "it_kw": it_kw,
        "ambient_c": raw["Twb_outside"] - 273.15,
        "setpoint_c": raw["Tchws_set_CHI"] - 273.15,
        "chw_flow_kg_s": raw["Mchw"],
        "chw_return_c": raw["Tchw_r"] - 273.15,
        "head_pressure_m": raw["Head_required"],
        "pump_vfd_rpm": raw["CWP_speedInput"],
        "pumps_active": raw["CWP_activatedNumber"],
    })
    chi_cols = [c for c in ["CHI01", "CHI02", "CHI03", "CHI04", "CHI05", "CHI06"] if c in raw.columns]
    if chi_cols:
        processed["chillers_active"] = raw[chi_cols].sum(axis=1)

    if progress:
        progress(0.5, f"Aggregating to {freq} resolution...")

    if freq not in ("5min", "5m"):
        agg = processed.set_index("timestamp").resample(freq).mean().dropna().reset_index()
    else:
        agg = processed.dropna().reset_index(drop=True)

    agg["pue"] = 1 + agg["cooling_kw"] / np.maximum(agg["it_kw"], 10.0)
    agg.to_parquet(cf)

    step_h = float(agg.timestamp.diff().median() / pd.Timedelta(hours=1)) if len(agg) > 1 else 1.0
    rep = {"buckets": len(agg), "rows": len(agg), "dropped_missing": 0, "dropped_invalid": 0,
           "dropped_outliers": 0, "start": agg.timestamp.min(), "end": agg.timestamp.max(),
           "step_hours": step_h}
    meta = {"rows_power": len(raw), "temp_unit": "C"}
    return agg, rep, meta


def load_synthetic(seed=7, mean_temp=18.0, noise=0.03, gap=0.5, n_servers=200):
    from simulate import generate_dataset
    d = generate_dataset(seed=seed, n_servers=n_servers, mean_temp=mean_temp, noise=noise, gap=gap)
    out = d[["timestamp", "it_kw", "ambient_c", "humidity_pct", "setpoint_c", "cooling_kw"]].copy()
    out["pue"] = 1 + out.cooling_kw / out.it_kw
    rep = {"buckets": len(out), "rows": len(out), "dropped_missing": 0, "dropped_invalid": 0,
           "dropped_outliers": 0, "step_hours": 1.0, "start": out.timestamp.min(), "end": out.timestamp.max()}
    return out, rep


def validate_telemetry(df, features):
    """Validate sensor telemetry for physical sanity, frozen sensors, and dropout rates."""
    issues = []
    for col in features:
        if col not in df:
            issues.append(f"Missing required channel: '{col}'")
            continue
        s = df[col].dropna()
        if len(s) == 0:
            issues.append(f"Channel '{col}' has 100% missing values")
        elif s.std() == 0:
            issues.append(f"Channel '{col}' is frozen (zero variance: {s.iloc[0]})")
    if "ambient_c" in df and (df["ambient_c"].min() < -60 or df["ambient_c"].max() > 60):
        issues.append(f"Ambient temperature out of physical range: [{df['ambient_c'].min():.1f}, {df['ambient_c'].max():.1f}] °C")
    if "it_kw" in df and (df["it_kw"].min() <= 0):
        issues.append("Non-positive IT load detected (data center powered off or sensor fault)")
    return issues

