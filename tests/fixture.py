"""TEST FIXTURE ONLY — writes small parquet files that mimic the *schema* of the NREL ESIF dataset
(wide layout: ts, it_power_kw, cooling_kw, hvac_kw, pump_kw, ...; weather in °F). The numbers are fabricated for
testing the pipeline offline. The real app downloads the real measurements instead."""
import numpy as np
import pandas as pd
from wetbulb_stub import wb


def make_esif_like(dirpath, days=900, step_min=5, seed=0, outdoor_names=False):
    rng = np.random.default_rng(seed)
    raw = dirpath / "raw"; raw.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range("2021-01-01", periods=days * 24 * 60 // step_min, freq=f"{step_min}min", tz="UTC")
    h = ts.hour.to_numpy() + ts.minute.to_numpy() / 60; doy = ts.dayofyear.to_numpy()
    it = 700 + 120 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 25, len(ts))
    t_c = 9 + 14 * np.sin(2 * np.pi * (doy - 110) / 365) + 6 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 1.2, len(ts))
    rh = np.clip(55 - 1.0 * (t_c - 9) + rng.normal(0, 8, len(ts)), 8, 99)
    w = wb(t_c, rh)
    cooling = 0.02 * it + 0.35 * np.maximum(w - 2, 0) ** 1.4 * (it / 700) + rng.normal(0, 0.6, len(ts))
    pd.DataFrame({"ts": ts, "it_power_kw": it, "cooling_kw": cooling, "hvac_kw": 14 + 0.004 * it + rng.normal(0, .4, len(ts)),
                  "pump_kw": 9 + 0.006 * it + rng.normal(0, .3, len(ts)), "plug_and_light_kw": 20.0,
                  "pue": 1.05, "day": ts.day}).to_parquet(raw / "esif_power.parquet", row_group_size=50_000)
    sel = np.arange(0, len(ts), 2)   # weather sampled at half the rate
    t_name, h_name = ("outdoor_air_temp", "outdoor_air_humidity") if outdoor_names else ("outside_air_temp", "outside_air_humidity")
    wdf = pd.DataFrame({"ts": ts[sel], t_name: t_c[sel] * 9 / 5 + 32, h_name: rh[sel], "day": ts[sel].day})
    if outdoor_names:                      # the real file also carries `tags` and `unit` columns
        wdf["tags"] = "{}"; wdf["unit"] = "degF"
    wdf.to_parquet(raw / "esif_weather.parquet", row_group_size=50_000)
