"""Synthetic data-center generator (ground truth).

Cooling power = chiller + facility fans + pumps + server-fan penalty + hotspot penalty.
Inputs: IT utilization, ambient temp, supply-air setpoint, relative humidity.
`gap` ("reality gap") switches on an effect that simple physics models do not know about
(hotspot recirculation at high load), so ML/hybrid models have something to learn.
"""
import numpy as np
import pandas as pd

P_IDLE_W, P_MAX_W = 120.0, 350.0   # per-server power at 0% / 100% utilization


def it_power_kw(u, n_servers=200):
    """Linear server power model: P = N (P_idle + (P_max - P_idle) u)."""
    return n_servers * (P_IDLE_W + (P_MAX_W - P_IDLE_W) * np.asarray(u, dtype=float)) / 1000.0


def cooling_components(u, t_amb, t_set, n_servers=200, rh=50.0, gap=0.0):
    u, t_amb, t_set, rh = (np.asarray(a, dtype=float) for a in (u, t_amb, t_set, rh))
    it = it_power_kw(u, n_servers)
    it_max = it_power_kw(1.0, n_servers)

    lift = np.maximum(t_amb - t_set, 0.0)
    cop = (np.maximum(6.5 - 0.11 * lift, 2.0) * (1 + 0.025 * (t_set - 18))
           * (1 - 0.0015 * np.maximum(rh - 50, 0)))                       # humid air = latent load
    econ = np.clip((t_set - 6 - t_amb) / 8, 0, 1) * (1 - 0.3 * np.maximum(rh - 60, 0) / 40)
    chiller = it / cop * (1 - 0.85 * econ)
    fans = 0.06 * it_max * (it / it_max) ** 3                               # cube law
    pumps = 1.5 * np.ones_like(chiller)
    server_fans = 0.012 * it * np.maximum(t_set - 24, 0) ** 1.5             # warm inlet -> fans ramp
    hotspot = gap * 0.10 * it * np.clip((u - 0.6) / 0.38, 0, 1) ** 2        # unmodelled effect
    return {"chiller": chiller, "fans": fans, "pumps": pumps,
            "server_fans": server_fans, "hotspot": hotspot}


def cooling_power_kw(u, t_amb, t_set, n_servers=200, rh=50.0, gap=0.0):
    return sum(cooling_components(u, t_amb, t_set, n_servers, rh, gap).values())


def generate_dataset(seed=7, hours=8760, n_servers=200, mean_temp=18.0, noise=0.03, gap=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    hod, day = t % 24, t // 24
    seasonal = -np.cos(2 * np.pi * (t / 24) / 365)
    ambient = (mean_temp + 9 * seasonal + 5 * np.sin(2 * np.pi * (hod - 9) / 24)
               + rng.normal(0, 1.5, hours))
    util = (0.45 + 0.20 * np.sin(2 * np.pi * (hod - 8) / 24)
            - 0.08 * ((day % 7) >= 5) + rng.normal(0, 0.07, hours))
    util = np.clip(util, 0.10, 0.98)
    sp_day = np.round(rng.uniform(18, 27, day.max() + 1) * 2) / 2
    setpoint = sp_day[day]
    rh = np.clip(65 - 0.8 * (ambient - mean_temp) + rng.normal(0, 8, hours), 20, 95)

    truth = cooling_power_kw(util, ambient, setpoint, n_servers, rh, gap)
    observed = truth * (1 + rng.normal(0, noise, hours))
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=hours, freq="h"),
        "utilization": util, "ambient_c": ambient, "setpoint_c": setpoint, "humidity_pct": rh,
        "it_kw": it_power_kw(util, n_servers),
        "cooling_kw": observed, "cooling_true_kw": truth,
    })


def climate_scenarios(temps, n_servers=200, gap=0.0, seed=7, fixed_setpoint=22.0):
    """Annual cooling energy & PUE per climate: fixed setpoint vs best-possible setpoint (oracle)."""
    grid = np.arange(18, 27.01, 0.5)
    rows = []
    for mt in temps:
        d = generate_dataset(seed, n_servers=n_servers, mean_temp=mt, noise=0, gap=gap)
        u, ta, rh = d.utilization.values, d.ambient_c.values, d.humidity_pct.values
        fixed = cooling_power_kw(u, ta, fixed_setpoint, n_servers, rh, gap)
        best = np.min([cooling_power_kw(u, ta, g, n_servers, rh, gap) for g in grid], axis=0)
        it = d.it_kw.values
        rows.append({"mean_temp_c": mt, "fixed_MWh": fixed.sum() / 1000, "optimal_MWh": best.sum() / 1000,
                     "PUE_fixed": 1 + fixed.sum() / it.sum(), "PUE_optimal": 1 + best.sum() / it.sum()})
    return pd.DataFrame(rows)
