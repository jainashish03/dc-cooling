"""Generate a realistic, high-fidelity data center test dataset.

Models a 1.2 MW enterprise data center across a complete 8,760-hour annual cycle.
Features thermodynamic chiller COP curves, wet-bulb heat rejection, economizer free-cooling,
diurnal server workload swings, and dynamic cooling setpoints.
"""
import math
import random
import datetime
import csv
from pathlib import Path

def stull_wet_bulb(t, rh):
    """Wet bulb temperature (°C) via Stull (2011) approximation."""
    tw = (t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
          + math.atan(t + rh)
          - math.atan(rh - 1.676331)
          + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
          - 4.686035)
    return tw

def generate():
    rng = random.Random(42)
    start_dt = datetime.datetime(2025, 1, 1, 0, 0)
    hours = 8760
    
    # 2,000 servers @ 150W idle, 450W peak = 300 kW idle, 900 kW peak base IT
    # + 300 kW storage / network = 600 kW to 1200 kW range
    
    rows = []
    
    for h in range(hours):
        current_dt = start_dt + datetime.timedelta(hours=h)
        doy = current_dt.timetuple().tm_yday
        hod = current_dt.hour
        weekday = current_dt.weekday() # 0 = Monday, 6 = Sunday
        
        # ── Outdoor Weather Model ──────────────────────────────────────────
        # Annual seasonal wave (min in Jan/Feb, max in July/August)
        seasonal_temp = 14.0 - 12.0 * math.cos(2 * math.pi * (doy - 15) / 365.0)
        # Daily diurnal wave (peak around 15:00, trough around 05:00)
        diurnal_temp = 5.5 * math.sin(2 * math.pi * (hod - 9) / 24.0)
        # Weather synoptic wave (3-5 day weather front variability)
        synoptic = 3.0 * math.sin(2 * math.pi * h / 96.0) + rng.gauss(0, 0.8)
        ambient_c = round(seasonal_temp + diurnal_temp + synoptic, 2)
        
        # Relative humidity (%): anti-correlated with temperature + rain fronts
        base_rh = 68.0 - 1.2 * (ambient_c - 14.0) + 10.0 * math.sin(2 * math.pi * h / 120.0) + rng.gauss(0, 3.5)
        humidity_pct = round(max(15.0, min(98.0, base_rh)), 1)
        
        wet_bulb_c = round(stull_wet_bulb(ambient_c, humidity_pct), 2)
        
        # ── IT Server Workload Model ────────────────────────────────────────
        # Diurnal user traffic curve (peak 14:00, trough 04:00)
        traffic_diurnal = 0.22 * math.sin(2 * math.pi * (hod - 8) / 24.0)
        # Weekday vs Weekend effect
        weekend_drop = -0.08 if weekday >= 5 else 0.04
        # Nightly batch analytics jobs (spikes between 01:00 and 04:00)
        batch_job = 0.12 if 1 <= hod <= 4 else 0.0
        # Random bursty user demand
        burst = rng.gauss(0, 0.035)
        
        utilization = max(0.18, min(0.96, 0.52 + traffic_diurnal + weekend_drop + batch_job + burst))
        
        # IT power draw: 550 kW baseline + 650 kW proportional to workload
        it_kw = round(550.0 + 650.0 * utilization + rng.gauss(0, 5.0), 1)
        
        # ── Cooling Plant & Setpoint Physics ────────────────────────────────
        # Supply air / chilled water setpoint (20.0°C to 24.5°C with scheduled resets)
        # Higher setpoints in winter to maximize free cooling; adjusted periodically
        day_idx = h // 24
        setpoint_c = round(21.0 + 1.5 * ((day_idx % 14) / 14.0) + (1.5 if ambient_c < 8.0 else 0.0), 1)
        
        # Thermodynamic lift (difference between heat sink ambient and indoor setpoint)
        lift = max(0.0, ambient_c - setpoint_c)
        # COP (Coefficient of Performance) curve
        # Higher setpoint -> lower lift -> higher efficiency COP
        base_cop = 5.8 - 0.09 * lift + 0.08 * (setpoint_c - 20.0)
        cop = max(2.1, min(7.5, base_cop + rng.gauss(0, 0.05)))
        
        # Economizer / Free cooling potential
        # When wet bulb is sufficiently below setpoint, water-side economizer activates
        if wet_bulb_c < (setpoint_c - 4.0):
            economizer_ratio = min(1.0, (setpoint_c - 4.0 - wet_bulb_c) / 6.0)
            economizer_mode = "Full" if economizer_ratio >= 0.9 else "Partial"
        else:
            economizer_ratio = 0.0
            economizer_mode = "Off"
            
        # Chiller mechanical compression power
        chiller_load = it_kw * (1.0 - 0.85 * economizer_ratio)
        chiller_kw = round(max(15.0, chiller_load / cop), 1)
        
        # Heat rejection fans (cooling towers) and condenser pumps
        cooling_tower_kw = round(18.0 + 0.035 * it_kw * max(0.5, (wet_bulb_c + 5) / 25.0), 1)
        
        # Chilled water loop circulation pumps
        chw_pumps_kw = round(14.0 + 0.015 * it_kw, 1)
        
        # CRAH (Computer Room Air Handler) fans inside the whitespace (cube law with air volume)
        crah_fans_kw = round(25.0 + 35.0 * (utilization ** 2.5), 1)
        
        # Total cooling power (sum of chiller + cooling tower + pumps + CRAH fans)
        cooling_kw = round(chiller_kw + cooling_tower_kw + chw_pumps_kw + crah_fans_kw, 1)
        
        # PUE calculation: Total Facility Power / IT Power
        pue = round(1.0 + (cooling_kw + 20.0) / it_kw, 3) # 20kW lights/aux
        
        rows.append({
            "timestamp": current_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "it_kw": it_kw,
            "cooling_kw": cooling_kw,
            "ambient_c": ambient_c,
            "humidity_pct": humidity_pct,
            "setpoint_c": setpoint_c,
            "wet_bulb_c": wet_bulb_c,
            "chiller_kw": chiller_kw,
            "cooling_tower_kw": cooling_tower_kw,
            "pumps_kw": chw_pumps_kw,
            "crah_fans_kw": crah_fans_kw,
            "pue": pue,
            "economizer_mode": economizer_mode,
        })
        
    out_path = Path("test_datacenter_new.csv")
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Generated {len(rows):,} rows in {out_path.resolve()}")
    return out_path

if __name__ == "__main__":
    generate()

