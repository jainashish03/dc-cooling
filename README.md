# ❄️ Data Center Cooling Energy Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](http://localhost:8501)
[![Tests Passing](https://img.shields.io/badge/tests-10%2F10%20passing-brightgreen.svg)]()

> **Real DC Telemetry → Thermodynamic Feature Engineering → Deep Learning & Physics-Informed Neural Networks (PINN / LSTM) → Sensitivity & What-If Planning → Anomaly Detection** in a unified, interactive Streamlit application.

---

## 🚀 Key Capabilities

- **Real-World High-Performance Data Center Telemetry**:
  1. **🏭 Real DC Full BMS Telemetry** (`data/raw/alpha_dc_real_bms.csv`): 100,330 operational rows of continuous chilled-water SCADA telemetry (DeepMind / LBNL paradigm) covering IT compute power, cooling plant load, wet-bulb temperature, chilled water flow, return temperature, pump head, pump VFD RPM, active pumps, and active chillers.
  2. **🏛️ NREL ESIF HPC Telemetry**: Multi-year facility archive from the National Renewable Energy Laboratory HPC data center. Dedicated historical era isolation for the **⭐ Modern Supercomputer Era (Kestrel: Sep 2022 – Mar 2024)**, legacy Peregrine era (2016–2022), and pre-shift baselines.
- **Thermodynamic & Physics-Informed Feature Engineering**:
  - **Stull (2011) Psychrometric Wet-Bulb ($T_{\text{wb}}$)**: Thermodynamic limit of evaporative cooling tower heat rejection.
  - **Approach Temperature**: $T_{\text{approach}} = T_{\text{ambient}} - T_{\text{wb}}$.
  - **Diurnal & Seasonal Solar Harmonics**: $\sin(2\pi \cdot \text{hour}/24)$, $\cos(2\pi \cdot \text{hour}/24)$, $\sin(2\pi \cdot \text{doy}/365.25)$, $\cos(2\pi \cdot \text{doy}/365.25)$.
  - **Hydronic Thermal Inertia Lags**: 1h, 2h, and 3h lags of ambient temperature and IT load capturing transport delay in chilled/warm water loops.
  - **Thermal Mass Rolling Averages**: 6-hour rolling means representing heat accumulation in building envelope and cooling loop.
  - **Autoregressive Plant Operational Feedback**: `cooling_lag1`, `cooling_lag2`, and `cooling_roll6` tracking continuous plant equilibrium, enabling high accuracy forward forecasting.
  - **Thermodynamic Cross-Interactions**: $IT \times T_{\text{wb}}$ and $IT \times T_{\text{ambient}}$.
- **8 Benchmark Models + Anomaly Detection**:
  1. **Constant-PUE Baseline**: Standard facility rule of thumb ($Q_{\text{cool}} = k \cdot IT$).
  2. **Polynomial OLS (deg 2)**: Second-order polynomial regression with physical features.
  3. **Ridge + Physics Features**: L2-regularized linear model over cross-interaction features.
  4. **Random Forest**: Non-linear ensemble with leaf regularization and multi-threaded tree inference.
  5. **Gradient Boosting (HistGBDT)**: Fast histogram-based gradient-boosted decision trees with monotonic physical constraints.
  6. **Stacked Ensemble**: Non-negative constrained meta-blender (`Ridge(alpha=10.0, positive=True)`) combining Ridge, HistGBDT, and Random Forest.
  7. **⭐ LSTM Neural Network**: PyTorch recurrent architecture processing sequential time-series windows to capture thermal momentum.
  8. **⚡ Physics-Informed NN (PINN)**: PyTorch architecture trained with composite loss enforcing the Carnot thermodynamic lower bound ($\hat{y} \ge 0.02 \times IT$) and autograd monotonicity ($\frac{\partial \hat{y}}{\partial IT} \ge 0$).
  9. **🚨 LSTM Autoencoder**: Unsupervised multi-channel temporal reconstruction model for zero-shot equipment fault and anomaly detection.

---

## 📊 Benchmark Leaderboards

### 1. NREL ESIF HPC Telemetry (Modern Kestrel Era — Chronological 80/20 Forward Split)
*Evaluates forward-in-time into unseen winter operational months ($N = 2,544$ hold-out hours):*

| Model | Hold-out $R^2$ | Hold-out RMSE | Hold-out MAE | MAPE % | WMAPE % | CV $R^2$ | CV RMSE | Skill vs Baseline | Fit (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Physics-Informed NN (PINN)** | **$+0.873$** | **$3.31\text{ kW}$** | $1.28\text{ kW}$ | **$2.11\%$** | $2.00\%$ | **$0.959$** | $3.91\text{ kW}$ | **$+93.5\%$** | $2.01\text{ s}$ |
| **LSTM Neural Network** | **$+0.869$** | **$3.37\text{ kW}$** | $1.20\text{ kW}$ | **$1.89\%$** | $1.88\%$ | **$0.959$** | $3.89\text{ kW}$ | **$+93.4\%$** | $1.64\text{ s}$ |
| **Random Forest** | **$+0.814$** | **$4.01\text{ kW}$** | $1.32\text{ kW}$ | **$2.01\%$** | $2.06\%$ | **$0.959$** | $3.90\text{ kW}$ | **$+92.1\%$** | $0.65\text{ s}$ |
| **Gradient Boosting** | **$+0.814$** | **$4.01\text{ kW}$** | $1.26\text{ kW}$ | **$1.86\%$** | $1.97\%$ | **$0.961$** | $3.80\text{ kW}$ | **$+92.1\%$** | $0.32\text{ s}$ |
| **Stacked Ensemble** | **$+0.815$** | **$4.00\text{ kW}$** | $1.34\text{ kW}$ | **$2.04\%$** | $2.10\%$ | **$0.961$** | $3.80\text{ kW}$ | **$+92.1\%$** | $6.24\text{ s}$ |
| **Ridge + physics features** | **$+0.779$** | **$4.37\text{ kW}$** | $1.59\text{ kW}$ | **$2.63\%$** | $2.49\%$ | **$0.956$** | $4.06\text{ kW}$ | **$+91.4\%$** | $0.11\text{ s}$ |
| **Polynomial OLS (deg 2)** | **$+0.767$** | **$4.49\text{ kW}$** | $1.77\text{ kW}$ | **$2.92\%$** | $2.76\%$ | **$0.944$** | $4.54\text{ kW}$ | **$+91.1\%$** | $0.07\text{ s}$ |
| *Constant-PUE baseline* | *$-28.729$* | *$50.69\text{ kW}$* | *$46.95\text{ kW}$* | *$76.13\%$* | *$73.46\%$* | *$-1.457$* | *$30.21\text{ kW}$* | *$0.0\%$* | *$0.00\text{ s}$* |

---

### 2. NREL ESIF HPC Telemetry (Modern Kestrel Era — Representative 80/20 Multi-Season Split)

| Model | Hold-out $R^2$ | Hold-out RMSE | Hold-out MAE | MAPE % | WMAPE % | CV $R^2$ | Skill vs Baseline | Fit (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stacked Ensemble** | **$0.961$** | **$3.71\text{ kW}$** | **$1.87\text{ kW}$** | **$3.64\%$** | $2.99\%$ | **$0.919$** | **$+87.7\%$** | $3.65\text{ s}$ |
| **Gradient Boosting** | **$0.959$** | **$3.84\text{ kW}$** | **$2.06\text{ kW}$** | **$4.03\%$** | $3.31\%$ | **$0.919$** | **$+87.3\%$** | $0.68\text{ s}$ |
| **Random Forest** | **$0.957$** | **$3.91\text{ kW}$** | **$1.87\text{ kW}$** | **$3.59\%$** | $2.99\%$ | **$0.910$** | **$+87.1\%$** | $0.57\text{ s}$ |
| **LSTM Neural Network** | **$0.834$** | **$7.70\text{ kW}$** | **$4.67\text{ kW}$** | **$10.51\%$** | $7.49\%$ | **$0.843$** | **$+74.5\%$** | $1.15\text{ s}$ |
| **Ridge + physics features** | **$0.823$** | **$7.94\text{ kW}$** | **$5.09\text{ kW}$** | **$11.03\%$** | $8.16\%$ | **$0.809$** | **$+73.8\%$** | $0.08\text{ s}$ |
| **Polynomial OLS (deg 2)** | **$0.822$** | **$7.97\text{ kW}$** | **$5.11\text{ kW}$** | **$11.09\%$** | $8.20\%$ | **$0.797$** | **$+73.7\%$** | $0.06\text{ s}$ |
| **Physics-Informed NN (PINN)** | **$0.822$** | **$7.97\text{ kW}$** | **$4.81\text{ kW}$** | **$10.84\%$** | $7.72\%$ | **$0.841$** | **$+73.6\%$** | $1.97\text{ s}$ |
| *Constant-PUE baseline* | *$-1.565$* | *$30.26\text{ kW}$* | *$26.00\text{ kW}$* | *$42.18\%$* | *$41.69\%$* | *$-1.457$* | *$0.0\%$* | *$0.00\text{ s}$* |

---

### 3. Real DC Full BMS Telemetry (100,330 Continuous SCADA Rows)

| Model | Hold-out $R^2$ | Hold-out RMSE | Hold-out MAE | MAPE % | CV $R^2$ | Skill vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **$0.998$** | **$1.54\text{ kW}$** | $0.86\text{ kW}$ | **$0.90\%$** | **$0.994$** | **$+95.4\%$** |
| **Stacked Ensemble** | **$0.998$** | **$1.73\text{ kW}$** | $1.02\text{ kW}$ | **$1.04\%$** | **$0.994$** | **$+94.8\%$** |
| **Random Forest** | **$0.998$** | **$1.76\text{ kW}$** | $0.79\text{ kW}$ | **$0.83\%$** | **$0.993$** | **$+94.7\%$** |
| **Ridge + physics features** | **$0.995$** | **$2.44\text{ kW}$** | $1.48\text{ kW}$ | **$1.49\%$** | **$0.991$** | **$+92.7\%$** |
| **LSTM Neural Network** | **$0.992$** | **$3.13\text{ kW}$** | $2.26\text{ kW}$ | **$2.32\%$** | **$0.992$** | **$+90.6\%$** |
| **Physics-Informed NN (PINN)** | **$0.983$** | **$4.61\text{ kW}$** | $3.28\text{ kW}$ | **$3.47\%$** | **$0.989$** | **$+86.2\%$** |

---

### 4. Actual vs. Predicted Cooling Power Distribution (Mean & Std on Test Sets)

When evaluating models on the unseen 20% chronological test set (future operational time horizon):

#### A. Commercial BMS Telemetry (Target Test Set: Mean = 108.07 kW, Std = 31.97 kW, N = 20,066)

| Model Architecture | Predicted Mean (kW) | Predicted Std (kW) | Bias Error (kW) | $R^2$ Score | RMSE (kW) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge + Physics** | **108.28** | **31.01** | **+0.21** | **0.9348** | **8.16** |
| **Gradient Boosting** | **111.21** | **30.79** | **+3.14** | **0.8435** | **12.65** |
| **Random Forest** | **114.89** | **27.78** | **+6.83** | **0.6853** | **17.94** |

#### B. Modern Data Center (Target Test Set: Mean = 121.80 kW, Std = 21.51 kW, N = 1,752)

| Model Architecture | Predicted Mean (kW) | Predicted Std (kW) | Bias Error (kW) | $R^2$ Score | RMSE (kW) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **121.88** | **20.48** | **+0.08** | **0.9323** | **5.60** |
| **Gradient Boosting** | **121.74** | **20.19** | **-0.06** | **0.9206** | **6.06** |
| **Ridge + Physics** | **123.85** | **25.13** | **+2.05** | **0.7215** | **11.35** |

> **💡 Top Recommendation — Facility-Specific Chronological Models ($R^2 = 0.92 - 0.93$, RMSE $\le 6 - 8	ext{ kW}$):**
> 
> Data centers have unique mechanical chillers, pumps, and pipe loops.
> Training on the facility's own telemetry with **autoregressive lags + physics features** gives **$R^2 > 0.93$** with virtually zero bias ($\pm 0.08	ext{ kW}$).

---

## ⚡ Quickstart

### 1. Installation
```bash
git clone https://github.com/jainashish03/dc-cooling.git
cd dc-cooling
pip install -r requirements.txt
pip install torch  # Optional: enables PyTorch LSTM, PINN, and Anomaly Detection
```

### 2. Fast Evaluation CLI
```bash
# Evaluate on Modern Kestrel Era (Chronological 80/20 split)
python3 eval_real.py --source esif --split chronological

# Evaluate on Modern Kestrel Era (Representative 80/20 split)
python3 eval_real.py --source esif --split representative

# Evaluate on Full BMS SCADA Telemetry
python3 eval_real.py --source bms

# Evaluate on Legacy Peregrine Era
python3 eval_real.py --source esif --esif-era peregrine
```

### 3. Launch Interactive Dashboard
```bash
streamlit run app.py --server.port 8501
```
Open **`http://localhost:8501`** in your browser to view:
- **🎯 Prediction & PUE Gauge**: Live cooling power inference, 90% confidence intervals, and daily operating cost/CO₂ footprint.
- **📊 Data Quality Report**: Sensor health verification, zero frozen channels check, physical range bounds, and statistical distributions.
- **🏆 Model Leaderboard**: Comparison across 9 metrics ($R^2$, RMSE, MAE, MAPE, WMAPE, CV $R^2$, CV RMSE, Skill vs Baseline, Fit Time), residual scatter plots, 7/14/30-day continuous tracking window, permutation feature importance, and `.joblib` model export.
- **🔬 Sensitivity**: Local gradients ($\partial \hat{y}/\partial x_i$), elasticities, tornado swing plots, and global Sobol $S_1 / S_T$ variance decomposition.
- **🧭 Scenarios**: Interactive what-if planning with IT scaling multipliers, ambient heatwaves, and setpoint optimizations.
- **🚨 Anomaly Detection**: Real-time multi-sensor reconstruction tracking with the PyTorch LSTM Autoencoder, error thresholding, and channel-level root-cause breakdown.

---

## 🧪 Testing

Execute the automated test suite covering all data transformations, models, sensitivity algorithms, PyTorch neural networks, and physics feature additions:
```bash
python3 -m pytest tests/
# 10 passed in ~22s
```

---

## 📄 License
This project is licensed under the Apache 2.0 / CC-BY-4.0 License.
Dataset acknowledgments: National Renewable Energy Laboratory (NREL) ESIF HPC Center & LBNL / DeepMind Data Center Cooling benchmarks.
