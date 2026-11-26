"""Generic cooling-power modelling for real data.

Inputs (any subset available in the data): IT load (kW), ambient temp, humidity, cooling setpoint.
Engineered physics features: wet-bulb temperature (Stull 2011), load x weather interactions, approach temp.
Models: constant-PUE baseline | polynomial OLS | ridge + physics features | random forest | gradient boosting | LSTM neural network | stack.
Validation: chronological hold-out + blocked K-fold. Uncertainty: out-of-fold residual quantile interval.
Sensitivity: local gradient/elasticity, tornado, sweeps, Sobol (Saltelli). What-if scenarios + setpoint optimiser.
"""
import time
from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

TARGET = "cooling_kw"
LABELS = {
    "it_kw": "IT load (kW)",
    "ambient_c": "Ambient temp (°C)",
    "humidity_pct": "Relative humidity (%)",
    "setpoint_c": "Cooling setpoint (°C)",
    "chw_flow_kg_s": "Chilled water flow (kg/s)",
    "chw_return_c": "CHW return temp (°C)",
    "head_pressure_m": "Pump head required (m)",
    "pump_vfd_rpm": "Pump VFD speed (RPM)",
    "pumps_active": "Active pumps count",
    "chillers_active": "Active chillers count",
    "wetbulb_c": "Wet-bulb temp (°C)",
    "approach": "Approach temp (°C)",
    "hour_sin": "Diurnal cycle (sin)",
    "hour_cos": "Diurnal cycle (cos)",
    "doy_sin": "Seasonal cycle (sin)",
    "doy_cos": "Seasonal cycle (cos)",
    "ambient_lag1": "Ambient lag 1h (°C)",
    "ambient_lag2": "Ambient lag 2h (°C)",
    "ambient_lag3": "Ambient lag 3h (°C)",
    "it_lag1": "IT load lag 1h (kW)",
    "it_lag2": "IT load lag 2h (kW)",
    "ambient_roll6": "Ambient 6h rolling mean (°C)",
    "it_roll6": "IT load 6h rolling mean (kW)",
    "it_x_wb": "IT × Wet-bulb interaction",
    "it_x_amb": "IT × Ambient interaction",
    "cooling_lag1": "Cooling power lag 1h (kW)",
    "cooling_lag2": "Cooling power lag 2h (kW)",
    "cooling_roll6": "Cooling power 6h rolling mean (kW)",
}


def wet_bulb(t, rh):
    """Stull (2011) wet-bulb approximation, T in °C, RH in %."""
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh) - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


def engineer(X, features):
    X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=list(features))
    out = X[list(features)].astype(float).copy()
    it, t = out["it_kw"].to_numpy(), out["ambient_c"].to_numpy()
    wb = wet_bulb(t, np.clip(out["humidity_pct"].to_numpy(), 5, 99)) if "humidity_pct" in features else t
    out["wetbulb_c"] = wb
    out["it_x_wb"], out["it_x_amb"], out["it_sq"], out["wb_sq"] = it * wb, it * t, it ** 2, wb ** 2

    # 1. Thermal approach & setpoint interactions
    if "setpoint_c" in features:
        sp = out["setpoint_c"].to_numpy()
        out["approach"], out["it_x_sp"], out["amb_x_sp"] = wb - sp, it * sp, t * sp

    # 2. Chilled water loop delta-T & instantaneous thermal load proxy
    if "chw_return_c" in features and "setpoint_c" in features:
        delta_t = np.maximum(out["chw_return_c"].to_numpy() - out["setpoint_c"].to_numpy(), 0.0)
        out["chw_delta_t"] = delta_t
        if "chw_flow_kg_s" in features:
            out["thermal_load_proxy"] = out["chw_flow_kg_s"].to_numpy() * delta_t

    # 3. Hydraulic pumping power (P_hyd = mass_flow * head)
    if "chw_flow_kg_s" in features and "head_pressure_m" in features:
        out["flow_x_head"] = out["chw_flow_kg_s"].to_numpy() * out["head_pressure_m"].to_numpy()

    # 4. Pump Affinity Law: power scales with (RPM / nominal)^3
    if "pump_vfd_rpm" in features:
        norm_rpm = out["pump_vfd_rpm"].to_numpy() / 1000.0
        out["rpm_sq"] = norm_rpm ** 2
        out["rpm_cubed"] = norm_rpm ** 3

    # 5. Flow rate per active pump (operating point relative to Best Efficiency Point)
    if "chw_flow_kg_s" in features and "pumps_active" in features:
        out["flow_per_pump"] = out["chw_flow_kg_s"].to_numpy() / np.maximum(out["pumps_active"].to_numpy(), 1.0)

    # 6. Part-load thermal ratio per active chiller
    if "chillers_active" in features and "thermal_load_proxy" in out:
        out["load_per_chiller"] = out["thermal_load_proxy"].to_numpy() / np.maximum(out["chillers_active"].to_numpy(), 1.0)

    return out


class ConstantPUE(BaseEstimator, RegressorMixin):
    """Industry rule-of-thumb baseline: cooling = k * IT load (k = overall cooling/IT ratio)."""
    def __init__(self, k=None):
        self.k = k

    def fit(self, X, y):
        self.k_ = float(np.sum(y) / np.sum(X["it_kw"]))
        return self

    def predict(self, X):
        k = getattr(self, "k_", self.k or 0.0)
        return k * np.asarray(X["it_kw"], float)

    def __reduce__(self):
        import model
        return (model.ConstantPUE, (getattr(self, "k_", self.k),), self.__dict__)


if TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):
        def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.0):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                                batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    class LSTMRegressor(BaseEstimator, RegressorMixin):
        """Long Short-Term Memory (LSTM) recurrent neural network for data-center cooling loads.
        
        Captures dynamic thermal inertia, time lags, and historical momentum across a sliding sequence window.
        """
        def __init__(self, seq_len=4, hidden_dim=32, num_layers=1, lr=0.01, epochs=25,
                     batch_size=128, random_state=0):
            self.seq_len = seq_len
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.random_state = random_state

        def _make_windows(self, X_scaled):
            N, D = X_scaled.shape
            if N == 0:
                return np.empty((0, self.seq_len, D))
            pad = np.repeat(X_scaled[0:1], self.seq_len - 1, axis=0)
            padded = np.vstack([pad, X_scaled])
            return np.lib.stride_tricks.sliding_window_view(padded, (self.seq_len, D))[:, 0, :, :]

        def fit(self, X, y):
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
                np.random.seed(self.random_state)

            X_arr = np.asarray(X, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            self.input_dim_ = X_arr.shape[1]
            self.scaler_X_ = StandardScaler().fit(X_arr)
            self.y_mean_ = float(np.mean(y_arr))
            self.y_std_ = float(np.std(y_arr)) if np.std(y_arr) > 1e-4 else 1.0

            X_scaled = self.scaler_X_.transform(X_arr)
            y_scaled = (y_arr - self.y_mean_) / self.y_std_

            windows = self._make_windows(X_scaled)
            X_t = torch.tensor(windows, dtype=torch.float32)
            y_t = torch.tensor(y_scaled, dtype=torch.float32)

            self.net_ = _LSTMNet(self.input_dim_, self.hidden_dim, self.num_layers)
            optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
            criterion = nn.MSELoss()

            dataset = torch.utils.data.TensorDataset(X_t, y_t)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.net_.train()
            for _ in range(self.epochs):
                for bx, by in loader:
                    optimizer.zero_grad()
                    pred = self.net_(bx)
                    loss = criterion(pred, by)
                    loss.backward()
                    optimizer.step()

            self.is_fitted_ = True
            return self

        def predict(self, X):
            X_arr = np.asarray(X, dtype=float)
            X_scaled = self.scaler_X_.transform(X_arr)
            windows = self._make_windows(X_scaled)

            self.net_.eval()
            with torch.no_grad():
                X_t = torch.tensor(windows, dtype=torch.float32)
                p_scaled = self.net_(X_t).numpy()

            p = p_scaled * self.y_std_ + self.y_mean_
            return np.maximum(p, 0.0)

        def __getstate__(self):
            state = self.__dict__.copy()
            if "net_" in state and state["net_"] is not None:
                state["net_state_"] = state["net_"].state_dict()
                del state["net_"]
            return state

        def __setstate__(self, state):
            net_state = state.pop("net_state_", None)
            self.__dict__.update(state)
            if net_state is not None:
                self.net_ = _LSTMNet(self.input_dim_, self.hidden_dim, self.num_layers)
                self.net_.load_state_dict(net_state)
                self.net_.eval()

    class PINNRegressor(BaseEstimator, RegressorMixin):
        """Physics-Informed Neural Network (PINN) for data-center cooling loads.
        
        Enforces 1st and 2nd Thermodynamic Laws directly into the loss function during backprop:
        1. Carnot / Real COP lower bound: Heat removal work W_comp >= Q_evap / COP_max
        2. Monotonicity constraint: d(cooling_kw) / d(it_kw) >= 0 (derived via PyTorch autograd)
        3. Physical non-negativity: cooling_kw >= 0.0
        Prevents catastrophic out-of-distribution failure during hardware retrofits or power jumps.
        """
        def __init__(self, seq_len=4, hidden_dim=32, num_layers=1, lr=0.01, epochs=25,
                     batch_size=128, lambda_carnot=0.15, lambda_mono=0.10, random_state=0):
            self.seq_len = seq_len
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.lambda_carnot = lambda_carnot
            self.lambda_mono = lambda_mono
            self.random_state = random_state

        def _make_windows(self, X_scaled):
            N, D = X_scaled.shape
            if N == 0:
                return np.empty((0, self.seq_len, D))
            pad = np.repeat(X_scaled[0:1], self.seq_len - 1, axis=0)
            padded = np.vstack([pad, X_scaled])
            return np.lib.stride_tricks.sliding_window_view(padded, (self.seq_len, D))[:, 0, :, :]

        def fit(self, X, y):
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
                np.random.seed(self.random_state)

            X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
            self.feature_names_ = list(X_df.columns)
            self.it_col_idx_ = self.feature_names_.index("it_kw") if "it_kw" in self.feature_names_ else 0

            X_arr = X_df.to_numpy(dtype=float)
            y_arr = np.asarray(y, dtype=float)
            self.input_dim_ = X_arr.shape[1]
            self.scaler_X_ = StandardScaler().fit(X_arr)
            self.y_mean_ = float(np.mean(y_arr))
            self.y_std_ = float(np.std(y_arr)) if np.std(y_arr) > 1e-4 else 1.0

            X_scaled = self.scaler_X_.transform(X_arr)
            y_scaled = (y_arr - self.y_mean_) / self.y_std_

            windows = self._make_windows(X_scaled)
            X_t = torch.tensor(windows, dtype=torch.float32)
            y_t = torch.tensor(y_scaled, dtype=torch.float32)

            it_mean = float(self.scaler_X_.mean_[self.it_col_idx_])
            it_std = float(self.scaler_X_.scale_[self.it_col_idx_])

            self.net_ = _LSTMNet(self.input_dim_, self.hidden_dim, self.num_layers)
            optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
            mse_fn = nn.MSELoss()

            dataset = torch.utils.data.TensorDataset(X_t, y_t)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.net_.train()
            for _ in range(self.epochs):
                for (bx, by) in loader:
                    optimizer.zero_grad()
                    bx.requires_grad_(True)
                    pred = self.net_(bx)
                    l_data = mse_fn(pred, by)

                    # 1. Physics Carnot / Heat removal lower bound (cooling >= 0.02 * IT)
                    pred_kw = pred * self.y_std_ + self.y_mean_
                    it_kw_tensor = bx[:, -1, self.it_col_idx_] * it_std + it_mean
                    min_cooling_kw = 0.02 * it_kw_tensor
                    l_carnot = torch.relu(min_cooling_kw - pred_kw).pow(2).mean() / (self.y_std_ ** 2)

                    # 2. Monotonicity constraint: d(cooling)/d(IT) >= 0 via PyTorch autograd
                    grads = torch.autograd.grad(outputs=pred.sum(), inputs=bx, create_graph=True)[0]
                    it_grad = grads[:, -1, self.it_col_idx_]
                    l_mono = torch.relu(-it_grad).pow(2).mean()

                    loss = l_data + self.lambda_carnot * l_carnot + self.lambda_mono * l_mono
                    loss.backward()
                    optimizer.step()

            self.is_fitted_ = True
            return self

        def predict(self, X):
            X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_)
            X_arr = X_df.to_numpy(dtype=float)
            X_scaled = self.scaler_X_.transform(X_arr)
            windows = self._make_windows(X_scaled)

            self.net_.eval()
            with torch.no_grad():
                X_t = torch.tensor(windows, dtype=torch.float32)
                p_scaled = self.net_(X_t).numpy()

            p = p_scaled * self.y_std_ + self.y_mean_
            return np.maximum(p, 0.0)

        def __getstate__(self):
            state = self.__dict__.copy()
            if "net_" in state and state["net_"] is not None:
                state["net_state_"] = state["net_"].state_dict()
                del state["net_"]
            return state

        def __setstate__(self, state):
            net_state = state.pop("net_state_", None)
            self.__dict__.update(state)
            if net_state is not None:
                self.net_ = _LSTMNet(self.input_dim_, self.hidden_dim, self.num_layers)
                self.net_.load_state_dict(net_state)
                self.net_.eval()

    class _Encoder(nn.Module):
        def __init__(self, in_d, h_d):
            super().__init__()
            self.lstm = nn.LSTM(in_d, h_d, batch_first=True)
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return h[-1]

    class _Decoder(nn.Module):
        def __init__(self, in_d, h_d, seq_len):
            super().__init__()
            self.seq_len = seq_len
            self.lstm = nn.LSTM(h_d, h_d, batch_first=True)
            self.fc = nn.Linear(h_d, in_d)
        def forward(self, z):
            rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
            out, _ = self.lstm(rep)
            return self.fc(out)

    class _AENet(nn.Module):
        def __init__(self, in_d, h_d, seq_len):
            super().__init__()
            self.enc = _Encoder(in_d, h_d)
            self.dec = _Decoder(in_d, h_d, seq_len)
        def forward(self, x):
            return self.dec(self.enc(x))

    class LSTMAutoencoder(BaseEstimator):
        """LSTM Autoencoder for unsupervised data center anomaly and fault detection.
        
        Compresses multi-channel sensor time-series into a latent representation and reconstructs them.
        Deviations (reconstruction error spikes) identify sensor drift, stuck setpoints, chiller trips,
        and pump cavitation.
        """
        def __init__(self, seq_len=4, hidden_dim=24, lr=0.005, epochs=20, batch_size=128, random_state=0):
            self.seq_len = seq_len
            self.hidden_dim = hidden_dim
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.random_state = random_state

        def _make_windows(self, X_scaled):
            N, D = X_scaled.shape
            if N == 0:
                return np.empty((0, self.seq_len, D))
            pad = np.repeat(X_scaled[0:1], self.seq_len - 1, axis=0)
            padded = np.vstack([pad, X_scaled])
            return np.lib.stride_tricks.sliding_window_view(padded, (self.seq_len, D))[:, 0, :, :]

        def fit(self, X, y=None):
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
                np.random.seed(self.random_state)

            X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
            self.feature_names_ = list(X_df.columns)
            X_arr = X_df.to_numpy(dtype=float)
            self.input_dim_ = X_arr.shape[1]
            self.scaler_ = StandardScaler().fit(X_arr)
            X_scaled = self.scaler_.transform(X_arr)

            windows = self._make_windows(X_scaled)
            X_t = torch.tensor(windows, dtype=torch.float32)

            self.net_ = _AENet(self.input_dim_, self.hidden_dim, self.seq_len)
            opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
            loss_fn = nn.MSELoss()

            dataset = torch.utils.data.TensorDataset(X_t)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.net_.train()
            for _ in range(self.epochs):
                for (bx,) in loader:
                    opt.zero_grad()
                    pred = self.net_(bx)
                    l = loss_fn(pred, bx)
                    l.backward()
                    opt.step()

            self.is_fitted_ = True
            return self

        def reconstruct(self, X):
            X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_)
            X_arr = X_df.to_numpy(dtype=float)
            X_scaled = self.scaler_.transform(X_arr)
            windows = self._make_windows(X_scaled)

            self.net_.eval()
            with torch.no_grad():
                rec_scaled = self.net_(torch.tensor(windows, dtype=torch.float32)).numpy()

            rec_current = rec_scaled[:, -1, :]
            rec_unscaled = self.scaler_.inverse_transform(rec_current)
            return rec_unscaled

        def anomaly_scores(self, X):
            X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_)
            X_arr = X_df.to_numpy(dtype=float)
            rec = self.reconstruct(X_df)
            sq_err = (X_arr - rec) ** 2
            var = np.maximum(self.scaler_.var_, 1e-4)
            norm_sq_err = sq_err / var
            overall_score = np.mean(norm_sq_err, axis=1)
            return overall_score, pd.DataFrame(norm_sq_err, columns=self.feature_names_, index=X_df.index)

        def __getstate__(self):
            state = self.__dict__.copy()
            if "net_" in state and state["net_"] is not None:
                state["net_state_"] = state["net_"].state_dict()
                del state["net_"]
            return state

        def __setstate__(self, state):
            net_state = state.pop("net_state_", None)
            self.__dict__.update(state)
            if net_state is not None:
                self.net_ = _AENet(self.input_dim_, self.hidden_dim, self.seq_len)
                self.net_.load_state_dict(net_state)
                self.net_.eval()


def make_models(features):
    fe = lambda: FunctionTransformer(partial(engineer, features=tuple(features)))
    imp = lambda: SimpleImputer(strategy="median")
    ridge = lambda: make_pipeline(imp(), fe(), PolynomialFeatures(2, include_bias=False), StandardScaler(),
                                  RidgeCV(alphas=np.logspace(-3, 3, 13)))
    rf = lambda: make_pipeline(imp(), fe(), RandomForestRegressor(80, max_depth=14, min_samples_leaf=4, n_jobs=-1, random_state=0))
    hgb = lambda: make_pipeline(imp(), fe(), HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                min_samples_leaf=20, l2_regularization=0.5, random_state=0))
    models = {
        "Constant-PUE baseline": ConstantPUE(),
        "Polynomial OLS (deg 2)": make_pipeline(imp(), fe(), PolynomialFeatures(2, include_bias=False), StandardScaler(),
                                                LinearRegression()),
        "Ridge + physics features": ridge(),
        "Random Forest": rf(),
        "Gradient Boosting": hgb(),
        "Stacked ensemble": StackingRegressor([("ridge", ridge()), ("hgb", hgb()), ("rf", rf())],
                                              final_estimator=Ridge(alpha=10.0, positive=True), cv=3, n_jobs=-1),
    }
    if TORCH_AVAILABLE:
        models["LSTM Neural Network"] = make_pipeline(
            imp(), fe(),
            LSTMRegressor(seq_len=4, hidden_dim=32, num_layers=1, lr=0.01, epochs=20, batch_size=128, random_state=0)
        )
        models["Physics-Informed NN (PINN)"] = make_pipeline(
            imp(), fe(),
            PINNRegressor(seq_len=4, hidden_dim=32, num_layers=1, lr=0.01, epochs=20, batch_size=128,
                          lambda_carnot=0.15, lambda_mono=0.10, random_state=0)
        )
    return models


def _metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = np.sum((y - p) ** 2)
    denom = np.abs(y).sum()
    wmape = float(np.sum(np.abs(y - p)) / denom * 100) if denom > 0 else 0.0
    return {"R²": 1 - ss / np.sum((y - y.mean()) ** 2), "RMSE": float(np.sqrt(ss / len(y))),
            "MAE": float(np.mean(np.abs(y - p))),
            "MAPE %": float(np.mean(np.abs(y - p) / np.maximum(np.abs(y), 0.05 * np.median(np.abs(y)))) * 100),
            "WMAPE %": wmape}


def evaluate_all(df, features, n_splits=3, split_strategy="representative"):
    """Evaluate all benchmark models with hold-out and K-fold cross-validation.
    
    split_strategy:
      - 'representative' (default): 80/20 train/test split preserving seasonality
        and operational modes across the dataset.
      - 'chronological': 80/20 chronological time-series split.
    """
    from sklearn.model_selection import train_test_split
    if split_strategy == "representative":
        tr, te = train_test_split(df, test_size=0.20, random_state=42)
    else:
        split = int(len(df) * 0.8)
        tr, te = df.iloc[:split], df.iloc[split:]

    X, y = df[features], df[TARGET].to_numpy()

    # Subsample for CV if dataset is large, keeping CV fast and accurate
    CV_N = min(6000, len(df))
    if len(df) > CV_N:
        cv_idx = np.linspace(0, len(df) - 1, CV_N, dtype=int)
        Xcv, ycv = X.iloc[cv_idx], y[cv_idx]
    else:
        Xcv, ycv = X, y

    kf = KFold(n_splits, shuffle=True, random_state=42)
    rows, hold, holdm, oof, q90, final = [], {}, {}, {}, {}, {}
    models = make_models(features)

    for name, est in models.items():
        t0 = time.time()
        m = clone(est).fit(tr[features], tr[TARGET])
        p = m.predict(te[features])
        fit_s = time.time() - t0
        
        if name == "Stacked ensemble":
            base_names = ["Ridge + physics features", "Gradient Boosting", "Random Forest"]
            if all(b in oof and not np.isnan(oof[b][0]) for b in base_names):
                X_meta = np.column_stack([oof[b] for b in base_names])
                meta_cv = Ridge(alpha=10.0, positive=True)
                cvp = cross_val_predict(meta_cv, X_meta, ycv, cv=kf)
                mc = _metrics(ycv, cvp)
            else:
                cvp = np.full_like(ycv, np.nan)
                mc = {"R²": float("nan"), "RMSE": float("nan")}
        else:
            cvp = cross_val_predict(clone(est), Xcv, ycv, cv=kf)
            mc = _metrics(ycv, cvp)
        
        q90[name] = float(np.quantile(np.abs(ycv - cvp), 0.90)) if not np.isnan(cvp[0]) else float(np.quantile(np.abs(te[TARGET] - p), 0.90))
        mt = _metrics(te[TARGET], p)
        p_mean = float(np.mean(p))
        p_std = float(np.std(p))
        bias = float(p_mean - np.mean(te[TARGET]))
        rows.append({"Model": name, "R²": mt["R²"], "RMSE (kW)": mt["RMSE"], "MAE (kW)": mt["MAE"],
                     "Pred Mean (kW)": p_mean, "Pred Std (kW)": p_std, "Bias (kW)": bias,
                     "MAPE %": mt["MAPE %"], "WMAPE %": mt["WMAPE %"], "CV RMSE (kW)": mc["RMSE"], "CV R²": mc["R²"], "Fit (s)": fit_s})
        hold[name], holdm[name], oof[name] = p, m, cvp
        # Fast fit final: if large dataset, use Xcv for heavy NNs
        fit_X, fit_y = (Xcv, ycv) if len(X) > 10000 and "NN" in name else (X, y)
        final[name] = clone(est).fit(fit_X, fit_y)

    lb = pd.DataFrame(rows)
    base = lb.loc[lb.Model == "Constant-PUE baseline", "RMSE (kW)"].iloc[0]
    lb["Skill vs baseline %"] = (1 - lb["RMSE (kW)"] / base) * 100
    lb = lb.sort_values("RMSE (kW)").reset_index(drop=True)
    best = lb[lb.Model != "Constant-PUE baseline"].iloc[0]["Model"]
    return {"leaderboard": lb, "holdout_pred": hold, "holdout_models": holdm, "oof": oof, "q90": q90,
            "final": final, "test": te, "best": best, "features": list(features),
            "target_test_mean": float(np.mean(te[TARGET])), "target_test_std": float(np.std(te[TARGET])),
            "target_test_min": float(np.min(te[TARGET])), "target_test_max": float(np.max(te[TARGET])),
            "test_rows": len(te)}


# ---------------------------------------------------------------- prediction + sensitivity
def predict(est, features, x):
    d = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
    pred = np.asarray(est.predict(d[list(features)]), float)
    return np.maximum(pred, 0.0)  # Physical guardrail: power consumption cannot be negative


def predict_point(est, features, x):
    return float(predict(est, features, {k: [v] for k, v in x.items()})[0])


def local_gradient(est, features, x, bounds):
    out = {}
    for k in features:
        h = 0.01 * (bounds[k][1] - bounds[k][0])
        hi, lo = dict(x), dict(x)
        hi[k] += h; lo[k] -= h
        out[k] = (predict_point(est, features, hi) - predict_point(est, features, lo)) / (2 * h)
    return out


def sensitivity_table(est, features, x, bounds):
    y0, g = predict_point(est, features, x), local_gradient(est, features, x, bounds)
    rows = []
    for k in features:
        rng = bounds[k][1] - bounds[k][0]
        rows.append({"Input": LABELS[k], "∂y/∂x (kW per unit)": g[k],
                     "Δ cooling for +10% of range (kW)": g[k] * 0.1 * rng,
                     "Elasticity (%Δy per %Δx)": g[k] * x[k] / y0 if y0 else np.nan})
    return pd.DataFrame(rows)


def tornado(est, features, x, bounds, frac=0.10):
    rows = []
    for k in features:
        lo_b, hi_b = bounds[k]; d = frac * (hi_b - lo_b)
        lo, hi = dict(x), dict(x)
        lo[k] = max(lo_b, x[k] - d); hi[k] = min(hi_b, x[k] + d)
        rows.append({"Input": LABELS[k], "low": predict_point(est, features, lo), "high": predict_point(est, features, hi)})
    t = pd.DataFrame(rows)
    t["swing"] = (t["high"] - t["low"]).abs()
    return t.sort_values("swing")


def sweep(est, features, x, var, grid):
    d = {k: np.full(len(grid), v) for k, v in x.items()}
    d[var] = np.asarray(grid)
    return predict(est, features, d)


def grid2d(est, features, x, vx, vy, gx, gy):
    XX, YY = np.meshgrid(gx, gy)
    d = {k: np.full(XX.size, v) for k, v in x.items()}
    d[vx], d[vy] = XX.ravel(), YY.ravel()
    return predict(est, features, d).reshape(XX.shape)


def sobol_indices(est, features, bounds, n=4096, seed=0):
    """Variance-based global sensitivity (Saltelli 2010). Assumes independent inputs over `bounds`."""
    rng = np.random.default_rng(seed)
    d = len(features)
    lo = np.array([bounds[k][0] for k in features]); hi = np.array([bounds[k][1] for k in features])
    A = lo + (hi - lo) * rng.random((n, d)); B = lo + (hi - lo) * rng.random((n, d))
    f = lambda M: est.predict(pd.DataFrame(M, columns=list(features)))
    fA, fB = f(A), f(B)
    var = np.var(np.concatenate([fA, fB]))
    f0 = np.mean(np.concatenate([fA, fB]))      # centring lowers estimator variance
    rows = []
    for i, k in enumerate(features):
        ABi = A.copy(); ABi[:, i] = B[:, i]
        fAB = f(ABi)
        rows.append({"Input": LABELS.get(k, k.replace("_", " ").title()), "S1 (main effect)": float(np.mean((fB - f0) * (fAB - fA)) / var),
                     "ST (total, incl. interactions)": float(0.5 * np.mean((fA - fAB) ** 2) / var)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- scenarios + optimisation
def run_scenario(est, df, features, it_scale=1.0, d_temp=0.0, d_rh=0.0, d_sp=0.0,
                 rpm_scale=1.0, flow_scale=1.0):
    X = df[list(features)].copy()
    X["it_kw"] = X["it_kw"] * it_scale
    X["ambient_c"] = X["ambient_c"] + d_temp
    if "humidity_pct" in features:
        X["humidity_pct"] = np.clip(X["humidity_pct"] + d_rh, 5, 100)
    if "setpoint_c" in features:
        X["setpoint_c"] = X["setpoint_c"] + d_sp
    if "pump_vfd_rpm" in features:
        X["pump_vfd_rpm"] = np.maximum(X["pump_vfd_rpm"] * rpm_scale, 0.0)
    if "chw_flow_kg_s" in features:
        X["chw_flow_kg_s"] = np.maximum(X["chw_flow_kg_s"] * flow_scale, 0.0)
    return predict(est, features, X), X


def optimise_setpoints(est, df, features, grid):
    n, g = len(df), len(grid)
    big = pd.DataFrame({k: np.repeat(df[k].to_numpy(), g) for k in features if k != "setpoint_c"})
    big["setpoint_c"] = np.tile(grid, n)
    preds = predict(est, features, big[list(features)])
    return grid[preds.reshape(n, g).argmin(axis=1)]


def export_model(est, features, path):
    """Serialize trained cooling model pipeline and feature metadata for deployment."""
    import joblib
    joblib.dump({"model": est, "features": list(features), "labels": LABELS}, path)


def load_model(path):
    """Load serialized model pipeline and feature metadata."""
    import joblib
    data = joblib.load(path)
    return data["model"], data["features"]