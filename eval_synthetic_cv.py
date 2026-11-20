"""Cross-validation on synthetic data.

Uses the physics simulator as ground-truth.
Reports blocked K-fold CV metrics + per-fold breakdown + OOF residual quantiles.
"""
import warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict

from data import load_synthetic, select_features
from model import make_models, _metrics, TARGET

# ── generate synthetic dataset ───────────────────────────────────────────────
print("Generating synthetic data...")
df, rep = load_synthetic(seed=7, n_servers=200, mean_temp=18.0, noise=0.03, gap=0.5)
features = select_features(df)
print(f"Rows     : {len(df):,}  ({rep['start'].date()} → {rep['end'].date()})")
print(f"Features : {features}")
print(f"Target   : mean={df[TARGET].mean():.1f} kW, std={df[TARGET].std():.1f} kW\n")

X, y = df[features], df[TARGET].to_numpy()
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=False)   # blocked / chronological folds

# ── per-model CV ─────────────────────────────────────────────────────────────
print(f"Running {N_SPLITS}-fold blocked cross-validation on {len(df):,} rows...\n")
print(f"{'Model':<28} {'CV R²':>8} {'CV RMSE':>9} {'CV MAE':>8} {'MAPE%':>7} {'90th-pct err':>13} {'Time(s)':>8}")
print("-"*90)

rows, oof_all = [], {}
for name, est in make_models(features).items():
    t0 = time.time()
    oof = cross_val_predict(clone(est), X, y, cv=kf)
    elapsed = time.time() - t0
    m = _metrics(y, oof)
    q90 = float(np.quantile(np.abs(y - oof), 0.90))
    rows.append({**{"Model": name}, **m, "90th-pct abs err (kW)": q90, "CV time (s)": elapsed})
    oof_all[name] = oof
    print(f"{name:<28} {m['R²']:>8.4f} {m['RMSE']:>9.2f} {m['MAE']:>8.2f} {m['MAPE %']:>7.2f} {q90:>13.2f} {elapsed:>8.1f}")

lb = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
best = lb.iloc[0]["Model"]

# ── per-fold breakdown for the best model ────────────────────────────────────
print(f"\n{'='*90}")
print(f"Per-fold breakdown — {best}")
print(f"{'='*90}")
best_est = clone(make_models(features)[best])
print(f"{'Fold':>5} {'Train rows':>11} {'Val rows':>9} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'MAPE%':>7}")
print("-"*65)
fold_rows = []
for i, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    Xtr, ytr = X.iloc[tr_idx], y[tr_idx]
    Xva, yva = X.iloc[va_idx], y[va_idx]
    m_fold = clone(best_est).fit(Xtr, ytr)
    p = m_fold.predict(Xva)
    fm = _metrics(yva, p)
    fold_rows.append(fm)
    print(f"{i:>5} {len(tr_idx):>11,} {len(va_idx):>9,} {fm['R²']:>8.4f} {fm['RMSE']:>8.2f} {fm['MAE']:>8.2f} {fm['MAPE %']:>7.2f}")

fold_df = pd.DataFrame(fold_rows)
print("-"*65)
print(f"{'MEAN':>5} {' ':>11} {' ':>9} {fold_df['R²'].mean():>8.4f} "
      f"{fold_df['RMSE'].mean():>8.2f} {fold_df['MAE'].mean():>8.2f} {fold_df['MAPE %'].mean():>7.2f}")
print(f"{'STD':>5} {' ':>11} {' ':>9} {fold_df['R²'].std():>8.4f} "
      f"{fold_df['RMSE'].std():>8.2f} {fold_df['MAE'].std():>8.2f} {fold_df['MAPE %'].std():>7.2f}")

# ── summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("FULL LEADERBOARD (sorted by CV R²)")
print(f"{'='*90}")
print(lb[["Model","R²","RMSE","MAE","MAPE %","90th-pct abs err (kW)","CV time (s)"]].to_string(index=False))
print(f"\n✓ Best model: {best}  (CV R² = {lb.iloc[0]['R²']:.4f})")
