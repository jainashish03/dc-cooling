"""pytest tests/ — pipeline tests on a schema-matching fixture (no network needed)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent)); sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, pandas as pd, pytest
import data as D, model as M
from fixture import make_esif_like


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    import os
    p = tmp_path_factory.mktemp("d"); os.environ["DC_DATA_DIR"] = str(p)
    make_esif_like(p, days=500)
    D.BATCH_ROWS = 40_000                      # force multi-batch streaming
    raw, meta = D.load_esif_raw("1h", "America/Denver")
    df, rep = D.build_dataset(raw, {"it": "it_power_kw", "cooling": ["cooling_kw", "hvac_kw", "pump_kw"],
                                    "temp": "outside_air_temp", "temp_unit": "F", "humidity": "outside_air_humidity"})
    return df, rep, meta


def test_streaming_aggregation(ds):
    df, rep, meta = ds
    assert meta["rows_power"] == 500 * 288 and 8000 < len(df) < 12001
    assert df.ambient_c.between(-30, 45).all() and (df.cooling_kw > 0).all()


def test_features(ds):
    assert D.select_features(ds[0]) == ["it_kw", "ambient_c", "humidity_pct"]


def test_models_beat_baseline(ds):
    df = ds[0]; f = D.select_features(df)
    R = M.evaluate_all(df, f)
    lb = R["leaderboard"].set_index("Model")
    assert lb.loc[R["best"], "Skill vs baseline %"] > 20
    assert lb.loc[R["best"], "R²"] > 0.6


def test_sensitivity_and_sobol(ds):
    df = ds[0]; f = D.select_features(df)
    est = M.make_models(f)["Gradient Boosting"].fit(df[f], df.cooling_kw)
    b = {k: (df[k].quantile(.005), df[k].quantile(.995)) for k in f}
    x = {k: float(df[k].median()) for k in f}
    g = M.local_gradient(est, f, x, b)
    assert g["it_kw"] > 0 and g["ambient_c"] > 0
    s = M.sobol_indices(est, f, b, n=4096)
    assert 0.7 < s["S1 (main effect)"].sum() < 1.2


def test_scenario(ds):
    df = ds[0]; f = D.select_features(df)
    est = M.make_models(f)["Ridge + physics features"].fit(df[f], df.cooling_kw)
    base, _ = M.run_scenario(est, df, f)
    hot, _ = M.run_scenario(est, df, f, d_temp=5)
    assert hot.sum() > base.sum()


def test_real_file_layout_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("DC_DATA_DIR", str(tmp_path))
    make_esif_like(tmp_path, days=60, outdoor_names=True)
    raw, meta = D.load_esif_raw("1h", "America/Denver")
    assert meta["temp_unit"] == "F" and {"outside_air_temp", "outside_air_humidity"} <= set(raw.columns)
    df, _ = D.build_dataset(raw, {"it": "it_power_kw", "cooling": ["cooling_kw", "hvac_kw", "pump_kw"],
                                  "temp": "outside_air_temp", "temp_unit": meta["temp_unit"],
                                  "humidity": "outside_air_humidity"})
    assert len(df) > 1000 and df.ambient_c.between(-30, 45).all()


def test_lstm_model(ds):
    if not M.TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    df = ds[0].iloc[:500]
    f = D.select_features(df)
    models = M.make_models(f)
    assert "LSTM Neural Network" in models
    est = models["LSTM Neural Network"].fit(df[f], df.cooling_kw)
    preds = est.predict(df[f])
    assert len(preds) == len(df)
    assert (preds >= 0.0).all()


def test_pinn_model(ds):
    if not M.TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    df = ds[0].iloc[:500]
    f = D.select_features(df)
    models = M.make_models(f)
    assert "Physics-Informed NN (PINN)" in models
    est = models["Physics-Informed NN (PINN)"].fit(df[f], df.cooling_kw)
    preds = est.predict(df[f])
    assert len(preds) == len(df)
    assert (preds >= 0.0).all()


def test_lstm_autoencoder(ds):
    if not M.TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    df = ds[0].iloc[:500]
    f = D.select_features(df)
    cols = list(f) + ["cooling_kw"]
    ae = M.LSTMAutoencoder(seq_len=4, hidden_dim=16, epochs=5, random_state=0)
    ae.fit(df[cols])
    scores, feat_errs = ae.anomaly_scores(df[cols])
    assert len(scores) == len(df)
    assert (scores >= 0.0).all()
    assert feat_errs.shape == (len(df), len(cols))


def test_temporal_physics_features(ds):
    raw_df = ds[0]
    enhanced = D.add_temporal_physics_features(raw_df)
    assert "wetbulb_c" in enhanced.columns
    assert "approach" in enhanced.columns
    assert "hour_sin" in enhanced.columns
    assert "hour_cos" in enhanced.columns
    assert "doy_sin" in enhanced.columns
    assert "doy_cos" in enhanced.columns
    assert "ambient_lag1" in enhanced.columns
    assert "it_lag1" in enhanced.columns
    assert "ambient_roll6" in enhanced.columns
    assert "it_roll6" in enhanced.columns
    assert "it_x_wb" in enhanced.columns
    assert "it_x_amb" in enhanced.columns
    feats = D.select_features(enhanced)
    assert len(feats) > 10
    assert "wetbulb_c" in feats


