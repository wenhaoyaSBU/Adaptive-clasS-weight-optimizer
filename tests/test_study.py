import json
import numpy as np
import pytest
import uproot
from cmok_toy.config import Config
from cmok_toy.study import run_study
from cmok_toy.inference import load_predict
from cmok_toy.binning import fit_binning
from cmok_toy.truth import Truth, generate


def test_end_to_end_checkpoint_and_resume(tmp_path):
    cfg = Config(replicas=1, n_background=400, n_signal=40, epochs=2, weight_epochs=2,
                 batch_size=64, models=['linear', 'mlp'], widths=[8, 4], scan_deltas=[1.],
                 meta_every=1, meta_warmup_epochs=0, bootstrap_samples=20, plots=False)
    rows = run_study(cfg, tmp_path)
    assert len(rows) == 14
    assert all(np.isfinite(r['physical_auc']) for r in rows)
    toy = tmp_path / f'toy_{cfg.seed}'
    binnings = json.loads((toy / 'binning.json').read_text())
    details = json.loads((toy / 'metric_details.json').read_text())
    assert len(binnings) == 14
    validation = generate(Truth(), cfg.seed, cfg.n_background, cfg.n_signal)['validation']
    for model in cfg.models:
        for method in ('plain_ce', 'weighted_ce', 'cmok', 'adaptive', 'balanced_cmok', 'scan_best'):
            key = f'{model}/{method}'
            val_score = load_predict(toy / f'{model}_{method}.pt', validation.x)
            expected = fit_binning(validation, val_score, cfg)
            np.testing.assert_allclose(binnings[key]['edges'], expected['edges'], atol=1e-7)
            assert details[key]['asimov']['edges'] == binnings[key]['edges']
            assert binnings[key]['optimization_sample'] == 'validation'
    predictions = np.load(toy / 'predictions.npz')
    with uproot.open(toy / 'events.root') as root:
        assert root['Events'].classname == 'TTree'
        assert root['Events'].num_entries == cfg.n_background + cfg.n_signal
        events = root['Events'].arrays(library='np')
    test = events['split'] == 2
    np.testing.assert_array_equal(events['event_id'][test], predictions['ids'])
    np.testing.assert_array_equal(events['event_weight'][test], predictions['event_weight'])
    for i in range(4):
        np.testing.assert_array_equal(events[f'x{i + 1}'][test], predictions['x'][:, i])
    root_mtime = (toy / 'events.root').stat().st_mtime_ns
    for model in cfg.models:
        for method in ('plain_ce', 'weighted_ce', 'cmok', 'adaptive', 'balanced_cmok', 'scan_best'):
            restored = load_predict(toy / f'{model}_{method}.pt', predictions['x'])
            np.testing.assert_allclose(restored, predictions[f'{model}__{method}'], rtol=1e-6, atol=1e-6)
    before = (toy / 'metrics.json').stat().st_mtime_ns
    assert run_study(cfg, tmp_path, resume=True) == rows
    assert (toy / 'metrics.json').stat().st_mtime_ns == before
    assert (toy / 'events.root').stat().st_mtime_ns == root_mtime
    with pytest.raises(FileExistsError):
        run_study(cfg, tmp_path)
    cfg.epochs += 1
    with pytest.raises(ValueError, match='different configuration'):
        run_study(cfg, tmp_path, resume=True)


def test_oracle_constant_signed_controls(tmp_path):
    cfg = Config(replicas=1, n_background=400, n_signal=40, epochs=1, weight_epochs=1,
                 batch_size=128, models=['linear'], scan_deltas=[], oracle_weights=True,
                 constant_weights=True, signed_weights=True, save_predictions=False, bootstrap_samples=20, plots=False)
    run_study(cfg, tmp_path)
    diagnostic = json.loads((tmp_path / f'toy_{cfg.seed}' / 'diagnostics.json').read_text())
    with uproot.open(tmp_path / f'toy_{cfg.seed}' / 'events.root') as root:
        assert np.any(root['Events']['event_weight'].array(library='np') < 0)
    assert not (tmp_path / f'toy_{cfg.seed}' / 'predictions.npz').exists()
    assert diagnostic['negative_weight_fraction'] > 0
    assert all(abs(row['relative_mean_bias']) < 1e-7 for row in diagnostic['weight_closure'])
