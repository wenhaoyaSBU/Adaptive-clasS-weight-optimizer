import json
import numpy as np
import pytest
import uproot
from cmok_toy.config import Config
from cmok_toy.root_io import write_events_root, SPLITS
from cmok_toy.truth import Truth, generate


@pytest.mark.parametrize('signed,constant', [(False, False), (True, True)])
def test_all_generated_events_round_trip(tmp_path, signed, constant):
    cfg = Config(n_background=400, n_signal=40, signed_weights=signed, constant_weights=constant)
    truth = Truth(constant)
    samples = generate(truth, cfg.seed, cfg.n_background, cfg.n_signal,
                       cfg.sigma_background, cfg.sigma_signal, signed)
    output = tmp_path / 'events.root'
    write_events_root(output, samples, truth, cfg, cfg.seed, chunk_size=17)
    assert not output.with_suffix('.root.tmp').exists()
    with uproot.open(output) as root:
        tree = root['Events']
        assert tree.classname == 'TTree'
        assert tree.num_entries == 440
        events = tree.arrays(library='np')
        metadata = json.loads(str(root['metadata']))
    assert metadata['seed'] == cfg.seed
    assert metadata['config'] == cfg.to_dict()
    assert metadata['splits'] == SPLITS
    assert len(np.unique(events['event_id'])) == 440
    for name, split_id in SPLITS.items():
        sample = samples[name]
        mask = events['split'] == split_id
        np.testing.assert_array_equal(events['event_id'][mask], sample.ids)
        np.testing.assert_array_equal(events['class_id'][mask], sample.y)
        np.testing.assert_array_equal(events['event_weight'][mask], sample.w)
        for i in range(4):
            np.testing.assert_array_equal(events[f'x{i + 1}'][mask], sample.x[:, i])
            assert events[f'x{i + 1}'].dtype == np.dtype('float64')
        mean_w = np.exp(truth.log_mean_weights(sample.x))
        np.testing.assert_allclose(events['mean_weight'][mask], mean_w[np.arange(len(sample.y)), sample.y], rtol=1e-14)
        np.testing.assert_allclose(events['mean_weight_background'][mask], mean_w[:, 0], rtol=1e-14)
        np.testing.assert_allclose(events['mean_weight_signal'][mask], mean_w[:, 1], rtol=1e-14)
        np.testing.assert_allclose(events['exact_generation_log_lr'][mask], truth.log_lr(sample.x, physical=False), atol=1e-14)
        np.testing.assert_allclose(events['exact_physical_log_lr'][mask], truth.log_lr(sample.x), atol=1e-14)
        np.testing.assert_array_equal(metadata['counts'][name], np.bincount(sample.y, minlength=2))
