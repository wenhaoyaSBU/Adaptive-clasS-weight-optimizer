import numpy as np
import pytest
from scipy.special import expit
from cmok_toy.binning import fit_binning, trafo_d_edges
from cmok_toy.config import Config
from cmok_toy.metrics import evaluate, asimov
from cmok_toy.truth import Truth, generate


def reference_scan(values, y, weights, config):
    """Independent event-level scan from the workspace's slow TrafoD recipe."""
    cuts = [values.max()]
    total_s, total_b = weights[y == 1].sum(), weights[y == 0].sum()
    for point in np.linspace(values.min(), values.max(), config.trafo_initial_bins)[1:-1][::-1]:
        edges = [point, *cuts]
        s = np.histogram(values[y == 1], edges, weights=weights[y == 1])[0][0]
        b = np.histogram(values[y == 0], edges, weights=weights[y == 0])[0][0]
        b2 = np.histogram(values[y == 0], edges, weights=weights[y == 0]**2)[0][0]
        occupancy = config.trafo_z_signal * s / total_s + config.trafo_z_background * b / total_b
        if (occupancy > 1 and b > 0 and np.sqrt(b2) / b < config.trafo_max_background_relative_error
                and s + b >= config.trafo_min_yield):
            cuts.insert(0, point)
    edges = np.array([values.min(), *cuts])
    edges[0], edges[-1] = 0., 1.
    return edges


@pytest.mark.parametrize('signed', [False, True])
def test_matches_event_level_reference(signed):
    cfg = Config(n_background=8000, n_signal=800, trafo_initial_bins=73, signed_weights=signed)
    truth = Truth()
    sample = generate(truth, 38, cfg.n_background, cfg.n_signal, signed=signed)['validation']
    score = truth.log_lr(sample.x)
    weights = sample.w.copy()
    for k, total in ((0, cfg.background_yield), (1, cfg.signal_yield)):
        weights[sample.y == k] *= total / weights[sample.y == k].sum()
    fitted = fit_binning(sample, score, cfg)
    expected = reference_scan(expit(score), sample.y, weights, cfg)
    assert len(expected) > 2
    np.testing.assert_allclose(fitted['edges'], expected, atol=1e-14)
    assert fitted['optimization_sample'] == 'validation'
    assert sum(fitted['validation_signal']) == pytest.approx(cfg.signal_yield)
    assert sum(fitted['validation_background']) == pytest.approx(cfg.background_yield)
    # Squared observed noisy weights, including signs, control the MC uncertainty.
    assert sum(fitted['validation_background_sumw2']) == pytest.approx(np.sum(weights[sample.y == 0]**2))


def test_thresholds_and_low_score_remainder():
    edges = np.array([0., .5, 1.])
    # Occupancy exactly 1 is not sufficient, and relative error exactly 0.2 fails.
    np.testing.assert_array_equal(trafo_d_edges(edges, [1, 1], [25, 25], [25, 25],
                                              z_signal=1, z_background=1, max_background_relative_error=.3), [0, 1])
    np.testing.assert_array_equal(trafo_d_edges(edges, [1, 1], [25, 25], [25, 25]), [0, 1])
    np.testing.assert_array_equal(trafo_d_edges(edges, [1, 1], [26, 26], [26, 26], min_yield=28), [0, 1])
    # Yield equal to the minimum is accepted. Empty low-score remainder is retained.
    np.testing.assert_array_equal(trafo_d_edges(edges, [0, 1], [0, 26], [0, 26], min_yield=27), edges)


def test_noisy_effective_statistics_merge_bins():
    edges = np.array([0., .5, 1.])
    np.testing.assert_array_equal(trafo_d_edges(edges, [1, 1], [100, 100], [100, 100]), edges)
    np.testing.assert_array_equal(trafo_d_edges(edges, [1, 1], [100, 100], [10000, 10000]), [0, 1])


def test_constant_score_and_insufficient_statistics():
    sample = generate(Truth(), 19, 100, 20)['validation']
    fitted = fit_binning(sample, np.zeros(len(sample.y)), Config())
    assert fitted['edges'] == [0, 1]
    assert fitted['fallback'] == 'constant_validation_score'
    fitted = fit_binning(sample, Truth().log_lr(sample.x), Config())
    assert fitted['edges'] == [0, 1]  # Fewer than 25 background events cannot pass 20%.


def test_test_tails_kept_and_edges_never_refitted():
    cfg, truth = Config(n_background=4000, n_signal=200), Truth()
    samples = generate(truth, 37, cfg.n_background, cfg.n_signal)
    fitted = fit_binning(samples['validation'], truth.log_lr(samples['validation'].x), cfg)
    score = truth.log_lr(samples['test'].x)
    with pytest.raises(ValueError, match='validation'):
        evaluate(samples['test'], score, truth, np.array([.95, .05]), cfg)
    for test_score in (score, score * 20):
        _, detail = evaluate(samples['test'], test_score, truth, np.array([.95, .05]), cfg, binning=fitted)
        assert detail['asimov']['edges'] == fitted['edges']
        assert sum(detail['asimov']['signal']) == pytest.approx(cfg.signal_yield)
        assert sum(detail['asimov']['background']) == pytest.approx(cfg.background_yield)
    _, detail = asimov(np.array([0, 1]), np.array([-1000., 1000.]), np.ones(2), [0., .5, 1.], 100, 1000)
    assert detail['signal'] == [0., 100.]
    assert detail['background'] == [1000., 0.]


@pytest.mark.parametrize('kwargs', [dict(binning_method='unknown'), dict(trafo_initial_bins=2),
                                   dict(trafo_max_background_relative_error=0), dict(trafo_min_yield=-1)])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs).validate()
