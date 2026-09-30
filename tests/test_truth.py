import numpy as np
from cmok_toy.truth import Truth, generate, validate_truth
from cmok_toy.inference import corrected_log_lr


def test_invertible_unit_jacobian():
    u = np.random.default_rng(7).normal(size=(50, 4))
    np.testing.assert_allclose(Truth.inverse(Truth.forward(u)), u, atol=1e-14)
    point, step = u[0], 1e-5
    jac = np.stack([(Truth.forward(point + step * np.eye(4)[k]) - Truth.forward(point - step * np.eye(4)[k])) / (2 * step) for k in range(4)], axis=1)
    np.testing.assert_allclose(np.linalg.det(jac), 1, atol=1e-9)


def test_exact_normalization_matches_mc():
    for check in validate_truth(n=100_000).values():
        assert abs(check['pull']) < 5


def test_physical_sampling_and_density():
    truth, rng = Truth(), np.random.default_rng(24)
    for k in range(2):
        x = truth.sample_x(rng, 80_000, k, physical=True)
        ratio = np.exp(truth.log_z[k] - truth.log_mean_weights(x)[:, k])
        assert abs(ratio.mean() - 1) < 5 * ratio.std(ddof=1) / np.sqrt(len(x))


def test_population_corrections():
    truth = Truth()
    x = truth.sample_x(np.random.default_rng(12), 100, 0)
    rho = np.array([20 / 21, 1 / 21])
    q, log_w = truth.log_q(x), truth.log_mean_weights(x)
    gen_odds = q[:, 1] - q[:, 0] + np.log(rho[1] / rho[0])
    for delta in (-2., 0., 3.):
        result = corrected_log_lr(gen_odds + delta, 'adaptive', delta, rho, truth.log_z, log_w)
        np.testing.assert_allclose(result, truth.log_lr(x), atol=1e-12)
    direct_odds = q[:, 1] - q[:, 0] + log_w[:, 1] - log_w[:, 0]
    result = corrected_log_lr(direct_odds, 'weighted_ce', 0, rho, truth.log_z, log_w)
    np.testing.assert_allclose(result, truth.log_lr(x), atol=1e-12)
    result = corrected_log_lr(gen_odds, 'plain_ce', 0, rho, truth.log_z, log_w)
    np.testing.assert_allclose(result, truth.log_lr(x, physical=False), atol=1e-12)


def test_constant_weight_limit():
    truth = Truth(constant_weights=True)
    x = np.random.default_rng(1).normal(size=(100, 4))
    np.testing.assert_allclose(truth.log_lr(x), truth.log_lr(x, physical=False), atol=1e-12)


def test_reproducible_disjoint_stratified_splits():
    first, second = generate(Truth(), 19, 2000, 100), generate(Truth(), 19, 2000, 100)
    expected = {'train': [1400, 70], 'validation': [300, 15], 'test': [300, 15]}
    for name, sample in first.items():
        np.testing.assert_array_equal(sample.x, second[name].x)
        np.testing.assert_array_equal(np.bincount(sample.y), expected[name])
        assert np.all(sample.w > 0)
    ids = np.concatenate([s.ids for s in first.values()])
    assert len(np.unique(ids)) == len(ids) == 2100


def test_noisy_weights_have_conditional_mean():
    for sample in generate(Truth(), 25, 100_000, 100_000).values():
        mean = np.exp(Truth().log_mean_weights(sample.x)[np.arange(len(sample.y)), sample.y])
        noise = sample.w / mean
        for k in range(2):
            v = noise[sample.y == k]
            assert abs(v.mean() - 1) < 5 * v.std(ddof=1) / np.sqrt(len(v))
