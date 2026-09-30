import copy
import numpy as np
import pytest
import torch
from cmok_toy.training import meta_objective, class_factors
from cmok_toy.metrics import evaluate, summarize, asimov, weighted_rank
from cmok_toy.truth import Truth, generate
from cmok_toy.config import Config


@pytest.mark.parametrize('kind', ['linear', 'mlp'])
def test_meta_gradient_matches_finite_difference(kind):
    from cmok_toy.models import classifier
    torch.manual_seed(31)
    model = classifier(kind, [5, 4]).double()
    x, vx = torch.randn(12, 4, dtype=torch.float64), torch.randn(10, 4, dtype=torch.float64)
    y, vy = torch.tensor([0] * 9 + [1] * 3), torch.tensor([0] * 7 + [1] * 3)
    rho, w = torch.tensor([.75, .25], dtype=torch.float64), torch.linspace(.4, 1.6, 10, dtype=torch.float64)
    log_w = torch.randn(10, 2, dtype=torch.float64) * .3
    def objective(d):
        return meta_objective(model, d, x, y, vx, vy, w, log_w, rho, .07, .003, .01)
    delta = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    grad, = torch.autograd.grad(objective(delta), delta)
    eps = 1e-5
    numeric = (objective(delta.detach() + eps) - objective(delta.detach() - eps)) / (2 * eps)
    torch.testing.assert_close(grad, numeric, rtol=1e-6, atol=1e-8)
    def actual_step(d):
        candidate = copy.deepcopy(model)
        optimizer = torch.optim.SGD(candidate.parameters(), lr=.07, weight_decay=.003)
        ce = torch.nn.functional.cross_entropy(candidate(x), y, reduction='none')
        (ce * class_factors(d, rho)[y]).mean().backward()
        optimizer.step()
        logits = candidate(vx) - torch.stack((-d / 2, d / 2)) + log_w
        ce = torch.nn.functional.cross_entropy(logits, vy, reduction='none')
        return (ce * w).mean() + .01 * d.square()
    actual_numeric = (actual_step(delta.detach() + eps) - actual_step(delta.detach() - eps)) / (2 * eps)
    torch.testing.assert_close(grad, actual_numeric, rtol=1e-6, atol=1e-8)


def test_class_balance_means_event_average():
    rho = torch.tensor([.95, .05], dtype=torch.float64)
    torch.testing.assert_close(class_factors(torch.tensor(0., dtype=torch.float64), rho), torch.ones(2, dtype=torch.float64))
    delta = torch.log(rho[0] / rho[1])
    torch.testing.assert_close(rho * class_factors(delta, rho), torch.full((2,), .5, dtype=torch.float64))


def test_oracle_metrics_and_constant_shift():
    truth = Truth()
    sample = generate(truth, 42, 4000, 200)['test']
    rho = np.array([20 / 21, 1 / 21])
    target = truth.log_lr(sample.x)
    metric, _ = evaluate(sample, target, truth, rho, Config(binning_method='fixed', bins=3))
    shifted, _ = evaluate(sample, target + 5, truth, rho, Config(binning_method='fixed', bins=3))
    assert metric['log_lr_mse'] < 1e-25
    assert metric['posterior_kl'] == 0
    assert metric['posterior_mse'] == 0
    assert metric['pearson'] == pytest.approx(1)
    assert shifted['log_lr_mse'] < 1e-25
    assert shifted['log_lr_mse_raw'] == pytest.approx(25)
    assert shifted['posterior_kl'] > 0
    assert shifted['physical_auc'] == metric['physical_auc']


def test_empty_background_not_infinite_significance():
    z, detail = asimov(np.array([0, 1]), np.array([-10., 10.]), np.ones(2), [0., .5, 1.], 100, 1000)
    assert z is None
    assert detail['empty_background_with_signal'] == 1


def test_paired_summary_and_one_toy_uncertainty():
    rows = [dict(seed=i, model='linear', method=m, physical_auc=i + delta) for i in range(4) for m, delta in [('cmok', 0), ('adaptive', .2)]]
    summary = summarize(rows, 100)
    paired = next(r for r in summary if r['method'] == 'adaptive' and r['comparison'] == 'minus_cmok')
    assert paired['mean'] == pytest.approx(.2)
    assert paired['standard_error'] == pytest.approx(0, abs=1e-15)
    one = summarize(rows[:1], 100)[0]
    assert one['standard_error'] is None and one['ci_low'] is None


def test_ties_in_weighted_ranks():
    ranks = weighted_rank(np.array([1., 1., 3.]), np.array([1., 3., 4.]))
    np.testing.assert_allclose(ranks, [.25, .25, .75])
