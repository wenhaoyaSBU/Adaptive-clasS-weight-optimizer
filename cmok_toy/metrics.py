"""Physical-law metrics. Truth weights reduce evaluation noise, not training noise."""
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, roc_curve


def average(x, w):
    return float(np.average(x, weights=w))


def correlation(x, y, w):
    dx, dy = x - average(x, w), y - average(y, w)
    denom = np.sqrt(average(dx**2, w) * average(dy**2, w))
    return average(dx * dy, w) / denom if denom > 0 else None


def weighted_rank(x, w):
    """Weighted mid-CDF rank; equal scores receive equal ranks."""
    order = np.argsort(x, kind='stable')
    values, weights = x[order], w[order]
    starts = np.r_[0, np.flatnonzero(np.diff(values)) + 1]
    mass = np.add.reduceat(weights, starts)
    ranks = (np.cumsum(mass) - .5 * mass) / mass.sum()
    output = np.empty(len(x))
    output[order] = np.repeat(ranks, np.diff(np.r_[starts, len(x)]))
    return output


def physical_weights(sample, truth, rho):
    observed_rho = np.bincount(sample.y, minlength=2) / len(sample.y)
    w = np.exp(truth.log_mean_weights(sample.x)[np.arange(len(sample.y)), sample.y])
    return w * rho[sample.y] / observed_rho[sample.y]


def asimov(y, score, w, edges, signal_yield, background_yield):
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] != 0 or edges[-1] != 1 or np.any(np.diff(edges) <= 0)):
        raise ValueError('Asimov edges must increase strictly from 0 to 1')
    shapes = []
    for k, total in ((0, background_yield), (1, signal_yield)):
        mask = y == k
        counts = np.histogram(expit(score[mask]), bins=edges, weights=w[mask])[0]
        shapes.append(total * counts / counts.sum())
    b, s = shapes
    # Empty-MC bins do not establish a truly zero background. Never invent a floor.
    invalid = (b == 0) & (s > 0)
    nonzero = b > 0
    z = None if invalid.any() else float(np.sqrt(2 * np.sum(
        (s[nonzero] + b[nonzero]) * np.log1p(s[nonzero] / b[nonzero]) - s[nonzero])))
    return z, dict(edges=edges.tolist(), signal=s.tolist(), background=b.tolist(),
                   empty_background_with_signal=int(invalid.sum()))


def evaluate(sample, score, truth, rho, config, *, binning=None):
    if binning is None:
        if config.binning_method != 'fixed':
            raise ValueError('TrafoD evaluation requires binning fitted on validation events')
        binning = dict(method='fixed', optimization_sample=None,
                       edges=np.linspace(0, 1, config.bins + 1).tolist())
    score = np.asarray(score, dtype=np.float64)
    if not np.all(np.isfinite(score)):
        raise FloatingPointError('Nonfinite learned score')
    target = truth.log_lr(sample.x)
    w = physical_weights(sample, truth, rho)
    residual = score - target
    offset = average(residual, w)
    log_odds = score + truth.physical_log_prior_odds(rho)
    truth_prob = truth.posterior(sample.x, rho)
    predicted = expit(log_odds)
    label_ce = np.logaddexp(0, log_odds) - sample.y * log_odds
    truth_ce = np.logaddexp(0, log_odds) - truth_prob * log_odds
    entropy = np.logaddexp(0, target + truth.physical_log_prior_odds(rho)) - truth_prob * (target + truth.physical_log_prior_odds(rho))
    calibration = []
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (predicted >= lo) & (predicted < hi if hi < 1 else predicted <= hi)
        if mask.any():
            calibration.append(dict(predicted=average(predicted[mask], w[mask]),
                                    observed=average(sample.y[mask], w[mask]),
                                    truth=average(truth_prob[mask], w[mask]),
                                    mass=float(w[mask].sum() / w.sum())))
    za, histogram = asimov(sample.y, score, w, binning['edges'], config.signal_yield, config.background_yield)
    histogram.update(method=binning['method'], optimization_sample=binning['optimization_sample'])
    metrics = dict(physical_auc=float(roc_auc_score(sample.y, score, sample_weight=w)),
                   log_lr_mse=average((residual - offset)**2, w),
                   log_lr_mse_raw=average(residual**2, w), log_lr_offset=offset,
                   pearson=correlation(score, target, w),
                   spearman=correlation(weighted_rank(score, w), weighted_rank(target, w), w),
                   physical_ce=average(label_ce, w), truth_posterior_ce=average(truth_ce, w),
                   posterior_kl=average(truth_ce - entropy, w),
                   posterior_mse=average((predicted - truth_prob)**2, w),
                   calibration_ece=sum(row['mass'] * abs(row['predicted'] - row['observed']) for row in calibration),
                   asimov_z=za, asimov_n_bins=len(binning['edges']) - 1,
                   binning_method=binning['method'])
    fpr, tpr, _ = roc_curve(sample.y, score, sample_weight=w)
    detail = dict(calibration=calibration, asimov=histogram,
                  roc_fpr=fpr.tolist(), roc_tpr=tpr.tolist())
    return metrics, detail


def summarize(rows, bootstrap_samples=2000, seed=19):
    """Across-toy uncertainty, including paired method-minus-CMOK differences."""
    rng = np.random.default_rng(seed)
    metrics = ('physical_auc', 'log_lr_mse', 'asimov_z', 'physical_ce', 'posterior_kl', 'alpha_ratio')
    output = []
    for model, method in sorted({(r['model'], r['method']) for r in rows}):
        group = [r for r in rows if r['model'] == model and r['method'] == method]
        baseline = {r['seed']: r for r in rows if r['model'] == model and r['method'] == 'cmok'}
        for metric in metrics:
            for paired in (False, True):
                values = []
                for row in group:
                    value = row.get(metric)
                    base = baseline.get(row['seed'], {}).get(metric)
                    if value is None or (paired and base is None):
                        continue
                    values.append(value - base if paired else value)
                a = np.asarray(values, dtype=float)
                n = len(a)
                if not n:
                    continue
                boot = rng.choice(a, size=(bootstrap_samples, n), replace=True).mean(axis=1) if n > 1 else None
                output.append(dict(model=model, method=method, metric=metric,
                                   comparison='minus_cmok' if paired else 'absolute',
                                   n=n, n_missing=len(group) - n, mean=float(a.mean()),
                                   standard_error=float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
                                   ci_low=float(np.quantile(boot, .025)) if n > 1 else None,
                                   ci_high=float(np.quantile(boot, .975)) if n > 1 else None))
    return output
