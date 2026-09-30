"""Validation-only TrafoD binning for significance templates.

Adapted from VHbbccTransformer/salt-framework/scr/binning_study_utils.py,
trafoD_optimized_binning (standard z_signal=10, z_background=5 settings).
"""
import numpy as np
from scipy.special import expit


def trafo_d_edges(fine_edges, signal, background, background_sumw2, *,
                  z_signal=10., z_background=5., max_background_relative_error=.2,
                  min_yield=3.):
    """Scan high to low, closing a bin only when all three conditions pass.

    As in the workspace reference, the low-score remainder is retained as its
    own bin and is not required to pass the conditions. No events are dropped.
    """
    fine_edges = np.asarray(fine_edges, dtype=float)
    signal, background, background_sumw2 = (
        np.asarray(values, dtype=float) for values in (signal, background, background_sumw2))
    if (fine_edges.ndim != 1 or len(fine_edges) < 2
            or not np.isfinite(fine_edges).all() or np.any(np.diff(fine_edges) <= 0)):
        raise ValueError('Fine edges must be finite and strictly increasing')
    for values in (signal, background, background_sumw2):
        if values.shape != (len(fine_edges) - 1,) or not np.isfinite(values).all():
            raise ValueError('Invalid fine-bin histogram')
    if np.any(background_sumw2 < 0):
        raise ValueError('Background sumw2 cannot be negative')
    total_s, total_b = signal.sum(), background.sum()
    if total_s <= 0 or total_b <= 0:
        return fine_edges[[0, -1]]
    cuts = [len(fine_edges) - 1]
    s = b = b2 = 0.
    for index in range(len(signal) - 1, 0, -1):
        s += signal[index]
        b += background[index]
        b2 += background_sumw2[index]
        occupancy = z_signal * s / total_s + z_background * b / total_b
        relative_error = np.sqrt(b2) / b if b > 0 else np.inf
        if occupancy > 1 and relative_error < max_background_relative_error and s + b >= min_yield:
            cuts.append(index)
            s = b = b2 = 0.
    return fine_edges[[0, *reversed(cuts)]]


def fit_binning(sample, score, config):
    """Use noisy validation W (including sum W²), never test data or truth weights.

    Normalize each class to the same configured analysis yields used for test
    templates. Extend the outer edges to [0,1] so unseen test tails are retained.
    """
    if config.binning_method == 'fixed':
        return dict(method='fixed', optimization_sample=None,
                    edges=np.linspace(0, 1, config.bins + 1).tolist())
    score = np.asarray(score, dtype=float)
    if score.shape != sample.y.shape or not len(score) or not np.isfinite(score).all():
        raise ValueError('Expected one finite validation score per event')
    values = expit(score)
    settings = dict(initial_bins=config.trafo_initial_bins, z_signal=config.trafo_z_signal,
                    z_background=config.trafo_z_background,
                    max_background_relative_error=config.trafo_max_background_relative_error,
                    min_yield=config.trafo_min_yield)
    result = dict(method='trafoD', optimization_sample='validation', weight_source='observed_event_weight',
                  settings=settings, edges=[0., 1.], fallback=None)
    weights = np.zeros(len(score))
    for k, total in ((0, config.background_yield), (1, config.signal_yield)):
        mask = sample.y == k
        class_sum = sample.w[mask].sum()
        if not np.isfinite(class_sum) or class_sum <= 0:
            result['fallback'] = 'nonpositive_validation_class_weight_sum'
            return result
        weights[mask] = sample.w[mask] * total / class_sum
    lo, hi = float(values.min()), float(values.max())
    result['validation_score_range'] = [lo, hi]
    if lo == hi:
        result['fallback'] = 'constant_validation_score'
        edges = np.array([0., 1.])
    else:
        # initial_bins is the number of grid POINTS, matching the local reference.
        fine_edges = np.linspace(lo, hi, config.trafo_initial_bins)
        mask = sample.y == 1
        s = np.histogram(values[mask], fine_edges, weights=weights[mask])[0]
        b = np.histogram(values[~mask], fine_edges, weights=weights[~mask])[0]
        b2 = np.histogram(values[~mask], fine_edges, weights=weights[~mask]**2)[0]
        edges = trafo_d_edges(fine_edges, s, b, b2,
                             z_signal=config.trafo_z_signal, z_background=config.trafo_z_background,
                             max_background_relative_error=config.trafo_max_background_relative_error,
                             min_yield=config.trafo_min_yield)
        edges[0], edges[-1] = 0., 1.
    result['edges'] = edges.tolist()
    mask = sample.y == 1
    s = np.histogram(values[mask], edges, weights=weights[mask])[0]
    b = np.histogram(values[~mask], edges, weights=weights[~mask])[0]
    b2 = np.histogram(values[~mask], edges, weights=weights[~mask]**2)[0]
    result['validation_signal'] = s.tolist()
    result['validation_background'] = b.tolist()
    result['validation_background_sumw2'] = b2.tolist()
    result['validation_background_relative_error'] = [float(np.sqrt(v) / w) if w > 0 else None for v, w in zip(b2, b)]
    return result
