"""Matched-seed sample-size, model-capacity, and weight-noise control sweeps."""
from dataclasses import replace
from pathlib import Path
import numpy as np
from .study import run_study, write_csv
from .metrics import summarize
import matplotlib.pyplot as plt


def run_sweep(config, output, axis, values, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for value in values:
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Sweep multipliers must be finite and positive')
        if axis == 'samples':
            variant = replace(config, n_background=round(config.n_background * value), n_signal=round(config.n_signal * value))
        elif axis == 'capacity':
            variant = replace(config, models=['mlp'], widths=[max(1, round(w * value)) for w in config.widths])
        else:
            variant = replace(config, sigma_signal=config.sigma_signal * value, sigma_background=config.sigma_background * value)
        results = run_study(variant, output / f'{axis}_{value:g}', resume)
        rows.extend(dict(axis=axis, multiplier=value, **r) for r in summarize(results, config.bootstrap_samples, config.seed))
        write_csv(output / 'sweep.csv', rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for model in config.models:
        for method in ('cmok', 'adaptive', 'weighted_ce'):
            for ax, metric in zip(axes, ('physical_auc', 'log_lr_mse', 'asimov_z')):
                series = sorted([r for r in rows if r['model'] == model and r['method'] == method and r['metric'] == metric and r['comparison'] == 'absolute'], key=lambda r: r['multiplier'])
                if series:
                    ax.errorbar([r['multiplier'] for r in series], [r['mean'] for r in series], yerr=[r['standard_error'] or 0 for r in series], marker='o', label=f'{model}/{method}')
                ax.set(xlabel=f'{axis} multiplier', ylabel=metric)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / f'{axis}_performance.png', dpi=140)
    plt.close(fig)
