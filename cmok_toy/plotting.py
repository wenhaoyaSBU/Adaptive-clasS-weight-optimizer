"""Noninteractive, standalone diagnostic figures (PNG)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


METHODS = ('plain_ce', 'weighted_ce', 'cmok', 'adaptive', 'balanced_cmok', 'scan_best')


def plot_toy(directory, truth_lr, scores, details, histories, scans, weight_truth, weight_pred, y):
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    models = sorted({key.split('/')[0] for key in scores})
    for model in models:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for method in METHODS:
            key = f'{model}/{method}'
            if key not in scores:
                continue
            score, detail = scores[key], details[key]
            axes[0, 0].plot(detail['roc_fpr'], detail['roc_tpr'], label=method)
            # Display deterministic subsample; metric calculations use all events.
            idx = np.linspace(0, len(score) - 1, min(800, len(score))).astype(int)
            axes[0, 1].scatter(truth_lr[idx], score[idx], s=3, alpha=.2, label=method)
            edges = np.linspace(np.quantile(truth_lr, .01), np.quantile(truth_lr, .99), 20)
            centers, residual = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (truth_lr >= lo) & (truth_lr < hi)
                if mask.any():
                    centers.append((lo + hi) / 2)
                    residual.append((score[mask] - truth_lr[mask]).mean())
            axes[1, 0].plot(centers, residual, label=method)
            cal = detail['calibration']
            axes[1, 1].plot([r['predicted'] for r in cal], [r['observed'] for r in cal], '.-', label=method)
        axes[0, 0].set(xlabel='Physical background efficiency', ylabel='Physical signal efficiency', title='Physical ROC')
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].set(xlabel='Exact physical log LR', ylabel='Learned log LR')
        axes[1, 0].set(xlabel='Exact physical log LR', ylabel='Mean residual (Q bins)')
        axes[1, 0].axhline(0, color='black', ls='--')
        axes[1, 1].plot([0, 1], [0, 1], 'k--')
        axes[1, 1].set(xlabel='Predicted physical posterior', ylabel='Physical signal fraction')
        fig.suptitle(model)
        fig.tight_layout()
        fig.savefig(directory / f'{model}_scores.png', dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for key, history in histories.items():
            if key.startswith(model + '/'):
                axes[0].plot([r['epoch'] for r in history], [r['validation_physical_ce'] for r in history], label=key.split('/')[1])
        history = histories[f'{model}/adaptive']
        axes[1].plot([r['epoch'] for r in history], [r['alpha_ratio'] for r in history])
        points = sorted(scans.get(model, []), key=lambda p: p['delta'])
        if points:
            axes[2].plot([p['delta'] for p in points], [p['validation_physical_ce'] for p in points], 'o-')
        chosen = details[f'{model}/adaptive']
        axes[2].plot(chosen['delta'], chosen['validation_physical_ce'], '*', ms=13, label='adaptive checkpoint')
        axes[0].set(xlabel='Epoch', ylabel='Validation physical CE')
        axes[0].legend(fontsize=7)
        axes[1].set(xlabel='Epoch', ylabel='Adaptive alpha_S / alpha_B')
        axes[2].set(xlabel='Fixed delta', ylabel='Best validation physical CE')
        axes[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(directory / f'{model}_training.png', dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for k, ax in enumerate(axes):
        mask = y == k
        t, p = weight_truth[mask, k], weight_pred[mask, k]
        idx = np.linspace(0, len(t) - 1, min(1500, len(t))).astype(int)
        ax.scatter(t[idx], p[idx], s=4, alpha=.3)
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lim, lim, 'k--')
        ax.set(xlabel='Exact conditional mean weight', ylabel='Predicted mean weight', title=('Background', 'Signal')[k])
    fig.tight_layout()
    fig.savefig(directory / 'weight_closure.png', dpi=140)
    plt.close(fig)


def plot_summary(directory, rows):
    models = sorted({r['model'] for r in rows if r['model'] != 'truth'})
    fig, axes = plt.subplots(len(models), 3, figsize=(13, 4 * len(models)), squeeze=False)
    for i, model in enumerate(models):
        group = [r for r in rows if r['model'] == model]
        adaptive = [r for r in group if r['method'] == 'adaptive']
        axes[i, 0].hist([r['alpha_ratio'] for r in adaptive], bins=min(15, max(1, len(adaptive))))
        axes[i, 0].set(xlabel='Adaptive alpha_S / alpha_B', ylabel='Toys', title=model)
        baseline = {r['seed']: r for r in group if r['method'] == 'cmok'}
        delta = [r['log_lr_mse'] - baseline[r['seed']]['log_lr_mse'] for r in adaptive]
        axes[i, 1].hist(delta, bins=min(15, max(1, len(delta))))
        axes[i, 1].axvline(0, color='black', ls='--')
        axes[i, 1].set(xlabel='Adaptive minus CMOK log-LR MSE', ylabel='Toys')
        axes[i, 2].scatter([r['neff_signal'] / r['neff_background'] for r in adaptive], delta)
        axes[i, 2].set(xlabel='Training N_eff,S / N_eff,B', ylabel='Adaptive minus CMOK MSE')
    fig.tight_layout()
    fig.savefig(Path(directory) / 'replica_summary.png', dpi=140)
    plt.close(fig)
