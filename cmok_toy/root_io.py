"""Portable ROOT TTrees of every generated event, before any preprocessing."""
import json
from pathlib import Path
import numpy as np
import uproot


SPLITS = {'train': 0, 'validation': 1, 'test': 2}
BRANCH_TYPES = {
    'event_id': 'int64',
    'class_id': 'int32',
    'split': 'int32',
    **{f'x{i}': 'float64' for i in range(1, 5)},
    'event_weight': 'float64',
    'mean_weight': 'float64',
    'mean_weight_background': 'float64',
    'mean_weight_signal': 'float64',
    'exact_generation_log_lr': 'float64',
    'exact_physical_log_lr': 'float64',
}


def write_events_root(path, samples, truth, config, seed, chunk_size=100_000):
    """Save all splits as one TTree, using bounded-memory, lossless chunks.

    Called immediately after generation, so events survive a later training failure.
    A temporary file is atomically renamed only after ROOT closes successfully.
    No random numbers are consumed and the input samples are never modified.
    """
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    metadata = {
        'schema_version': 1,
        'seed': seed,
        'config': config.to_dict(),
        'classes': {'background': 0, 'signal': 1},
        'splits': SPLITS,
        'counts': {name: np.bincount(sample.y, minlength=2).tolist()
                   for name, sample in samples.items()},
        'log_z': truth.log_z.tolist(),
        'means': truth.means.tolist(),
        'covariances': truth.covariances.tolist(),
        'weight_coefficients': truth.coefficients.tolist(),
        'feature_space': 'raw generated observables, before standardization',
        'mean_weight': 'exact E[W|X,Y] for the event class; not a fitted prediction',
    }
    try:
        with uproot.recreate(temporary, compression=uproot.ZLIB(4)) as output:
            output['metadata'] = json.dumps(metadata, allow_nan=False)
            # Explicit mktree guarantees a TTree, including with newer uproot defaults.
            tree = output.mktree('Events', BRANCH_TYPES, title='All generated toy MC events')
            for name, split_id in SPLITS.items():
                sample = samples[name]
                for start in range(0, len(sample.y), chunk_size):
                    block = slice(start, start + chunk_size)
                    x, y = sample.x[block], sample.y[block]
                    mean_weights = np.exp(truth.log_mean_weights(x))
                    tree.extend({
                        'event_id': sample.ids[block],
                        'class_id': y.astype(np.int32),
                        'split': np.full(len(y), split_id, dtype=np.int32),
                        **{f'x{i + 1}': x[:, i] for i in range(4)},
                        'event_weight': sample.w[block],
                        'mean_weight': mean_weights[np.arange(len(y)), y],
                        'mean_weight_background': mean_weights[:, 0],
                        'mean_weight_signal': mean_weights[:, 1],
                        'exact_generation_log_lr': truth.log_lr(x, physical=False),
                        'exact_physical_log_lr': truth.log_lr(x),
                    })
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
