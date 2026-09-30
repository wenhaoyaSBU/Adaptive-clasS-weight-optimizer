from dataclasses import dataclass, field, asdict
import json
import math


@dataclass
class Config:
    seed: int = 1729
    replicas: int = 50
    n_background: int = 1_000_000
    n_signal: int = 50_000
    sigma_background: float = .4
    sigma_signal: float = 1.
    epochs: int = 30
    weight_epochs: int = 30
    batch_size: int = 2048
    learning_rate: float = .05
    weight_learning_rate: float = .002
    weight_decay: float = .0001
    meta_learning_rate: float = .005
    meta_every: int = 5
    meta_warmup_epochs: int = 3
    meta_ema: float = .9
    meta_penalty: float = .001
    delta_clip: float = 5.
    delta_initial: float = 0.
    scan_deltas: list[float] = field(default_factory=lambda: [-2., 0., 1., 2., 3., 4.])
    models: list[str] = field(default_factory=lambda: ['linear', 'mlp'])
    widths: list[int] = field(default_factory=lambda: [64, 64, 32])
    constant_weights: bool = False
    oracle_weights: bool = False
    signed_weights: bool = False
    binning_method: str = 'trafoD'
    bins: int = 8  # Only used for optional legacy fixed-binning comparisons.
    trafo_initial_bins: int = 500  # Candidate edge points, not final bin count.
    trafo_z_signal: float = 10.
    trafo_z_background: float = 5.
    trafo_max_background_relative_error: float = .2
    trafo_min_yield: float = 3.
    signal_yield: float = 100.
    background_yield: float = 10_000.
    bootstrap_samples: int = 2000
    threads: int = 2
    device: str = 'cpu'
    save_predictions: bool = True
    plots: bool = True

    def validate(self):
        integer_positive = ('n_background', 'n_signal', 'replicas', 'epochs', 'weight_epochs', 'batch_size', 'meta_every',
                            'bins', 'trafo_initial_bins', 'bootstrap_samples', 'threads')
        for name in integer_positive:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.binning_method not in ('trafoD', 'fixed'):
            raise ValueError('binning_method must be trafoD or fixed')
        if self.trafo_initial_bins < 3 or self.trafo_min_yield < 0:
            raise ValueError('TrafoD needs at least 3 grid points and nonnegative minimum yield')
        if min(self.n_background, self.n_signal) < 20:
            raise ValueError('Need at least 20 generated events per class')
        for name in ('learning_rate', 'weight_learning_rate', 'meta_learning_rate', 'delta_clip',
                     'signal_yield', 'background_yield', 'trafo_z_signal', 'trafo_z_background',
                     'trafo_max_background_relative_error'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if not 0 <= self.meta_ema < 1 or self.meta_warmup_epochs < 0:
            raise ValueError('Invalid meta EMA or warmup')
        if min(self.sigma_background, self.sigma_signal, self.weight_decay, self.meta_penalty) < 0:
            raise ValueError('Noise and regularization must be nonnegative')
        if abs(self.delta_initial) > self.delta_clip:
            raise ValueError('Initial delta is outside clipping bounds')
        if not self.models or len(set(self.models)) != len(self.models) or any(m not in ('linear', 'mlp') for m in self.models):
            raise ValueError('models must contain linear and/or mlp')
        if not self.widths or any(not isinstance(w, int) or w < 1 for w in self.widths):
            raise ValueError('widths must be positive')
        for name, value in self.to_dict().items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if any(not math.isfinite(d) or abs(d) > 30 for d in self.scan_deltas):
            raise ValueError('scan_deltas must be finite and within [-30, 30]')
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        return self

    @classmethod
    def read(cls, path=None):
        if path is None:
            return cls().validate()
        with open(path) as f:
            return cls(**json.load(f)).validate()

    def to_dict(self):
        return asdict(self)
