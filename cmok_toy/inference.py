"""Apply saved models and the matching CMOK weight head to raw features."""
from pathlib import Path
import numpy as np
import torch
from .models import classifier, WeightHead
from .training import predict
from .truth import Truth


def corrected_log_lr(odds, method, delta, rho, log_z, log_weights):
    log_rho = np.log(rho[1] / rho[0])
    if method == 'plain_ce':
        return odds - log_rho
    norm = log_z[0] - log_z[1]
    if method == 'weighted_ce':
        return odds - np.log(rho[0] / rho[1]) - log_rho + norm
    return odds - delta - log_rho + log_weights[:, 1] - log_weights[:, 0] + norm


def load_predict(checkpoint, x):
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all():
        raise ValueError('Expected finite input with shape (n_events, 4)')
    if not len(x):
        return np.empty(0)
    model = classifier(saved['model'], saved['widths'])
    model.load_state_dict(saved['state_dict'])
    inputs = torch.tensor((x - saved['mean']) / saved['scale'], dtype=torch.float32)
    logits = predict(model, inputs).numpy().astype(np.float64)
    log_w = np.zeros((len(x), 2))
    if saved['method'] not in ('plain_ce', 'weighted_ce'):
        if saved['oracle_weights']:
            log_w = Truth(saved['constant_weights']).log_mean_weights(x)
        else:
            weight = torch.load(checkpoint.parent / 'weight_head.pt', map_location='cpu', weights_only=True)
            head = WeightHead(weight['state_dict']['scales'])
            head.load_state_dict(weight['state_dict'])
            wx = torch.tensor((x - weight['mean']) / weight['scale'], dtype=torch.float32)
            log_w = predict(head, wx).numpy()
    return corrected_log_lr(logits[:, 1] - logits[:, 0], saved['method'], saved['delta'],
                            saved['rho'], saved['log_z'], log_w)
