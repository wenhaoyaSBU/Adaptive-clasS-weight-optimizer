"""Training and a fully differentiated one-step SGD hypergradient."""
import copy
import numpy as np
import torch
from torch.func import functional_call
from torch.nn import functional as F
from .models import classifier, WeightHead


def log_alpha(delta):
    return torch.stack((-delta / 2, delta / 2))


def class_factors(delta, rho):
    a = log_alpha(delta).exp()
    # Fix expected loss scale as delta changes; this normalization is differentiated.
    return a / (rho * a).sum()


def weighted_ce(logits, y, weights):
    return (F.cross_entropy(logits, y, reduction='none') * weights).mean()


def meta_objective(model, delta, train_x, train_y, val_x, val_y, val_w,
                   val_log_weights, rho, learning_rate, weight_decay=0., penalty=0.):
    """Explicit prior derivative AND mixed derivative through theta' are retained.

    The optimizer matched here is ordinary SGD (no momentum); train minibatches
    follow Q, so class frequencies are already present in the sample average.
    """
    params = dict(model.named_parameters())
    loss = weighted_ce(model(train_x), train_y, class_factors(delta, rho)[train_y])
    grads = torch.autograd.grad(loss, tuple(params.values()), create_graph=True)
    fast = {name: p - learning_rate * (g + weight_decay * p)
            for (name, p), g in zip(params.items(), grads)}
    logits = functional_call(model, fast, (val_x,))
    physical_logits = logits - log_alpha(delta) + val_log_weights.detach()
    return weighted_ce(physical_logits, val_y, val_w) + penalty * delta.square()


def tensors(sample, mean, scale, device):
    return {
        'x': torch.tensor((sample.x - mean) / scale, dtype=torch.float32, device=device),
        'y': torch.tensor(sample.y, dtype=torch.long, device=device),
        'w': torch.tensor(sample.w, dtype=torch.float32, device=device),
    }


@torch.no_grad()
def predict(model, x, batch_size=8192):
    model.eval()
    return torch.cat([model(batch) for batch in x.split(batch_size)])


def fit_weight_head(train, validation, config, seed):
    torch.manual_seed(seed)
    scales = torch.stack([train['w'][train['y'] == k].mean() for k in range(2)])
    if torch.any(scales <= 0):
        raise ValueError('Nonpositive empirical class mean: increase statistics for signed-weight study')
    model = WeightHead(scales.cpu()).to(train['x'].device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.weight_learning_rate)
    rho = torch.bincount(train['y'], minlength=2) / len(train['y'])
    rng = np.random.default_rng(seed)
    history, best, best_state = [], float('inf'), None
    for epoch in range(config.weight_epochs):
        model.train()
        total = 0.
        for idx in np.array_split(rng.permutation(len(train['y'])),
                                  max(1, int(np.ceil(len(train['y']) / config.batch_size)))):
            y = train['y'][idx]
            log_w = model(train['x'][idx]).gather(1, y[:, None]).squeeze(1)
            # MSE in LINEAR weight space, scaled by a class constant only.
            residual = (log_w.exp() - train['w'][idx]) / scales[y]
            loss = (residual.square() / (2 * rho[y])).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Weight regression diverged')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        with torch.no_grad():
            y = validation['y']
            estimate = predict(model, validation['x']).gather(1, y[:, None]).squeeze(1).exp()
            residual = (estimate - validation['w']) / scales[y]
            val = sum(residual[y == k].square().mean().item() for k in range(2)) / 2
        history.append(dict(epoch=epoch + 1, train_mse=total / len(train['y']), validation_mse=val))
        if val < best:
            best, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, history


def physical_logits(logits, method, delta, rho, log_weights):
    if method == 'weighted_ce':
        # Remove the balancing tilt; event weights have already entered training.
        c = 1 / (2 * rho)
        return logits - c.log()
    return logits - log_alpha(delta) + log_weights


def fit_classifier(kind, method, train, validation, val_log_weights, rho, config, seed,
                   fixed_delta=0.):
    torch.manual_seed(seed)
    model = classifier(kind, config.widths).to(train['x'].device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                                weight_decay=config.weight_decay)
    adaptive = method == 'adaptive'
    delta = torch.tensor(config.delta_initial if adaptive else fixed_delta,
                         dtype=torch.float32, device=train['x'].device)
    rng = np.random.default_rng(seed + 1)
    meta_rng = np.random.default_rng(seed + 2)
    # Validation samples may have slightly different fractions after integer splitting.
    val_rho = torch.bincount(validation['y'], minlength=2) / len(validation['y'])
    val_w = validation['w'] * rho[validation['y']] / val_rho[validation['y']]
    if val_w.mean() <= 0:
        raise ValueError('Physical validation measure has nonpositive total weight')
    val_w = val_w / val_w.mean()
    direct_w = train['w'] / (2 * rho[train['y']])
    if direct_w.mean() <= 0:
        raise ValueError('Weighted training measure has nonpositive total weight')
    direct_w = direct_w / direct_w.mean()
    history, best_state, best_delta = [], None, None
    best, ema, step = float('inf'), 0., 0
    for epoch in range(config.epochs):
        model.train()
        total, total_ce, hypergrads = 0., 0., []
        for idx in np.array_split(rng.permutation(len(train['y'])),
                                  max(1, int(np.ceil(len(train['y']) / config.batch_size)))):
            x, y = train['x'][idx], train['y'][idx]
            if adaptive and epoch >= config.meta_warmup_epochs and step % config.meta_every == 0:
                vi = meta_rng.choice(len(validation['y']), size=min(config.batch_size, len(validation['y'])), replace=False)
                d = delta.detach().requires_grad_(True)
                outer = meta_objective(model, d, x, y, validation['x'][vi], validation['y'][vi],
                                       val_w[vi], val_log_weights[vi], rho,
                                       config.learning_rate, config.weight_decay, config.meta_penalty)
                hypergrad, = torch.autograd.grad(outer, d)
                if not torch.isfinite(hypergrad):
                    raise FloatingPointError('Nonfinite adaptive hypergradient')
                ema = config.meta_ema * ema + (1 - config.meta_ema) * hypergrad.item()
                delta = (delta - config.meta_learning_rate * ema).clamp(-config.delta_clip, config.delta_clip).detach()
                hypergrads.append(hypergrad.item())
            logits = model(x)
            ce = F.cross_entropy(logits, y, reduction='none')
            factors = direct_w[idx] if method == 'weighted_ce' else class_factors(delta, rho)[y]
            loss = (ce * factors).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Classifier loss diverged: {kind}/{method}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
            total_ce += ce.detach().sum().item()
            step += 1
        with torch.no_grad():
            logits = predict(model, validation['x'])
            phys = physical_logits(logits, method, delta, rho, val_log_weights)
            val = weighted_ce(phys, validation['y'], val_w).item()
            gen_losses = F.cross_entropy(logits - log_alpha(delta), validation['y'], reduction='none')
            if method == 'weighted_ce':
                gen_losses = None  # Event-weighted CE does not estimate Q.
        row = dict(epoch=epoch + 1, train_loss=total / len(train['y']),
                   train_ce=total_ce / len(train['y']), validation_physical_ce=val,
                   delta=delta.item(), alpha_ratio=float(delta.exp()),
                   hypergradient=float(np.mean(hypergrads)) if hypergrads else None)
        if gen_losses is not None:
            row['validation_generation_ce'] = gen_losses.mean().item()
            for k in range(2):
                row[f'validation_ce_class_{k}'] = gen_losses[validation['y'] == k].mean().item()
        # Per-class gradient norms from a deterministic training-only diagnostic subset.
        for k in range(2):
            sub = torch.where(train['y'] == k)[0][:min(1024, config.batch_size)]
            loss = F.cross_entropy(model(train['x'][sub]), train['y'][sub])
            row[f'train_ce_class_{k}'] = loss.item()
            grads = torch.autograd.grad(loss, tuple(model.parameters()))
            row[f'gradient_norm_class_{k}'] = sum(g.square().sum().item() for g in grads)**.5
        history.append(row)
        if val < best:
            best, best_state, best_delta = val, copy.deepcopy(model.state_dict()), delta.item()
    model.load_state_dict(best_state)
    return model, best_delta, history, best
