"""Reproducible orchestration, checkpoints, resumable replicas, and reports."""
from pathlib import Path
import csv
import hashlib
import json
import platform
import time
import numpy as np
import torch
import uproot
from .truth import Truth, generate, effective_size
from .root_io import write_events_root
from .training import tensors, fit_weight_head, fit_classifier, predict
from .metrics import evaluate, summarize
from .binning import fit_binning
from .inference import corrected_log_lr
from .plotting import plot_toy, plot_summary


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_toy(config, seed, directory):
    start = time.monotonic()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    truth = Truth(config.constant_weights)
    samples = generate(truth, seed, config.n_background, config.n_signal,
                       config.sigma_background, config.sigma_signal, config.signed_weights)
    write_events_root(directory / 'events.root', samples, truth, config, seed)
    train_sample, val_sample = samples['train'], samples['validation']
    mean, scale = train_sample.x.mean(axis=0), train_sample.x.std(axis=0)
    train = tensors(train_sample, mean, scale, config.device)
    validation = tensors(val_sample, mean, scale, config.device)
    rho = torch.bincount(train['y'], minlength=2) / len(train['y'])
    rho_np = rho.cpu().numpy().astype(float)
    diagnostics = dict(seed=seed, rho=rho_np.tolist(), log_z=truth.log_z.tolist(),
                       physical_prior=(rho_np * np.exp(truth.log_z) / np.dot(rho_np, np.exp(truth.log_z))).tolist(),
                       counts={name: np.bincount(s.y, minlength=2).tolist() for name, s in samples.items()},
                       training_neff=[effective_size(train_sample.w[train_sample.y == k]) for k in range(2)],
                       negative_weight_fraction=float((train_sample.w < 0).mean()))
    weight_history = []
    if config.oracle_weights:
        head = None
        val_log_w = torch.tensor(truth.log_mean_weights(val_sample.x), dtype=torch.float32, device=config.device)
    else:
        head, weight_history = fit_weight_head(train, validation, config, seed + 100)
        val_log_w = predict(head, validation['x']).detach()
        torch.save({'state_dict': head.cpu().state_dict(), 'mean': mean.tolist(), 'scale': scale.tolist()},
                   directory / 'weight_head.pt')
        head.to(config.device)
    write_json(directory / 'weight_history.json', weight_history)
    fits, histories, scans = {}, {}, {}
    for model_index, kind in enumerate(config.models):
        model_seed = seed + 1000 * (model_index + 1)
        print(f'  seed={seed} model={kind}: CMOK, weighted CE, adaptive, fixed controls', flush=True)
        base = fit_classifier(kind, 'cmok', train, validation, val_log_w, rho, config, model_seed)
        fits[f'{kind}/cmok'] = base
        # Plain and CMOK use exactly the same generation classifier and checkpoint.
        fits[f'{kind}/plain_ce'] = base
        fits[f'{kind}/weighted_ce'] = fit_classifier(kind, 'weighted_ce', train, validation, val_log_w, rho, config, model_seed)
        fits[f'{kind}/adaptive'] = fit_classifier(kind, 'adaptive', train, validation, val_log_w, rho, config, model_seed)
        balance_delta = float(np.log(rho_np[0] / rho_np[1]))
        balanced = fit_classifier(kind, 'fixed', train, validation, val_log_w, rho, config, model_seed, balance_delta)
        fits[f'{kind}/balanced_cmok'] = balanced
        scan_fits = {0.: base, balance_delta: balanced}
        for delta in config.scan_deltas:
            if float(delta) not in scan_fits:
                scan_fits[float(delta)] = fit_classifier(kind, 'fixed', train, validation, val_log_w, rho, config, model_seed, float(delta))
        scans[kind] = [dict(delta=d, validation_physical_ce=fit[3]) for d, fit in scan_fits.items()]
        # Selection uses noisy held-out physical CE only, never a test/truth metric.
        best_scan = min(scan_fits, key=lambda d: scan_fits[d][3])
        fits[f'{kind}/scan_best'] = scan_fits[best_scan]
        for delta, fit in scan_fits.items():
            write_json(directory / f'{kind}_scan_delta_{delta:.6g}_history.json', fit[2])
    # Freeze method-specific significance edges using validation events only.
    binnings = {}
    validation_log_w = val_log_w.cpu().numpy()
    for key, (model, delta, _, _) in fits.items():
        logits = predict(model, validation['x']).cpu().numpy().astype(np.float64)
        score = corrected_log_lr(logits[:, 1] - logits[:, 0], key.split('/')[1],
                                 delta, rho_np, truth.log_z, validation_log_w)
        binnings[key] = fit_binning(val_sample, score, config)
    for physical, label in ((False, 'exact_generation'), (True, 'exact_physical')):
        binnings[f'truth/{label}'] = fit_binning(val_sample, truth.log_lr(val_sample.x, physical=physical), config)
    write_json(directory / 'binning.json', binnings)
    # Training, model/scan selection and significance binning are complete before testing.
    test_sample = samples['test']
    test = tensors(test_sample, mean, scale, config.device)
    log_w = truth.log_mean_weights(test_sample.x) if head is None else predict(head, test['x']).cpu().numpy()
    weight_truth, weight_pred = np.exp(truth.log_mean_weights(test_sample.x)), np.exp(log_w)
    closure = []
    for k in range(2):
        mask = test_sample.y == k
        expected, prediction = weight_truth[mask, k], weight_pred[mask, k]
        closure.append(dict(class_id=k, relative_mean_bias=float(prediction.mean() / expected.mean() - 1),
                            mean_squared_error=float(np.mean((prediction - expected)**2)),
                            log_mean_squared_error=float(np.mean((np.log(prediction) - np.log(expected))**2)),
                            observed_mean=float(test_sample.w[mask].mean()),
                            predicted_mean=float(prediction.mean()), exact_mean=float(expected.mean())))
    diagnostics['weight_closure'] = closure
    if head is not None:
        centered_log_w = log_w - head.scales.log().cpu().numpy()
        diagnostics['weight_head_saturation_fraction'] = float((np.abs(centered_log_w) >= 14.999).mean())
    scores, details, rows = {}, {}, []
    for key, (model, delta, history, objective) in fits.items():
        kind, method = key.split('/')
        logits = predict(model, test['x']).cpu().numpy().astype(np.float64)
        odds = logits[:, 1] - logits[:, 0]
        alpha_ratio = float(np.exp(delta))
        if method == 'weighted_ce':
            alpha_ratio = float(rho_np[0] / rho_np[1])
        score = corrected_log_lr(odds, method, delta, rho_np, truth.log_z, log_w)
        metrics, detail = evaluate(test_sample, score, truth, rho_np, config, binning=binnings[key])
        row = dict(seed=seed, model=kind, method=method, **metrics, alpha_ratio=alpha_ratio,
                   delta=float(np.log(alpha_ratio)), validation_physical_ce=objective,
                   neff_background=diagnostics['training_neff'][0], neff_signal=diagnostics['training_neff'][1])
        rows.append(row)
        detail.update(delta=float(delta), validation_physical_ce=objective)
        details[key], scores[key], histories[key] = detail, score, history
        torch.save({'state_dict': {name: value.cpu() for name, value in model.state_dict().items()},
                    'model': kind, 'method': method, 'widths': config.widths,
                    'delta': delta, 'alpha_ratio': alpha_ratio, 'rho': rho_np.tolist(),
                    'log_z': truth.log_z.tolist(), 'mean': mean.tolist(), 'scale': scale.tolist(),
                    'constant_weights': config.constant_weights, 'oracle_weights': config.oracle_weights},
                   directory / f'{kind}_{method}.pt')
    for physical, label in ((False, 'exact_generation'), (True, 'exact_physical')):
        score = truth.log_lr(test_sample.x, physical=physical)
        metrics, _ = evaluate(test_sample, score, truth, rho_np, config, binning=binnings[f'truth/{label}'])
        rows.append(dict(seed=seed, model='truth', method=label, **metrics, alpha_ratio=1.))
    write_json(directory / 'diagnostics.json', diagnostics)
    write_json(directory / 'histories.json', histories)
    write_json(directory / 'scans.json', scans)
    # Downsample only stored ROC coordinates, after all metrics were evaluated.
    for detail in details.values():
        idx = np.unique(np.linspace(0, len(detail['roc_fpr']) - 1, min(2000, len(detail['roc_fpr']))).astype(int))
        for name in ('roc_fpr', 'roc_tpr'):
            detail[name] = np.asarray(detail[name])[idx].tolist()
    write_json(directory / 'metric_details.json', details)
    if config.save_predictions:
        np.savez_compressed(directory / 'predictions.npz', x=test_sample.x, y=test_sample.y,
                            event_weight=test_sample.w, ids=test_sample.ids,
                            exact_physical_log_lr=truth.log_lr(test_sample.x),
                            exact_generation_log_lr=truth.log_lr(test_sample.x, physical=False),
                            log_mean_weight=log_w, **{key.replace('/', '__'): value for key, value in scores.items()})
    if config.plots:
        plot_toy(directory, truth.log_lr(test_sample.x), scores, details, histories, scans,
                 weight_truth, weight_pred, test_sample.y)
    write_json(directory / 'timing.json', {'seconds': time.monotonic() - start})
    # Completion marker written last, so interrupted replicas are safely rerun.
    write_json(directory / 'metrics.json', rows)
    return rows


def make_report(output, rows, config):
    output = Path(output)
    summary = summarize(rows, config.bootstrap_samples, config.seed)
    write_csv(output / 'metrics.csv', rows)
    write_csv(output / 'summary.csv', summary)
    write_json(output / 'summary.json', summary)
    lines = ['# CMOK toy study', '',
             f"Completed {len(set(r['seed'] for r in rows))} independent toys; "
             f'{config.n_background:,} background and {config.n_signal:,} signal events per toy.', '',
             'Values are means ± standard errors across toys. Paired bootstrap intervals and '
             'adaptive-minus-CMOK differences are in `summary.csv`. Missing values are not replaced with zeros.', '',
             '| Model | Method | Physical AUC | Centered log-LR MSE | Asimov Z | alpha S/B |',
             '|---|---|---:|---:|---:|---:|']
    lookup = {(r['model'], r['method'], r['metric']): r for r in summary if r['comparison'] == 'absolute'}
    for model, method in sorted({(r['model'], r['method']) for r in rows}):
        cells = []
        for metric in ('physical_auc', 'log_lr_mse', 'asimov_z', 'alpha_ratio'):
            item = lookup.get((model, method, metric))
            if item is None:
                cells.append('unavailable')
            else:
                se = item['standard_error']
                cells.append(f"{item['mean']:.5g}" + (f' ± {se:.2g}' if se is not None else ' (one toy)')
                             + (f" [n={item['n']}]" if item['n_missing'] else ''))
        lines.append('| ' + ' | '.join([model, method, *cells]) + ' |')
    methods = sorted({row.get('binning_method', 'fixed') for row in rows})
    binning_note = ('Asimov binning: ' + ', '.join(methods) + '. ')
    if 'trafoD' in methods:
        binning_note += ('TrafoD edges are fitted separately for each score on noisy validation events '
                         'and frozen before test evaluation. Edges and validation diagnostics are saved '
                         'in each toy/binning.json. ')
    else:
        binning_note += 'Templates use fixed bins in sigmoid(log LR). '
    binning_note += ('All methods use the same configured total yields. If a test bin contains signal '
                     'but no sampled background, Z is unavailable; no background floor is inserted.')
    lines += ['', binning_note, '',
              'Truth-weighted test metrics integrate the physical law using exact conditional means. '
              'Training and meta-validation use noisy event weights. The centered MSE offset is a diagnostic '
              'computed on test data and is never fed back into inference.', '',
              'A short run verifies execution, not the gain hypothesis. Adaptive weights may worsen performance; '
              'the literal one-step objective includes an explicit calibration derivative through prior removal. '
              'Compare the validation scan and paired intervals before interpreting representation gains.', '']
    (output / 'report.md').write_text('\n'.join(lines))
    if config.plots:
        plot_summary(output, rows)
    return summary


def run_study(config, output, resume=False):
    config.validate()
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'config.json'
    if config_path.exists():
        if not resume:
            raise FileExistsError(f'{output} already contains a run; use --resume or a new output directory')
        if json.loads(config_path.read_text()) != config.to_dict():
            raise ValueError('Cannot resume with a different configuration')
    elif resume and any(output.iterdir()):
        raise ValueError('Cannot resume a directory without its saved config')
    write_json(config_path, config.to_dict())
    metadata = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, uproot=uproot.__version__,
                    device=config.device, sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                   for p in Path(__file__).parent.glob('*.py')})
    metadata_path = output / 'environment.json'
    if resume and metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError('Code or environment changed; start a new output directory')
    write_json(metadata_path, metadata)
    rows = []
    for replica in range(config.replicas):
        seed = config.seed + replica * 100_003
        directory = output / f'toy_{seed}'
        complete = directory / 'metrics.json'
        if resume and complete.exists():
            rows.extend(json.loads(complete.read_text()))
            print(f'Reusing completed toy {replica + 1}/{config.replicas}, seed={seed}', flush=True)
        else:
            print(f'Toy {replica + 1}/{config.replicas}, seed={seed}', flush=True)
            rows.extend(run_toy(config, seed, directory))
        # A usable tabular report remains available after each replica.
        make_report(output, rows, config)
    return rows
