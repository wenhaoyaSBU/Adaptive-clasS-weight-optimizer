import argparse
import json
from pathlib import Path
from .config import Config


def main():
    parser = argparse.ArgumentParser(description='CMOK and adaptive class-weight toy MC')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='Train and evaluate independent toys')
    run.add_argument('--config', type=Path)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--resume', action='store_true')
    for name in ('replicas', 'seed', 'n-background', 'n-signal', 'epochs', 'weight-epochs', 'threads'):
        run.add_argument('--' + name, type=int)
    run.add_argument('--device')
    for name in ('oracle-weights', 'constant-weights', 'signed-weights'):
        run.add_argument('--' + name, action='store_true', default=None)
    run.add_argument('--no-plots', action='store_true')
    check = sub.add_parser('validate-truth', help='Check analytic normalizations against independent Monte Carlo')
    check.add_argument('--events', type=int, default=200_000)
    check.add_argument('--seed', type=int, default=123)
    check.add_argument('--output', type=Path)
    report = sub.add_parser('summarize', help='Rebuild reports from completed toys')
    report.add_argument('output', type=Path)
    sweep = sub.add_parser('sweep', help='Run sample-size, capacity, or noise controls')
    sweep.add_argument('--config', type=Path, required=True)
    sweep.add_argument('--output', type=Path, required=True)
    sweep.add_argument('--axis', choices=['samples', 'capacity', 'noise'], required=True)
    sweep.add_argument('--values', type=float, nargs='+', required=True)
    sweep.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'validate-truth':
        from .truth import validate_truth
        from .study import write_json
        if args.events < 2:
            parser.error('--events must be at least 2')
        result = validate_truth(args.seed, args.events)
        print(json.dumps(result, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
    elif args.command == 'sweep':
        from .controls import run_sweep
        run_sweep(Config.read(args.config), args.output, args.axis, args.values, args.resume)
    elif args.command == 'summarize':
        from .study import make_report
        config = Config.read(args.output / 'config.json')
        rows = [row for path in sorted(args.output.glob('toy_*/metrics.json')) for row in json.loads(path.read_text())]
        if not rows:
            parser.error('No completed toys in that output directory')
        make_report(args.output, rows, config)
    else:
        from .study import run_study
        config = Config.read(args.config)
        for name in ('replicas', 'seed', 'n_background', 'n_signal', 'epochs', 'weight_epochs',
                     'threads', 'device', 'oracle_weights', 'constant_weights', 'signed_weights'):
            if getattr(args, name) is not None:
                setattr(config, name, getattr(args, name))
        if args.no_plots:
            config.plots = False
        run_study(config.validate(), args.output, args.resume)
        print(f'Report: {args.output / "report.md"}')


if __name__ == '__main__':
    main()
