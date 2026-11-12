import argparse
import json
from pathlib import Path
from .data import load_events


def load_config(path):
    config = json.loads(Path(path).read_text())
    for key in ['radii', 'graph_radius', 'neighbors', 'hidden_dim', 'embedding_dim', 'layers', 'max_clusters', 'cpu_threads', 'learning_rate']:
        values = config[key] if isinstance(config[key], list) else [config[key]]
        if not values or any(v <= 0 for v in values):
            raise ValueError(f'{key} must be positive.')
    if len(set(config['optional_features'])) != len(config['optional_features']):
        raise ValueError('Duplicate optional features.')
    reserved = {'event_id', 'cluster_id', 'label', 'candidate_id', 'split', 'split_group', 'energy', 'eta', 'phi', 'signal_score', 'membership_affinity'}
    if reserved.intersection(config['optional_features']):
        raise ValueError('Optional features must not contain labels, IDs, predictions, or required inputs.')
    if not 0 < config['mask_fraction'] < 1:
        raise ValueError('mask_fraction must be between zero and one.')
    if config['angle_jitter'] < 0 or config['energy_jitter'] < 0:
        raise ValueError('Jitter must be nonnegative.')
    g = config['grouping']
    for name in ['spatial_scale', 'embedding_scale', 'max_link_dr', 'eps', 'min_samples', 'min_clusters']:
        if g[name] <= 0:
            raise ValueError(f'grouping.{name} must be positive.')
    if g['min_samples'] < 2 or g['min_clusters'] < 2:
        raise ValueError('Require at least two clusters per candidate; isolated deposits are not jets by default.')
    if g['embedding_weight'] < 0 or g['min_candidate_et'] < 0 or not 0 <= g['signal_threshold'] <= 1:
        raise ValueError('Invalid grouping weight, ET threshold, or signal threshold.')
    return config


def main():
    parser = argparse.ArgumentParser(description='Contextual EMCal cluster embedding research pipeline')
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='Generate artificial software-test patterns, not physics events')
    demo.add_argument('--output', required=True)
    demo.add_argument('--events', type=int, default=60)
    root = sub.add_parser('convert-root')
    root.add_argument('--input', required=True)
    root.add_argument('--tree', required=True)
    root.add_argument('--mapping', required=True)
    root.add_argument('--layout', choices=['jagged', 'flat'], default='jagged')
    root.add_argument('--output', required=True)
    for name in ['validate', 'train', 'predict']:
        p = sub.add_parser(name)
        p.add_argument('--data', required=True)
        p.add_argument('--events', help='Event registry CSV, including zero-cluster events and optional splits')
        p.add_argument('--config', default='examples/config.json')
        if name != 'validate':
            p.add_argument('--output', required=True)
            p.add_argument('--device', default='auto')
        if name == 'train':
            p.add_argument('--mode', choices=['self', 'weak'], default='self')
            p.add_argument('--epochs', type=int, default=30)
        if name == 'predict':
            p.add_argument('--method', choices=['learned', 'spatial', 'features'], default='learned')
            p.add_argument('--checkpoint')
            p.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test')
    ev = sub.add_parser('evaluate')
    ev.add_argument('--predictions', required=True)
    ev.add_argument('--output')
    plot = sub.add_parser('plot')
    plot.add_argument('--predictions', required=True)
    plot.add_argument('--event-id', required=True)
    plot.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'demo':
        from .demo import make_demo
        print(make_demo(args.output, args.events))
    elif args.command == 'convert-root':
        from .root_io import convert_root
        print(convert_root(args.input, args.tree, args.mapping, args.output, args.layout))
    elif args.command == 'evaluate':
        from .evaluate import evaluate
        print(json.dumps(evaluate(args.predictions, args.output), indent=2))
    elif args.command == 'plot':
        from .evaluate import plot_event
        plot_event(args.predictions, args.event_id, args.output)
    else:
        config = load_config(args.config)
        # Learned inference validates inputs against the checkpoint feature schema.
        input_config = config
        if args.command == 'predict' and args.method == 'learned':
            if not args.checkpoint:
                parser.error('--method learned requires --checkpoint')
            import torch
            input_config = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['config']
        events = load_events(args.data, input_config, args.events)
        if args.command == 'validate':
            print(json.dumps({'events': len(events), 'clusters': sum(len(e.frame) for e in events),
                              'empty_events': sum(len(e.frame) == 0 for e in events),
                              'splits': {s: sum(e.split == s for e in events) for s in ['train', 'val', 'test']}}, indent=2))
        elif args.command == 'train':
            from .train import train
            print(train(events, config, args.output, args.epochs, args.mode, args.device))
        else:
            from .inference import predict
            print(predict(events, config, args.output, args.method, args.checkpoint, args.device, args.split))


if __name__ == '__main__':
    main()
