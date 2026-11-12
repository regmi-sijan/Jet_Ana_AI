import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import read_table, wrap_phi


def evaluate(prediction_dir, output=None):
    directory = Path(prediction_dir)
    clusters = read_table(directory / 'clusters.csv')
    events = read_table(directory / 'events.csv')
    report = {'n_events': len(events), 'mean_candidates_per_event': float(events.n_candidates.mean()),
              'zero_candidate_fraction': float((events.n_candidates == 0).mean()),
              'n_clusters': len(clusters),
              'unassigned_cluster_fraction': float((clusters.candidate_id < 0).mean()) if len(clusters) else None,
              'candidate_count_distribution': {str(k): int(v) for k, v in events.n_candidates.value_counts().sort_index().items()},
              'note': 'Labels, if present, are reference/weak labels, not verified real-data ancestry. Candidate ET is a scalar cluster ET sum, not calibrated full-jet pT.'}
    if 'label' in clusters:
        known = clusters.label >= -1
        if known.any():
            y = clusters.label.to_numpy()
            p = clusters.candidate_id.to_numpy()
            tp = ((y >= 0) & (p >= 0)).sum()
            fp = ((y == -1) & (p >= 0)).sum()
            fn = ((y >= 0) & (p == -1)).sum()
            report['reference_node_precision'] = float(tp / (tp + fp)) if tp + fp else None
            report['reference_node_recall'] = float(tp / (tp + fn)) if tp + fn else None
            pair_tp = pair_fp = pair_fn = 0
            for _, frame in clusters.groupby('event_id'):
                y, p = frame.label.to_numpy(), frame.candidate_id.to_numpy()
                i, j = np.triu_indices(len(y), 1)
                valid = (y[i] >= -1) & (y[j] >= -1)
                same_y = (y[i] >= 0) & (y[i] == y[j])
                same_p = (p[i] >= 0) & (p[i] == p[j])
                pair_tp += int((valid & same_y & same_p).sum())
                pair_fp += int((valid & ~same_y & same_p).sum())
                pair_fn += int((valid & same_y & ~same_p).sum())
            report['reference_pair_precision'] = pair_tp / (pair_tp + pair_fp) if pair_tp + pair_fp else None
            report['reference_pair_recall'] = pair_tp / (pair_tp + pair_fn) if pair_tp + pair_fn else None
            # Only fully labeled nonempty background events establish a reference zero-jet control.
            control = []
            for eid, frame in clusters.groupby('event_id'):
                if (frame.label == -1).all():
                    control.append(eid)
            control_events = events[events.event_id.isin(control)]
            report['n_fully_labeled_background_events'] = len(control_events)
            report['background_control_any_candidate_fraction'] = float((control_events.n_candidates > 0).mean()) if len(control_events) else None
    target = Path(output) if output else directory / 'report.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def plot_event(prediction_dir, event_id, output):
    import os
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(target.resolve().parent / '.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(prediction_dir)
    frame = read_table(directory / 'clusters.csv')
    registry = read_table(directory / 'events.csv')
    if str(event_id) not in set(registry.event_id.astype(str)):
        raise ValueError(f'Unknown event {event_id}')
    frame = frame[frame.event_id.astype(str) == str(event_id)]
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, group in frame.groupby('candidate_id'):
        et = group.energy.to_numpy() / np.cosh(group.eta.to_numpy())
        ax.scatter(group.eta, wrap_phi(group.phi.to_numpy()), s=15 + 35 * np.log1p(et),
                   label='unassigned' if label < 0 else f'candidate {label}',
                   c='lightgray' if label < 0 else None, alpha=.8)
    ax.set(xlabel='eta', ylabel='phi [rad]', title=f'Event {event_id}: EMCal-visible candidates (marker area scales with log ET)')
    if len(frame):
        ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)
