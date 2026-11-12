"""Artificial patterns for software smoke tests only; not a physics simulation."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data import wrap_phi


def make_demo(output, n_events=60, seed=42):
    if n_events < 6:
        raise ValueError('Use at least six demo events.')
    rng = np.random.default_rng(seed)
    rows, events = [], []
    for i in range(n_events):
        # Explicit deterministic partitions ensure small demos have all splits.
        split = 'train' if i % 10 < 7 else ('val' if i % 10 < 9 else 'test')
        events.append({'event_id': f'demo_{i}', 'split': split})
        n_groups = (i // 10 + i % 10) % 4
        cid = 0
        # Periodic empty events exercise zero-input handling.
        if i % 17 == 0:
            continue
        for _ in range(int(rng.integers(15, 30))):
            rows.append({'event_id': f'demo_{i}', 'cluster_id': cid, 'energy': float(rng.exponential(.5) + .05),
                         'eta': float(rng.uniform(-.7, .7)), 'phi': float(rng.uniform(-np.pi, np.pi)), 'label': -1})
            cid += 1
        for g in range(n_groups):
            eta, phi = rng.uniform(-.4, .4), wrap_phi(-2.4 + g * 2.1 + rng.normal(0, .1))
            for _ in range(int(rng.integers(6, 12))):
                rows.append({'event_id': f'demo_{i}', 'cluster_id': cid, 'energy': float(rng.exponential(2) + .5),
                             'eta': float(eta + rng.normal(0, .06)), 'phi': float(wrap_phi(phi + rng.normal(0, .06))), 'label': g})
                cid += 1
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'clusters.csv', index=False)
    pd.DataFrame(events).to_csv(out / 'events.csv', index=False)
    return out
