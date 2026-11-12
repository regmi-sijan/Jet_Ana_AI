from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ['event_id', 'cluster_id', 'energy', 'eta', 'phi']


def wrap_phi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def read_table(path):
    path = Path(path)
    if path.suffix == '.csv':
        return pd.read_csv(path, dtype={'event_id': str, 'cluster_id': str, 'split_group': str})
    if path.suffix in ('.parquet', '.pq'):
        return pd.read_parquet(path)
    raise ValueError('Expected CSV or Parquet (Parquet requires pyarrow).')


def split_for(key, seed):
    u = int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:8], 16) / 2**32
    return 'train' if u < .7 else ('val' if u < .85 else 'test')


@dataclass
class Event:
    event_id: str
    frame: pd.DataFrame
    split: str


def load_events(path, config, event_table=None):
    df = read_table(path)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if df[REQUIRED].isna().any().any():
        raise ValueError('Required columns contain missing values.')
    df['event_id'] = df.event_id.astype(str)
    df['cluster_id'] = df.cluster_id.astype(str)
    if df.duplicated(['event_id', 'cluster_id']).any():
        raise ValueError('Duplicate (event_id, cluster_id); combine run and event IDs if necessary.')
    numeric = ['energy', 'eta', 'phi'] + config['optional_features']
    for name in numeric:
        if name not in df:
            raise ValueError(f'Configured optional feature missing: {name}')
        df[name] = pd.to_numeric(df[name], errors='raise')
        if not np.isfinite(df[name].to_numpy(dtype=float)).all():
            raise ValueError(f'Nonfinite values in {name}; clean data explicitly.')
    if (df.energy <= 0).any():
        raise ValueError('energy must be positive in GeV; apply detector quality selections before input.')
    df.phi = wrap_phi(df.phi.to_numpy())
    if 'label' not in df:
        df['label'] = -2
    label = pd.to_numeric(df.label, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(label).all() or (label < -2).any() or (label != np.floor(label)).any():
        raise ValueError('label must be an integer: -2 unknown, -1 background, >=0 event-local group.')
    df['label'] = label.astype(int)
    registry = read_table(event_table) if event_table else pd.DataFrame({'event_id': df.event_id.unique()})
    if 'event_id' not in registry or registry.event_id.isna().any():
        raise ValueError('Event table requires nonmissing event_id.')
    registry['event_id'] = registry.event_id.astype(str)
    if registry.event_id.duplicated().any():
        raise ValueError('Duplicate event IDs in event table.')
    if not set(df.event_id).issubset(set(registry.event_id)):
        raise ValueError('Event table omits events in cluster table.')
    # Splits belong in the event registry so empty events have the same metadata.
    group_splits = {}
    events = []
    grouped = {key: g.reset_index(drop=True) for key, g in df.groupby('event_id', sort=False)}
    for row in registry.to_dict('records'):
        eid = row['event_id']
        key = str(row.get('split_group', eid))
        if pd.isna(row.get('split_group', eid)):
            raise ValueError('split_group cannot be missing.')
        split = row.get('split', split_for(key, config['seed']))
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val, or test.')
        if key in group_splits and group_splits[key] != split:
            raise ValueError('A split_group crosses dataset splits.')
        group_splits[key] = split
        frame = grouped.get(eid, df.iloc[:0].copy())
        if len(frame) > config['max_clusters']:
            raise ValueError(f'Event {eid} has {len(frame)} clusters; max_clusters={config["max_clusters"]}. No truncation performed.')
        events.append(Event(eid, frame, split))
    return events


def features(frame, config):
    """Recompute all neighborhood features from this view; no label information."""
    eta = frame.eta.to_numpy(dtype=np.float32)
    phi = frame.phi.to_numpy(dtype=np.float32)
    energy = frame.energy.to_numpy(dtype=np.float32)
    # Input eta is required to be finite and in the detector's physical range.
    if np.any(np.abs(eta) > 15):
        raise ValueError('|eta| > 15; check units/coordinates.')
    et = energy / np.cosh(eta)
    de = eta[:, None] - eta[None, :]
    dp = wrap_phi(phi[:, None] - phi[None, :])
    dr = np.hypot(de, dp)
    diag = np.eye(len(frame), dtype=bool)
    cols = [np.log1p(et), eta]
    names = ['log1p_et', 'eta']
    for r in config['radii']:
        mask = (dr < r) & ~diag
        total = mask @ et
        count = mask.sum(1)
        cols.extend([np.log1p(total), np.log1p(count), et / (et + total + 1e-8)])
        names.extend([f'log1p_neighbor_et_{r}', f'log1p_neighbors_{r}', f'local_fraction_{r}'])
    nearest = np.where(diag, np.inf, dr).min(1) if len(frame) else np.array([])
    nearest = np.minimum(nearest, config['graph_radius'])
    cols.append(nearest)
    names.append('nearest_dr_capped')
    for name in config['optional_features']:
        cols.append(frame[name].to_numpy(dtype=np.float32))
        names.append(name)
    x = np.stack(cols, axis=1).astype(np.float32)
    return x, np.stack([eta, phi], axis=1), et, names


class Scaler:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    @classmethod
    def fit(cls, events, config):
        arrays = [features(e.frame, config)[0] for e in events if len(e.frame)]
        if not arrays:
            raise ValueError('No nonempty training events.')
        # Sufficient statistics, avoiding concatenating a second full dataset.
        n = sum(len(a) for a in arrays)
        mean = sum(a.astype(float).sum(0) for a in arrays) / n
        var = sum(((a - mean)**2).sum(0) for a in arrays) / n
        return cls(mean.tolist(), np.maximum(np.sqrt(var), .01).tolist())

    def transform(self, x):
        return np.clip((x - self.mean) / self.std, -20, 20).astype(np.float32)

    def state(self):
        return {'mean': self.mean.tolist(), 'std': self.std.tolist()}
