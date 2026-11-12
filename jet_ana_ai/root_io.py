import json
from pathlib import Path
import numpy as np
import pandas as pd


def convert_root(path, tree, mapping_path, output, layout='jagged'):
    """Jagged: one TTree entry/event; flat: one entry/cluster. Never guess layout."""
    import awkward as ak
    import uproot
    mapping = json.loads(Path(mapping_path).read_text())
    required = {'event_id', 'energy', 'eta', 'phi'}
    if not required.issubset(mapping):
        raise ValueError(f'ROOT mapping requires {sorted(required)}')
    if layout == 'flat' and 'cluster_id' not in mapping:
        raise ValueError('Flat ROOT layout requires cluster_id mapping.')
    frames, event_frames = [], []
    # Chunked reads; final tables are materialized. Use per-file shards for large datasets.
    with uproot.open(path) as file:
        for arrays in file[tree].iterate(list(set(mapping.values())), library='ak', step_size='100 MB'):
            if layout == 'flat':
                data = {name: ak.to_numpy(arrays[branch]) for name, branch in mapping.items()}
                if any(value.ndim != 1 for value in data.values()):
                    raise ValueError('Flat layout requires scalar branches.')
                frame = pd.DataFrame(data)
                frames.append(frame)
                event_frames.append(frame[['event_id']].drop_duplicates())
            else:
                events = arrays[mapping['event_id']]
                if ak.to_numpy(events).ndim != 1:
                    raise ValueError('event_id must be scalar per event.')
                event_frames.append(pd.DataFrame({'event_id': ak.to_list(events)}))
                counts = ak.to_numpy(ak.num(arrays[mapping['energy']], axis=1))
                data = {'event_id': np.repeat(ak.to_numpy(events), counts)}
                for name, branch in mapping.items():
                    if name == 'event_id':
                        continue
                    if not np.array_equal(ak.to_numpy(ak.num(arrays[branch], axis=1)), counts):
                        raise ValueError(f'Jagged cluster lengths differ for {name}')
                    data[name] = ak.to_numpy(ak.flatten(arrays[branch], axis=1))
                if 'cluster_id' not in data:
                    data['cluster_id'] = ak.to_numpy(ak.flatten(ak.local_index(arrays[mapping['energy']], axis=1)))
                frames.append(pd.DataFrame(data))
    if not frames:
        raise ValueError('Empty ROOT tree.')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True)
    if layout == 'jagged' and events.event_id.duplicated().any():
        raise ValueError('Duplicate event IDs; create a globally unique run/event key before conversion.')
    if frame.duplicated(['event_id', 'cluster_id']).any():
        raise ValueError('Duplicate event/cluster IDs in ROOT input.')
    frame.to_csv(output, index=False)
    events.drop_duplicates().to_csv(output.with_name(output.stem + '_events.csv'), index=False)
    return output
