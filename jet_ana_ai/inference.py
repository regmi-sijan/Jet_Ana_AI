import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import Scaler, features
from .grouping import group_clusters, candidate_summary
from .train import device_for, load_checkpoint, tensor_event


def predict(events, config, output, method='learned', checkpoint_path=None, device_name='auto', split='test'):
    device = device_for(device_name)
    model = checkpoint = scaler = None
    if method == 'learned':
        if not checkpoint_path:
            raise ValueError('learned method requires --checkpoint')
        model, scaler, checkpoint = load_checkpoint(checkpoint_path, device)
        # Feature/model definitions always come from checkpoint; grouping can be tuned separately.
        model_config = checkpoint['config']
    else:
        model_config = config
        if method == 'features':
            scaler = Scaler.fit([e for e in events if e.split == 'train'], config)
    selected = [e for e in events if split == 'all' or e.split == split]
    if not selected:
        raise ValueError(f'No events selected for split {split}.')
    rows, summaries, event_rows, embeddings = [], [], [], []
    for event in selected:
        x, pos, et, _ = features(event.frame, model_config)
        representation = None
        score = None
        if len(x) and model is not None:
            xt, pt = tensor_event(event, model_config, scaler, device)
            with torch.no_grad():
                result = model(xt, pt)
            representation = result['z'].cpu().numpy()
            if checkpoint['signal_trained']:
                score = result['signal_logit'].sigmoid().cpu().numpy()
        elif method == 'features':
            representation = scaler.transform(x)
        labels, affinities = group_clusters(pos, et, representation, config['grouping'], score)
        for i, original in enumerate(event.frame.to_dict('records')):
            row = dict(original, candidate_id=int(labels[i]), membership_affinity=float(affinities[i]),
                       signal_score=float(score[i]) if score is not None else np.nan)
            rows.append(row)
            if representation is not None:
                embeddings.append(dict(event_id=event.event_id, cluster_id=original['cluster_id'],
                                       **{f'z{j}': float(z) for j, z in enumerate(representation[i])}))
        candidates = candidate_summary(event.frame, labels, et)
        summaries.extend(dict(event_id=event.event_id, **r) for r in candidates)
        train_seen = checkpoint is not None and event.event_id in checkpoint['train_event_ids']
        val_seen = checkpoint is not None and event.event_id in checkpoint['validation_event_ids']
        event_rows.append({'event_id': event.event_id, 'split': event.split, 'n_input_clusters': len(x),
                           'n_candidates': len(candidates), 'n_unassigned': int((labels < 0).sum()),
                           'seen_in_training': train_seen, 'seen_in_validation': val_seen})
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    columns = list(selected[0].frame.columns) + ['candidate_id', 'membership_affinity', 'signal_score']
    pd.DataFrame(rows, columns=columns).to_csv(out / 'clusters.csv', index=False)
    candidate_columns = ['event_id', 'candidate_id', 'n_clusters', 'sum_cluster_et', 'eta_centroid', 'phi_centroid', 'et_weighted_rms_dr', 'max_centroid_dr']
    pd.DataFrame(summaries, columns=candidate_columns).to_csv(out / 'candidates.csv', index=False)
    pd.DataFrame(event_rows).to_csv(out / 'events.csv', index=False)
    if embeddings:
        pd.DataFrame(embeddings).to_csv(out / 'embeddings.csv', index=False)
    else:
        (out / 'embeddings.csv').unlink(missing_ok=True)
    metadata = {'method': method, 'training_mode': checkpoint['mode'] if checkpoint else None,
                'checkpoint': str(checkpoint_path) if checkpoint_path else None,
                'grouping': config['grouping'], 'model_config': model_config,
                'interpretation': 'Exploratory EMCal-visible candidates. Affinity is not a probability. Signal score, when available, is uncalibrated weak-label classifier output. No ancestry is inferred.'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return out
