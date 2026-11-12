import numpy as np
from sklearn.cluster import DBSCAN
from .data import wrap_phi


def group_clusters(pos, et, representation, settings, signal_score=None):
    """Conservative hybrid metric. Scores are affinities, NOT probabilities."""
    n = len(et)
    labels = np.full(n, -1, dtype=int)
    affinity = np.zeros(n, dtype=float)
    if n == 0:
        return labels, affinity
    de = pos[:, None, 0] - pos[None, :, 0]
    dp = wrap_phi(pos[:, None, 1] - pos[None, :, 1])
    dr = np.hypot(de, dp)
    spatial = dr / settings['spatial_scale']
    if representation is None:
        distance = spatial
    else:
        # RMS distance avoids implicit scaling with embedding dimension.
        latent = np.sqrt(np.mean((representation[:, None] - representation[None, :])**2, axis=-1))
        distance = np.hypot(spatial, settings['embedding_weight'] * latent / settings['embedding_scale'])
    # Angular gating applies to pairwise edges; it is not a bound on full group diameter.
    distance[dr > settings['max_link_dr']] = 1e6
    np.fill_diagonal(distance, 0)
    eligible = np.ones(n, dtype=bool)
    if signal_score is not None:
        eligible &= signal_score >= settings['signal_threshold']
    idx = np.flatnonzero(eligible)
    if len(idx) < settings['min_samples']:
        return labels, affinity
    sub = distance[np.ix_(idx, idx)]
    algorithm = DBSCAN(eps=settings['eps'], min_samples=settings['min_samples'], metric='precomputed').fit(sub)
    raw = algorithm.labels_
    candidates = []
    for label in sorted(set(raw) - {-1}):
        members = idx[raw == label]
        if len(members) < settings['min_clusters'] or et[members].sum() < settings['min_candidate_et']:
            continue
        # No automatic bridging fix: expose radius diagnostics for validation.
        candidates.append(members)
    candidates.sort(key=lambda members: -et[members].sum())
    for label, members in enumerate(candidates):
        labels[members] = label
        core = np.intersect1d(idx[algorithm.core_sample_indices_], members)
        # Exclude self so a core node does not get an automatic affinity of one.
        for i in members:
            reference = core[core != i]
            if not len(reference):
                reference = members[members != i]
            if len(reference):
                affinity[i] = np.exp(-.5 * (distance[i, reference].min() / settings['eps'])**2)
    return labels, affinity


def candidate_summary(frame, labels, et):
    rows = []
    for group in sorted(set(labels) - {-1}):
        mask = labels == group
        w = et[mask]
        eta = frame.eta.to_numpy()[mask]
        phi = frame.phi.to_numpy()[mask]
        center_eta = np.average(eta, weights=w)
        center_phi = np.arctan2(np.sum(w * np.sin(phi)), np.sum(w * np.cos(phi)))
        dr = np.hypot(eta - center_eta, wrap_phi(phi - center_phi))
        rows.append({'candidate_id': int(group), 'n_clusters': int(mask.sum()),
                     'sum_cluster_et': float(w.sum()), 'eta_centroid': float(center_eta),
                     'phi_centroid': float(center_phi), 'et_weighted_rms_dr': float(np.sqrt(np.average(dr**2, weights=w))),
                     'max_centroid_dr': float(dr.max())})
    return rows
