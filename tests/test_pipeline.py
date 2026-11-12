import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from jet_ana_ai.cli import load_config
from jet_ana_ai.data import features, load_events
from jet_ana_ai.demo import make_demo
from jet_ana_ai.grouping import group_clusters
from jet_ana_ai.model import ClusterEncoder, weak_loss
from jet_ana_ai.train import train
from jet_ana_ai.inference import predict
from jet_ana_ai.evaluate import evaluate
from jet_ana_ai.root_io import convert_root

@pytest.fixture
def config():
    c = load_config(Path(__file__).parents[1] / 'examples/config.json')
    c.update(hidden_dim=16, embedding_dim=4, layers=1, neighbors=4)
    return c

def frame():
    return pd.DataFrame({'event_id': ['a'] * 3, 'cluster_id': ['0', '1', '2'], 'energy': [3., 2., 1.],
                         'eta': [0., .01, .02], 'phi': [np.pi-.01, -np.pi+.01, 0], 'label': [0, 0, -1]})

def test_phi_wrap_and_permutation(config):
    x, pos, _, _ = features(frame(), config)
    assert x[0, -1] < .03
    model = ClusterEncoder(x.shape[1], config).eval()
    with torch.no_grad():
        a = model(torch.tensor(x), torch.tensor(pos))['z']
        perm = [2, 0, 1]
        b = model(torch.tensor(x[perm]), torch.tensor(pos[perm]))['z']
    torch.testing.assert_close(a[perm], b, atol=1e-6, rtol=1e-5)

def test_grouping_zero_variable_and_phi(config):
    g = config['grouping'].copy()
    g.update(min_samples=2, min_clusters=2, min_candidate_et=0)
    labels, _ = group_clusters(np.empty((0, 2)), np.array([]), None, g)
    assert len(labels) == 0
    p = np.array([[0, np.pi-.01], [0, -np.pi+.01], [.5, 0], [.51, .01], [-.6, 1.5]])
    labels, _ = group_clusters(p, np.ones(5), None, g)
    assert labels[0] == labels[1] >= 0
    assert labels[2] == labels[3] >= 0
    assert labels[0] != labels[2]
    assert labels[4] == -1
    labels, _ = group_clusters(p, np.ones(5), None, g, np.zeros(5))
    assert (labels == -1).all()

def test_unknown_labels_no_weak_loss():
    out = {'z': torch.randn(4, 4, requires_grad=True), 'signal_logit': torch.randn(4, requires_grad=True)}
    a, b = weak_loss(out, torch.full((4,), -2))
    assert a.item() == b.item() == 0
    (a+b).backward()
    assert torch.isfinite(out['z'].grad).all()

def test_input_integrity(tmp_path, config):
    df = frame()
    df.loc[1, 'cluster_id'] = '0'
    path = tmp_path / 'bad.csv'
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        load_events(path, config)
    df = frame()
    df.loc[0, 'energy'] = np.nan
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_events(path, config)

@pytest.mark.parametrize('mode', ['self', 'weak'])
def test_end_to_end(tmp_path, config, mode):
    folder = make_demo(tmp_path / 'data', 20)
    events = load_events(folder / 'clusters.csv', config, folder / 'events.csv')
    if mode == 'self':
        for e in events:
            e.frame['label'] = -2
    checkpoint = train(events, config, tmp_path / 'model', 1, mode)
    for method in ['learned', 'spatial', 'features']:
        out = predict(events, config, tmp_path / method, method, checkpoint, split='all')
        report = evaluate(out)
        assert report['n_events'] == 20
        assert np.isfinite(pd.read_csv(out / 'clusters.csv').membership_affinity).all()
        if mode == 'self':
            assert pd.read_csv(out / 'clusters.csv').signal_score.isna().all()
            assert 'reference_node_precision' not in report
    assert np.isfinite(json.loads((tmp_path / 'model/history.json').read_text())[0]['val_loss'])

def test_root_jagged_preserves_empty(tmp_path, config):
    import awkward as ak
    import uproot
    path = tmp_path / 'input.root'
    with uproot.recreate(path) as f:
        f['events'] = {'event_id': np.array([10, 11]), 'e': ak.Array([[2., 1.], []]),
                       'eta': ak.Array([[.1, .2], []]), 'phi': ak.Array([[.2, .3], []])}
    mapping = tmp_path / 'map.json'
    mapping.write_text(json.dumps({'event_id': 'event_id', 'energy': 'e', 'eta': 'eta', 'phi': 'phi'}))
    out = convert_root(path, 'events', mapping, tmp_path / 'converted.csv')
    events = load_events(out, config, tmp_path / 'converted_events.csv')
    assert [len(e.frame) for e in events] == [2, 0]

def test_split_group_leakage(tmp_path, config):
    frame().to_csv(tmp_path / 'clusters.csv', index=False)
    pd.DataFrame({'event_id': ['a', 'b'], 'split_group': ['run1', 'run1'], 'split': ['train', 'test']}).to_csv(tmp_path / 'events.csv', index=False)
    with pytest.raises(ValueError, match='crosses'):
        load_events(tmp_path / 'clusters.csv', config, tmp_path / 'events.csv')

def test_optional_label_leakage(tmp_path):
    c = json.loads((Path(__file__).parents[1] / 'examples/config.json').read_text())
    c['optional_features'] = ['label']
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match='Optional'):
        load_config(path)

def test_identical_weak_embeddings_have_finite_gradients():
    out = {'z': torch.zeros(3, 4, requires_grad=True), 'signal_logit': torch.zeros(3, requires_grad=True)}
    a, b = weak_loss(out, torch.tensor([0, 0, -1]))
    (a + b).backward()
    assert torch.isfinite(out['z'].grad).all()

def test_root_flat(tmp_path, config):
    import uproot
    path = tmp_path / 'input.root'
    with uproot.recreate(path) as f:
        f['events'] = {'event': np.array([1, 1, 2]), 'cluster': np.array([0, 1, 0]),
                       'energy': np.array([2., 1., 3.]), 'eta': np.array([.1, .2, .3]), 'phi': np.array([.2, .3, .4])}
    mapping = tmp_path / 'map.json'
    mapping.write_text(json.dumps({'event_id': 'event', 'cluster_id': 'cluster', 'energy': 'energy', 'eta': 'eta', 'phi': 'phi'}))
    out = convert_root(path, 'events', mapping, tmp_path / 'converted.csv', 'flat')
    events = load_events(out, config, tmp_path / 'converted_events.csv')
    assert [len(e.frame) for e in events] == [2, 1]

def test_singleton_and_global_phi_shift(config):
    df = frame().iloc[:1].copy()
    x, pos, _, _ = features(df, config)
    model = ClusterEncoder(x.shape[1], config).eval()
    with torch.no_grad():
        out = model(torch.tensor(x), torch.tensor(pos))
    assert torch.isfinite(out['z']).all()
    df = frame()
    a, pa, _, _ = features(df, config)
    df['phi'] += .8
    b, pb, _, _ = features(df, config)
    np.testing.assert_allclose(a, b, atol=2e-6)
    with torch.no_grad():
        za = model(torch.tensor(a), torch.tensor(pa))['z']
        zb = model(torch.tensor(b), torch.tensor(pb))['z']
    torch.testing.assert_close(za, zb, atol=1e-5, rtol=1e-5)
