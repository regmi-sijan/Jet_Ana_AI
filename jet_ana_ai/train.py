import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import Scaler, features, wrap_phi
from .model import ClusterEncoder, representation_loss, weak_loss


def device_for(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    # CPU is the reproducible reference; use --device mps explicitly on Apple Silicon.
    return torch.device('cpu')


def tensor_event(event, config, scaler, device, rng=None):
    frame = event.frame.copy()
    if rng is not None:
        # Small measurement-like perturbations only; feature summaries recomputed.
        # Defaults are demonstration settings, not detector resolution claims.
        frame['energy'] *= np.exp(rng.normal(0, config['energy_jitter'], len(frame)))
        frame['eta'] += rng.normal(0, config['angle_jitter'], len(frame))
        frame['phi'] = wrap_phi(frame.phi.to_numpy() + rng.normal(0, config['angle_jitter'], len(frame)))
    x, pos, _, _ = features(frame, config)
    return (torch.tensor(scaler.transform(x), device=device), torch.tensor(pos, device=device))


def train(events, config, out_dir, epochs, mode, device_name='auto'):
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config['cpu_threads'])
    device = device_for(device_name)
    train_events = [e for e in events if e.split == 'train' and len(e.frame)]
    val_events = [e for e in events if e.split == 'val' and len(e.frame)]
    if not train_events or not val_events:
        raise ValueError('Need nonempty train and validation events. Supply an event table with explicit splits for small datasets.')
    if epochs < 1:
        raise ValueError('epochs must be >=1')
    signal_trained = False
    if mode == 'weak':
        labels = np.concatenate([e.frame.label.to_numpy() for e in train_events])
        if not (labels >= 0).any() or not (labels == -1).any():
            raise ValueError('Weak mode requires both trusted jet-group and background labels in training events.')
        signal_trained = True
    scaler = Scaler.fit(train_events, config)
    _, _, _, names = features(train_events[0].frame, config)
    model = ClusterEncoder(len(names), config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    best = float('inf')
    history = []

    def event_loss(event, rng):
        target, pos = tensor_event(event, config, scaler, device)
        xa, pa = tensor_event(event, config, scaler, device, rng)
        xb, pb = tensor_event(event, config, scaler, device, rng)
        ma = torch.tensor(rng.random(xa.shape) < config['mask_fraction'], device=device)
        mb = torch.tensor(rng.random(xb.shape) < config['mask_fraction'], device=device)
        a = model(xa.masked_fill(ma, 0), pa)
        b = model(xb.masked_fill(mb, 0), pb)
        # Reconstruct masked node features, plus matching-view representation loss.
        rec = sum(F.mse_loss(o['reconstruction'][m], target[m]) if m.any() else o['reconstruction'].sum() * 0
                  for o, m in ((a, ma), (b, mb))) / 2
        loss = config['reconstruction_weight'] * rec + representation_loss(a, b, config)
        if mode == 'weak':
            clean = model(target, pos)
            labels = torch.tensor(event.frame.label.to_numpy(), device=device)
            node, pair = weak_loss(clean, labels)
            loss = loss + config['node_weight'] * node + config['pair_weight'] * pair
        return loss

    for epoch in range(epochs):
        model.train()
        order = list(train_events)
        random.shuffle(order)
        rng = np.random.default_rng(seed + epoch)
        train_losses = []
        for event in order:
            opt.zero_grad(set_to_none=True)
            loss = event_loss(event, rng)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss; inspect inputs and configuration.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        val_losses = []
        # Fixed corruption stream makes validation comparable across epochs.
        vrng = np.random.default_rng(seed + 100000)
        with torch.no_grad():
            for event in val_events:
                val_losses.append(float(event_loss(event, vrng)))
        record = {'epoch': epoch + 1, 'train_loss': float(np.mean(train_losses)), 'val_loss': float(np.mean(val_losses))}
        history.append(record)
        print(json.dumps(record), flush=True)
        if record['val_loss'] < best:
            best = record['val_loss']
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'model': state, 'config': config, 'scaler': scaler.state(), 'features': names,
                        'mode': mode, 'signal_trained': signal_trained, 'best_epoch': epoch + 1,
                        'train_event_ids': [e.event_id for e in train_events],
                        'validation_event_ids': [e.event_id for e in val_events]}, out / 'best.pt')
        (out / 'history.json').write_text(json.dumps(history, indent=2))
    return out / 'best.pt'


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = ClusterEncoder(len(checkpoint['features']), checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    return model, Scaler(**checkpoint['scaler']), checkpoint
