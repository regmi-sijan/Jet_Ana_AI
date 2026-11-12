import torch
from torch import nn
from torch.nn import functional as F


def geometry(pos):
    de = pos[:, None, 0] - pos[None, :, 0]
    dp = pos[:, None, 1] - pos[None, :, 1]
    dp = torch.atan2(torch.sin(dp), torch.cos(dp))
    dr = torch.sqrt(de.square() + dp.square() + 1e-12)
    return de, dp, dr


class GraphBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(2 * hidden + 3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.update = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, edge, src, dst):
        agg = torch.zeros_like(h)
        if len(src):
            msg = self.message(torch.cat([h[src], h[dst], edge], dim=1))
            agg.index_add_(0, src, msg)
            count = torch.bincount(src, minlength=len(h)).clamp_min(1).to(h.dtype)
            agg = agg / count[:, None]
        return self.norm(h + self.update(torch.cat([h, agg], dim=1)))


class ClusterEncoder(nn.Module):
    """Permutation-equivariant local message passing; event-by-event, no padding."""
    def __init__(self, input_dim, config):
        super().__init__()
        h = config['hidden_dim']
        self.config = config
        self.input = nn.Sequential(nn.Linear(input_dim, h), nn.SiLU(), nn.LayerNorm(h))
        self.blocks = nn.ModuleList([GraphBlock(h) for _ in range(config['layers'])])
        self.context = nn.Sequential(nn.Linear(h * 2, h), nn.SiLU())
        self.embedding = nn.Linear(h, config['embedding_dim'])
        self.reconstruct = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, input_dim))
        self.signal = nn.Linear(h, 1)

    def forward(self, x, pos):
        n = len(x)
        if not n:
            raise ValueError('Skip empty events before calling encoder.')
        de, dp, dr = geometry(pos)
        dist = dr.masked_fill(torch.eye(n, device=x.device, dtype=torch.bool), float('inf'))
        k = min(self.config['neighbors'], max(n - 1, 0))
        if k:
            values, indices = dist.topk(k, largest=False, dim=1)
            source = torch.arange(n, device=x.device)[:, None].expand(n, k)
            keep = values <= self.config['graph_radius']
            src, dst = source[keep], indices[keep]
        else:
            src = dst = torch.empty(0, dtype=torch.long, device=x.device)
        radius = self.config['graph_radius']
        edge = torch.stack([de[src, dst] / radius, dp[src, dst] / radius, dr[src, dst] / radius], dim=1)
        h = self.input(x)
        for block in self.blocks:
            h = block(h, edge, src, dst)
        h = self.context(torch.cat([h, h.mean(0, keepdim=True).expand_as(h)], dim=1))
        return {'z': self.embedding(h), 'reconstruction': self.reconstruct(h), 'signal_logit': self.signal(h).squeeze(-1)}


def representation_loss(a, b, config):
    """VICReg-style consistency/variance/covariance, on corresponding nodes."""
    za, zb = a['z'], b['z']
    inv = F.mse_loss(za, zb)
    var = za.new_zeros(())
    cov = za.new_zeros(())
    if len(za) > 1:
        for z in (za, zb):
            var = var + F.relu(1 - torch.sqrt(z.var(0, unbiased=False) + 1e-4)).mean()
            centered = z - z.mean(0)
            c = centered.T @ centered / (len(z) - 1)
            cov = cov + (c - torch.diag(torch.diag(c))).square().sum() / z.shape[1]
    return config['consistency_weight'] * inv + config['variance_weight'] * var + config['covariance_weight'] * cov


def weak_loss(output, labels, margin=1.0):
    """Labels -2 unknown, -1 background, >=0 event-local group ID."""
    known = labels >= -1
    zero = output['z'].sum() * 0
    node = F.binary_cross_entropy_with_logits(output['signal_logit'][known], (labels[known] >= 0).float()) if known.any() else zero
    # Background is not one common object; never pull background nodes together.
    i, j = torch.triu_indices(len(labels), len(labels), offset=1, device=labels.device)
    valid = known[i] & known[j] & ((labels[i] >= 0) | (labels[j] >= 0))
    i, j = i[valid], j[valid]
    if not len(i):
        return node, zero
    distance = (output['z'][i] - output['z'][j]).square().mean(1).add(1e-8).sqrt()
    same = (labels[i] == labels[j]) & (labels[i] >= 0)
    # Class-balanced within each event; labels have no cross-event meaning.
    terms = []
    if same.any():
        terms.append(distance[same].square().mean())
    if (~same).any():
        terms.append(F.relu(margin - distance[~same]).square().mean())
    return node, sum(terms) / len(terms)
