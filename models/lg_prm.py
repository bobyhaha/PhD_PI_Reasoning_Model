import torch
import torch.nn as nn
from .common import TokenEncoder

class Explorer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = max(8, int(cfg.d_model * cfg.d_explorer_mult))
        self.net = nn.Sequential(nn.LayerNorm(2*cfg.d_model), nn.Linear(2*cfg.d_model, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, cfg.d_model))
    def forward(self, state, xs):
        return self.net(torch.cat([state, xs], dim=-1))

class PISynthesizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_proj = nn.Linear(3*cfg.d_model, cfg.d_model)
        layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.dim_ff_mult*cfg.d_model, dropout=cfg.dropout, batch_first=True, activation='gelu')
        self.tr = nn.TransformerEncoder(layer, num_layers=cfg.pi_layers)
        self.out = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model))
    def forward(self, state, xs, explorer_summary):
        z = self.in_proj(torch.cat([state, xs, explorer_summary], dim=-1)).unsqueeze(1)
        return state + self.out(self.tr(z).squeeze(1))

class PhDPIPRM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = TokenEncoder(cfg)
        self.init = nn.Linear(cfg.d_model, cfg.d_model)
        self.explorers = nn.ModuleList([Explorer(cfg) for _ in range(cfg.n_explorers)])
        self.pi = PISynthesizer(cfg)
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.num_classes))

    def forward(self, x):
        token_features = self.encoder(x)
        xs = token_features.mean(dim=1)
        state = self.init(xs)
        props_all = []
        for _ in range(self.cfg.lg_steps):
            props = torch.stack([e(state, xs) for e in self.explorers], dim=1)
            props_all.append(props)
            ps = props.mean(dim=1)
            state = self.pi(state, xs, ps)
        token_logits = self.head(token_features + state.unsqueeze(1))
        out = {
            'logits': self.head(state),
            'token_logits': token_logits,
            'proposals': torch.cat(props_all, dim=1),
        }
        return out
