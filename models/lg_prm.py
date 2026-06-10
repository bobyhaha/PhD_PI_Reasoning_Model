import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import TokenEncoder

class Explorer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = max(8, int(cfg.d_model * cfg.d_explorer_mult))
        self.net = nn.Sequential(nn.LayerNorm(2*cfg.d_model), nn.Linear(2*cfg.d_model, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, cfg.d_model))
    def forward(self, state, xs):
        return self.net(torch.cat([state, xs], dim=-1))

class Library(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        size = getattr(cfg, 'rag_library_size', cfg.library_size)
        self.keys = nn.Parameter(torch.randn(size, cfg.d_model) / math.sqrt(cfg.d_model))
        self.values = nn.Parameter(torch.randn(size, cfg.d_model) / math.sqrt(cfg.d_model))
    def forward(self, q):
        attn = F.softmax(q @ self.keys.T / math.sqrt(q.shape[-1]), dim=-1)
        return attn @ self.values, attn

class NeuralLibrary(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg.mlp_library_mult * cfg.d_model
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_hidden, cfg.d_model),
        )
    def forward(self, q):
        return self.net(q)

class Gate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(2*cfg.d_model), nn.Linear(2*cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 1))
    def forward(self, state, xs):
        return torch.sigmoid(self.net(torch.cat([state, xs], dim=-1)))

class PISynthesizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_proj = nn.Linear(4*cfg.d_model, cfg.d_model)
        layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.dim_ff_mult*cfg.d_model, dropout=cfg.dropout, batch_first=True, activation='gelu')
        self.tr = nn.TransformerEncoder(layer, num_layers=cfg.pi_layers)
        self.out = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model))
    def forward(self, state, xs, explorer_summary, libvec):
        z = self.in_proj(torch.cat([state, xs, explorer_summary, libvec], dim=-1)).unsqueeze(1)
        return state + self.out(self.tr(z).squeeze(1))

class LibraryGatedPRM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = TokenEncoder(cfg)
        self.init = nn.Linear(cfg.d_model, cfg.d_model)
        self.explorers = nn.ModuleList([Explorer(cfg) for _ in range(cfg.n_explorers)])
        self.rag_library = Library(cfg)
        self.mlp_library = NeuralLibrary(cfg)
        self.gate = Gate(cfg)
        self.pi = PISynthesizer(cfg)
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.num_classes))

    def _gate_mask(self, gate_prob):
        hard = torch.ones_like(gate_prob) if self.cfg.forced_library else (gate_prob >= self.cfg.gate_threshold).to(gate_prob.dtype)
        if self.training and self.cfg.straight_through_gate:
            return hard + gate_prob - gate_prob.detach()
        return hard

    def _retrieve(self, state, gate_prob):
        gate = self._gate_mask(gate_prob) if self.cfg.hard_library_gate or self.cfg.forced_library else gate_prob
        active = gate.squeeze(-1) > 0
        libvec = torch.zeros_like(state)
        entropy = state.new_tensor(0.0)
        if active.any():
            ret, attn = self.rag_library(state[active])
            mlp_ret = self.mlp_library(state[active])
            libvec[active] = gate[active] * 0.5 * (ret + mlp_ret)
            entropy = -(attn * attn.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return libvec, entropy, gate

    def forward(self, x):
        token_features = self.encoder(x)
        xs = token_features.mean(dim=1)
        state = self.init(xs)
        gates, hard_gates, props_all, ents = [], [], [], []
        for _ in range(self.cfg.lg_steps):
            props = torch.stack([e(state, xs) for e in self.explorers], dim=1)
            props_all.append(props)
            ps = props.mean(dim=1)
            if self.cfg.use_library:
                gate = self.gate(state, xs)
                if self.cfg.forced_library:
                    gate = torch.ones_like(gate)
                libvec, ent, hard_gate = self._retrieve(state, gate)
                ents.append(ent)
            else:
                gate = torch.zeros(state.shape[0], 1, device=state.device)
                hard_gate = gate
                libvec = torch.zeros_like(state)
            gates.append(gate)
            hard_gates.append(hard_gate.detach())
            state = self.pi(state, xs, ps, libvec)
        token_logits = self.head(token_features + state.unsqueeze(1))
        out = {
            'logits': self.head(state),
            'token_logits': token_logits,
            'gate_probs': torch.stack(gates, dim=1),
            'gate_hard': torch.stack(hard_gates, dim=1),
            'proposals': torch.cat(props_all, dim=1),
        }
        out['library_entropy'] = torch.stack(ents).mean() if ents else torch.tensor(0.0, device=x.device)
        return out
