from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    model_type: str = 'transformer'
    vocab_size: int = 128
    num_classes: int = 17
    max_seq_len: int = 31
    d_model: int = 128
    n_heads: int = 4
    dropout: float = 0.1
    n_layers: int = 4
    dim_ff_mult: int = 4
    h_cycles: int = 2
    l_cycles: int = 4
    h_layers: int = 1
    l_layers: int = 1
    trm_steps: int = 8
    trm_layers: int = 2
    n_explorers: int = 8
    d_explorer_mult: float = 0.5
    pi_layers: int = 2
    library_size: int = 128
    rag_library_size: int = 128
    mlp_library_mult: int = 4
    lg_steps: int = 4
    use_library: bool = True
    forced_library: bool = False
    hard_library_gate: bool = False
    gate_threshold: float = 0.5
    straight_through_gate: bool = True
    use_diversity_loss: bool = True

class TokenEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.ln = nn.LayerNorm(cfg.d_model)
    def forward(self, x):
        _, l = x.shape
        pos = torch.arange(l, device=x.device)
        return self.ln(self.token(x) + self.pos(pos)[None, :, :])

class MLPBlock(nn.Module):
    def __init__(self, d_model, mult=4, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.net = nn.Sequential(nn.Linear(d_model, mult*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(mult*d_model, d_model))
    def forward(self, x):
        return x + self.net(self.ln(x))

class TinyCore(nn.Module):
    def __init__(self, d_model, n_layers=2, dropout=0.1, mult=4):
        super().__init__()
        self.blocks = nn.ModuleList([MLPBlock(d_model, mult, dropout) for _ in range(n_layers)])
    def forward(self, z):
        for b in self.blocks:
            z = b(z)
        return z

class TokenReasoningBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_ln = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ff_ln = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_ff_mult * cfg.d_model),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_ff_mult * cfg.d_model, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)
    def forward(self, z):
        h = self.attn_ln(z)
        attn, _ = self.attn(h, h, h, need_weights=False)
        z = z + self.dropout(attn)
        return z + self.dropout(self.ff(self.ff_ln(z)))

class TokenReasoningCore(nn.Module):
    def __init__(self, cfg, n_layers):
        super().__init__()
        self.blocks = nn.ModuleList([TokenReasoningBlock(cfg) for _ in range(n_layers)])
    def forward(self, z, injection=None):
        if injection is not None:
            z = z + injection
        for block in self.blocks:
            z = block(z)
        return z

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def diversity_loss(proposals):
    if proposals.shape[1] <= 1:
        return proposals.new_tensor(0.0)
    z = F.normalize(proposals, dim=-1)
    sim = torch.einsum('bed,bfd->bef', z, z)
    e = sim.shape[1]
    mask = ~torch.eye(e, dtype=torch.bool, device=sim.device)
    return sim[:, mask].pow(2).mean()
