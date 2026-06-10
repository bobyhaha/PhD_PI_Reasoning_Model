import torch.nn as nn
from .common import TokenEncoder

class StandardTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = TokenEncoder(cfg)
        layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.dim_ff_mult*cfg.d_model, dropout=cfg.dropout, batch_first=True, activation='gelu')
        self.tr = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.num_classes))
    def forward(self, x):
        h = self.tr(self.encoder(x))
        token_logits = self.head(h)
        return {'logits': token_logits[:, 0], 'token_logits': token_logits, 'token_state': h}
