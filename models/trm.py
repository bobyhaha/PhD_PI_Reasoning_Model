import torch
import torch.nn as nn
from .common import TokenEncoder, TokenReasoningCore

class TRM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = TokenEncoder(cfg)
        self.init = nn.Parameter(torch.randn(1, 1, cfg.d_model) / (cfg.d_model ** 0.5))
        self.core = TokenReasoningCore(cfg, cfg.trm_layers)
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.num_classes))

    def _step(self, z, xs):
        return self.core(z, xs)

    def forward(self, x):
        xs = self.encoder(x)
        z = self.init.expand(xs.shape[0], xs.shape[1], -1)
        with torch.no_grad():
            for _ in range(max(0, self.cfg.trm_steps - 1)):
                z = self._step(z, xs)
        z = self._step(z, xs)
        token_logits = self.head(z)
        return {'logits': token_logits[:, 0], 'token_logits': token_logits, 'token_state': z}
