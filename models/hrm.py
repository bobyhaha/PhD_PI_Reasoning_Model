import torch
import torch.nn as nn
from .common import TokenEncoder, TokenReasoningCore

class HRM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = TokenEncoder(cfg)
        self.h_init = nn.Parameter(torch.randn(1, 1, cfg.d_model) / (cfg.d_model ** 0.5))
        self.l_init = nn.Parameter(torch.randn(1, 1, cfg.d_model) / (cfg.d_model ** 0.5))
        self.low = TokenReasoningCore(cfg, cfg.l_layers)
        self.high = TokenReasoningCore(cfg, cfg.h_layers)
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.num_classes))

    def _cycle(self, h, l, xs):
        for _ in range(self.cfg.l_cycles):
            l = self.low(l, h + xs)
        h = self.high(h, l)
        return h, l

    def forward(self, x):
        xs = self.encoder(x)
        h = self.h_init.expand(xs.shape[0], xs.shape[1], -1)
        l = self.l_init.expand_as(h)
        with torch.no_grad():
            for _ in range(max(0, self.cfg.h_cycles - 1)):
                h, l = self._cycle(h, l, xs)
        h, l = self._cycle(h, l, xs)
        z = h + l
        token_logits = self.head(z)
        return {'logits': token_logits[:, 0], 'token_logits': token_logits, 'token_state': z}
