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

    def _initial_states(self, xs, init_noise_std=0.0):
        h = self.h_init.expand(xs.shape[0], xs.shape[1], -1)
        l = self.l_init.expand_as(h)
        if init_noise_std and init_noise_std > 0:
            h = h + torch.randn_like(h) * float(init_noise_std)
            l = l + torch.randn_like(l) * float(init_noise_std)
        return h, l

    def forward_depth(self, x, eval_depth=1, init_noise_std=0.0, noise_scale=None, grad_last_only=False, residual_window=0):
        del noise_scale
        xs = self.encoder(x)
        h, l = self._initial_states(xs, init_noise_std=init_noise_std)
        total_cycles = max(1, int(eval_depth)) * max(1, int(self.cfg.h_cycles))
        residual_window = max(0, int(residual_window or 0))
        residuals = []
        if grad_last_only and total_cycles > 1:
            with torch.no_grad():
                for _ in range(total_cycles - 1):
                    h, l = self._cycle(h, l, xs)
        for _ in range(1 if grad_last_only and total_cycles > 1 else total_cycles):
            prev_h, prev_l = h, l
            h, l = self._cycle(h, l, xs)
            if residual_window:
                residuals.append(((h - prev_h).pow(2).mean(dim=(1, 2)) + (l - prev_l).pow(2).mean(dim=(1, 2))) * 0.5)
        z = h + l
        token_logits = self.head(z)
        out = {'logits': token_logits[:, 0], 'token_logits': token_logits, 'token_state': z}
        if residuals:
            out['residual_score'] = torch.stack(residuals[-residual_window:], dim=0).mean(dim=0)
        return out

    def forward(self, x):
        xs = self.encoder(x)
        h, l = self._initial_states(xs)
        with torch.no_grad():
            for _ in range(max(0, self.cfg.h_cycles - 1)):
                h, l = self._cycle(h, l, xs)
        h, l = self._cycle(h, l, xs)
        z = h + l
        token_logits = self.head(z)
        return {'logits': token_logits[:, 0], 'token_logits': token_logits, 'token_state': z}
