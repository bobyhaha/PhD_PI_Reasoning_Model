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

    def _initial_state(self, xs, init_noise_std=0.0):
        z = self.init.expand(xs.shape[0], xs.shape[1], -1)
        if init_noise_std and init_noise_std > 0:
            z = z + torch.randn_like(z) * float(init_noise_std)
        return z

    def forward_depth(self, x, eval_depth=1, init_noise_std=0.0, noise_scale=None, grad_last_only=False, residual_window=0):
        del noise_scale
        xs = self.encoder(x)
        z = self._initial_state(xs, init_noise_std=init_noise_std)
        total_steps = max(1, int(eval_depth)) * max(1, int(self.cfg.trm_steps))
        residual_window = max(0, int(residual_window or 0))
        residuals = []
        if grad_last_only and total_steps > 1:
            with torch.no_grad():
                for _ in range(total_steps - 1):
                    z = self._step(z, xs)
        for _ in range(1 if grad_last_only and total_steps > 1 else total_steps):
            prev = z
            z = self._step(z, xs)
            if residual_window:
                residuals.append((z - prev).pow(2).mean(dim=(1, 2)))
        token_logits = self.head(z)
        out = {'logits': token_logits[:, 0], 'token_logits': token_logits, 'token_state': z}
        if residuals:
            out['residual_score'] = torch.stack(residuals[-residual_window:], dim=0).mean(dim=0)
        return out

    def forward(self, x):
        xs = self.encoder(x)
        z = self._initial_state(xs)
        with torch.no_grad():
            for _ in range(max(0, self.cfg.trm_steps - 1)):
                z = self._step(z, xs)
        z = self._step(z, xs)
        token_logits = self.head(z)
        return {'logits': token_logits[:, 0], 'token_logits': token_logits, 'token_state': z}
