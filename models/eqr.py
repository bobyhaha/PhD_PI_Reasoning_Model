import math

import torch
import torch.nn as nn


def _trunc_normal(shape, *, std, device, dtype):
    x = torch.empty(shape, device=device, dtype=dtype)
    return nn.init.trunc_normal_(x, mean=0.0, std=float(std), a=-2 * float(std), b=2 * float(std))


def _rms_norm(x, eps=1e-5):
    return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True).add(eps)).to(dtype=x.dtype)


class EqRBlock(nn.Module):
    """Official EqR-style recurrent block with optional token-mixing MLP."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        ff_d = cfg.dim_ff_mult * d
        if cfg.mlp_t:
            self.mlp_t = nn.Sequential(
                nn.LayerNorm(cfg.max_seq_len),
                nn.Linear(cfg.max_seq_len, cfg.dim_ff_mult * cfg.max_seq_len),
                nn.SiLU(),
                nn.Linear(cfg.dim_ff_mult * cfg.max_seq_len, cfg.max_seq_len),
            )
        else:
            self.attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
            self.attn_ln = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, ff_d),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(ff_d, d),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, z):
        if self.cfg.mlp_t:
            mixed = self.mlp_t(z.transpose(1, 2)).transpose(1, 2)
            z = _rms_norm(z + self.drop(mixed))
        else:
            h = self.attn_ln(z)
            attn, _ = self.attn(h, h, h, need_weights=False)
            z = _rms_norm(z + self.drop(attn))
        return _rms_norm(z + self.drop(self.ff(z)))


class NoisyReasoningModule(nn.Module):
    def __init__(self, cfg, n_layers):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([EqRBlock(cfg) for _ in range(n_layers)])
        self.state_max_rms = float(getattr(cfg, "state_max_rms", 0.0) or 0.0)

    def _limit_state(self, z):
        if self.state_max_rms <= 0:
            return z
        rms = z.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        scale = torch.clamp(self.state_max_rms / rms, max=1.0).to(dtype=z.dtype)
        return z * scale

    def forward(self, z, injection, noise_scale=None):
        updated = self._limit_state(z + injection)
        for block in self.blocks:
            updated = self._limit_state(block(updated))
        out = (1.0 - float(self.cfg.phd_lambda)) * z + float(self.cfg.phd_lambda) * updated
        scale = float(self.cfg.phd_noise_scale if noise_scale is None else noise_scale)
        if scale > 0:
            out = out + torch.randn_like(out) * scale
        return self._limit_state(out)


class EqR(nn.Module):
    """EqR attractor model following the released locuslab algorithmic recipe.

    This keeps the repository's simple tensor API, but uses the important EqR
    ingredients: random latent resets, damped/noisy recurrences, H/L latent
    hierarchy, and a q-halt head trained by the ACT-style loss in train.py.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_scale = math.sqrt(cfg.d_model)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model) if not cfg.mlp_t else None
        self.level = NoisyReasoningModule(cfg, cfg.l_layers)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes, bias=False)
        self.q_head = nn.Linear(cfg.d_model, 1)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)

    def _input_embeddings(self, x):
        emb = self.embed_scale * self.embed(x)
        if self.pos is not None:
            pos = torch.arange(x.shape[1], device=x.device)
            emb = (emb + self.embed_scale * self.pos(pos)[None, :, :]) * 0.707106781
        return emb

    def _initial_states(self, xs, init_noise_std=None):
        std = float(self.cfg.init_std if init_noise_std is None else init_noise_std)
        h = _trunc_normal(xs.shape, std=std, device=xs.device, dtype=xs.dtype)
        l = _trunc_normal(xs.shape, std=std, device=xs.device, dtype=xs.dtype)
        return h, l

    def _outer_step(self, h, l, xs, noise_scale=None):
        for _ in range(self.cfg.l_cycles):
            l = self.level(l, h + xs, noise_scale=noise_scale)
        h = self.level(h, l, noise_scale=noise_scale)
        return h, l

    def _readout(self, h, l, residuals=None):
        del l
        token_logits = self.head(h)
        out = {
            "logits": token_logits[:, 0],
            "token_logits": token_logits,
            "token_state": h,
            "q_halt_logits": self.q_head(h[:, 0]).squeeze(-1).to(torch.float32),
        }
        if residuals:
            out["residual_score"] = torch.stack(residuals, dim=0).mean(dim=0)
        return out

    def forward_depth(
        self,
        x,
        eval_depth=1,
        init_noise_std=None,
        noise_scale=None,
        grad_last_only=False,
        residual_window=0,
    ):
        xs = self._input_embeddings(x)
        h, l = self._initial_states(xs, init_noise_std=init_noise_std)
        steps = max(1, int(eval_depth))
        residual_window = max(0, int(residual_window or 0))
        residuals = []
        if grad_last_only and steps > 1:
            with torch.no_grad():
                for _ in range(steps - 1):
                    h, l = self._outer_step(h, l, xs, noise_scale=noise_scale)
        for _ in range(1 if grad_last_only and steps > 1 else steps):
            prev_h, prev_l = h, l
            h, l = self._outer_step(h, l, xs, noise_scale=noise_scale)
            if residual_window:
                score = ((h - prev_h).pow(2).mean(dim=(1, 2)) + (l - prev_l).pow(2).mean(dim=(1, 2))) * 0.5
                residuals.append(score)
                residuals = residuals[-residual_window:]
        return self._readout(h, l, residuals)

    def forward_act(self, x, halt_max_steps=None, init_noise_std=None, noise_scale=None, exploration_prob=None):
        xs = self._input_embeddings(x)
        h, l = self._initial_states(xs, init_noise_std=init_noise_std)
        max_steps = max(1, int(halt_max_steps or self.cfg.eqr_halt_max_steps))
        exploration_prob = float(self.cfg.halt_exploration_prob if exploration_prob is None else exploration_prob)
        final_out = None
        chosen = None
        min_steps = torch.ones(x.shape[0], device=x.device, dtype=torch.long)
        if self.training and max_steps > 1 and exploration_prob > 0:
            explore = torch.rand(x.shape[0], device=x.device) < exploration_prob
            random_steps = torch.randint(2, max_steps + 1, (x.shape[0],), device=x.device)
            min_steps = torch.where(explore, random_steps, min_steps)

        for step in range(1, max_steps + 1):
            h, l = self._outer_step(h, l, xs, noise_scale=noise_scale)
            out = self._readout(h, l)
            final_out = out
            halt = out["q_halt_logits"] > 0
            halt = halt & (step >= min_steps)
            if step == max_steps:
                halt = torch.ones_like(halt)
            if chosen is None:
                chosen = {k: v for k, v in out.items()}
                chosen_mask = halt
            else:
                update = halt & ~chosen_mask
                chosen_mask = chosen_mask | halt
                for key, value in out.items():
                    if isinstance(value, torch.Tensor) and key in chosen and value.shape[:1] == update.shape:
                        shape = (update.shape[0],) + (1,) * (value.ndim - 1)
                        chosen[key] = torch.where(update.view(shape), value, chosen[key])
            if bool(chosen_mask.all().item()):
                break
        return chosen if chosen is not None else final_out

    def forward(self, x):
        if self.training:
            return self.forward_depth(
                x,
                eval_depth=self.cfg.eqr_halt_max_steps,
                init_noise_std=self.cfg.init_std,
                noise_scale=self.cfg.phd_noise_scale,
                grad_last_only=True,
            )
        return self.forward_depth(x, eval_depth=1, init_noise_std=self.cfg.init_std, noise_scale=0.0)
