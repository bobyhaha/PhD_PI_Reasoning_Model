"""
HRM meta-model variants.

Architecture recap (base HRM):
    - encoder:  TokenEncoder
    - low core: TokenReasoningCore(l_layers)  – iterated l_cycles times per h_cycle
    - high core: TokenReasoningCore(h_layers) – iterated once per h_cycle
    - h_cycles h-cycles total; early cycles run under torch.no_grad in base model

Meta variants use full BPTT (no torch.no_grad) for proper gradient flow to the
meta-modules.  This uses more memory but is required for correct training.

HRM_V1 – PoLar controller
    Input → xs_pooled → PoLarController → flat gate vector.
    Gates are indexed as:  [h_cycles × (l_cycles×l_layers + h_layers)].
    At each h-cycle c:
      - l_cycles × l_layers low-gates  (same gate for all l-cycles in c)
      - h_layers high-gates

HRM_V2 – LoRA HyperNet per h-cycle
    At each h-cycle, two separate LoRAHyperNets produce adapters for the
    low core (context = xs_pooled + h_t_pooled) and the high core
    (context = xs_pooled + l_t_pooled).
    Adapters are shared across the l_cycles inner iterations of the low core.

HRM_V3 – Combined
    Both PoLar gates and LoRA adapters active simultaneously.
"""

import torch
import torch.nn as nn
from .common import TokenEncoder
from .meta_modules import (
    PoLarController,
    LoRAReasoningCore,
    LoRAHyperNet,
    lora_delta_norm,
)

def _adapter_norm_mean(*adapter_lists):
    norms = [lora_delta_norm(adapters) for adapters in adapter_lists if adapters]
    if norms:
        return torch.stack(norms).mean()
    return None


class HRM_V1(nn.Module):
    """HRM with PoLar layer-program selection over low and high cores."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg     = cfg
        self.encoder = TokenEncoder(cfg)
        self.h_init  = nn.Parameter(
            torch.randn(1, 1, cfg.d_model) / (cfg.d_model ** 0.5)
        )
        self.l_init  = nn.Parameter(
            torch.randn(1, 1, cfg.d_model) / (cfg.d_model ** 0.5)
        )
        self.low  = LoRAReasoningCore(cfg, cfg.l_layers)
        self.high = LoRAReasoningCore(cfg, cfg.h_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )
        # Gate layout per h-cycle:
        #   [l_cycles * l_layers] low gates  +  [h_layers] high gates
        gates_per_cycle = cfg.l_cycles * cfg.l_layers + cfg.h_layers
        n_decisions     = cfg.h_cycles * gates_per_cycle
        self.polar      = PoLarController(cfg.d_model, n_decisions)
        self._gpc       = gates_per_cycle   # gates per h-cycle (cached)

    def _cycle(self, h, l, xs, low_gates, high_gates):
        for lc in range(self.cfg.l_cycles):
            seg      = low_gates[lc * self.cfg.l_layers :
                                 (lc + 1) * self.cfg.l_layers]
            l = self.low(l, h + xs, gate_list=seg)
        h = self.high(h, l, gate_list=high_gates)
        return h, l

    def forward(self, x):
        xs        = self.encoder(x)
        h         = self.h_init.expand(xs.shape[0], xs.shape[1], -1)
        l         = self.l_init.expand_as(h)
        xs_pooled = xs.mean(dim=1)
        all_gates = self.polar(xs_pooled)    # [B, h_cycles * gpc]
        gpc       = self._gpc
        ll, hl    = self.cfg.l_cycles * self.cfg.l_layers, self.cfg.h_layers

        for c in range(self.cfg.h_cycles):
            seg        = all_gates[:, c * gpc : (c + 1) * gpc]  # [B, gpc]
            low_gates  = [seg[:, b] for b in range(ll)]
            high_gates = [seg[:, ll + b] for b in range(hl)]
            h, l       = self._cycle(h, l, xs, low_gates, high_gates)

        z            = h + l
        token_logits = self.head(z)
        return {
            'logits'       : token_logits[:, 0],
            'token_logits' : token_logits,
            'token_state'  : z,
            'polar_gates'  : all_gates,
            'polar_usage'  : all_gates.mean(),
        }

    def forward_depth(self, x, eval_depth=1, init_noise_std=0.0, noise_scale=None, grad_last_only=False, residual_window=0):
        del noise_scale
        xs = self.encoder(x)
        h = self.h_init.expand(xs.shape[0], xs.shape[1], -1)
        l = self.l_init.expand_as(h)
        if init_noise_std and init_noise_std > 0:
            h = h + torch.randn_like(h) * float(init_noise_std)
            l = l + torch.randn_like(l) * float(init_noise_std)
        xs_pooled = xs.mean(dim=1)
        all_gates = self.polar(xs_pooled)
        gpc = self._gpc
        ll, hl = self.cfg.l_cycles * self.cfg.l_layers, self.cfg.h_layers
        total_cycles = max(1, int(eval_depth)) * max(1, int(self.cfg.h_cycles))
        residual_window = max(0, int(residual_window or 0))
        residuals = []
        start_c = 0
        if grad_last_only and total_cycles > 1:
            with torch.no_grad():
                for c in range(total_cycles - 1):
                    slot = c % max(1, int(self.cfg.h_cycles))
                    seg = all_gates[:, slot * gpc : (slot + 1) * gpc]
                    low_gates = [seg[:, b] for b in range(ll)]
                    high_gates = [seg[:, ll + b] for b in range(hl)]
                    h, l = self._cycle(h, l, xs, low_gates, high_gates)
            start_c = total_cycles - 1
        for c in range(start_c, total_cycles):
            slot = c % max(1, int(self.cfg.h_cycles))
            seg = all_gates[:, slot * gpc : (slot + 1) * gpc]
            low_gates = [seg[:, b] for b in range(ll)]
            high_gates = [seg[:, ll + b] for b in range(hl)]
            prev_h, prev_l = h, l
            h, l = self._cycle(h, l, xs, low_gates, high_gates)
            if residual_window:
                residuals.append(((h - prev_h).pow(2).mean(dim=(1, 2)) + (l - prev_l).pow(2).mean(dim=(1, 2))) * 0.5)
        z = h + l
        token_logits = self.head(z)
        out = {
            'logits': token_logits[:, 0],
            'token_logits': token_logits,
            'token_state': z,
            'polar_gates': all_gates,
            'polar_usage': all_gates.mean(),
        }
        if residuals:
            out['residual_score'] = torch.stack(residuals[-residual_window:], dim=0).mean(dim=0)
        return out


class HRM_V2(nn.Module):
    """HRM with separate LoRA hypernetworks for low and high cores per h-cycle."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg     = cfg
        self.encoder = TokenEncoder(cfg)
        self.h_init  = nn.Parameter(
            torch.randn(1, 1, cfg.d_model) / (cfg.d_model ** 0.5)
        )
        self.l_init  = nn.Parameter(
            torch.randn(1, 1, cfg.d_model) / (cfg.d_model ** 0.5)
        )
        self.low      = LoRAReasoningCore(cfg, cfg.l_layers)
        self.high     = LoRAReasoningCore(cfg, cfg.h_layers)
        self.low_hyper  = LoRAHyperNet(cfg, n_blocks=cfg.l_layers)
        self.high_hyper = LoRAHyperNet(cfg, n_blocks=cfg.h_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

    def _cycle(self, h, l, xs):
        xs_p = xs.mean(dim=1)
        h_p  = h.mean(dim=1)

        # Low adapters: conditioned on (xs, h) – h drives the low exploration
        low_adapters = self.low_hyper(
            torch.cat([xs_p, h_p], dim=-1)
        )
        for _ in range(self.cfg.l_cycles):
            l = self.low(l, h + xs, lora_list=low_adapters)

        # High adapters: conditioned on (xs, l) – l carries the low-level result
        l_p = l.mean(dim=1)
        high_adapters = self.high_hyper(
            torch.cat([xs_p, l_p], dim=-1)
        )
        h = self.high(h, l, lora_list=high_adapters)
        norm = _adapter_norm_mean(low_adapters, high_adapters)
        if norm is None:
            norm = h.new_tensor(0.0)
        return h, l, norm

    def forward(self, x):
        xs = self.encoder(x)
        h  = self.h_init.expand(xs.shape[0], xs.shape[1], -1)
        l  = self.l_init.expand_as(h)

        lora_norms = []
        for _ in range(self.cfg.h_cycles):
            h, l, norm = self._cycle(h, l, xs)
            lora_norms.append(norm)

        z            = h + l
        token_logits = self.head(z)
        return {
            'logits'       : token_logits[:, 0],
            'token_logits' : token_logits,
            'token_state'  : z,
            'lora_delta_norm': torch.stack(lora_norms).mean(),
        }

    def forward_depth(self, x, eval_depth=1, init_noise_std=0.0, noise_scale=None, grad_last_only=False, residual_window=0):
        del noise_scale
        xs = self.encoder(x)
        h = self.h_init.expand(xs.shape[0], xs.shape[1], -1)
        l = self.l_init.expand_as(h)
        if init_noise_std and init_noise_std > 0:
            h = h + torch.randn_like(h) * float(init_noise_std)
            l = l + torch.randn_like(l) * float(init_noise_std)
        lora_norms = []
        total_cycles = max(1, int(eval_depth)) * max(1, int(self.cfg.h_cycles))
        residual_window = max(0, int(residual_window or 0))
        residuals = []
        if grad_last_only and total_cycles > 1:
            with torch.no_grad():
                for _ in range(total_cycles - 1):
                    h, l, _ = self._cycle(h, l, xs)
        for _ in range(1 if grad_last_only and total_cycles > 1 else total_cycles):
            prev_h, prev_l = h, l
            h, l, norm = self._cycle(h, l, xs)
            lora_norms.append(norm)
            if residual_window:
                residuals.append(((h - prev_h).pow(2).mean(dim=(1, 2)) + (l - prev_l).pow(2).mean(dim=(1, 2))) * 0.5)
        z = h + l
        token_logits = self.head(z)
        out = {
            'logits': token_logits[:, 0],
            'token_logits': token_logits,
            'token_state': z,
            'lora_delta_norm': torch.stack(lora_norms).mean(),
        }
        if residuals:
            out['residual_score'] = torch.stack(residuals[-residual_window:], dim=0).mean(dim=0)
        return out


class HRM_V3(nn.Module):
    """HRM with both PoLar gates and LoRA per-cycle adapters."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg     = cfg
        self.encoder = TokenEncoder(cfg)
        self.h_init  = nn.Parameter(
            torch.randn(1, 1, cfg.d_model) / (cfg.d_model ** 0.5)
        )
        self.l_init  = nn.Parameter(
            torch.randn(1, 1, cfg.d_model) / (cfg.d_model ** 0.5)
        )
        self.low  = LoRAReasoningCore(cfg, cfg.l_layers)
        self.high = LoRAReasoningCore(cfg, cfg.h_layers)
        self.low_hyper  = LoRAHyperNet(cfg, n_blocks=cfg.l_layers)
        self.high_hyper = LoRAHyperNet(cfg, n_blocks=cfg.h_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )
        gates_per_cycle = cfg.l_cycles * cfg.l_layers + cfg.h_layers
        n_decisions     = cfg.h_cycles * gates_per_cycle
        self.polar      = PoLarController(cfg.d_model, n_decisions)
        self._gpc       = gates_per_cycle
        self._ll        = cfg.l_cycles * cfg.l_layers

    def _cycle(self, h, l, xs, low_gates, high_gates, low_adapters, high_adapters):
        for lc in range(self.cfg.l_cycles):
            seg = low_gates[lc * self.cfg.l_layers :
                            (lc + 1) * self.cfg.l_layers]
            l = self.low(l, h + xs, lora_list=low_adapters, gate_list=seg)
        h = self.high(h, l, lora_list=high_adapters, gate_list=high_gates)
        return h, l

    def forward(self, x):
        xs        = self.encoder(x)
        h         = self.h_init.expand(xs.shape[0], xs.shape[1], -1)
        l         = self.l_init.expand_as(h)
        xs_pooled = xs.mean(dim=1)
        all_gates = self.polar(xs_pooled)
        gpc, ll, hl = self._gpc, self._ll, self.cfg.h_layers
        lora_norms = []

        for c in range(self.cfg.h_cycles):
            h_p  = h.mean(dim=1)
            seg  = all_gates[:, c * gpc : (c + 1) * gpc]

            low_gates    = [seg[:, b] for b in range(ll)]
            high_gates   = [seg[:, ll + b] for b in range(hl)]
            low_adapters = self.low_hyper(torch.cat([xs_pooled, h_p], dim=-1))
            l_after_low = l
            for lc in range(self.cfg.l_cycles):
                low_seg = low_gates[lc * self.cfg.l_layers :
                                    (lc + 1) * self.cfg.l_layers]
                l_after_low = self.low(
                    l_after_low,
                    h + xs,
                    lora_list=low_adapters,
                    gate_list=low_seg,
                )
            l_p_after_low = l_after_low.mean(dim=1)
            high_adapters = self.high_hyper(torch.cat([xs_pooled, l_p_after_low], dim=-1))

            h = self.high(h, l_after_low, lora_list=high_adapters, gate_list=high_gates)
            l = l_after_low
            norm = _adapter_norm_mean(low_adapters, high_adapters)
            if norm is not None:
                lora_norms.append(norm)

        z            = h + l
        token_logits = self.head(z)
        return {
            'logits'       : token_logits[:, 0],
            'token_logits' : token_logits,
            'token_state'  : z,
            'polar_gates'  : all_gates,
            'polar_usage'  : all_gates.mean(),
            'lora_delta_norm': torch.stack(lora_norms).mean(),
        }

    def forward_depth(self, x, eval_depth=1, init_noise_std=0.0, noise_scale=None, grad_last_only=False, residual_window=0):
        del noise_scale
        xs = self.encoder(x)
        h = self.h_init.expand(xs.shape[0], xs.shape[1], -1)
        l = self.l_init.expand_as(h)
        if init_noise_std and init_noise_std > 0:
            h = h + torch.randn_like(h) * float(init_noise_std)
            l = l + torch.randn_like(l) * float(init_noise_std)
        xs_pooled = xs.mean(dim=1)
        all_gates = self.polar(xs_pooled)
        gpc, ll, hl = self._gpc, self._ll, self.cfg.h_layers
        lora_norms = []
        total_cycles = max(1, int(eval_depth)) * max(1, int(self.cfg.h_cycles))
        residual_window = max(0, int(residual_window or 0))
        residuals = []
        start_c = 0
        if grad_last_only and total_cycles > 1:
            with torch.no_grad():
                for c in range(total_cycles - 1):
                    slot = c % max(1, int(self.cfg.h_cycles))
                    seg = all_gates[:, slot * gpc : (slot + 1) * gpc]
                    low_gates = [seg[:, b] for b in range(ll)]
                    high_gates = [seg[:, ll + b] for b in range(hl)]
                    low_adapters = self.low_hyper(torch.cat([xs_pooled, h.mean(dim=1)], dim=-1))
                    l_after_low = l
                    for lc in range(self.cfg.l_cycles):
                        low_seg = low_gates[lc * self.cfg.l_layers : (lc + 1) * self.cfg.l_layers]
                        l_after_low = self.low(l_after_low, h + xs, lora_list=low_adapters, gate_list=low_seg)
                    high_adapters = self.high_hyper(torch.cat([xs_pooled, l_after_low.mean(dim=1)], dim=-1))
                    h = self.high(h, l_after_low, lora_list=high_adapters, gate_list=high_gates)
                    l = l_after_low
            start_c = total_cycles - 1
        for c in range(start_c, total_cycles):
            slot = c % max(1, int(self.cfg.h_cycles))
            seg = all_gates[:, slot * gpc : (slot + 1) * gpc]
            low_gates = [seg[:, b] for b in range(ll)]
            high_gates = [seg[:, ll + b] for b in range(hl)]
            low_adapters = self.low_hyper(torch.cat([xs_pooled, h.mean(dim=1)], dim=-1))
            l_after_low = l
            prev_h, prev_l = h, l
            for lc in range(self.cfg.l_cycles):
                low_seg = low_gates[lc * self.cfg.l_layers : (lc + 1) * self.cfg.l_layers]
                l_after_low = self.low(l_after_low, h + xs, lora_list=low_adapters, gate_list=low_seg)
            high_adapters = self.high_hyper(torch.cat([xs_pooled, l_after_low.mean(dim=1)], dim=-1))
            h = self.high(h, l_after_low, lora_list=high_adapters, gate_list=high_gates)
            l = l_after_low
            norm = _adapter_norm_mean(low_adapters, high_adapters)
            if norm is not None:
                lora_norms.append(norm)
            if residual_window:
                residuals.append(((h - prev_h).pow(2).mean(dim=(1, 2)) + (l - prev_l).pow(2).mean(dim=(1, 2))) * 0.5)
        z = h + l
        token_logits = self.head(z)
        out = {
            'logits': token_logits[:, 0],
            'token_logits': token_logits,
            'token_state': z,
            'polar_gates': all_gates,
            'polar_usage': all_gates.mean(),
            'lora_delta_norm': torch.stack(lora_norms).mean() if lora_norms else torch.tensor(0.0, device=x.device),
        }
        if residuals:
            out['residual_score'] = torch.stack(residuals[-residual_window:], dim=0).mean(dim=0)
        return out
