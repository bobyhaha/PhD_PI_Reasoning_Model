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
        norm = torch.stack([
            lora_delta_norm(low_adapters),
            lora_delta_norm(high_adapters),
        ]).mean()
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
            lora_norms.append(torch.stack([
                lora_delta_norm(low_adapters),
                lora_delta_norm(high_adapters),
            ]).mean())

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
