"""
LG-PRM meta-model variants.

Architecture recap (base LG-PRM):
    encoder → init → [lg_steps × (PhD explorer bank + PI synthesizer)]
    → head

The PI synthesizer is the primary computation at each step.  We target it for
PoLar gating (V1) and LoRA adaptation (V2/V3) because it contains the majority
of reasoning capacity (pi_layers transformer layers).

The LoRAPISynthesizer (from meta_modules) replaces nn.TransformerEncoder with
LoRAReasoningCore so that the same module handles V1, V2, and V3 transparently.
Note: at L=1 (single-token PI input), MHA is trivially identity, so the
      effective computation equals the original PI synthesizer.

LGP_V1 – PoLar over PI layers each step
    PoLarController predicts (lg_steps × pi_layers) gates; at each step the
    corresponding pi_layers gates are passed to LoRAPISynthesizer.

LGP_V2 – LoRA HyperNet per step for PI layers
    At each step a LoRAHyperNet(context=(xs, state)) generates adapters for
    the pi_layers blocks inside LoRAPISynthesizer.

LGP_V3 – Combined
"""

import torch
import torch.nn as nn
from .common import TokenEncoder
from .meta_modules import (
    PoLarController,
    LoRAHyperNet,
    LoRAPISynthesizer,
    lora_delta_norm,
)


# ---------------------------------------------------------------------------
# Sub-modules shared with base LG-PRM
# ---------------------------------------------------------------------------

class _Explorer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = max(8, int(cfg.d_model * cfg.d_explorer_mult))
        self.net = nn.Sequential(
            nn.LayerNorm(2 * cfg.d_model),
            nn.Linear(2 * cfg.d_model, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, cfg.d_model),
        )

    def forward(self, state, xs):
        return self.net(torch.cat([state, xs], dim=-1))


# ---------------------------------------------------------------------------
# Shared LGP body (PhD explorers + output head)
# ---------------------------------------------------------------------------

class _LGPBase(nn.Module):
    """Shared PhD explorer and output-head logic for all LG-PRM variants."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg         = cfg
        self.encoder     = TokenEncoder(cfg)
        self.init        = nn.Linear(cfg.d_model, cfg.d_model)
        self.explorers   = nn.ModuleList(
            [_Explorer(cfg) for _ in range(cfg.n_explorers)]
        )
        self.head        = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

    def _explore(self, state, xs):
        """Runs the PhD explorer bank. Returns all proposals and their mean."""
        props = torch.stack([e(state, xs) for e in self.explorers], dim=1)
        ps    = props.mean(dim=1)
        return props, ps

    def _build_output(self, token_features, state, props_all):
        token_logits = self.head(token_features + state.unsqueeze(1))
        out = {
            'logits'           : self.head(state),
            'token_logits'     : token_logits,
            'proposals'        : torch.cat(props_all, dim=1),
        }
        return out


# ---------------------------------------------------------------------------
# V1 – PoLar
# ---------------------------------------------------------------------------

class LGP_V1(_LGPBase):
    """LG-PRM with PoLar gates over PI layers at each reasoning step."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.pi    = LoRAPISynthesizer(cfg)
        n_decisions = cfg.lg_steps * cfg.pi_layers
        self.polar  = PoLarController(cfg.d_model, n_decisions)

    def forward(self, x):
        token_features = self.encoder(x)
        xs      = token_features.mean(dim=1)
        state   = self.init(xs)
        xs_pool = xs                               # already pooled
        all_gates = self.polar(xs_pool)            # [B, lg_steps * pi_layers]
        pl = self.cfg.pi_layers

        props_all = []
        for s in range(self.cfg.lg_steps):
            props, ps = self._explore(state, xs)
            props_all.append(props)
            seg       = all_gates[:, s * pl : (s + 1) * pl]   # [B, pi_layers]
            gate_list = [seg[:, b] for b in range(pl)]
            state     = self.pi(state, xs, ps, gate_list=gate_list)

        out = self._build_output(token_features, state, props_all)
        out['polar_gates'] = all_gates
        out['polar_usage'] = all_gates.mean()
        return out


# ---------------------------------------------------------------------------
# V2 – LoRA HyperNet
# ---------------------------------------------------------------------------

class LGP_V2(_LGPBase):
    """LG-PRM with LoRA hypernetwork generating PI-layer adapters per step."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.pi    = LoRAPISynthesizer(cfg)
        self.hyper = LoRAHyperNet(cfg, n_blocks=cfg.pi_layers)

    def forward(self, x):
        token_features = self.encoder(x)
        xs    = token_features.mean(dim=1)
        state = self.init(xs)

        props_all, lora_norms = [], []
        for _ in range(self.cfg.lg_steps):
            props, ps = self._explore(state, xs)
            props_all.append(props)
            context  = torch.cat([xs, state], dim=-1)   # [B, 2d]
            adapters = self.hyper(context)
            lora_norms.append(lora_delta_norm(adapters))
            state    = self.pi(state, xs, ps, lora_list=adapters)

        out = self._build_output(token_features, state, props_all)
        out['lora_delta_norm'] = torch.stack(lora_norms).mean()
        return out


# ---------------------------------------------------------------------------
# V3 – Combined
# ---------------------------------------------------------------------------

class LGP_V3(_LGPBase):
    """LG-PRM with both PoLar PI-layer gating and LoRA per-step adapters."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.pi    = LoRAPISynthesizer(cfg)
        self.hyper = LoRAHyperNet(cfg, n_blocks=cfg.pi_layers)
        n_decisions = cfg.lg_steps * cfg.pi_layers
        self.polar  = PoLarController(cfg.d_model, n_decisions)

    def forward(self, x):
        token_features = self.encoder(x)
        xs        = token_features.mean(dim=1)
        state     = self.init(xs)
        all_gates = self.polar(xs)
        pl        = self.cfg.pi_layers

        props_all, lora_norms = [], []
        for s in range(self.cfg.lg_steps):
            props, ps = self._explore(state, xs)
            props_all.append(props)
            context   = torch.cat([xs, state], dim=-1)
            adapters  = self.hyper(context)
            lora_norms.append(lora_delta_norm(adapters))
            seg       = all_gates[:, s * pl : (s + 1) * pl]
            gate_list = [seg[:, b] for b in range(pl)]
            state     = self.pi(state, xs, ps, lora_list=adapters, gate_list=gate_list)

        out = self._build_output(token_features, state, props_all)
        out['polar_gates'] = all_gates
        out['polar_usage'] = all_gates.mean()
        out['lora_delta_norm'] = torch.stack(lora_norms).mean()
        return out
