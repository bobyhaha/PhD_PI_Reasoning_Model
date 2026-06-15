"""
LG-PRM meta-model variants.

Architecture recap (base LibraryGatedPRM):
    encoder → init → [lg_steps × (n_explorers + library gate + PI synthesizer)]
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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class _Library(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        size = getattr(cfg, 'rag_library_size', cfg.library_size)
        self.keys   = nn.Parameter(
            torch.randn(size, cfg.d_model) / math.sqrt(cfg.d_model)
        )
        self.values = nn.Parameter(
            torch.randn(size, cfg.d_model) / math.sqrt(cfg.d_model)
        )

    def forward(self, q):
        attn = F.softmax(q @ self.keys.T / math.sqrt(q.shape[-1]), dim=-1)
        return attn @ self.values, attn


class _NeuralLibrary(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_h = cfg.mlp_library_mult * cfg.d_model
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, d_h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_h, cfg.d_model),
        )

    def forward(self, q):
        return self.net(q)


class _Gate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * cfg.d_model),
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, state, xs):
        return torch.sigmoid(self.net(torch.cat([state, xs], dim=-1)))


# ---------------------------------------------------------------------------
# Shared LGP body (explorers + library + gating logic)
# ---------------------------------------------------------------------------

class _LGPBase(nn.Module):
    """Shared initialisation and retrieval logic for all LGP variants."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg         = cfg
        self.encoder     = TokenEncoder(cfg)
        self.init        = nn.Linear(cfg.d_model, cfg.d_model)
        self.explorers   = nn.ModuleList(
            [_Explorer(cfg) for _ in range(cfg.n_explorers)]
        )
        self.rag_library = _Library(cfg)
        self.mlp_library = _NeuralLibrary(cfg)
        self.gate        = _Gate(cfg)
        self.head        = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

    # ---- retrieval helpers (identical to base LGP) ----

    def _gate_mask(self, gate_prob):
        hard = (
            torch.ones_like(gate_prob)
            if self.cfg.forced_library
            else (gate_prob >= self.cfg.gate_threshold).to(gate_prob.dtype)
        )
        if self.training and self.cfg.straight_through_gate:
            return hard + gate_prob - gate_prob.detach()
        return hard

    def _retrieve(self, state, gate_prob):
        gate   = (
            self._gate_mask(gate_prob)
            if (self.cfg.hard_library_gate or self.cfg.forced_library)
            else gate_prob
        )
        active  = gate.squeeze(-1) > 0
        libvec  = torch.zeros_like(state)
        entropy = state.new_tensor(0.0)
        if active.any():
            ret, attn         = self.rag_library(state[active])
            mlp_ret           = self.mlp_library(state[active])
            libvec[active]    = gate[active] * 0.5 * (ret + mlp_ret)
            entropy           = -(attn * attn.clamp_min(1e-8).log()).sum(-1).mean()
        return libvec, entropy, gate

    def _explore_and_retrieve(self, state, xs):
        """Runs explorers + optional library.  Returns (xs, ps, libvec, gate, hgate, ent)."""
        props = torch.stack([e(state, xs) for e in self.explorers], dim=1)
        ps    = props.mean(dim=1)
        if self.cfg.use_library:
            gate_prob = self.gate(state, xs)
            if self.cfg.forced_library:
                gate_prob = torch.ones_like(gate_prob)
            libvec, ent, hard_gate = self._retrieve(state, gate_prob)
        else:
            gate_prob  = torch.zeros(state.shape[0], 1, device=state.device)
            hard_gate  = gate_prob
            libvec     = torch.zeros_like(state)
            ent        = state.new_tensor(0.0)
        return props, ps, libvec, gate_prob, hard_gate, ent

    def _build_output(self, token_features, state, gates, hard_gates,
                      props_all, ents):
        token_logits = self.head(token_features + state.unsqueeze(1))
        out = {
            'logits'           : self.head(state),
            'token_logits'     : token_logits,
            'gate_probs'       : torch.stack(gates, dim=1),
            'gate_hard'        : torch.stack(hard_gates, dim=1),
            'proposals'        : torch.cat(props_all, dim=1),
            'library_entropy'  : (
                torch.stack(ents).mean() if ents
                else torch.tensor(0.0, device=state.device)
            ),
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

        gates, hard_gates, props_all, ents = [], [], [], []
        for s in range(self.cfg.lg_steps):
            props, ps, libvec, gp, hg, ent = self._explore_and_retrieve(state, xs)
            props_all.append(props)
            gates.append(gp)
            hard_gates.append(hg.detach())
            ents.append(ent)
            seg       = all_gates[:, s * pl : (s + 1) * pl]   # [B, pi_layers]
            gate_list = [seg[:, b] for b in range(pl)]
            state     = self.pi(state, xs, ps, libvec, gate_list=gate_list)

        out = self._build_output(token_features, state, gates, hard_gates,
                                 props_all, ents)
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

        gates, hard_gates, props_all, ents, lora_norms = [], [], [], [], []
        for _ in range(self.cfg.lg_steps):
            props, ps, libvec, gp, hg, ent = self._explore_and_retrieve(state, xs)
            props_all.append(props)
            gates.append(gp)
            hard_gates.append(hg.detach())
            ents.append(ent)
            context  = torch.cat([xs, state], dim=-1)   # [B, 2d]
            adapters = self.hyper(context)
            lora_norms.append(lora_delta_norm(adapters))
            state    = self.pi(state, xs, ps, libvec, lora_list=adapters)

        out = self._build_output(token_features, state, gates, hard_gates,
                                 props_all, ents)
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

        gates, hard_gates, props_all, ents, lora_norms = [], [], [], [], []
        for s in range(self.cfg.lg_steps):
            props, ps, libvec, gp, hg, ent = self._explore_and_retrieve(state, xs)
            props_all.append(props)
            gates.append(gp)
            hard_gates.append(hg.detach())
            ents.append(ent)
            context   = torch.cat([xs, state], dim=-1)
            adapters  = self.hyper(context)
            lora_norms.append(lora_delta_norm(adapters))
            seg       = all_gates[:, s * pl : (s + 1) * pl]
            gate_list = [seg[:, b] for b in range(pl)]
            state     = self.pi(state, xs, ps, libvec,
                                lora_list=adapters, gate_list=gate_list)

        out = self._build_output(token_features, state, gates, hard_gates,
                                 props_all, ents)
        out['polar_gates'] = all_gates
        out['polar_usage'] = all_gates.mean()
        out['lora_delta_norm'] = torch.stack(lora_norms).mean()
        return out
