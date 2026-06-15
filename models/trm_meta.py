"""
TRM meta-model variants.

TRM_V1 – PoLar layer-program selection
    A PoLarController reads the pooled input once and predicts a binary
    execution gate for every (step, block) slot.  Skipped blocks are replaced
    by identity.  Full BPTT (no torch.no_grad loop) so the controller gets
    clean gradients at every step.

TRM_V2 – LoRA HyperNet per step
    At each recurrent step a LoRAHyperNet takes (xs_pooled, z_pooled) and
    generates per-block LoRA adapters.  Full BPTT ensures the hypernet is
    trained end-to-end.  Training objective: standard cross-entropy (same as
    base TRM).  The zero-init of A-projections guarantees ΔW=0 at init, so
    the model starts from the same effective computation as base TRM.

TRM_V3 – Combined (PoLar + LoRA HyperNet)
    Both the PoLarController (which blocks to run) and the LoRAHyperNet (how
    to run them) are active simultaneously.  Blocks gated to 0 are skipped
    regardless of the generated adapters.
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


# ---------------------------------------------------------------------------
# V1 – PoLar
# ---------------------------------------------------------------------------

class TRM_V1(nn.Module):
    """TRM with PoLar layer-program controller."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg   = cfg
        self.encoder = TokenEncoder(cfg)
        self.init  = nn.Parameter(
            torch.randn(1, 1, cfg.d_model) / (cfg.d_model ** 0.5)
        )
        self.core  = LoRAReasoningCore(cfg, cfg.trm_layers)
        self.polar = PoLarController(
            cfg.d_model, cfg.trm_steps * cfg.trm_layers
        )
        self.head  = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

    def forward(self, x):
        xs         = self.encoder(x)                # [B, L, d]
        z          = self.init.expand(xs.shape[0], xs.shape[1], -1)
        xs_pooled  = xs.mean(dim=1)                 # [B, d]
        # Predict the full execution program from the input (once)
        all_gates  = self.polar(xs_pooled)          # [B, T*n_layers]
        T, nb      = self.cfg.trm_steps, self.cfg.trm_layers

        for t in range(T):
            seg        = all_gates[:, t * nb : (t + 1) * nb]   # [B, n_layers]
            gate_list  = [seg[:, b] for b in range(nb)]
            z          = self.core(z, xs, gate_list=gate_list)

        token_logits = self.head(z)
        return {
            'logits'       : token_logits[:, 0],
            'token_logits' : token_logits,
            'token_state'  : z,
            'polar_gates'  : all_gates,
            'polar_usage'  : all_gates.mean(),
        }


# ---------------------------------------------------------------------------
# V2 – LoRA HyperNet
# ---------------------------------------------------------------------------

class TRM_V2(nn.Module):
    """TRM with per-step LoRA hypernetwork."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg     = cfg
        self.encoder = TokenEncoder(cfg)
        self.init    = nn.Parameter(
            torch.randn(1, 1, cfg.d_model) / (cfg.d_model ** 0.5)
        )
        self.core    = LoRAReasoningCore(cfg, cfg.trm_layers)
        self.hyper   = LoRAHyperNet(cfg, n_blocks=cfg.trm_layers)
        self.head    = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

    def forward(self, x):
        xs        = self.encoder(x)
        z         = self.init.expand(xs.shape[0], xs.shape[1], -1)
        xs_pooled = xs.mean(dim=1)                  # [B, d]
        lora_norms = []

        for _ in range(self.cfg.trm_steps):
            z_pooled = z.mean(dim=1)                # [B, d]
            context  = torch.cat([xs_pooled, z_pooled], dim=-1)  # [B, 2d]
            adapters = self.hyper(context)          # list[n_layers] of (dW_up, dW_dn)
            lora_norms.append(lora_delta_norm(adapters))
            z        = self.core(z, xs, lora_list=adapters)

        token_logits = self.head(z)
        return {
            'logits'       : token_logits[:, 0],
            'token_logits' : token_logits,
            'token_state'  : z,
            'lora_delta_norm': torch.stack(lora_norms).mean(),
        }


# ---------------------------------------------------------------------------
# V3 – Combined
# ---------------------------------------------------------------------------

class TRM_V3(nn.Module):
    """TRM with both PoLar layer-program selection and LoRA per-step adapters."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg     = cfg
        self.encoder = TokenEncoder(cfg)
        self.init    = nn.Parameter(
            torch.randn(1, 1, cfg.d_model) / (cfg.d_model ** 0.5)
        )
        self.core    = LoRAReasoningCore(cfg, cfg.trm_layers)
        self.polar   = PoLarController(
            cfg.d_model, cfg.trm_steps * cfg.trm_layers
        )
        self.hyper   = LoRAHyperNet(cfg, n_blocks=cfg.trm_layers)
        self.head    = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

    def forward(self, x):
        xs        = self.encoder(x)
        z         = self.init.expand(xs.shape[0], xs.shape[1], -1)
        xs_pooled = xs.mean(dim=1)
        all_gates = self.polar(xs_pooled)           # [B, T*n_layers]
        T, nb     = self.cfg.trm_steps, self.cfg.trm_layers
        lora_norms = []

        for t in range(T):
            z_pooled  = z.mean(dim=1)
            context   = torch.cat([xs_pooled, z_pooled], dim=-1)
            adapters  = self.hyper(context)
            lora_norms.append(lora_delta_norm(adapters))
            seg       = all_gates[:, t * nb : (t + 1) * nb]
            gate_list = [seg[:, b] for b in range(nb)]
            z         = self.core(z, xs, lora_list=adapters, gate_list=gate_list)

        token_logits = self.head(z)
        return {
            'logits'       : token_logits[:, 0],
            'token_logits' : token_logits,
            'token_state'  : z,
            'polar_gates'  : all_gates,
            'polar_usage'  : all_gates.mean(),
            'lora_delta_norm': torch.stack(lora_norms).mean(),
        }
