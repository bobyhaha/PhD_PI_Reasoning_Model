"""
Shared meta-learning modules used by V1, V2, and V3 model variants.

V1 – PoLar (Program-of-Layers):
    A lightweight PoLarController takes a condensed input summary and predicts
    per-input binary execution gates for each (step, block) slot.
    Skipped blocks are replaced by an identity pass-through.
    Inspired by "Skip a Layer or Loop It?" (arXiv 2606.06574, ICML 2026).

V2 – LoRA HyperNet per loop:
    A LoRAHyperNet takes (input_pooled, state_pooled) context and generates
    rank-r LoRA A-matrices for each block at each reasoning step.
    B-matrices are fixed learnable parameters (not context-dependent).
    Delta W = B @ A_t  →  zero-delta at init (A-projections zero-initialized).

V3 – Combined:
    Both PoLarController (gate: skip/run) and LoRAHyperNet (how: adapter weight)
    are active.  Blocks that are gated-off receive identity; running blocks also
    get per-step LoRA deltas.
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# PoLar controller
# ---------------------------------------------------------------------------

class PoLarController(nn.Module):
    """
    Predicts a per-input execution program as n_decisions soft binary gates.

    Architecture:  LayerNorm → Linear(d, d//4) → GELU → Linear(d//4, n_decisions)

    Training  : straight-through sigmoid  (hard gate + soft gradient).
    Inference : hard threshold at 0.5.

    Zero-initialised output layer → gates start at sigmoid(0) = 0.5 and the
    hard straight-through path initially keeps all blocks.
    """

    def __init__(self, d_model: int, n_decisions: int):
        super().__init__()
        d_h = max(16, d_model // 4)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_h),
            nn.GELU(),
            nn.Linear(d_h, n_decisions),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : [B, d_model]  –  condensed input summary (e.g. mean of token embeddings).
        Returns [B, n_decisions] differentiable binary gates ∈ {0, 1} (approx).
        """
        logits = self.net(z)                      # [B, n_decisions]
        soft = torch.sigmoid(logits)
        if self.training:
            hard = (logits >= 0).float()
            return hard + soft - soft.detach()    # straight-through estimator
        return (soft > 0.5).float()


# ---------------------------------------------------------------------------
# LoRA-capable reasoning block
# ---------------------------------------------------------------------------

class LoRAReasoningBlock(nn.Module):
    """
    A full transformer-style reasoning block with optional per-sample LoRA
    on both FF projections and an optional PoLar skip gate.

    LoRA targets (applied to FF sub-layer only for clean, exact LoRA semantics):
        ff_up   : Linear(d, ff_d)  –  ΔW_up   ∈ [B, ff_d, d]
        ff_down : Linear(ff_d, d)  –  ΔW_down ∈ [B, d, ff_d]

    Contribution:  output += input @ ΔW.T   (standard additive LoRA).

    Gate semantics:
        gate = 1  →  full block computation used.
        gate = 0  →  identity pass-through  (block is skipped).
    """

    def __init__(self, cfg):
        super().__init__()
        d    = cfg.d_model
        ff_d = cfg.dim_ff_mult * d
        self.attn_ln = nn.LayerNorm(d)
        self.attn    = nn.MultiheadAttention(d, cfg.n_heads,
                                             dropout=cfg.dropout,
                                             batch_first=True)
        self.ff_ln   = nn.LayerNorm(d)
        self.ff_up   = nn.Linear(d, ff_d)
        self.ff_act  = nn.SiLU()
        self.ff_drop = nn.Dropout(cfg.dropout)
        self.ff_down = nn.Linear(ff_d, d)
        self.drop    = nn.Dropout(cfg.dropout)

    def forward(self,
                z: torch.Tensor,
                lora=None,
                gate: torch.Tensor = None) -> torch.Tensor:
        """
        z    : [B, L, d]
        lora : None | (dW_up [B, ff_d, d], dW_down [B, d, ff_d])
        gate : None | [B]  binary gate  (0 = skip, 1 = run)
        """
        # Attention (unchanged – no LoRA on MHA to keep backward-compat)
        h        = self.attn_ln(z)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        z1       = z + self.drop(attn_out)

        # Feed-forward with optional LoRA
        h2   = self.ff_ln(z1)
        up   = self.ff_up(h2)
        if lora is not None:
            # dW_up [B, ff_d, d]: contribution = h2 @ dW_up.T  → [B, L, ff_d]
            up = up + torch.einsum('bld,bod->blo', h2, lora[0])
        up   = self.ff_drop(self.ff_act(up))
        down = self.ff_down(up)
        if lora is not None:
            # dW_down [B, d, ff_d]: contribution = up @ dW_down.T → [B, L, d]
            down = down + torch.einsum('bld,bod->blo', up, lora[1])
        z_out = z1 + self.drop(down)

        # PoLar gate: 0 → identity skip, 1 → keep block output
        if gate is not None:
            g     = gate.view(-1, 1, 1)            # [B, 1, 1]
            z_out = g * z_out + (1.0 - g) * z     # soft/hard mix

        return z_out


# ---------------------------------------------------------------------------
# LoRA-capable reasoning core (wraps multiple LoRAReasoningBlocks)
# ---------------------------------------------------------------------------

class LoRAReasoningCore(nn.Module):
    """
    Drop-in replacement for TokenReasoningCore that supports per-block
    LoRA adapters and PoLar gates.
    """

    def __init__(self, cfg, n_layers: int):
        super().__init__()
        self.state_max_rms = float(getattr(cfg, 'state_max_rms', 0.0) or 0.0)
        self.blocks = nn.ModuleList(
            [LoRAReasoningBlock(cfg) for _ in range(n_layers)]
        )

    def _limit_state(self, z):
        if self.state_max_rms <= 0:
            return z
        rms = z.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        scale = torch.clamp(self.state_max_rms / rms, max=1.0).to(dtype=z.dtype)
        return z * scale

    def forward(self, z, injection=None, lora_list=None, gate_list=None):
        """
        injection : [B, L, d]  added to z before the first block.
        lora_list : list[n_layers] of (dW_up, dW_down) tuples | None per block.
        gate_list : list[n_layers] of [B] binary gates | None per block.
        """
        if injection is not None:
            z = z + injection
            z = self._limit_state(z)
        for i, block in enumerate(self.blocks):
            lora = lora_list[i] if lora_list is not None else None
            gate = gate_list[i] if gate_list is not None else None
            z    = block(z, lora=lora, gate=gate)
            z    = self._limit_state(z)
        return z


# ---------------------------------------------------------------------------
# LoRA hypernetwork
# ---------------------------------------------------------------------------

class LoRAHyperNet(nn.Module):
    """
    Generates per-step LoRA adapters for a LoRAReasoningCore.

    Design
    ------
    context  →  encoder  →  code  ∈ ℝ^{d_code}
    For each block b:
        A_up_b  = a_up_proj_b(code)    ∈ [B, r, d]
        A_dn_b  = a_dn_proj_b(code)    ∈ [B, r, ff_d]
        ΔW_up_b = B_up_b @ A_up_b      ∈ [B, ff_d, d]     (outer-product style)
        ΔW_dn_b = B_dn_b @ A_dn_b      ∈ [B, d,    ff_d]

    B matrices are fixed learnable (Kaiming init).
    A projections are zero-initialized → ΔW = 0 at the start of training.

    Parameters (base config: d=128, ff_d=512, r=2, d_code=32, n_blocks=2)
    ~~~~~~~~~~~
        encoder   : 2d * d_code  ≈ 8K
        a_up_proj : n_blocks * d_code * r * d    ≈ 16K
        a_dn_proj : n_blocks * d_code * r * ff_d ≈ 65K
        B mats    : n_blocks * (ff_d*r + d*r)    ≈  3K
        Total     : ~92K  (≈22 % of base TRM at d=128)

    For parameter-comparable configs use param_match.py to find the d_model
    that makes total V2-model params equal to the target base model params.
    """

    def __init__(self, cfg, n_blocks: int):
        super().__init__()
        d     = cfg.d_model
        ff_d  = cfg.dim_ff_mult * d
        r     = cfg.lora_rank
        d_c   = cfg.meta_d_code

        # Encoder: [2d] → [d_code]
        self.encoder = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, d_c),
            nn.GELU(),
        )

        # A-matrix projectors: code → flat A  (zero-init for stable start)
        self.a_up = nn.ModuleList(
            [nn.Linear(d_c, r * d,    bias=False) for _ in range(n_blocks)]
        )
        self.a_dn = nn.ModuleList(
            [nn.Linear(d_c, r * ff_d, bias=False) for _ in range(n_blocks)]
        )
        for proj in (*self.a_up, *self.a_dn):
            nn.init.zeros_(proj.weight)

        # Fixed B matrices  (Kaiming uniform)
        self.b_up = nn.ParameterList(
            [nn.Parameter(torch.empty(ff_d, r)) for _ in range(n_blocks)]
        )
        self.b_dn = nn.ParameterList(
            [nn.Parameter(torch.empty(d,    r)) for _ in range(n_blocks)]
        )
        for p in (*self.b_up, *self.b_dn):
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))

        self.n_blocks = n_blocks
        self.r  = r
        self.d  = d
        self.ff_d = ff_d
        self.base_scale = float(getattr(cfg, 'lora_scale', 0.05))
        self.max_delta = float(getattr(cfg, 'lora_max', 0.1))
        self.register_buffer('runtime_scale', torch.tensor(self.base_scale), persistent=False)

    def set_runtime_step(self, step: int, train_cfg=None):
        train_cfg = train_cfg or {}
        scale = float(train_cfg.get('lora_scale', self.base_scale))
        warmup = int(train_cfg.get('lora_warmup_steps') or 0)
        if warmup > 0:
            scale *= min(1.0, max(0.0, float(step + 1) / float(warmup)))
        self.runtime_scale.fill_(scale)

    def _stabilize(self, delta):
        scale = self.runtime_scale.to(device=delta.device, dtype=delta.dtype)
        max_delta = float(self.max_delta)
        if max_delta > 0:
            delta = torch.tanh(delta / max_delta) * max_delta
        return delta * scale

    def forward(self, context: torch.Tensor):
        """
        context : [B, 2*d_model]  = cat(xs_pooled, z_pooled)
        Returns : list of n_blocks tuples
                  (dW_up [B, ff_d, d], dW_dn [B, d, ff_d])
        """
        code = self.encoder(context)          # [B, d_code]
        B    = code.shape[0]
        out  = []
        for i in range(self.n_blocks):
            A_up = self.a_up[i](code).view(B, self.r, self.d)        # [B,r,d]
            A_dn = self.a_dn[i](code).view(B, self.r, self.ff_d)     # [B,r,ff_d]
            # ΔW_up = B_up [ff_d, r] @ A_up [r, d]  →  [B, ff_d, d]
            dW_up = torch.einsum('or,bri->boi', self.b_up[i], A_up)
            # ΔW_dn = B_dn [d, r]   @ A_dn [r, ff_d] →  [B, d, ff_d]
            dW_dn = torch.einsum('or,bri->boi', self.b_dn[i], A_dn)
            out.append((self._stabilize(dW_up), self._stabilize(dW_dn)))
        return out


def lora_delta_norm(adapters) -> torch.Tensor:
    """Mean squared generated LoRA delta, used as a gentle stability cost."""
    if not adapters:
        return torch.tensor(0.0)
    terms = []
    for dW_up, dW_dn in adapters:
        terms.append(dW_up.pow(2).mean())
        terms.append(dW_dn.pow(2).mean())
    return torch.stack(terms).mean()


# ---------------------------------------------------------------------------
# LoRA-capable PI Synthesizer  (replaces PISynthesizer in LGP variants)
# ---------------------------------------------------------------------------

class LoRAPISynthesizer(nn.Module):
    """
    Replaces the nn.TransformerEncoder-based PISynthesizer with a
    LoRAReasoningCore so the PI layers can receive LoRA adapters and/or
    PoLar gates.

    On single-token sequences (L=1) the MHA in each LoRAReasoningBlock is
    trivially identity, so the net computation is identical to the original
    (just the FF layers matter), while gaining LoRA+gate support.
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.in_proj = nn.Linear(3 * d, d)
        self.core    = LoRAReasoningCore(cfg, cfg.pi_layers)
        self.out     = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
        )

    def forward(self, state, xs, explorer_summary, lora_list=None, gate_list=None):
        """
        state, xs, explorer_summary : [B, d]
        lora_list / gate_list       : passed through to LoRAReasoningCore
        Returns updated state [B, d].
        """
        z = self.in_proj(
            torch.cat([state, xs, explorer_summary], dim=-1)
        ).unsqueeze(1)                               # [B, 1, d]
        z = self.core(z, lora_list=lora_list, gate_list=gate_list)   # [B, 1, d]
        return state + self.out(z.squeeze(1))        # [B, d]
