"""
Shared multi-rate plumbing for phases 1-4.

Phase 5/6 (FederatedDeepMMNet) has always owned the multi-rate topology; this
module generalizes the same task contract to every other phase so that all
phases can train and run inference on the multi-rate chunk store
(:mod:`fmu2ml.surrogate.data.federated_store`) and be compared on the
identical task.

Contract every multi-rate model satisfies (matched to the federated model, so
the trainer/evaluator dispatch is phase-agnostic):

- ``model.multi_rate is True``
- ``model.config`` — the phase config (a :class:`MultiRateSpec` subclass)
- ``model.branches`` — mapping whose iteration yields branch ids (a
  ``nn.ModuleDict`` keyed by branch id)
- ``model.forward_multirate({branch_id: (u_hist_b, y_hist_b)})`` returns
  ``{'pred_T': (B, K_T, n_T), ...}`` — per-group consecutive-delta
  predictions on each group's own ``K`` grid.

What differs per phase is HOW each encodes and decodes (its identity):

- Phase 1 LSTM: per-branch LSTM+attention context, direct MLP decode per head
  (no basis expansion) — see :class:`fmu2ml.surrogate.architecture.lstm.MultiRateLSTM`.
- Phase 2 DeepONet: basis-coefficient branch x Fourier trunk inner product
  (``head_mode='plain'``).
- Phase 3 Hybrid: phase 2 + its learnable per-output blend gate
  (``head_mode='alpha'``).
- Phase 4 Domain: phase 2 + per-output scale/bias and sigmoid skip gate,
  its domain-head signature (``head_mode='skip'``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .configs import zero_persistence_heads
from .federated import FourierTrunkNetwork, make_temporal_encoder

# head -> (batch-key suffix, column_info index key, output key)
HEAD_META: Dict[str, Tuple[str, str, str]] = {
    'G_T':  ('T',  'temp_indices',         'pred_T'),
    'G_V':  ('V',  'flow_indices',         'pred_V'),
    'G_p':  ('p',  'pressure_indices',     'pred_p'),
    'G_Vs': ('Vs', 'flow_sec_indices',     'pred_Vs'),
    'G_ps': ('ps', 'pressure_sec_indices', 'pred_ps'),
    'G_W':  ('W',  'power_indices',        'pred_W'),
}


def head_layout(config: Any, column_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per-head layout resolved from (config, column_info).

    Returns ``{head: {suffix, pred_key, branch, K, n_outputs, col_idx}}`` for
    the heads present in ``config.head_outputs`` (canonical 6-group names).
    """
    layout: Dict[str, Dict[str, Any]] = {}
    for h in config.head_outputs:
        if h not in HEAD_META:
            raise KeyError(
                f"Unknown head {h!r}; multi-rate mode supports the canonical "
                f"groups {list(HEAD_META)}.")
        suffix, idx_key, pred_key = HEAD_META[h]
        col_idx = list(column_info[idx_key])
        layout[h] = {
            'suffix': suffix,
            'pred_key': pred_key,
            'branch': config.encoder_groups[h],
            'K': int(config.head_prediction_steps[h]),
            'n_outputs': len(col_idx),
            'col_idx': col_idx,
        }
    return layout


class BranchEncoderBank(nn.Module):
    """One temporal encoder per branch id over concat(u_hist, y_hist).

    Each branch runs at its own (rate, history); the encoder maps
    ``(B, H_b, n_inputs + n_dynamic) -> (B, out_dim)``.
    """

    def __init__(self, config: Any, n_inputs: int, n_dynamic: int,
                 hidden: int, n_layers: int, out_dim: int, dropout: float):
        super().__init__()
        branch_ids = sorted(set(config.encoder_groups.values()))
        enc_type = getattr(config, 'encoder_type', 'lstm')
        self.encoders = nn.ModuleDict({
            b: make_temporal_encoder(enc_type, n_inputs + n_dynamic,
                                     hidden, n_layers, out_dim, dropout)
            for b in branch_ids
        })
        self.branch_ids = branch_ids
        self.out_dim = out_dim

    def forward(
        self, branch_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        return {
            b: self.encoders[b](torch.cat([u, y], dim=-1))
            for b, (u, y) in branch_inputs.items()
        }


class OperatorHead(nn.Module):
    """DeepONet-family decoder head: basis coefficients x Fourier trunk.

    ``pred[b, k, o] = sum_n A[b, n, o] * T[k, n]`` with per-phase decoration:

    - ``'plain'``: raw inner product (phase 2 identity)
    - ``'alpha'``: per-output learnable sigmoid blend gate (phase 3 identity)
    - ``'skip'``:  per-output scale+bias then sigmoid skip gate
      (phase 4 domain-head identity)
    """

    def __init__(self, in_dim: int, n_basis: int, n_outputs: int, K: int,
                 n_fourier: int, trunk_hidden: int, head_mode: str = 'plain'):
        super().__init__()
        assert head_mode in ('plain', 'alpha', 'skip'), head_mode
        self.n_basis = n_basis
        self.n_outputs = n_outputs
        self.head_mode = head_mode
        self.proj = nn.Linear(in_dim, n_basis * n_outputs)
        self.trunk = FourierTrunkNetwork(
            n_fourier=n_fourier, hidden_size=trunk_hidden,
            n_basis=n_basis, prediction_steps=K,
        )
        if head_mode in ('alpha', 'skip'):
            # sigmoid(0) = 0.5: the phases' documented initial blend
            self.gate_logit = nn.Parameter(torch.zeros(n_outputs))
        if head_mode == 'skip':
            self.output_scale = nn.Parameter(torch.ones(n_outputs) * 0.1)
            self.output_bias = nn.Parameter(torch.zeros(n_outputs))

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        B = ctx.shape[0]
        A = self.proj(ctx).view(B, self.n_basis, self.n_outputs)
        T = self.trunk()                                   # (K, n_basis)
        out = torch.einsum('bno,kn->bko', A, T)            # (B, K, n_out)
        if self.head_mode == 'skip':
            out = out * self.output_scale + self.output_bias
        if self.head_mode in ('alpha', 'skip'):
            out = torch.sigmoid(self.gate_logit).view(1, 1, -1) * out
        return out


class MultiRateOperatorModel(nn.Module):
    """Shared base for the phase 2/3/4 multi-rate variants.

    Subclasses only choose the head decoration (``head_mode``) and map their
    config fields onto encoder/trunk sizes.
    """

    def __init__(self, config: Any, column_info: Dict[str, Any], *,
                 head_mode: str, encoder_hidden: int, encoder_layers: int,
                 n_basis: int, trunk_hidden: int, n_fourier: int,
                 dropout: float):
        super().__init__()
        self.config = config
        self.multi_rate = True
        self.layout = head_layout(config, column_info)
        n_inputs = len(column_info['input_cols'])
        n_dynamic = len(column_info['dynamic_cols'])

        self.encoder_bank = BranchEncoderBank(
            config, n_inputs, n_dynamic,
            hidden=encoder_hidden, n_layers=encoder_layers,
            out_dim=n_basis, dropout=dropout,
        )
        self.heads = nn.ModuleDict({
            h: OperatorHead(
                in_dim=n_basis, n_basis=n_basis,
                n_outputs=spec['n_outputs'], K=spec['K'],
                n_fourier=n_fourier, trunk_hidden=trunk_hidden,
                head_mode=head_mode,
            )
            for h, spec in self.layout.items()
        })
        self._init_weights()

    @property
    def branches(self) -> nn.ModuleDict:
        """Iteration yields branch ids (trainer/evaluator contract)."""
        return self.encoder_bank.encoders

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=0.3)
            elif ('bias' in name and 'output_bias' not in name
                  and 'gate_logit' not in name):
                nn.init.zeros_(param)

    def forward_multirate(
        self, branch_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        enc = self.encoder_bank(branch_inputs)
        out: Dict[str, torch.Tensor] = {}
        for h, spec in self.layout.items():
            out[spec['pred_key']] = self.heads[h](enc[spec['branch']])
        return zero_persistence_heads(
            out, {h: s['pred_key'] for h, s in self.layout.items()},
            self.config.persistence_heads)

    def forward(self, *args, **kwargs):
        # DistributedDataParallel installs its gradient-sync hooks on __call__,
        # which dispatches to forward(). Under DDP the trainer therefore cannot
        # reach forward_multirate() directly — calling model.module.
        # forward_multirate() would run and silently skip the all-reduce, so
        # every rank would train a divergent copy. Accepting the branch_inputs
        # dict here routes the multi-rate call through DDP properly.
        if args and isinstance(args[0], dict):
            return self.forward_multirate(*args, **kwargs)
        raise RuntimeError(
            f"{type(self).__name__} is a multi-rate model; use "
            f"forward_multirate({{branch_id: (u_hist_b, y_hist_b)}}).")


__all__ = [
    'HEAD_META',
    'head_layout',
    'BranchEncoderBank',
    'OperatorHead',
    'MultiRateOperatorModel',
]
