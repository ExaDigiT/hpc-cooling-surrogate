"""Export a trained multi-rate Phase 5 run as a self-contained bundle for RAPS.

RAPS cannot import fmu2ml (fmu2ml bundles its own top-level ``raps`` package,
which would shadow the real one), so the bundle carries everything inference
needs with only torch and numpy on the RAPS side:

    <out>/model.ts      TorchScript module: forward(u_fast, y_fast, u_slow, y_slow)
                        -> six normalized-delta tensors, heads in bundle order
    <out>/bundle.json   columns, per-head rate/history/horizon/indices, input and
                        output z-score statistics, per-rate delta scales, the
                        training input ranges, and a steady-state initializer

Usage (from the repository root; same env as training):

    PHASE_SYSTEM=frontier python surrogate_models/scripts/export_for_raps.py \\
        --run surrogate_models/runs/frontier_phase5_mr_allheads_200h \\
        --store /abs/path/systematic-200/store --out /abs/path/bundle

The steady-state initializer is a per-output quadratic in the 20 min trailing
means of (own cabinet Q_flow, wet-bulb temperature). RAPS uses it only to fill
the model's history at start-up, then relaxes the surrogate under constant inputs.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import fmu2ml.surrogate as surrogate  # noqa: E402
from fmu2ml.surrogate.data import store as _store  # noqa: E402
from fmu2ml.surrogate.data.normalizer import FederatedNormalizer  # noqa: E402

HEADS = ["G_T", "G_V", "G_p", "G_Vs", "G_ps", "G_W"]
PRED_KEY = {"G_T": "pred_T", "G_V": "pred_V", "G_p": "pred_p",
            "G_Vs": "pred_Vs", "G_ps": "pred_ps", "G_W": "pred_W"}
IDX_KEY = {"G_T": "temp_indices", "G_V": "flow_indices", "G_p": "pressure_indices",
           "G_Vs": "flow_sec_indices", "G_ps": "pressure_sec_indices", "G_W": "power_indices"}
FORMAT_VERSION = 1


class FlatPhase5(torch.nn.Module):
    """Positional wrapper so the multi-rate model traces to TorchScript."""

    def __init__(self, model, fast: str, slow: str):
        super().__init__()
        self.model, self.fast, self.slow = model, fast, slow

    def forward(self, u_fast, y_fast, u_slow, y_slow):
        out = self.model({self.fast: (u_fast, y_fast), self.slow: (u_slow, y_slow)})
        return tuple(out[PRED_KEY[h]] for h in HEADS)


def load_run(run_dir: Path, store_dir: Path):
    raw = json.loads((run_dir / "config.json").read_text())
    cls = surrogate.Phase5Config
    cfg = cls(**{k: v for k, v in raw.items() if k in set(cls.__dataclass_fields__)})
    assert cfg.multi_rate, "export supports multi-rate runs only"

    man = _store.load_store_manifest(str(store_dir))
    import pandas as pd
    schema = pd.DataFrame(columns=list(man["input_cols"]) + list(man["dynamic_cols"]))
    ci = surrogate.build_federated_column_info(schema, cfg, verbose=False)
    assert list(ci["input_cols"]) == list(man["input_cols"])
    assert list(ci["dynamic_cols"]) == list(man["dynamic_cols"])

    model = surrogate.get_architecture("federated")(cfg, column_info=ci)
    ck = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    nrm = FederatedNormalizer.load(run_dir / "normalizer.json")
    return cfg, ci, man, model, nrm


def fit_steady_state(store_dir, man, chunk_ids, n_cdus, q_cols, towb_col, dyn_cols,
                     rate=30, window_s=1200, holdout=None):
    """Per-output quadratic y_t = f(mean own Q_flow, mean Towb over the last 20 min).

    The systematic data never holds its inputs constant, so there is no true
    steady state to fit. Trailing 20 min means stand in for the equilibrium
    forcing (the thermal response lags are 10-17 min). RAPS uses this only as
    the starting history and then relaxes the surrogate under constant inputs.
    Returns coefficients, sample counts and, if ``holdout`` chunks are given,
    the held-out RMSE per output.
    """
    n_in = man["n_input"]
    in_cols = list(man["input_cols"])
    q_idx = [in_cols.index(q_cols[i]) for i in range(n_cdus)]
    w_idx = in_cols.index(towb_col)
    win = window_s // rate
    cdu_of = np.array([int(c.split("computeBlock[")[1].split("]")[0]) - 1 for c in dyn_cols])

    def samples(cids):
        X = {c: [] for c in range(n_cdus)}; Y = []
        for cid in cids:
            a = np.asarray(_store.open_chunk_array(str(store_dir), cid, rate), dtype=np.float64)
            k = np.ones(win) / win
            wm = np.convolve(a[:, w_idx], k, mode="valid")
            Y.append(a[win - 1:, n_in:])
            for c in range(n_cdus):
                qm = np.convolve(a[:, q_idx[c]], k, mode="valid")
                X[c].append(np.stack([qm, wm], axis=1))
        return {c: np.concatenate(v) for c, v in X.items()}, np.concatenate(Y)

    X, Y = samples(chunk_ids)
    coef = []
    for j in range(len(dyn_cols)):
        coef.append(np.linalg.lstsq(_ss_design(X[cdu_of[j]]), Y[:, j], rcond=None)[0])
    rmse = None
    if holdout:
        Xh, Yh = samples(holdout)
        rmse = [float(np.sqrt(np.mean((_ss_design(Xh[cdu_of[j]]) @ coef[j] - Yh[:, j]) ** 2)))
                for j in range(len(dyn_cols))]
    return [c.tolist() for c in coef], len(Y), rmse


def _ss_design(X):
    """Design matrix of the steady-state quadratic; Q in MW, Towb offset from 290 K."""
    q = X[:, 0] / 1e6
    w = X[:, 1] - 290.0
    return np.stack([np.ones_like(q), q, w, q * q, q * w, w * w], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg, ci, man, model, nrm = load_run(args.run, args.store)
    data_man = json.loads((args.run / "data_manifest.json").read_text())
    in_cols, dyn_cols = list(ci["input_cols"]), list(ci["dynamic_cols"])
    n_cdus = int(cfg.num_cdus)

    branches = {}
    for h in HEADS:
        b = cfg.encoder_groups[h]
        spec = {"rate": int(cfg.head_subsample[h]), "history": int(cfg.head_history_steps[h])}
        assert branches.setdefault(b, spec) == spec, f"branch {b} mixes rates"
    rates = sorted({s["rate"] for s in branches.values()})
    assert len(rates) == 2, "bundle format assumes one fast and one slow branch"
    fast = next(b for b, s in branches.items() if s["rate"] == rates[0])
    slow = next(b for b, s in branches.items() if s["rate"] == rates[1])

    flat = FlatPhase5(model, fast, slow).eval()
    ex = (torch.randn(2, branches[fast]["history"], len(in_cols)),
          torch.randn(2, branches[fast]["history"], len(dyn_cols)),
          torch.randn(2, branches[slow]["history"], len(in_cols)),
          torch.randn(2, branches[slow]["history"], len(dyn_cols)))
    with torch.no_grad():
        ts = torch.jit.trace(flat, ex, strict=False)
        ref, got = flat(*ex), ts(*ex)
        err = max(float((a - b).abs().max()) for a, b in zip(ref, got))
    assert err < 1e-5, f"TorchScript mismatch {err}"

    q_cols = [c for c in in_cols if c.endswith("_Q_flow_total")]
    towb = next(c for c in in_cols if not c.endswith("_Q_flow_total"))
    assert len(q_cols) == n_cdus

    train_ids = list(data_man["train_chunks"])
    coef, n_used, ss_rmse = fit_steady_state(args.store, man, train_ids, n_cdus, q_cols, towb,
                                             dyn_cols, holdout=list(data_man["val_chunks"]))

    # training input ranges at the base rate, for out-of-distribution warnings
    lo = np.full(len(in_cols), np.inf); hi = np.full(len(in_cols), -np.inf)
    for cid in train_ids:
        a = np.asarray(_store.open_chunk_array(str(args.store), cid, rates[0]))[:, :len(in_cols)]
        lo = np.minimum(lo, a.min(0)); hi = np.maximum(hi, a.max(0))

    in_norm, out_norm = nrm.input_normalizer, nrm.output_normalizer
    bundle = {
        "format_version": FORMAT_VERSION,
        "source_run": args.run.name,
        "system": cfg.system_name,
        "num_cdus": n_cdus,
        "input_cols": in_cols,
        "dynamic_cols": dyn_cols,
        "branches": {"fast": {"name": fast, **branches[fast]}, "slow": {"name": slow, **branches[slow]}},
        "stride_s": int(np.lcm.reduce(rates)),
        "heads": [{"name": h,
                   "branch": "fast" if cfg.encoder_groups[h] == fast else "slow",
                   "rate": int(cfg.head_subsample[h]),
                   "horizon": int(cfg.head_prediction_steps[h]),
                   "persistence": h in cfg.persistence_heads,
                   "indices": [int(i) for i in ci[IDX_KEY[h]]]} for h in HEADS],
        "input_norm": {"mean": [in_norm.get_mean(c) for c in in_cols],
                       "std": [in_norm.get_std(c) for c in in_cols]},
        "output_norm": {"mean": [out_norm.get_mean(c) for c in dyn_cols],
                        "std": [out_norm.get_std(c) for c in dyn_cols]},
        "delta_scale": {str(r): [nrm._delta_for(r).get_scale(c) for c in dyn_cols] for r in rates},
        "input_range": {"min": lo.tolist(), "max": hi.tolist()},
        "steady_state": {"features": ["1", "Q_MW", "Towb-290", "Q_MW^2", "Q_MW*(Towb-290)",
                                      "(Towb-290)^2"],
                         "inputs": "20 min trailing means of own cabinet Q_flow (W) and Towb (K)",
                         "coef": coef, "n_samples": n_used, "val_rmse": ss_rmse},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    ts.save(str(args.out / "model.ts"))
    (args.out / "bundle.json").write_text(json.dumps(bundle))
    print(f"exported {args.run.name} -> {args.out}")
    print(f"  TorchScript max |diff| vs eager: {err:.2e}")
    print(f"  heads: " + ", ".join(f"{h['name']}@{h['rate']}s" for h in bundle["heads"]))
    print(f"  steady-state initializer: {n_used} samples; val RMSE by output type:")
    for t in ["T_prim_s_C", "T_prim_r_C", "T_sec_s_C", "T_sec_r_C", "V_flow_prim_GPM",
              "V_flow_sec_GPM", "p_prim_s_psig", "p_prim_r_psig", "p_sec_s_psig",
              "p_sec_r_psig", "W_flow_CDUP_kW"]:
        r = [e for c, e in zip(dyn_cols, ss_rmse) if c.endswith(t)]
        sd = [s for c, s in zip(dyn_cols, bundle["output_norm"]["std"]) if c.endswith(t)]
        print(f"    {t:16s} RMSE {np.mean(r):8.3f}   (output std {np.mean(sd):8.3f})")


if __name__ == "__main__":
    main()
