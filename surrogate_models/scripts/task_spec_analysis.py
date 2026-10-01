"""Measurements that set a new system's surrogate task spec (paper Sec. IV-A, Tables II-III).

    python surrogate_models/scripts/task_spec_analysis.py /abs/path/systematic-200 [--test-chunk 0]

1. Spectral budget: fraction of each output's variance above the Nyquist of
   candidate sampling rates (3, 30, 60 s). Pick the slowest rate that keeps
   this below about 1% for a group (the paper's criterion).
2. Response delay: lag of the peak cross-correlation between input and output
   increments (own CDU Q_flow, and the plant-wide ambient input). Look-backs
   must cover the longest delay; Summit/Frontier thermal lags are 10-25 min.
3. Learnability: a ridge model on (last state, own Q_flow history, ambient
   history) against persistence, per output type, at a 60 s and a 600 s
   horizon, trained on every other chunk and scored on --test-chunk. A group
   whose outputs the ridge cannot beat (skill <= ~0.05 at both horizons) is a
   candidate persistence-locked group (head_loss_weights 0); anything clearly
   positive should be trained. Do not copy another system's choice.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # repo root
from fmu2ml.config.fmu_io import Q_FLOW_PATTERN, is_ambient_col  # noqa: E402

S = "simulator[1].datacenter[1].computeBlock[{}].cdu[1].summary.{}"
TYPES = {"T_prim_s": "T_prim_s_C", "T_prim_r": "T_prim_r_C", "T_sec_s": "T_sec_s_C",
         "T_sec_r": "T_sec_r_C", "V_flow_prim": "V_flow_prim_GPM", "V_flow_sec": "V_flow_sec_GPM",
         "p_prim_s": "p_prim_s_psig", "p_prim_r": "p_prim_r_psig", "p_sec_s": "p_sec_s_psig",
         "p_sec_r": "p_sec_r_psig", "W_flow": "W_flow_CDUP_kW"}
GROUP = {"T_prim_s": "G_T", "T_prim_r": "G_T", "T_sec_s": "G_T", "T_sec_r": "G_T",
         "V_flow_prim": "G_V", "p_prim_s": "G_p", "p_prim_r": "G_p", "V_flow_sec": "G_Vs",
         "p_sec_s": "G_ps", "p_sec_r": "G_ps", "W_flow": "G_W"}
SPECS = {"60s": (3, 100, 20), "600s": (30, 40, 20)}   # (rate s, history steps, horizon steps)


def load(data_dir):
    manifest = json.load(open(os.path.join(data_dir, "systematic_chunk_manifest.json")))
    n = int(manifest["n_cdus"])
    chunks, amb = {}, None
    for ch in manifest["chunks"]:
        f = glob.glob(os.path.join(ch["directory"], "fmu_output_*_operational.parquet"))[0]
        if amb is None:
            import pyarrow.parquet as pq
            amb = next(c for c in pq.read_schema(f).names if is_ambient_col(c))
        cols = [amb] + [Q_FLOW_PATTERN.format(i) for i in range(1, n + 1)] + \
               [S.format(i, v) for i in range(1, n + 1) for v in TYPES.values()]
        chunks[ch["chunk_id"]] = pd.read_parquet(f, columns=cols)
    return n, amb, chunks


def spectral_budget(n, chunks, rates=(3, 30, 60)):
    rows = []
    for t, v in TYPES.items():
        fr = {r: [] for r in rates}
        for df in chunks.values():
            for i in range(1, n + 1):
                x = df[S.format(i, v)].to_numpy()
                f, p = welch(x - x.mean(), fs=1.0, nperseg=4096)
                tot = p[1:].sum()
                for r in rates:
                    fr[r].append(100 * p[f > 1 / (2 * r)].sum() / tot if tot > 0 else 0)
        rows.append([t, GROUP[t]] + [f"{np.mean(fr[r]):.2f} ({np.max(fr[r]):.2f})" for r in rates])
    return pd.DataFrame(rows, columns=["output", "group"] + [f"R={r}s" for r in rates])


def peak_lag(u, y, max_lag=1800, block=10):
    """Lag (s) maximizing |corr(du(t), dy(t+lag))| on block means."""
    m = len(u) // block * block
    du = np.diff(u[:m].reshape(-1, block).mean(1)); dy = np.diff(y[:m].reshape(-1, block).mean(1))
    du = (du - du.mean()) / (du.std() + 1e-12); dy = (dy - dy.mean()) / (dy.std() + 1e-12)
    best, lag = 0.0, 0
    for k in range(max_lag // block):
        c = np.mean(du[:len(du) - k] * dy[k:]) if k else np.mean(du * dy)
        if abs(c) > abs(best):
            best, lag = c, k * block
    return lag, best


def response_delays(n, amb, chunks):
    rows = []
    for t, v in TYPES.items():
        lq, cq, la, ca = [], [], [], []
        for df in chunks.values():
            for i in range(1, n + 1):
                a, b = peak_lag(df[Q_FLOW_PATTERN.format(i)].to_numpy(), df[S.format(i, v)].to_numpy())
                lq.append(a); cq.append(b)
            a, b = peak_lag(df[amb].to_numpy(), df[S.format(1, v)].to_numpy())
            la.append(a); ca.append(b)
        rows.append([t, GROUP[t], int(np.median(lq)), f"{np.median(cq):+.2f}",
                     int(np.median(la)), f"{np.median(ca):+.2f}"])
    return pd.DataFrame(rows, columns=["output", "group", "lag_Q (s)", "corr_Q",
                                       "lag_ambient (s)", "corr_ambient"])


def windows(df, i, v, amb, rate, hist, horizon):
    y = df[S.format(i, v)].to_numpy()
    q = df[Q_FLOW_PATTERN.format(i)].to_numpy() / 1e5
    w = df[amb].to_numpy() - 280
    m = len(y) // rate * rate
    yb, qb, wb = (a[:m].reshape(-1, rate).mean(1) for a in (y, q, w))
    X, Y = [], []
    for t in range(hist, len(yb) - horizon, 2):
        X.append(np.r_[yb[t - hist:t + 1] - yb[t], qb[t - hist:t + 1], wb[t - hist:t + 1], 1.0])
        Y.append(yb[t + horizon] - yb[t])
    return np.array(X), np.array(Y)


def learnability(n, amb, chunks, test_id, cdu_stride=3):
    rows = []
    for t, v in TYPES.items():
        rec = [t, GROUP[t]]
        for rate, hist, hor in SPECS.values():
            sk = []
            for i in range(1, n + 1, cdu_stride):
                tr = [windows(df, i, v, amb, rate, hist, hor) for c, df in chunks.items() if c != test_id]
                Xtr, ytr = np.vstack([a for a, _ in tr]), np.concatenate([b for _, b in tr])
                Xte, yte = windows(chunks[test_id], i, v, amb, rate, hist, hor)
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
                mu[-1], sd[-1] = 0.0, 1.0
                A = (Xtr - mu) / sd
                wgt = np.linalg.solve(A.T @ A + np.eye(A.shape[1]), A.T @ ytr)
                pred = ((Xte - mu) / sd) @ wgt
                mse_p = np.mean(yte ** 2)
                sk.append(1 - np.mean((yte - pred) ** 2) / mse_p if mse_p > 0 else 0.0)
            rec.append(float(np.median(sk)))
        rec.append("persistence-locked candidate" if max(rec[2:]) <= 0.05 else "learnable")
        rows.append(rec)
    df = pd.DataFrame(rows, columns=["output", "group"] + [f"skill @{k}" for k in SPECS] + ["verdict"])
    return df.round({f"skill @{k}": 3 for k in SPECS})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--test-chunk", type=int, default=None, help="held-out chunk (default: first)")
    args = ap.parse_args()
    n, amb, chunks = load(os.path.expanduser(args.data_dir))
    test_id = args.test_chunk if args.test_chunk is not None else min(chunks)
    print(f"{n} CDUs, {len(chunks)} chunks, ambient input {amb.split('_sources_')[-1]}, "
          f"test chunk {test_id}\n")
    print("1. Variance above the Nyquist of rate R (%), mean (max) over CDUs x chunks")
    print(spectral_budget(n, chunks).to_string(index=False))
    print("\n2. Response lag of output increments to input increments, median over CDUs x chunks")
    print(response_delays(n, amb, chunks).to_string(index=False))
    print(f"\n3. Ridge skill vs persistence on chunk {test_id} (1 - MSE_ridge/MSE_persistence)")
    print(learnability(n, amb, chunks, test_id).to_string(index=False))


if __name__ == "__main__":
    main()
