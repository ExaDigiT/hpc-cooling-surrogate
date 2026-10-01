"""Sanity checks on a systematic FMU dataset before building a store or training.

    python surrogate_models/scripts/validate_dataset.py /abs/path/systematic-200

Reads <data>/systematic_chunk_manifest.json and each chunk's
fmu_output_*_operational.parquet. Per chunk: row count against the input, time
step, missing per-CDU outputs, NaN/inf, and whether the inputs the FMU recorded
match the generated inputs. Over all chunks: per-output ranges and the
temperature-ordering / heat-exchanger checks of the paper's physics screen.
Exits 1 if any chunk fails, so a pipeline can stop on bad data.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

S = "simulator[1].datacenter[1].computeBlock[{}].cdu[1].summary.{}"
VARS = ["T_prim_s_C", "T_prim_r_C", "T_sec_s_C", "T_sec_r_C", "p_prim_s_psig", "p_prim_r_psig",
        "p_sec_s_psig", "p_sec_r_psig", "V_flow_prim_GPM", "V_flow_sec_GPM", "W_flow_CDUP_kW"]


def output_file(chunk):
    found = glob.glob(os.path.join(chunk["directory"], "fmu_output_*_operational.parquet"))
    return found[0] if found else None


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    data_dir = os.path.expanduser(sys.argv[1])
    manifest = json.load(open(os.path.join(data_dir, "systematic_chunk_manifest.json")))
    n_cdu = int(manifest["n_cdus"])
    step = int(manifest.get("timestep_seconds", 1))
    print(f"{manifest.get('system_name', '?')}: {n_cdu} CDUs, {len(manifest['chunks'])} chunks, "
          f"{step} s steps\n")

    rows, per_var, ok = [], {v: [] for v in VARS}, True
    viol = {"T_prim_r > T_prim_s": 0, "T_sec_r > T_sec_s": 0, "T_sec_s > T_prim_s (HX)": 0}
    for ch in manifest["chunks"]:
        f = output_file(ch)
        if f is None:
            rows.append(dict(chunk=ch["chunk_id"], status="MISSING")); ok = False
            continue
        df = pd.read_parquet(f)
        inp = pd.read_parquet(ch["input_file"])
        dt = np.unique(np.round(np.diff(df["time"].to_numpy()), 6))
        cols = [S.format(i, v) for i in range(1, n_cdu + 1) for v in VARS]
        missing = sum(c not in df.columns for c in cols)
        num = df.select_dtypes("number")
        nan, inf = int(num.isna().sum().sum()), int(np.isinf(num.to_numpy()).sum())
        common = [c for c in inp.columns if c in df.columns]
        inputs_match = len(common) == inp.shape[1] and np.allclose(df[common].to_numpy(),
                                                                   inp[common].to_numpy())
        good = (len(df) == ch["rows"] and list(dt) == [step] and not missing and not nan
                and not inf and inputs_match)
        ok &= good
        rows.append(dict(chunk=ch["chunk_id"], rows=len(df), expected=ch["rows"],
                         dt=",".join(map(str, dt)), hours=round(len(df) * step / 3600, 2),
                         missing_cols=missing, nan=nan, inf=inf, inputs_match=inputs_match,
                         status="ok" if good else "FAIL"))
        if missing:
            continue
        for v in VARS:
            x = df[[S.format(i, v) for i in range(1, n_cdu + 1)]].to_numpy()
            per_var[v].append((x.min(), x.max(), x.std(axis=0).mean()))
        for i in range(1, n_cdu + 1):
            g = lambda v: df[S.format(i, v)].to_numpy()  # noqa: E731
            viol["T_prim_r > T_prim_s"] += int((g("T_prim_r_C") <= g("T_prim_s_C")).sum())
            viol["T_sec_r > T_sec_s"] += int((g("T_sec_r_C") <= g("T_sec_s_C")).sum())
            viol["T_sec_s > T_prim_s (HX)"] += int((g("T_sec_s_C") <= g("T_prim_s_C")).sum())

    print(pd.DataFrame(rows).to_string(index=False))
    print("\nper-output range over all chunks and CDUs (min, max, mean per-CDU std):")
    for v, vals in per_var.items():
        if vals:
            a = np.array(vals)
            print(f"  {v:16s} {a[:, 0].min():10.2f} {a[:, 1].max():10.2f}   std {a[:, 2].mean():9.4f}")
    print("\nphysics violations (row x CDU counts):", viol)
    ok &= not any(viol.values())
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
