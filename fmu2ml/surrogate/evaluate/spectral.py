"""
Per-group spectral budget diagnostic (W7).

Answers, *empirically*, the question that motivates multi-rate sampling: for each
decoder group, how much signal variance lives **above** the Nyquist frequency of a
candidate coarse sampling rate? Plain striding folds that energy back into the
passband (aliasing); a large fraction (especially for ``V_flow_sec``) is direct
evidence that a single global rate corrupts the fast channels and that a finer rate
for that group is worth its cost.

This is the companion to the autocorrelation/near-unit-root analysis: it converts
"the hydraulic channels look aliased" from an assertion into a measurement.

Method: Welch PSD per channel on the **native** (rate-1) series, computed per chunk
and averaged (chunks are independently re-stabilized, so we never FFT across a
boundary). For a candidate decimation rate ``R`` with native step ``dt`` the new
Nyquist is ``f_nyq = 1/(2*R*dt)``; the diagnostic reports the fraction of variance
above ``f_nyq`` per channel, aggregated per group.
"""

from typing import Dict, List, Optional, Any, Sequence
import numpy as np
from scipy.signal import welch

from ..data import store as _store


# head -> (suffix, column_info index key); mirrors the model's group split
_GROUP_INDEX_KEYS = {
    'G_T':  ('T',  'temp_indices'),
    'G_V':  ('V',  'flow_indices'),
    'G_p':  ('p',  'pressure_indices'),
    'G_Vs': ('Vs', 'flow_sec_indices'),
    'G_ps': ('ps', 'pressure_sec_indices'),
    'G_W':  ('W',  'power_indices'),
}


def channel_variance_above(
    x: np.ndarray, fs: float, cutoff_hz: float, nperseg: Optional[int] = None,
) -> float:
    """
    Fraction of a 1-D signal's variance above ``cutoff_hz`` via Welch PSD.

    Returns a value in ``[0, 1]`` (0 if the signal is constant). ``fs`` is the
    native sampling frequency (Hz); ``cutoff_hz`` is the candidate-rate Nyquist.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8 or np.allclose(x, x[0]):
        return 0.0
    nseg = nperseg or min(len(x), 1024)
    f, pxx = welch(x, fs=fs, nperseg=nseg, detrend='constant')
    total = np.trapz(pxx, f)
    if total <= 0:
        return 0.0
    mask = f > cutoff_hz
    above = np.trapz(pxx[mask], f[mask]) if mask.any() else 0.0
    return float(np.clip(above / total, 0.0, 1.0))


def per_group_psd_budget(
    dynamic_chunks: Sequence[np.ndarray],
    dynamic_cols: List[str],
    column_info: Dict[str, Any],
    candidate_rates: Sequence[int],
    dt: float = 1.0,
    nperseg: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Per-group fraction-of-variance-above-Nyquist for each candidate decimation rate.

    Parameters
    ----------
    dynamic_chunks : list of (T_i, n_dynamic) native arrays (one per chunk).
    dynamic_cols : column names matching the array columns.
    column_info : provides the per-group index lists (``temp_indices`` ...).
    candidate_rates : decimation rates R to evaluate (e.g. ``[10, 30, 60]``).
    dt : native sample period in seconds (1.0 for 1 Hz data).

    Returns
    -------
    dict
        ``{'fs': fs, 'groups': {suffix: {rate: {'mean_frac_above', 'max_frac_above',
        'worst_channel', 'cutoff_hz', 'period_s'}}}, 'channels': {col: {rate: frac}}}``.
        A large ``mean/max_frac_above`` at a rate means that rate aliases that group.
    """
    fs = 1.0 / dt
    # per-channel fraction-above, averaged over chunks (Welch per chunk)
    n_cols = len(dynamic_cols)
    chan_frac = {int(R): np.zeros(n_cols) for R in candidate_rates}
    for R in candidate_rates:
        cutoff = 1.0 / (2.0 * R * dt)
        accum = np.zeros(n_cols)
        counts = np.zeros(n_cols)
        for arr in dynamic_chunks:
            arr = np.asarray(arr)
            for j in range(n_cols):
                accum[j] += channel_variance_above(arr[:, j], fs, cutoff, nperseg)
                counts[j] += 1
        chan_frac[int(R)] = accum / np.maximum(counts, 1)

    groups: Dict[str, Any] = {}
    for head, (suf, idx_key) in _GROUP_INDEX_KEYS.items():
        idx = column_info.get(idx_key)
        if not idx:
            continue
        per_rate = {}
        for R in candidate_rates:
            fr = chan_frac[int(R)][idx]
            worst_local = int(np.argmax(fr))
            per_rate[int(R)] = {
                'mean_frac_above': float(np.mean(fr)),
                'max_frac_above': float(np.max(fr)),
                'worst_channel': dynamic_cols[idx[worst_local]],
                'cutoff_hz': 1.0 / (2.0 * R * dt),
                'period_s': float(R * dt),
            }
        groups[suf] = per_rate

    channels = {
        dynamic_cols[j]: {int(R): float(chan_frac[int(R)][j]) for R in candidate_rates}
        for j in range(n_cols)
    }
    return {'fs': fs, 'groups': groups, 'channels': channels}


def psd_budget_from_store(
    store_dir: str,
    column_info: Dict[str, Any],
    candidate_rates: Sequence[int],
    dt: float = 1.0,
    nperseg: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run :func:`per_group_psd_budget` from a memmap store that has the **native**
    (rate-1) arrays. Requires rate 1 in the store manifest.
    """
    manifest = _store.load_store_manifest(store_dir)
    if 1 not in manifest['rates']:
        raise ValueError(
            "psd_budget_from_store needs the native rate-1 arrays in the store; "
            f"manifest rates are {manifest['rates']}. Rebuild with rate 1 included."
        )
    nin = manifest['n_input']
    chunks = [np.asarray(_store.open_chunk_array(store_dir, cid, 1)[:, nin:])
              for cid in manifest['chunk_ids']]
    return per_group_psd_budget(
        chunks, manifest['dynamic_cols'], column_info, candidate_rates, dt, nperseg)


def print_psd_budget(report: Dict[str, Any], file=None) -> None:
    """Pretty-print the per-group spectral budget (largest aliasing first)."""
    def _p(*a):
        print(*a, file=file)
    _p(f"\nPer-group spectral budget (native fs = {report['fs']:.4g} Hz)")
    _p("fraction of variance above a candidate rate's Nyquist (higher = more aliasing):")
    for suf, per_rate in report['groups'].items():
        _p(f"  group {suf}:")
        for R in sorted(per_rate):
            d = per_rate[R]
            _p(f"    rate {R:>3} ({d['period_s']:.0f}s, f_nyq={d['cutoff_hz']:.4g} Hz): "
               f"mean={d['mean_frac_above']*100:5.1f}%  max={d['max_frac_above']*100:5.1f}% "
               f"(worst: {d['worst_channel']})")
