"""
Facility-level (global) input schedules, built independently of per-CDU scenarios.

Two signals in the FMU input are shared by the whole plant rather than owned by
any CDU:

* ``T_ext`` — the cooling-tower ambient temperature. There is exactly one T_ext
  column, so in independent-CDU mode the per-scenario T_ext traces of 256 of the
  257 CDUs were silently discarded (the builder exported CDU_01's trace). The
  exported signal was therefore an *arbitrary* byproduct of whichever scenarios
  CDU_01 happened to draw — mostly piecewise-constant holds — not a designed
  weather signal.

* The **aggregate load**. With N independent CDUs the common-mode component of
  total facility power shrinks like 1/sqrt(N) (~6% of per-CDU swing at N=257),
  so plant-level dynamics (cooling tower loop, primary supply temperature — the
  weakest surrogate output) are barely excited by independent sampling alone.

This module builds both signals as first-class, segment-composed schedules with
their own systematic coverage, and returns the segment metadata so the design is
auditable in the chunk manifest.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ─────────────────────────────────────────────────────────────────────────────
# Global T_ext schedule
# ─────────────────────────────────────────────────────────────────────────────

T_EXT_MIN_K = 265.0
T_EXT_MAX_K = 315.0


def build_global_t_ext_schedule(
    n_steps: int,
    timestep_seconds: int = 1,
    seed: int = 42,
    t_ext_min_k: float = T_EXT_MIN_K,
    t_ext_max_k: float = T_EXT_MAX_K,
    micro_sigma_k: float = 0.4,
    micro_tau_s: float = 600.0,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Compose a plant-wide ambient-temperature timeline from weather segments.

    Segment types (drawn round-robin-with-shuffle so every type keeps appearing
    over long sequences, with randomized parameters per draw):

    - ``hold``     15–45 min at the current level — steady-weather baselines.
    - ``front``    a weather-front step: ±4–18 K ramped over 5–20 min.
    - ``ramp``     slow trend at 0.3–2 K/min toward a new level (morning
                   warm-up / evening cool-down rates).
    - ``diurnal``  1–3 h compressed sinusoidal day/night arc spanning up to
                   ~60% of the full range.
    - ``excursion`` out-and-back spike toward a range extreme and back (cold
                   snap / heat burst), 20–60 min total.

    A slow OU micro-weather texture (sigma ``micro_sigma_k``, tau
    ``micro_tau_s``) is superimposed so T_ext is never exactly constant, then
    the whole signal is lightly smoothed (weather has no 1 Hz energy) and
    clipped to [t_ext_min_k, t_ext_max_k].

    Returns
    -------
    (t_ext, segments) : the schedule array (n_steps,) and per-segment metadata
    dicts (type, start_s, duration_s, from_k, to_k) for the manifest.
    """
    rng = np.random.RandomState(seed)
    dt = float(timestep_seconds)
    span = t_ext_max_k - t_ext_min_k

    out = np.empty(n_steps, dtype=np.float64)
    segments: List[Dict] = []
    seg_types = ["hold", "front", "ramp", "diurnal", "excursion"]

    level = t_ext_min_k + span * rng.uniform(0.3, 0.7)
    i = 0
    bag: List[str] = []
    while i < n_steps:
        if not bag:
            bag = list(seg_types)
            rng.shuffle(bag)
        seg = bag.pop()
        start = i

        if seg == "hold":
            dur = int(rng.uniform(15, 45) * 60 / dt)
            dur = min(dur, n_steps - i)
            out[i:i + dur] = level
            i += dur
            to_level = level

        elif seg == "front":
            # Signed step, biased back toward mid-range near the boundaries.
            headroom_up = t_ext_max_k - level
            headroom_dn = level - t_ext_min_k
            delta = rng.uniform(4.0, 18.0) * (1 if rng.rand() < headroom_up /
                                              (headroom_up + headroom_dn) else -1)
            to_level = float(np.clip(level + delta, t_ext_min_k, t_ext_max_k))
            ramp = int(rng.uniform(5, 20) * 60 / dt)
            settle = int(rng.uniform(10, 20) * 60 / dt)
            ramp = min(ramp, n_steps - i)
            r = np.linspace(0, np.pi, ramp)
            out[i:i + ramp] = level + (to_level - level) * 0.5 * (1 - np.cos(r))
            i += ramp
            settle = min(settle, n_steps - i)
            out[i:i + settle] = to_level
            i += settle
            dur = ramp + settle

        elif seg == "ramp":
            rate_k_min = rng.uniform(0.3, 2.0)
            to_level = t_ext_min_k + span * rng.uniform(0.1, 0.9)
            dur = int(abs(to_level - level) / rate_k_min * 60 / dt)
            dur = max(dur, 1)
            dur = min(dur, n_steps - i)
            out[i:i + dur] = np.linspace(level, to_level, dur)
            i += dur

        elif seg == "diurnal":
            dur = int(rng.uniform(60, 180) * 60 / dt)
            dur = min(dur, n_steps - i)
            amp = span * rng.uniform(0.15, 0.30)
            center = float(np.clip(level, t_ext_min_k + amp, t_ext_max_k - amp))
            t = np.arange(dur) * dt
            # full cosine arc so the segment returns near its starting level
            out[i:i + dur] = center - amp * np.cos(2 * np.pi * t / (dur * dt))
            level_end = out[i + dur - 1] if dur > 0 else level
            i += dur
            to_level = float(level_end)

        else:  # excursion
            extreme = t_ext_max_k if rng.rand() < 0.5 else t_ext_min_k
            peak = level + (extreme - level) * rng.uniform(0.6, 1.0)
            dur = int(rng.uniform(20, 60) * 60 / dt)
            dur = min(dur, n_steps - i)
            t = np.linspace(0, np.pi, dur)
            out[i:i + dur] = level + (peak - level) * np.sin(t)
            i += dur
            to_level = level

        segments.append({
            "type": seg,
            "start_s": int(start * dt),
            "duration_s": int((i - start) * dt),
            "from_k": round(float(level), 2),
            "to_k": round(float(to_level), 2),
        })
        level = float(to_level)

    # OU micro-weather texture on top.
    theta = dt / micro_tau_s
    sigma = micro_sigma_k * np.sqrt(2 * theta)
    ou = np.zeros(n_steps)
    for k in range(1, n_steps):
        ou[k] = ou[k - 1] * (1 - theta) + sigma * rng.normal()
    out = out + ou

    # Weather has no 1 Hz energy: smooth on a ~1 min scale.
    out = gaussian_filter1d(out, sigma=max(60.0 / dt, 1.0))
    return np.clip(out, t_ext_min_k, t_ext_max_k), segments


# ─────────────────────────────────────────────────────────────────────────────
# Global load factor (correlated facility-level events)
# ─────────────────────────────────────────────────────────────────────────────

def build_global_load_factor(
    n_steps: int,
    timestep_seconds: int = 1,
    seed: int = 43,
    event_rate_per_hour: float = 3.0,
    ou_sigma: float = 0.02,
    ou_tau_s: float = 300.0,
    factor_min: float = 0.60,
    factor_max: float = 1.15,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Multiplicative common-mode load factor applied to every CDU's Q_flow.

    Baseline 1.0 with (a) a small OU texture (sigma ``ou_sigma``, tau
    ``ou_tau_s``) so aggregate load always breathes, and (b) Poisson-arriving
    facility events (mean ``event_rate_per_hour``):

    - ``job_wave``     +8–20% surge, 30 s–5 min cosine ramp up, hold
                       2–15 min, ramp back — a large job (or wave of jobs)
                       starting across many cabinets at once.
    - ``drain``        −15–35% sag, 1–10 min ramps, hold 5–20 min — queue
                       drain / reservation gap / maintenance window.
    - ``quiet_shift``  ±5–12% level shift persisting until the next event —
                       shift-scale utilization change (night/weekend).

    Events overlap additively in log-space is unnecessary at these amplitudes;
    they are composed additively and clipped to [factor_min, factor_max].

    Rationale: with N independent CDUs the common-mode fraction of aggregate
    load variance is ~1/N; these events restore realistic facility-scale
    excitation so plant-loop outputs (cooling tower, primary supply temps,
    W_flow at high aggregate load) see real dynamics. Amplitudes are fractions
    of each CDU's own level, so per-CDU inputs stay within their design range.

    Returns
    -------
    (factor, events) : the factor array (n_steps,) and per-event metadata.
    """
    rng = np.random.RandomState(seed)
    dt = float(timestep_seconds)

    factor = np.ones(n_steps, dtype=np.float64)
    events: List[Dict] = []

    # Poisson event arrivals.
    p_event = event_rate_per_hour * dt / 3600.0
    i = 0
    while i < n_steps:
        if rng.rand() < p_event:
            kind = rng.choice(["job_wave", "drain", "quiet_shift"],
                              p=[0.45, 0.35, 0.20])
            if kind == "job_wave":
                amp = rng.uniform(0.08, 0.20)
                ramp = int(rng.uniform(30, 300) / dt)
                hold = int(rng.uniform(2, 15) * 60 / dt)
            elif kind == "drain":
                amp = -rng.uniform(0.15, 0.35)
                ramp = int(rng.uniform(60, 600) / dt)
                hold = int(rng.uniform(5, 20) * 60 / dt)
            else:  # quiet_shift: persists until the next event window ends
                amp = rng.uniform(0.05, 0.12) * (1 if rng.rand() < 0.5 else -1)
                ramp = int(rng.uniform(120, 600) / dt)
                hold = int(rng.uniform(20, 60) * 60 / dt)

            up = np.linspace(0, np.pi, max(ramp, 1))
            profile = np.concatenate([
                amp * 0.5 * (1 - np.cos(up)),          # ramp in
                np.full(max(hold, 0), amp),            # hold
                amp * 0.5 * (1 + np.cos(up)),          # ramp out
            ])
            end = min(i + len(profile), n_steps)
            factor[i:end] += profile[:end - i]
            events.append({
                "type": str(kind),
                "start_s": int(i * dt),
                "duration_s": int((end - i) * dt),
                "amplitude": round(float(amp), 3),
            })
            i = end
        else:
            i += 1

    # OU texture.
    theta = dt / ou_tau_s
    sigma = ou_sigma * np.sqrt(2 * theta)
    ou = np.zeros(n_steps)
    for k in range(1, n_steps):
        ou[k] = ou[k - 1] * (1 - theta) + sigma * rng.normal()
    factor = factor + ou

    return np.clip(factor, factor_min, factor_max), events


def fit_to_length(arr: np.ndarray, target_length: int) -> np.ndarray:
    """Trim, or edge-hold-pad, a schedule to exactly ``target_length`` steps."""
    if len(arr) >= target_length:
        return arr[:target_length]
    pad = np.full(target_length - len(arr), arr[-1] if len(arr) else 0.0)
    return np.concatenate([arr, pad])
