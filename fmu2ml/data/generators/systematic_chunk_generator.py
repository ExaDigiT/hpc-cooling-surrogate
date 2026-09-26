"""
Systematic Chunk Generator

Generates the full systematic input sequence ONCE (a long, coherently ordered
sweep through the state space: steady-state grid -> steps -> ramps -> sinusoids
-> random, each scenario sequentially appended), then chops that single sequence
into `num_chunks` contiguous slices saved as per-chunk input files.

The expensive FMU simulation is parallelized downstream: each chunk_<id>/ dir is
fed to the FMU simulator independently (see parallel_output_generator.process_chunk),
which re-stabilizes to the chunk's first row before simulating it forward.

Generation is intentionally sequential -- the scenario ordering and the smooth
transitions between scenarios are the whole point of the systematic sweep, so it
must be built as one sequence and then split, not generated in independent pieces.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .systematic_input_generator import generate_systematic_fmu_dataset
from raps.config import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('systematic_chunk_generator')


def generate_systematic_chunks(
    num_chunks: int = 24,
    n_cdus: int = 0,
    duration_hours: Optional[int] = None,
    timestep_seconds: int = 1,
    output_dir: str = "data",
    system_name: str = "marconi100",
    seed: int = 42,
    steady_state_duration: int = 600,
    step_ramp_duration: int = 30,
    include_random: bool = True,
    sequence_method: str = "monotonic",
    apply_jitter: bool = True,
    independent_cdus: bool = True,
    save_raw: bool = False,
    global_t_ext: bool = True,
    global_load_events: bool = True,
) -> Dict:
    """
    Generate the full systematic sequence, then chop it into per-chunk input files.

    Parameters
    ----------
    num_chunks : int
        Number of contiguous slices (and chunk_<id>/ directories) to split into.
    n_cdus : int
        Number of CDUs. If <= 0, read from the system config.
    duration_hours : int, optional
        Length of the full sequence in hours. If None, the generator emits the
        complete scenario suite (~73h).
    timestep_seconds : int
        Time step in seconds.
    output_dir : str
        Base output directory (chunks go in output_dir/chunk_0, chunk_1, ...).
    system_name : str
        System configuration name (resolved via raps ConfigManager).
    seed : int
        Random seed for the single generation pass.
    steady_state_duration, step_ramp_duration, include_random, sequence_method,
    apply_jitter, independent_cdus :
        Forwarded to generate_systematic_fmu_dataset (see that function).
    save_raw : bool
        Also save the raw per-CDU frame (phase/scenario metadata) for each chunk,
        aligned row-for-row with the input slice. Useful for later analysis.

    Returns
    -------
    Dict with the chunk manifest and statistics.
    """
    # Resolve n_cdus from system config if not explicitly provided
    system_config = ConfigManager(system_name=system_name).get_config()
    if n_cdus <= 0:
        n_cdus = system_config.get('NUM_CDUS', 49)
        logger.info(f"Using system config for {system_name}: {n_cdus} CDUs")
    else:
        logger.info(f"Using explicit n_cdus: {n_cdus} CDUs (overriding system config)")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Systematic Chunk Generation (generate full sequence, then chop)")
    logger.info(f"Config: {n_cdus} CDUs, {timestep_seconds}s timestep, seed={seed}")
    logger.info(f"Mode: {'Independent CDUs (LHS)' if independent_cdus else 'Shared sequence'}")
    logger.info(f"Sequence method: {sequence_method}")
    logger.info(f"Chunks: {num_chunks} -> {output_dir}/chunk_0/, ..., chunk_{num_chunks-1}/")
    logger.info("=" * 60)

    # ── 1. Generate the full long sequence ONCE (sequential) ──
    logger.info("\n1. Generating full systematic sequence...")
    result = generate_systematic_fmu_dataset(
        n_cdus=n_cdus,
        duration_hours=duration_hours,
        timestep_seconds=timestep_seconds,
        seed=seed,
        output_dir=output_dir,
        save_output=False,  # we save the chopped chunks ourselves
        config=system_config,
        steady_state_duration=steady_state_duration,
        step_ramp_duration=step_ramp_duration,
        include_random=include_random,
        sequence_method=sequence_method,
        apply_jitter=apply_jitter,
        independent_cdus=independent_cdus,
        global_t_ext=global_t_ext,
        global_load_events=global_load_events,
    )

    fmu_df = result['fmu_input']
    raw_df = result['raw_input'] if save_raw else None
    scenarios = result['scenarios']

    total_rows = len(fmu_df)
    total_hours = total_rows * timestep_seconds / 3600
    logger.info(f"   Full sequence: {total_rows} rows ({total_hours:.2f}h), {fmu_df.shape[1]} columns")

    if num_chunks > total_rows:
        logger.warning(
            f"num_chunks ({num_chunks}) > rows ({total_rows}); "
            f"some chunks would be empty. Clamping to {total_rows} chunks."
        )
        num_chunks = total_rows

    # ── 2. Chop into contiguous, near-equal slices ──
    logger.info(f"\n2. Chopping into {num_chunks} contiguous chunks...")
    bounds = np.linspace(0, total_rows, num_chunks + 1).astype(int)

    # ── 3. Save each slice as a per-chunk input file ──
    chunk_info: List[Dict] = []
    output_directories: List[str] = []

    for i in range(num_chunks):
        start, end = int(bounds[i]), int(bounds[i + 1])
        if end <= start:
            continue

        chunk_dir = Path(output_dir) / f"chunk_{i}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_df = fmu_df.iloc[start:end].reset_index(drop=True)
        chunk_hours = len(chunk_df) * timestep_seconds / 3600

        # Name so it sorts before any later 'fmu_output_*' (process_chunk picks
        # the alphabetically-first parquet): 'fmu_input' < 'fmu_output'.
        input_name = f"fmu_input_systematic_{chunk_hours:.1f}hrs.parquet"
        chunk_df.to_parquet(chunk_dir / input_name, index=False)

        if raw_df is not None:
            raw_chunk = raw_df.iloc[start:end].reset_index(drop=True)
            raw_chunk.to_parquet(chunk_dir / f"raw_systematic_{chunk_hours:.1f}hrs.parquet", index=False)

        info = {
            "chunk_id": i,
            "directory": str(chunk_dir),
            "input_file": str(chunk_dir / input_name),
            "rows": int(len(chunk_df)),
            "start_row": start,
            "end_row": end,
            "hours": round(chunk_hours, 4),
        }
        chunk_info.append(info)
        output_directories.append(str(chunk_dir))
        logger.info(f"   ✓ chunk_{i}: rows [{start}:{end}] ({len(chunk_df)} rows, {chunk_hours:.2f}h)")

    # ── Manifest ──
    scenario_counts: Dict[str, int] = {}
    for s in scenarios:
        t = s.scenario_type.value
        scenario_counts[t] = scenario_counts.get(t, 0) + 1

    manifest = {
        "generation_timestamp": datetime.now().isoformat(),
        "system_name": system_name,
        "n_cdus": n_cdus,
        "timestep_seconds": timestep_seconds,
        "seed": seed,
        "independent_cdus": independent_cdus,
        "sequence_method": sequence_method,
        "total_rows": total_rows,
        "total_hours": round(total_hours, 4),
        "num_chunks": len(chunk_info),
        "n_scenarios": len(scenarios),
        "scenario_counts": scenario_counts,
        "global_t_ext": global_t_ext,
        "global_load_events": global_load_events,
        "global_schedules": result.get('global_schedules', {}),
        "chunks": chunk_info,
    }
    manifest_path = os.path.join(output_dir, "systematic_chunk_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info(f"✓ Generated {len(chunk_info)} chunks from a {total_hours:.2f}h sequence")
    logger.info(f"  Manifest: {manifest_path}")
    logger.info("=" * 60)

    return {
        "total_rows": total_rows,
        "total_hours": total_hours,
        "num_chunks": len(chunk_info),
        "output_directories": output_directories,
        "chunks": chunk_info,
        "scenario_counts": scenario_counts,
        "manifest_file": manifest_path,
        "base_output_dir": output_dir,
    }


# Example command-line usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate one systematic sequence and chop it into chunk input files"
    )
    parser.add_argument("--chunks", type=int, default=24, help="Number of chunks")
    parser.add_argument("--cdus", type=int, default=0, help="Number of CDUs (0 = from config)")
    parser.add_argument("--duration-hours", type=int, default=None,
                        help="Length of the full sequence in hours (default: full suite)")
    parser.add_argument("--timestep", type=int, default=1, help="Timestep in seconds")
    parser.add_argument("--output", type=str, default="data", help="Base output directory")
    parser.add_argument("--system-name", type=str, default="marconi100", help="System configuration name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steady-state-duration", type=int, default=600,
                        help="Steady-state scenario duration in seconds")
    parser.add_argument("--step-ramp-duration", type=int, default=30,
                        help="Ramp duration in step scenarios in seconds")
    parser.add_argument("--no-random", action="store_true", help="Exclude random realistic scenarios")
    parser.add_argument("--no-jitter", action="store_true", help="Disable per-CDU jitter (shared mode only)")
    parser.add_argument("--sequence-method", type=str, default="monotonic",
                        choices=["monotonic", "greedy"], help="Scenario sequencing method")
    parser.add_argument("--shared-mode", action="store_true",
                        help="Use shared scenario sequence for all CDUs (legacy mode)")
    parser.add_argument("--save-raw", action="store_true",
                        help="Also save per-chunk raw phase/scenario frames")
    args = parser.parse_args()

    result = generate_systematic_chunks(
        num_chunks=args.chunks,
        n_cdus=args.cdus,
        duration_hours=args.duration_hours,
        timestep_seconds=args.timestep,
        output_dir=args.output,
        system_name=args.system_name,
        seed=args.seed,
        steady_state_duration=args.steady_state_duration,
        step_ramp_duration=args.step_ramp_duration,
        include_random=not args.no_random,
        sequence_method=args.sequence_method,
        apply_jitter=not args.no_jitter,
        independent_cdus=not args.shared_mode,
        save_raw=args.save_raw,
    )

    print(f"\n✓ Chopped a {result['total_hours']:.2f}h sequence into {result['num_chunks']} chunks")
    print(f"  Data saved in: {result['base_output_dir']}/chunk_*/ directories")
    print(f"  Manifest: {result['manifest_file']}")
