"""
FMU input column naming, per system.

The cooling FMUs do not share one input layout. The POWER9 models (Summit,
Lassen, Marconi100) take a per-CDU heat load and inlet air temperature plus a
plant-wide ``..._sources_T_ext``. The Frontier model (Simulator_olcf5_base)
has no per-CDU ``T_Air`` and names its plant-wide input ``..._sources_Towb``.
Everything that writes or reads FMU input columns goes through
``FMUInputSpec`` so those differences live in one place, derived from the
system's RAPS config (``NUM_CDUS``, ``RACKS_PER_CDU``, ``TEMPERATURE_KEYS``).

Heat load units: the input generators sample a per-CDU heat load between
``MIN_POWER`` and ``MAX_POWER`` (kW). RAPS feeds the FMU one representative
cabinet per CDU (``Q_flow_total = cdu_power / RACKS_PER_CDU``, see
``raps/cooling.py``), so ``q_flow_per_cabinet`` applies the same division. For
the one-rack-per-CDU POWER9 systems it is the identity.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

Q_FLOW_PATTERN = "simulator_1_datacenter_1_computeBlock_{}_cabinet_1_sources_Q_flow_total"
T_AIR_PATTERN = "simulator_1_datacenter_1_computeBlock_{}_cabinet_1_sources_T_Air"
T_EXT_COL = "simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext"
TOWB_COL = "simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb"

# Lower-case substrings that identify the plant-wide ambient input column.
AMBIENT_TOKENS = ("t_ext", "towb")


def with_placeholder_t_air(input_ranges):
    """Fill a constant ``T_Air`` range if a caller left it out.

    The analysis sweep builders always generate a ``T_Air`` series; for FMUs
    without that input it is dropped by ``FMUInputSpec.adapt_frame`` before
    simulation, so callers for Frontier need not supply a range. ``None``
    passes through so the builders' own defaults still apply.
    """
    if input_ranges is None or "T_Air" in input_ranges:
        return input_ranges
    return {**input_ranges, "T_Air": (298.15, 298.15)}


def is_ambient_col(col: str) -> bool:
    """True for the plant-wide ambient (external / wet-bulb) input column."""
    col_l = col.lower()
    return any(tok in col_l for tok in AMBIENT_TOKENS)


@dataclass(frozen=True)
class FMUInputSpec:
    """Input column layout of one system's cooling FMU."""
    n_cdus: int
    racks_per_cdu: int = 1
    t_ext_col: str = T_EXT_COL
    has_t_air: bool = True

    @classmethod
    def from_config(cls, config: Dict[str, Any], n_cdus: Optional[int] = None) -> "FMUInputSpec":
        """Build from a RAPS config dict. ``n_cdus`` overrides ``NUM_CDUS``."""
        keys = config.get("TEMPERATURE_KEYS") or []
        ambient = [k for k in keys if is_ambient_col(k)]
        return cls(
            n_cdus=int(n_cdus if n_cdus is not None else config["NUM_CDUS"]),
            racks_per_cdu=int(config.get("RACKS_PER_CDU", 1) or 1),
            t_ext_col=ambient[0] if ambient else T_EXT_COL,
            # Configs without TEMPERATURE_KEYS predate Frontier: assume POWER9 layout.
            has_t_air=any("T_Air" in k for k in keys) if keys else True,
        )

    @classmethod
    def from_system(cls, system_name: str, n_cdus: Optional[int] = None) -> "FMUInputSpec":
        from raps.config import ConfigManager
        return cls.from_config(ConfigManager(system_name=system_name).get_config(), n_cdus)

    @classmethod
    def from_config_dir(cls, system_dir, n_cdus: Optional[int] = None) -> "FMUInputSpec":
        """Build from ``<config>/<system>/{system,cooling}.json`` without RAPS.

        RAPS resolves its config directory against the working directory,
        which breaks for callers (the phase notebooks) that do not run from
        the repository root.
        """
        system_dir = Path(system_dir)
        config: Dict[str, Any] = {}
        for name in ("system.json", "cooling.json"):
            path = system_dir / name
            if path.exists():
                config.update(json.loads(path.read_text()))
        return cls.from_config(config, n_cdus)

    @staticmethod
    def q_flow_col(cdu_num: int) -> str:
        return Q_FLOW_PATTERN.format(cdu_num)

    def t_air_col(self, cdu_num: int) -> Optional[str]:
        return T_AIR_PATTERN.format(cdu_num) if self.has_t_air else None

    def q_flow_per_cabinet(self, cdu_heat_w):
        """Per-CDU heat load (W) -> the FMU's per-cabinet ``Q_flow_total``."""
        if self.racks_per_cdu == 1:
            return cdu_heat_w
        return cdu_heat_w / self.racks_per_cdu

    def input_columns(self) -> List[str]:
        """All FMU input columns: per CDU (Q_flow[, T_Air]), then the ambient input."""
        cols = []
        for i in range(1, self.n_cdus + 1):
            cols.append(self.q_flow_col(i))
            if self.has_t_air:
                cols.append(T_AIR_PATTERN.format(i))
        cols.append(self.t_ext_col)
        return cols

    def input_vars(self) -> List[str]:
        """Input variable names used by the analyses: Q_flow[, T_Air], T_ext."""
        return ["Q_flow"] + (["T_Air"] if self.has_t_air else []) + ["T_ext"]

    def adapt_frame(self, df):
        """Convert a frame built with POWER9 column names to this FMU's inputs.

        For callers that assemble rows as ``Q_flow`` (per-CDU heat, W),
        ``T_Air`` and ``..._T_ext``: renames the ambient column, drops
        ``T_Air`` if the FMU has no such input, and splits the heat load per
        cabinet. Returns a new frame; apply it once, right before simulating.
        """
        if self.has_t_air and self.t_ext_col == T_EXT_COL and self.racks_per_cdu == 1:
            return df
        df = df.rename(columns={T_EXT_COL: self.t_ext_col})
        if not self.has_t_air:
            df = df.drop(columns=[c for c in df.columns if "_sources_T_Air" in c])
        if self.racks_per_cdu != 1:
            q_cols = [c for c in df.columns if c.endswith("_sources_Q_flow_total")]
            df[q_cols] = df[q_cols] / self.racks_per_cdu
        return df

    def input_patterns(self) -> Dict[str, str]:
        """Pattern dict in the form ``SurrogateConfig.input_patterns`` expects.

        The ambient input keeps the ``T_ext`` key whatever its FMU name, since
        the surrogate code treats that key as the one global input.
        """
        patterns = {"Q_flow": Q_FLOW_PATTERN}
        if self.has_t_air:
            patterns["T_Air"] = T_AIR_PATTERN
        patterns["T_ext"] = self.t_ext_col
        return patterns
