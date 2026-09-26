"""
Column information builders for surrogate models.

Dynamically builds column_info dictionaries from dataframes and config,
mapping columns to CDUs and classifying as temporal/algebraic.
Supports domain-specific (Phase 4) and federated (Phase 5/6) column structures.
"""

from typing import Dict, List, Any, Optional, Tuple
import re
import pandas as pd

from ..architecture.configs import (
    SurrogateConfig,
    DomainDeepONetConfig,
    FederatedConfig,
    OutputType,
    OUTPUT_CLASSIFICATION,
    DOMAIN_OUTPUTS,
    FEDERATED_HEAD_OUTPUTS,
    HEAD_TYPES,
)


def _resolve_cdu_column(
    df: pd.DataFrame,
    pattern: str,
    cdu_id: int,
) -> Optional[str]:
    """
    Resolve a config pattern to an actual dataframe column for one CDU.

    Tries, in order:
    1. Format-string substitution (``pattern.format(cdu_id)``)
    2. Exact column name match
    3. Fuzzy match with known CDU token patterns
    4. Regex boundary match for the CDU ID

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe whose columns are searched
    pattern : str
        Column name pattern (may contain ``{}`` placeholder)
    cdu_id : int
        CDU identifier to resolve

    Returns
    -------
    Optional[str]
        Matched column name, or None if no match found
    """
    columns = df.columns

    if "{" in pattern:
        candidate = pattern.format(cdu_id)
        return candidate if candidate in columns else None

    if pattern in columns:
        return pattern

    pattern_l = pattern.lower()
    token_candidates = (
        f"computeblock_{cdu_id}",
        f"cdu_{cdu_id}",
        f"cabinet_{cdu_id}",
        f"_{cdu_id}_",
        f"_{cdu_id}",
        f"[{cdu_id}]",
    )

    matches = [
        col for col in columns
        if pattern_l in col.lower() and any(tok in col.lower() for tok in token_candidates)
    ]
    if matches:
        return min(matches, key=len)

    boundary = re.compile(rf"(^|[^0-9]){cdu_id}([^0-9]|$)")
    matches = [
        col for col in columns
        if pattern_l in col.lower() and boundary.search(col.lower())
    ]
    if matches:
        return min(matches, key=len)

    return None


def build_column_info(
    df: pd.DataFrame,
    config: SurrogateConfig,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Build column information dictionary from dataframe and config.
    
    Identifies input/output columns based on naming patterns and classifies
    outputs as temporal (dynamic) or algebraic (near-instantaneous).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with all columns
    config : SurrogateConfig
        Configuration with column patterns and CDU info
    verbose : bool
        Whether to print column identification summary
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'input_cols': List of input column names
        - 'output_cols': List of output column names
        - 'temporal_cols': List of temporal output columns
        - 'algebraic_cols': List of algebraic output columns
        - 'col_to_cdu': Mapping from column name to CDU index
        - 'col_to_type': Mapping from column name to output type name
        - 'temporal_indices': Indices of temporal cols in output_cols
        - 'algebraic_indices': Indices of algebraic cols in output_cols
        - 'n_inputs': Number of input columns
        - 'n_outputs': Number of output columns
        - 'n_temporal': Number of temporal output columns
        - 'n_algebraic': Number of algebraic output columns
    """
    result: Dict[str, Any] = {
        'input_cols': [],
        'output_cols': [],
        'temporal_cols': [],
        'algebraic_cols': [],
        'col_to_cdu': {},
        'col_to_type': {},
    }


    # Build input columns (per-CDU)
    for cdu_id in config.cdu_ids:
        for pattern_name, pattern in config.input_patterns.items():
            if pattern_name == 'T_ext':
                continue  # Handle global column separately
            col = _resolve_cdu_column(df, pattern, cdu_id)
            if col is not None:
                result['input_cols'].append(col)
    
    # Add global input column (T_ext)
    t_ext_pattern = config.input_patterns.get('T_ext', '')
    if t_ext_pattern and t_ext_pattern in df.columns:
        result['input_cols'].append(t_ext_pattern)
    else:
        # Fallback: search for T_ext-like column
        for col in df.columns:
            if 't_ext' in col.lower() and col not in result['input_cols']:
                result['input_cols'].append(col)
                break
    
    # Build output columns (per-CDU)
    for output_name, pattern in config.output_patterns.items():
        for cdu_id in config.cdu_ids:
            col = _resolve_cdu_column(df, pattern, cdu_id)
            if col is not None:
                result['output_cols'].append(col)
                result['col_to_cdu'][col] = cdu_id
                result['col_to_type'][col] = output_name
    
    # Remove duplicates while preserving order
    result['input_cols'] = list(dict.fromkeys(result['input_cols']))
    result['output_cols'] = list(dict.fromkeys(result['output_cols']))
    
    # Classify outputs as temporal or algebraic
    for col in result['output_cols']:
        output_type = config.get_output_type(col)
        if output_type == OutputType.ALGEBRAIC:
            result['algebraic_cols'].append(col)
        else:
            result['temporal_cols'].append(col)
    
    # Compute indices
    result['temporal_indices'] = [
        result['output_cols'].index(c) for c in result['temporal_cols']
    ]
    result['algebraic_indices'] = [
        result['output_cols'].index(c) for c in result['algebraic_cols']
    ]
    
    # Summary counts
    result['n_inputs'] = len(result['input_cols'])
    result['n_outputs'] = len(result['output_cols'])
    result['n_temporal'] = len(result['temporal_cols'])
    result['n_algebraic'] = len(result['algebraic_cols'])
    
    if verbose:
        print(f"\n{'='*60}")
        print("COLUMN IDENTIFICATION")
        print(f"{'='*60}")
        print(f"Input columns:      {result['n_inputs']}")
        print(f"Output columns:     {result['n_outputs']}")
        print(f"  - Temporal:       {result['n_temporal']} (dynamic)")
        print(f"  - Algebraic:      {result['n_algebraic']} (instantaneous)")
        print(f"  - Per CDU:        {config.n_outputs_per_cdu}")
        print(f"  - CDUs:           {config.num_cdus}")
    
    return result


def build_domain_column_info(
    df: pd.DataFrame,
    config: DomainDeepONetConfig,
    verbose: bool = True,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Build column information for a specific domain (Phase 4).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with all columns
    config : DomainDeepONetConfig
        Domain-specific configuration
    verbose : bool
        Whether to print column identification summary
        
    Returns
    -------
    Tuple[List[str], List[str], Dict[str, Any]]
        (input_cols, output_cols, column_info)
    """

    # Build input columns
    input_cols = []
    for cdu_id in config.cdu_ids:
        for pattern_name, pattern in config.input_patterns.items():
            if pattern_name == 'T_ext':
                continue
            col = _resolve_cdu_column(df, pattern, cdu_id)
            if col is not None:
                input_cols.append(col)
    
    # Add T_ext
    t_ext_pattern = config.input_patterns.get('T_ext', '')
    if t_ext_pattern and t_ext_pattern in df.columns:
        input_cols.append(t_ext_pattern)
    else:
        for col in df.columns:
            if 't_ext' in col.lower() and col not in input_cols:
                input_cols.append(col)
                break
    
    input_cols = list(dict.fromkeys(input_cols))
    
    # Build output columns for this domain
    output_cols = []
    column_info: Dict[str, Any] = {
        'primary_cols': [],
        'secondary_cols': [],
        'supply_cols': [],
        'return_cols': [],
        'col_to_cdu': {},
        'col_to_type': {},
    }
    
    for output_type in config.domain_outputs:
        pattern = config.output_patterns.get(output_type, '')
        if not pattern:
            # Prefix fallback: output_type may be a stem of the actual key
            # e.g. "T_prim_s" → "T_prim_s_C" when using suffixed output_patterns
            for key, val in config.output_patterns.items():
                if key.startswith(output_type) or output_type.startswith(key):
                    pattern = val
                    break
        if not pattern:
            continue
        for cdu_id in config.cdu_ids:
            col = _resolve_cdu_column(df, pattern, cdu_id)
            if col is not None:
                output_cols.append(col)
                column_info['col_to_cdu'][col] = cdu_id
                column_info['col_to_type'][col] = output_type
                
                # Classify primary/secondary
                if 'prim' in output_type.lower():
                    column_info['primary_cols'].append(col)
                elif 'sec' in output_type.lower():
                    column_info['secondary_cols'].append(col)
                
                # Classify supply/return (for pressure)
                if '_s_' in col or '_s_' in output_type:
                    column_info['supply_cols'].append(col)
                elif '_r_' in col or '_r_' in output_type:
                    column_info['return_cols'].append(col)
    
    # Compute indices
    column_info['primary_indices'] = [
        output_cols.index(c) for c in column_info['primary_cols']
    ]
    column_info['secondary_indices'] = [
        output_cols.index(c) for c in column_info['secondary_cols']
    ]
    
    if verbose:
        print(f"\n{config.domain.upper()} Domain:")
        print(f"  Input columns:     {len(input_cols)}")
        print(f"  Output columns:    {len(output_cols)}")
        print(f"  Primary outputs:   {len(column_info['primary_cols'])}")
        print(f"  Secondary outputs: {len(column_info['secondary_cols'])}")
        if column_info['supply_cols']:
            print(f"  Supply outputs:    {len(column_info['supply_cols'])}")
            print(f"  Return outputs:    {len(column_info['return_cols'])}")
    
    return input_cols, output_cols, column_info


def build_federated_column_info(
    df: pd.DataFrame,
    config: FederatedConfig,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Build column information for federated model (Phase 5/6).
    
    Categorizes outputs into 6 decoder head groups.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with all columns
    config : FederatedConfig
        Federated configuration
    verbose : bool
        Whether to print column identification summary
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'input_cols': List of input column names
        - 'temp_cols', 'flow_cols', etc.: Columns for each head
        - 'dynamic_cols': All dynamic output columns (ordered by head)
        - '*_indices': Index ranges for each head within dynamic_cols
        - 'head_groups': Dict mapping head name to cols/indices/type
        - 'type_indices': Dict mapping output_type to indices (for physics)
    """
    result: Dict[str, Any] = {
        'input_cols': [],
        'temp_cols': [],          # G_T
        'flow_cols': [],          # G_V
        'pressure_cols': [],      # G_p
        'flow_sec_cols': [],      # G_Vs
        'pressure_sec_cols': [],  # G_ps
        'power_cols': [],         # G_W
        'col_to_cdu': {},
        'col_to_type': {},
    }


    # Build input columns
    for cdu_id in config.cdu_ids:
        for pattern_name, pattern in config.input_patterns.items():
            if pattern_name == 'T_ext':
                continue
            col = _resolve_cdu_column(df, pattern, cdu_id)
            if col is not None:
                result['input_cols'].append(col)
    
    # Add T_ext
    t_ext_pattern = config.input_patterns.get('T_ext', '')
    if t_ext_pattern and t_ext_pattern in df.columns:
        result['input_cols'].append(t_ext_pattern)
    else:
        for col in df.columns:
            if 't_ext' in col.lower() and col not in result['input_cols']:
                result['input_cols'].append(col)
                break
    
    result['input_cols'] = list(dict.fromkeys(result['input_cols']))
    
    # Map head key to result key
    head_to_result_key = {
        'G_T': 'temp_cols',
        'G_V': 'flow_cols',
        'G_p': 'pressure_cols',
        'G_Vs': 'flow_sec_cols',
        'G_ps': 'pressure_sec_cols',
        'G_W': 'power_cols',
    }
    
    # Build output columns by head group
    for head_name, output_types in config.head_outputs.items():
        result_key = head_to_result_key[head_name]
        for output_type in output_types:
            pattern = config.output_patterns.get(output_type, '')
            if not pattern:
                continue
            for cdu_id in config.cdu_ids:
                col = _resolve_cdu_column(df, pattern, cdu_id)
                if col is not None:
                    result[result_key].append(col)
                    result['col_to_cdu'][col] = cdu_id
                    result['col_to_type'][col] = output_type
    
    # Combine all dynamic columns (ordered by head)
    result['dynamic_cols'] = (
        result['temp_cols'] +
        result['flow_cols'] +
        result['pressure_cols'] +
        result['flow_sec_cols'] +
        result['pressure_sec_cols'] +
        result['power_cols']
    )
    result['all_output_cols'] = result['dynamic_cols']
    
    # Compute index ranges for each head within dynamic_cols
    n_T = len(result['temp_cols'])
    n_V = len(result['flow_cols'])
    n_p = len(result['pressure_cols'])
    n_Vs = len(result['flow_sec_cols'])
    n_ps = len(result['pressure_sec_cols'])
    n_W = len(result['power_cols'])
    
    offset = 0
    result['temp_indices'] = list(range(offset, offset + n_T)); offset += n_T
    result['flow_indices'] = list(range(offset, offset + n_V)); offset += n_V
    result['pressure_indices'] = list(range(offset, offset + n_p)); offset += n_p
    result['flow_sec_indices'] = list(range(offset, offset + n_Vs)); offset += n_Vs
    result['pressure_sec_indices'] = list(range(offset, offset + n_ps)); offset += n_ps
    result['power_indices'] = list(range(offset, offset + n_W)); offset += n_W
    
    # Build head_groups mapping
    result['head_groups'] = {
        'G_T': {
            'cols': result['temp_cols'],
            'indices': result['temp_indices'],
            'type': 'standard',
        },
        'G_V': {
            'cols': result['flow_cols'],
            'indices': result['flow_indices'],
            'type': 'standard',
        },
        'G_p': {
            'cols': result['pressure_cols'],
            'indices': result['pressure_indices'],
            'type': 'standard',
        },
        'G_Vs': {
            'cols': result['flow_sec_cols'],
            'indices': result['flow_sec_indices'],
            'type': 'skip',
        },
        'G_ps': {
            'cols': result['pressure_sec_cols'],
            'indices': result['pressure_sec_indices'],
            'type': 'skip',
        },
        'G_W': {
            'cols': result['power_cols'],
            'indices': result['power_indices'],
            'type': 'skip',
        },
    }
    
    # Build type_indices for physics constraints (Phase 6)
    # Maps output_type -> list of indices in dynamic_cols, one per CDU
    result['type_indices'] = {}
    all_output_types = set()
    for output_types in config.head_outputs.values():
        all_output_types.update(output_types)
    
    for output_type in all_output_types:
        pattern = config.output_patterns.get(output_type, '')
        if not pattern:
            continue
        indices = []
        for cdu_id in config.cdu_ids:
            col = pattern.format(cdu_id)
            if col in result['dynamic_cols']:
                indices.append(result['dynamic_cols'].index(col))
        result['type_indices'][output_type] = indices
    
    if verbose:
        print(f"\n{'='*60}")
        print("COLUMN IDENTIFICATION (Federated)")
        print(f"{'='*60}")
        print(f"Input columns:              {len(result['input_cols'])}")
        print(f"Total dynamic outputs:      {len(result['dynamic_cols'])}")
        print(f"  G_T  (temperatures):      {n_T}")
        print(f"  G_V  (prim flow):         {n_V}")
        print(f"  G_p  (prim pressure):     {n_p}")
        print(f"  G_Vs (sec flow):          {n_Vs}")
        print(f"  G_ps (sec pressure):      {n_ps}")
        print(f"  G_W  (pump power):        {n_W}")
        print(f"  Per-type index maps:      {len(result['type_indices'])} types")
    
    return result


def identify_qflow_columns(
    input_cols: List[str],
    config: SurrogateConfig,
    verbose: bool = True,
) -> List[int]:
    """
    Identify indices of Q_flow (or similar) columns for algebraic pathway.
    
    Parameters
    ----------
    input_cols : List[str]
        List of input column names
    config : SurrogateConfig
        Configuration with algebraic_primary_input pattern
        
    Returns
    -------
    List[int]
        Indices of Q_flow columns in input_cols
    """
    qflow_indices = []
    
    # Get the primary input pattern for algebraic pathway
    algebraic_input = getattr(config, 'algebraic_primary_input', 'q_flow')
    
    for i, col in enumerate(input_cols):
        if algebraic_input.lower() in col.lower():
            qflow_indices.append(i)
    
    if verbose:
        preview = qflow_indices[:3] if len(qflow_indices) > 3 else qflow_indices
        print(f"  Q_flow columns:   {len(qflow_indices)} (indices: {preview}...)")
    
    return qflow_indices


def build_cdu_column_mapping(
    cols: List[str],
    cdu_ids: List[int],
) -> Dict[str, int]:
    """
    Map each column to its CDU index (0 to num_cdus-1).
    
    Parameters
    ----------
    cols : List[str]
        List of column names
    cdu_ids : List[int]
        List of CDU IDs
        
    Returns
    -------
    Dict[str, int]
        Mapping from column name to CDU index
    """
    col_to_cdu_idx = {}
    
    for col in cols:
        found = False
        for idx, cdu_id in enumerate(cdu_ids):
            # Check various naming patterns
            if f'[{cdu_id}]' in col or f'_{cdu_id}_' in col or col.endswith(f'_{cdu_id}'):
                col_to_cdu_idx[col] = idx
                found = True
                break
        if not found:
            # Default to CDU 0 if pattern not found
            col_to_cdu_idx[col] = 0
    
    return col_to_cdu_idx


def get_column_cdu_indices(
    cols: List[str],
    cdu_ids: List[int],
) -> List[int]:
    """
    Get CDU index for each column as a list.
    
    Parameters
    ----------
    cols : List[str]
        List of column names
    cdu_ids : List[int]
        List of CDU IDs
        
    Returns
    -------
    List[int]
        List of CDU indices corresponding to each column
    """
    mapping = build_cdu_column_mapping(cols, cdu_ids)
    return [mapping[col] for col in cols]


def get_system_column_info(
    system_name: str,
    config: Optional[SurrogateConfig] = None,
) -> Dict[str, Any]:
    """
    Get column info template for a specific system.
    
    Parameters
    ----------
    system_name : str
        HPC system name ('summit', 'lassen', 'marconi100')
    config : Optional[SurrogateConfig]
        Configuration (created from system if not provided)
        
    Returns
    -------
    Dict[str, Any]
        Column info template with expected column patterns
    """
    if config is None:
        config = SurrogateConfig.from_system(system_name)
    
    return {
        'system_name': system_name,
        'num_cdus': config.num_cdus,
        'cdu_ids': config.cdu_ids,
        'input_patterns': config.input_patterns,
        'output_patterns': config.output_patterns,
        'expected_inputs_per_cdu': config.n_inputs_per_cdu,
        'expected_outputs_per_cdu': config.n_outputs_per_cdu,
    }


__all__ = [
    'build_column_info',
    'build_domain_column_info',
    'build_federated_column_info',
    'identify_qflow_columns',
    'build_cdu_column_mapping',
    'get_column_cdu_indices',
    'get_system_column_info',
]