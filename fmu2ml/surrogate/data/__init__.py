"""
Data processing utilities for surrogate models.

Provides datasets, normalizers, and column information builders
for all surrogate model phases (1-6).
"""

from .column_info import (
    build_column_info,
    build_domain_column_info,
    build_federated_column_info,
    identify_qflow_columns,
    build_cdu_column_mapping,
    get_column_cdu_indices,
    get_system_column_info,
)

from .normalizer import (
    ZScoreNormalizer,
    DeltaNormalizer,
    InputWhitener,
    DomainNormalizer,
    FederatedNormalizer,
    SurrogateNormalizer,
)

from .dataset import (
    LSTMDataset,
    HybridDataset,
    DomainDataset,
    FederatedDataset,
    create_dataloaders_lstm,
    create_dataloaders_hybrid,
    create_dataloaders_domain,
    create_dataloaders_federated,
    create_surrogate_dataloaders,
    measure_dominant_cycle,
    suggest_chunk_size,
    chunk_shuffled_ranges,
    assign_chunk_ranges,
    build_split_arrays,
    concat_train_arrays,
    print_split_report,
    load_chunk_dataframes,
    concat_chunk_dataframes,
)

from .store import (
    build_chunk_store,
    discover_chunk_files,
    load_store_manifest,
    open_chunk_array,
    block_mean_decimate,
    assign_chunk_ids_explicit,
)

from .federated_store import (
    FederatedStoreDataset,
    build_store_for_config,
    create_dataloaders_federated_store,
    create_dataloaders_multirate,
    fit_store_normalizer,
)


__all__ = [
    # Column info
    'build_column_info',
    'build_domain_column_info',
    'build_federated_column_info',
    'identify_qflow_columns',
    'build_cdu_column_mapping',
    'get_column_cdu_indices',
    'get_system_column_info',
    # Normalizers
    'ZScoreNormalizer',
    'DeltaNormalizer',
    'InputWhitener',
    'DomainNormalizer',
    'FederatedNormalizer',
    'SurrogateNormalizer',
    # Datasets
    'LSTMDataset',
    'HybridDataset',
    'DomainDataset',
    'FederatedDataset',
    # Dataloader factories
    'create_dataloaders_lstm',
    'create_dataloaders_hybrid',
    'create_dataloaders_domain',
    'create_dataloaders_federated',
    'create_surrogate_dataloaders',
    # Split strategy
    'measure_dominant_cycle',
    'suggest_chunk_size',
    'chunk_shuffled_ranges',
    'assign_chunk_ranges',
    'build_split_arrays',
    'concat_train_arrays',
    'print_split_report',
    'load_chunk_dataframes',
    'concat_chunk_dataframes',
    # Out-of-core memmap store (recommended for full-data Phase 5/6)
    'build_chunk_store',
    'discover_chunk_files',
    'load_store_manifest',
    'open_chunk_array',
    'block_mean_decimate',
    'assign_chunk_ids_explicit',
    'FederatedStoreDataset',
    'build_store_for_config',
    'create_dataloaders_federated_store',
    'create_dataloaders_multirate',
    'fit_store_normalizer',
]