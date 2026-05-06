"""Public model API.

This file intentionally stays small and only re-exports the model pieces that
external callers use. The heavy implementation now lives in focused modules:
- src.model_ops
- src.model_graph
- src.model_personal
- src.model_structure
- src.model_cdm
"""

from src.model_cdm import CognitiveDiagnosisModel
from src.model_graph import ConceptGraphConv, MultiHeadRelationLearning, StudentKnowledgeEncoder
from src.model_ops import (
    _apply_sparse_local_posterior,
    _build_no_a_query_support_cache,
    _build_support_cache,
    _compute_sparse_local_messages,
    _gather_head_rows,
    _gather_head_support_features,
    _gather_row_support,
    _masked_absmax_or_zero,
    _masked_sparse_row_entropy,
    _masked_support_softmax,
    _normalize_sparse_scores,
    _pack_active_row_index,
    _safe_zero_preserving_sqrt,
)
from src.model_personal import AdaptiveGate, PersonalRelationGenerator
from src.model_structure import ConceptStructureModeling

__all__ = [
    'AdaptiveGate',
    'CognitiveDiagnosisModel',
    'ConceptGraphConv',
    'ConceptStructureModeling',
    'MultiHeadRelationLearning',
    'PersonalRelationGenerator',
    'StudentKnowledgeEncoder',
    '_apply_sparse_local_posterior',
    '_build_no_a_query_support_cache',
    '_build_support_cache',
    '_compute_sparse_local_messages',
    '_gather_head_rows',
    '_gather_head_support_features',
    '_gather_row_support',
    '_masked_absmax_or_zero',
    '_masked_sparse_row_entropy',
    '_masked_support_softmax',
    '_normalize_sparse_scores',
    '_pack_active_row_index',
    '_safe_zero_preserving_sqrt',
]
