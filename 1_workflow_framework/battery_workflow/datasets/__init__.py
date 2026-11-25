"""
Loaders for the published datasets consumed by Case 1 and Case 2 workflows.

Each module here wraps an upstream dataset release pinned by commit/version and
per-file SHA-256. The upstream GitHub Case 1 releases are never vendored; see
``_source`` for how a local cache directory is verified on load.

    ml_htvs   ML-HTVS-Li-SE   audit paper_id 10771  [@kim2023accelerating]
    lisse_mp  LiSSE-MP        audit paper_id 10664  [@wang2024high]
    ml_sse    ML-SSE          audit paper_id 10102  [@kurniawan2025unveiling]
    oedb      OEDB            CC BY 4.0 snapshot
"""

from ._source import (
    DatasetIntegrityError,
    MissingDatasetError,
    PublishedSource,
    to_float,
)

__all__ = [
    "DatasetIntegrityError",
    "MissingDatasetError",
    "PublishedSource",
    "to_float",
    "lisse_mp",
    "ml_htvs",
    "ml_sse",
    "oedb",
]
