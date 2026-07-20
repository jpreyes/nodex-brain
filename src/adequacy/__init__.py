"""Capa de adecuación del NDX-Coder: contrato + checker + score."""
from .schema import (
    CATEGORIES, SYSTEMS, SEVERITY_WEIGHT,
    Finding, RequiredSpec, adequacy_verdict, adequacy_score,
)

__all__ = [
    "CATEGORIES", "SYSTEMS", "SEVERITY_WEIGHT",
    "Finding", "RequiredSpec", "adequacy_verdict", "adequacy_score",
]
