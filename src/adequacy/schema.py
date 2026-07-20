"""Capa de ADECUACIÓN del NDX-Coder — vista Python del contrato COMPARTIDO.

La fuente de verdad vive en la AUTORIDAD COMPARTIDA que todos los repos ya
resuelven: nodex-compiler (paquete dual JS `nodex-compiler` / Py `nodex_compiler`).
    nodex-compiler/src/dsl/adequacy.contract.json
Se identifica por PAQUETE + `version`, no por path — por eso no hay "¿cuál copia
es la buena?": hay un solo nodex_compiler y trae su propio contrato, versionado.
Este módulo lo LEE (no lo redefine).

Resolución (igual patrón que run_ndx / NODEXC_CLI):
    env NODEX_ADEQUACY_CONTRACT
    → API del paquete: from nodex_compiler import adequacy_contract  (si existe)
    → path hermano:    ../nodex-compiler/src/dsl/adequacy.contract.json
    → fallback embebido (deja a brain funcionando standalone; si difiere, manda el JSON)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# ---- fallback embebido (si no se encuentra el contrato compartido) ----------
_FALLBACK = {
    "vocab": {
        "systems": ["frame", "truss", "continuous_beam", "cantilever", "arch", "cable",
                    "slab", "wall", "shell", "solid", "mixed"],
        "categories": ["missing_param", "wrong_value", "wrong_typology", "wrong_support",
                       "wrong_load", "wrong_material", "wrong_section", "topology_broken",
                       "extra_element", "wrong_analysis", "wrong_norm", "missing_check",
                       "physical_insane"],
        "severities": ["blocker", "major", "minor"],
        "severity_weight": {"blocker": 1.0, "major": 0.4, "minor": 0.1},
    }
}


EXPECTED_VERSION = "0.1.0"   # versión del contrato contra la que se construyó este código


def _contract_path() -> Path | None:
    env = os.environ.get("NODEX_ADEQUACY_CONTRACT")
    if env:
        return Path(env)
    # autoridad compartida: ../nodex-compiler/src/dsl/adequacy.contract.json (repos hermanos)
    sibling = (Path(__file__).resolve().parents[3]
               / "nodex-compiler" / "src" / "dsl" / "adequacy.contract.json")
    return sibling if sibling.is_file() else None


def load_contract() -> dict:
    # 1) API del paquete compartido (identidad por import, no por path)
    if not os.environ.get("NODEX_ADEQUACY_CONTRACT"):
        try:
            from nodex_compiler import adequacy_contract  # type: ignore
            return adequacy_contract()
        except Exception:
            pass
    # 2) path hermano / override por env
    p = _contract_path()
    if p and p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    # 3) fallback embebido
    return _FALLBACK


def check_version(strict: bool = False) -> str | None:
    """Avisa si el contrato resuelto no coincide con EXPECTED_VERSION."""
    v = load_contract().get("version")
    if v != EXPECTED_VERSION:
        msg = f"[adequacy] contrato v{v} ≠ esperado v{EXPECTED_VERSION}"
        if strict:
            raise RuntimeError(msg)
        import warnings
        warnings.warn(msg)
    return v


_C = load_contract()
_VOCAB = _C.get("vocab", _FALLBACK["vocab"])

SYSTEMS = _VOCAB["systems"]
CATEGORIES = _VOCAB["categories"]
SEVERITIES = _VOCAB["severities"]
SEVERITY_WEIGHT = _VOCAB["severity_weight"]


@dataclass
class Finding:
    category: str                       # ∈ CATEGORIES
    severity: str                       # ∈ SEVERITIES
    field: str                          # a qué parte del spec corresponde, p.ej. "geometry.spans_m"
    message: str
    expected: object = None
    actual: object = None
    location: dict | None = None        # {line, col, statement} de analyze()/AST
    fix: dict | None = None             # EditOp quirúrgico para nodex-code authoring


@dataclass
class RequiredSpec:
    pedido: str
    system: str | None = None
    geometry: dict = field(default_factory=dict)
    materials: list = field(default_factory=list)
    sections: list = field(default_factory=list)
    loads: list = field(default_factory=list)
    supports: list = field(default_factory=list)
    analysis: list = field(default_factory=list)
    norm: str | None = None
    checks: list = field(default_factory=list)
    tolerances: dict = field(default_factory=lambda: {"length_m": 0.01, "load_pct": 1.0})


def adequacy_verdict(findings: list[Finding]) -> bool:
    """Binaria: adecuado ⟺ ningún finding blocker/major."""
    return not any(f.severity in ("blocker", "major") for f in findings)


def adequacy_score(findings: list[Finding]) -> float:
    """Graduado 0..1 para rankear best-of-N: 1 - Σ peso(severidad)."""
    penalty = sum(SEVERITY_WEIGHT.get(f.severity, 0.0) for f in findings)
    return max(0.0, 1.0 - penalty)
