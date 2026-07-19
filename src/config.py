"""Carga de configuración YAML con herencia vía la clave `extends`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Fusiona `override` sobre `base` de forma recursiva."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Carga un YAML, resolviendo `extends` relativo al archivo.

    El config hijo se fusiona sobre el padre (deep merge).
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    parent_ref = cfg.pop("extends", None)
    if parent_ref:
        parent_cfg = load_config(path.parent / parent_ref)
        cfg = _deep_merge(parent_cfg, cfg)

    return cfg
