"""TSD — bias de atención ultramétrico para el NDX-Coder.

`ultrametric` es numpy puro; `collator` e `infer` necesitan torch. Se cargan de forma
PEREZOSA para que las herramientas de análisis —p.ej. `experiments/eval_tail.py`, que solo
cuenta statement kinds— corran sin torch instalado. Antes este archivo los importaba de
entrada, así que bastaba tocar `ultrametric` para exigir torch.
"""
from .ultrametric import kernel_bias, tsd_bias_matrix, ultrametric_matrix

__all__ = ["tsd_bias_matrix", "ultrametric_matrix", "kernel_bias", "TSDCollator", "generate_tsd"]

_LAZY = {"TSDCollator": ".collator", "generate_tsd": ".infer"}


def __getattr__(name):          # PEP 562
    if name in _LAZY:
        from importlib import import_module
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
