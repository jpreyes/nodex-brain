"""TSD — bias de atención ultramétrico para el NDX-Coder."""
from .ultrametric import tsd_bias_matrix, ultrametric_matrix, kernel_bias
from .collator import TSDCollator
from .infer import generate_tsd

__all__ = ["tsd_bias_matrix", "ultrametric_matrix", "kernel_bias", "TSDCollator", "generate_tsd"]
