"""Validación de código NDX contra el nodex-compiler.

Es el VERIFICADOR del proyecto: lo usan el eval (% de decks que compilan) y el
loop de auto-mejora (STaR) para filtrar generaciones. Llama al CLI del
compilador (JSON-in / JSON-out) y no requiere instalar el paquete Python.

`runNdx` compila **y resuelve** el deck → `{model, solves, results, warnings}`,
o `{error}` si algo falla. Aquí `ok=True` significa "compila y resuelve".

Resolución del CLI (igual que `nodex_compiler.runner`):
    env NODEXC_CLI  →  ../nodex-compiler/src/cli.js  →  `nodexc` en PATH

Requiere Node y (para resolver) el WASM de Nodex, igual que el lado JS.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# nodex-brain y nodex-compiler son hermanos bajo .../sistema/
_REPO_CLI = Path(__file__).resolve().parents[2] / "nodex-compiler" / "src" / "cli.js"


def _resolve_cli() -> list[str]:
    env = os.environ.get("NODEXC_CLI")
    if env:
        return ["node", env]
    if _REPO_CLI.exists():
        return ["node", str(_REPO_CLI)]
    on_path = shutil.which("nodexc")
    if on_path:
        return [on_path]
    raise FileNotFoundError(
        "CLI de nodex-compiler no encontrado — define NODEXC_CLI apuntando a "
        "src/cli.js, o instala `nodexc` en el PATH."
    )


@dataclass
class Validation:
    ok: bool
    error: str | None = None
    result: dict | None = None


def validate(ndx: str, timeout: float = 60.0) -> Validation:
    """Compila y resuelve un deck NDX. `ok=True` si corre sin error."""
    try:
        cmd = _resolve_cli()
    except FileNotFoundError as exc:
        return Validation(ok=False, error=str(exc))

    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps({"op": "runNdx", "ndx": ndx}),
            capture_output=True,
            text=True,
            encoding="utf-8",   # el CLI siempre emite UTF-8 (m², m⁴, símbolos)
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return Validation(ok=False, error=f"timeout tras {timeout}s")

    out = (proc.stdout or "").strip()
    try:
        result = json.loads(out) if out else {}
    except json.JSONDecodeError:
        return Validation(ok=False, error=(proc.stderr or out or "sin salida del CLI").strip())

    if isinstance(result, dict) and result.get("error"):
        return Validation(ok=False, error=str(result["error"]))
    return Validation(ok=True, result=result if isinstance(result, dict) else None)


def compile_rate(decks: list[str], max_workers: int = 8) -> tuple[float, list[Validation]]:
    """% de decks que compilan+resuelven, y el detalle por deck (en paralelo)."""
    if not decks:
        return 0.0, []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(validate, decks))
    ok = sum(1 for r in results if r.ok)
    return ok / len(results), results


if __name__ == "__main__":
    import sys

    src = Path(sys.argv[1]).read_text(encoding="utf-8") if len(sys.argv) > 1 else sys.stdin.read()
    v = validate(src)
    print(json.dumps({"ok": v.ok, "error": v.error}, ensure_ascii=False, indent=2))
