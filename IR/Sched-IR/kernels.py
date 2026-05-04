from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGET = HERE / "cost_evals" / "kernels.py"

spec = importlib.util.spec_from_file_location("_sched_ir_cost_evals_kernels", TARGET)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

_EXPORTS = [name for name in dir(mod) if not name.startswith("__")]
for name in _EXPORTS:
    if not name.startswith("__"):
        globals()[name] = getattr(mod, name)

__all__ = _EXPORTS
