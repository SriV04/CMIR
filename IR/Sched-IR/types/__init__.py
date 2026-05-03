"""Sched-IR schema type defaults.

This directory intentionally uses the name ``types`` to mirror NN-IR.
When scripts are executed from ``IR/Sched-IR/``, Python can resolve
``import types`` to this package before the stdlib module. To avoid
breaking stdlib imports such as ``importlib.util``, this package proxies
the real stdlib ``types`` module into its top-level namespace.
"""

from __future__ import annotations

import os as _os


_STDLIB_TYPES = _os.path.join(_os.path.dirname(_os.__file__), "types.py")
_ns: dict[str, object] = {}
with open(_STDLIB_TYPES, "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), _STDLIB_TYPES, "exec"), _ns)

for _key, _value in _ns.items():
    if _key in {"__builtins__", "__name__", "__file__", "__package__"}:
        continue
    globals()[_key] = _value

