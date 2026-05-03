from __future__ import annotations

import warnings
from typing import Iterable


def _warn(msg: str) -> None:
    warnings.warn(msg, stacklevel=2)


def _require_fields(props: dict, keys: Iterable[str], *, context: str) -> list[str]:
    missing = [key for key in keys if props.get(key) is None]
    if missing:
        raise ValueError(f"{context}: missing required fields {missing}")
    return missing


def validate_sched_decomposition(g) -> None:
    for vx in g.vertices:
        p = g.pmap[vx]
        op = p.get("op")
        params = p.get("op_params") or {}
        name = p.get("nn_layer_name") or f"vx{vx}"

        _require_fields(
            p,
            ["op", "op_params", "nn_layer_name", "inserted_by"],
            context=f"vertex {vx} ({name})",
        )

        if op == "dense":
            _require_fields(params, ["kernel_shape"], context=f"dense {name}")
            if params.get("qkernel_values") is None and params.get("kernel_values") is None:
                raise ValueError(f"dense {name}: missing qkernel_values/kernel_values")
            if params.get("input_kif") is None and params.get("input_qint") is None and params.get("in_bw") is None:
                _warn(f"dense {name}: no input precision metadata or legacy in_bw")

        elif op == "reduce":
            _require_fields(
                params,
                ["mode", "axes", "input_shape", "output_shape"],
                context=f"reduce {name}",
            )
            if params.get("reduction_width") is None:
                _warn(f"reduce {name}: reduction_width could not be inferred")

        elif op == "elementwise":
            _require_fields(
                params,
                ["op", "input_shapes", "output_shape", "n_inputs"],
                context=f"elementwise {name}",
            )
            if len(params.get("input_shapes") or []) != params.get("n_inputs"):
                raise ValueError(f"elementwise {name}: n_inputs does not match input_shapes")

        elif op == "activation":
            _require_fields(params, ["func", "input_shape"], context=f"activation {name}")

    for edge in g.edges:
        ep = g.pmap[edge]
        if ep.get("tensor_shape") is None:
            raise ValueError(f"edge {edge}: missing tensor_shape")
        if not any(
            ep.get(key) is not None
            for key in ("bitwidth", "qint", "kif", "src_qint", "src_kif", "dst_qint", "dst_kif")
        ):
            _warn(f"edge {edge}: missing precision metadata")

