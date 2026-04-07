#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference


class PostprocessError(RuntimeError):
    pass


@dataclass(frozen=True)
class PostprocessStats:
    folded_bn_count: int
    patched_reshape_count: int
    cleared_value_info_count: int
    updated_batch_axis_count: int
    original_output_name: str
    final_output_name: str


def _iter_value_infos(model: onnx.ModelProto) -> Iterable[onnx.ValueInfoProto]:
    yield from model.graph.input
    yield from model.graph.output
    yield from model.graph.value_info


def _get_tensor_names(model: onnx.ModelProto) -> set[str]:
    names: set[str] = set()
    for value_info in _iter_value_infos(model):
        if value_info.name:
            names.add(value_info.name)
    for initializer in model.graph.initializer:
        if initializer.name:
            names.add(initializer.name)
    for node in model.graph.node:
        for name in node.input:
            if name:
                names.add(name)
        for name in node.output:
            if name:
                names.add(name)
    return names


def _get_epsilon(node: onnx.NodeProto) -> float:
    epsilon = 1e-5
    for attr in node.attribute:
        if attr.name == "epsilon":
            epsilon = attr.f
            break
    return epsilon


def _make_unique_name(existing_names: set[str], prefix: str) -> str:
    if prefix not in existing_names:
        existing_names.add(prefix)
        return prefix
    index = 1
    while True:
        candidate = f"{prefix}_{index}"
        if candidate not in existing_names:
            existing_names.add(candidate)
            return candidate
        index += 1


def _remove_unused_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    used_names = {graph_input.name for graph_input in model.graph.input}
    for node in model.graph.node:
        used_names.update(name for name in node.input if name)

    kept_initializers = [
        initializer for initializer in model.graph.initializer if initializer.name in used_names
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    return model


def _rename_tensor(model: onnx.ModelProto, old_name: str, new_name: str) -> None:
    if old_name == new_name:
        return

    existing_names = _get_tensor_names(model)
    if new_name in existing_names and new_name != old_name:
        raise PostprocessError(f'Cannot rename tensor "{old_name}" to "{new_name}": name already exists')

    for value_info in _iter_value_infos(model):
        if value_info.name == old_name:
            value_info.name = new_name

    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == old_name:
                node.input[index] = new_name
        for index, output_name in enumerate(node.output):
            if output_name == old_name:
                node.output[index] = new_name


def _update_batch_dim_names(model: onnx.ModelProto, batch_dim_name: str) -> int:
    updated_names: set[str] = set()
    for value_info in _iter_value_infos(model):
        value_kind = value_info.type.WhichOneof("value")
        if value_kind != "tensor_type":
            continue

        shape = value_info.type.tensor_type.shape
        if len(shape.dim) == 0:
            continue

        dim = shape.dim[0]
        if dim.dim_param == batch_dim_name and not dim.HasField("dim_value"):
            continue

        dim.ClearField("dim_value")
        dim.dim_param = batch_dim_name
        updated_names.add(value_info.name)

    return len(updated_names)


def _patch_fixed_batch_reshape_initializers(model: onnx.ModelProto) -> int:
    initializer_map = {initializer.name: initializer for initializer in model.graph.initializer}
    patched_count = 0

    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) != 2:
            continue

        shape_initializer = initializer_map.get(node.input[1])
        if shape_initializer is None:
            continue

        shape_value = numpy_helper.to_array(shape_initializer)
        if shape_value.ndim != 1 or shape_value.size != 3:
            continue
        if int(shape_value[0]) != 1 or int(shape_value[2]) != -1:
            continue

        patched_shape = shape_value.copy()
        patched_shape[0] = 0
        shape_initializer.CopyFrom(
            numpy_helper.from_array(patched_shape.astype(shape_value.dtype, copy=False), name=shape_initializer.name)
        )
        patched_count += 1

    return patched_count


def _clear_value_infos(model: onnx.ModelProto) -> int:
    cleared_count = len(model.graph.value_info)
    del model.graph.value_info[:]
    return cleared_count


def _validate_fold_shapes(
    bn_node: onnx.NodeProto,
    matmul_weight: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
) -> None:
    if matmul_weight.ndim != 2:
        raise PostprocessError(
            f'Cannot fold "{bn_node.name or bn_node.output[0]}": MatMul weight must be 2D, got {matmul_weight.ndim}D'
        )

    bn_params = {
        "scale": scale,
        "bias": bias,
        "mean": mean,
        "var": var,
    }
    for label, value in bn_params.items():
        if value.ndim != 1:
            raise PostprocessError(
                f'Cannot fold "{bn_node.name or bn_node.output[0]}": BatchNormalization {label} must be 1D'
            )

    channel_dim = matmul_weight.shape[1]
    for label, value in bn_params.items():
        if value.shape[0] != channel_dim:
            raise PostprocessError(
                f'Cannot fold "{bn_node.name or bn_node.output[0]}": '
                f'MatMul output channels ({channel_dim}) do not match BatchNormalization {label} ({value.shape[0]})'
            )


def fold_batch_norms_after_matmul(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    graph = model.graph
    initializer_map = {initializer.name: initializer for initializer in graph.initializer}
    producer_by_output = {
        output_name: node
        for node in graph.node
        for output_name in node.output
        if output_name
    }
    existing_tensor_names = _get_tensor_names(model)
    existing_node_names = {node.name for node in graph.node if node.name}
    updated_nodes = []
    folded_bn_count = 0

    for node in graph.node:
        if node.op_type != "BatchNormalization":
            updated_nodes.append(node)
            continue

        producer = producer_by_output.get(node.input[0]) if node.input else None
        if producer is None or producer.op_type != "MatMul":
            updated_nodes.append(node)
            continue

        if len(node.input) != 5:
            raise PostprocessError(
                f'Cannot fold "{node.name or node.output[0]}": BatchNormalization must have 5 inputs'
            )
        if len(producer.input) != 2:
            raise PostprocessError(
                f'Cannot fold "{node.name or node.output[0]}": MatMul must have 2 inputs'
            )

        required_initializers = [producer.input[1], *node.input[1:]]
        missing = [name for name in required_initializers if name not in initializer_map]
        if missing:
            raise PostprocessError(
                f'Cannot fold "{node.name or node.output[0]}": missing initializer(s): {", ".join(missing)}'
            )

        weight = numpy_helper.to_array(initializer_map[producer.input[1]]).copy()
        scale = numpy_helper.to_array(initializer_map[node.input[1]])
        bias = numpy_helper.to_array(initializer_map[node.input[2]])
        mean = numpy_helper.to_array(initializer_map[node.input[3]])
        var = numpy_helper.to_array(initializer_map[node.input[4]])
        _validate_fold_shapes(node, weight, scale, bias, mean, var)

        epsilon = _get_epsilon(node)
        alpha = scale / np.sqrt(var + epsilon)
        folded_weight = weight * alpha.reshape((1, -1))
        folded_bias = (bias - mean * alpha).astype(weight.dtype, copy=False)

        initializer_map[producer.input[1]].CopyFrom(
            numpy_helper.from_array(folded_weight.astype(weight.dtype, copy=False), name=producer.input[1])
        )

        bias_name_prefix = f'{node.name or node.output[0]}_folded_bias'
        bias_name = _make_unique_name(existing_tensor_names, bias_name_prefix)
        graph.initializer.append(numpy_helper.from_array(folded_bias, name=bias_name))
        initializer_map[bias_name] = graph.initializer[-1]

        add_name_prefix = f'{node.name or node.output[0]}_folded_add'
        add_name = _make_unique_name(existing_node_names, add_name_prefix)
        updated_nodes.append(
            helper.make_node(
                "Add",
                inputs=[node.input[0], bias_name],
                outputs=list(node.output),
                name=add_name,
            )
        )
        folded_bn_count += 1

    if folded_bn_count == 0:
        return model, 0

    del graph.node[:]
    graph.node.extend(updated_nodes)
    model = _remove_unused_initializers(model)
    return model, folded_bn_count


def process_model(
    input_path: str | Path,
    output_path: str | Path,
    output_name: str = "feat",
    batch_dim_name: str = "N",
) -> PostprocessStats:
    model = onnx.load(str(input_path))
    onnx.checker.check_model(model)

    if len(model.graph.output) != 1:
        raise PostprocessError(f"Expected exactly 1 model output, but got {len(model.graph.output)}")

    model, folded_bn_count = fold_batch_norms_after_matmul(model)
    patched_reshape_count = _patch_fixed_batch_reshape_initializers(model)

    original_output_name = model.graph.output[0].name
    _rename_tensor(model, original_output_name, output_name)
    cleared_value_info_count = _clear_value_infos(model)

    try:
        model = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
    except Exception as exc:
        raise PostprocessError(f"Symbolic shape inference failed: {exc}") from exc

    updated_batch_axis_count = _update_batch_dim_names(model, batch_dim_name)
    onnx.checker.check_model(model)

    try:
        onnx.save(model, str(output_path))
    except Exception as exc:
        raise PostprocessError(f'Failed to save ONNX model to "{output_path}": {exc}') from exc

    return PostprocessStats(
        folded_bn_count=folded_bn_count,
        patched_reshape_count=patched_reshape_count,
        cleared_value_info_count=cleared_value_info_count,
        updated_batch_axis_count=updated_batch_axis_count,
        original_output_name=original_output_name,
        final_output_name=output_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process an ONNX model for TransFace inference")
    parser.add_argument("input", type=str, help="Input ONNX model path")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX model path")
    parser.add_argument("--output-name", type=str, default="feat", help='Renamed output tensor name')
    parser.add_argument(
        "--batch-dim-name",
        type=str,
        default="N",
        help='Batch dimension symbolic name to apply to input/output/value_info tensors',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        stats = process_model(
            input_path=args.input,
            output_path=args.output,
            output_name=args.output_name,
            batch_dim_name=args.batch_dim_name,
        )
    except PostprocessError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: Unexpected failure: {exc}", file=sys.stderr)
        return 1

    print(f"Folded BatchNormalization nodes: {stats.folded_bn_count}")
    print(f"Patched fixed-batch Reshape nodes: {stats.patched_reshape_count}")
    print(f"Cleared stale value_info entries: {stats.cleared_value_info_count}")
    print(f'Updated batch axis tensors: {stats.updated_batch_axis_count}')
    print(f'Output name: "{stats.original_output_name}" -> "{stats.final_output_name}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
