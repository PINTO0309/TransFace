from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from postprocess_onnx import PostprocessError, process_model


def _make_model(graph: onnx.GraphProto) -> onnx.ModelProto:
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="postprocess-onnx-tests",
    )
    model.ir_version = 10
    return model


def _save_model(path: Path, model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _run_model(path: Path, inputs: dict[str, np.ndarray]) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output_name = session.get_outputs()[0].name
    return session.run([output_name], inputs)[0]


def _build_matmul_bn_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 2])
    matmul_out = helper.make_tensor_value_info("matmul_out", TensorProto.FLOAT, ["batch", 2])

    weight = numpy_helper.from_array(
        np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=np.float32),
        name="weight",
    )
    scale = numpy_helper.from_array(np.array([1.25, 0.75], dtype=np.float32), name="scale")
    bias = numpy_helper.from_array(np.array([0.5, -0.25], dtype=np.float32), name="bias")
    mean = numpy_helper.from_array(np.array([0.1, -0.2], dtype=np.float32), name="mean")
    var = numpy_helper.from_array(np.array([0.3, 0.8], dtype=np.float32), name="var")

    matmul = helper.make_node("MatMul", ["input", "weight"], ["matmul_out"], name="matmul")
    batch_norm = helper.make_node(
        "BatchNormalization",
        ["matmul_out", "scale", "bias", "mean", "var"],
        ["output"],
        name="bn",
        epsilon=1e-5,
    )

    graph = helper.make_graph(
        [matmul, batch_norm],
        "matmul-bn-graph",
        [input_info],
        [output_info],
        [weight, scale, bias, mean, var],
        value_info=[matmul_out],
    )
    return _make_model(graph)


def _build_multi_output_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 2])
    output_a = helper.make_tensor_value_info("output_a", TensorProto.FLOAT, ["batch", 2])
    output_b = helper.make_tensor_value_info("output_b", TensorProto.FLOAT, ["batch", 2])
    identity_a = helper.make_node("Identity", ["input"], ["output_a"], name="identity_a")
    identity_b = helper.make_node("Identity", ["input"], ["output_b"], name="identity_b")
    graph = helper.make_graph([identity_a, identity_b], "multi-output-graph", [input_info], [output_a, output_b])
    return _make_model(graph)


def _build_bn_without_matmul_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 2])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 2])

    scale = numpy_helper.from_array(np.array([1.0, 0.9], dtype=np.float32), name="scale")
    bias = numpy_helper.from_array(np.array([0.1, -0.2], dtype=np.float32), name="bias")
    mean = numpy_helper.from_array(np.array([0.0, 0.0], dtype=np.float32), name="mean")
    var = numpy_helper.from_array(np.array([1.0, 1.0], dtype=np.float32), name="var")

    batch_norm = helper.make_node(
        "BatchNormalization",
        ["input", "scale", "bias", "mean", "var"],
        ["output"],
        name="bn_no_matmul",
    )
    graph = helper.make_graph(
        [batch_norm],
        "bn-without-matmul-graph",
        [input_info],
        [output_info],
        [scale, bias, mean, var],
    )
    return _make_model(graph)


class PostprocessOnnxTests(unittest.TestCase):
    def test_fold_matmul_batchnorm_and_preserve_output(self) -> None:
        input_array = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "input.onnx"
            output_path = tmpdir_path / "output.onnx"
            _save_model(input_path, _build_matmul_bn_model())

            expected = _run_model(input_path, {"input": input_array})
            stats = process_model(input_path, output_path)
            actual = _run_model(output_path, {"input": input_array})
            processed = onnx.load(str(output_path))

        self.assertEqual(stats.folded_bn_count, 1)
        self.assertEqual(stats.original_output_name, "output")
        self.assertEqual(stats.final_output_name, "feat")
        self.assertTrue(np.allclose(expected, actual, atol=1e-5))
        self.assertEqual(processed.graph.output[0].name, "feat")
        self.assertEqual([node.op_type for node in processed.graph.node], ["MatMul", "Add"])
        self.assertNotIn("BatchNormalization", [node.op_type for node in processed.graph.node])
        self.assertTrue(any(node.output[0] == "feat" for node in processed.graph.node))

        tensor_infos = list(processed.graph.input) + list(processed.graph.output) + list(processed.graph.value_info)
        self.assertTrue(tensor_infos)
        for value_info in tensor_infos:
            shape = value_info.type.tensor_type.shape
            if not shape.dim:
                continue
            self.assertEqual(shape.dim[0].dim_param, "N")
            self.assertFalse(shape.dim[0].HasField("dim_value"))

    def test_multiple_outputs_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "multi_output.onnx"
            output_path = tmpdir_path / "unused.onnx"
            _save_model(input_path, _build_multi_output_model())

            with self.assertRaises(PostprocessError):
                process_model(input_path, output_path)

    def test_batchnorm_without_direct_matmul_is_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "bn_input.onnx"
            output_path = tmpdir_path / "bn_output.onnx"
            _save_model(input_path, _build_bn_without_matmul_model())

            stats = process_model(input_path, output_path)
            processed = onnx.load(str(output_path))

        self.assertEqual(stats.folded_bn_count, 0)
        self.assertIn("BatchNormalization", [node.op_type for node in processed.graph.node])
        self.assertEqual(processed.graph.output[0].name, "feat")


if __name__ == "__main__":
    unittest.main()
