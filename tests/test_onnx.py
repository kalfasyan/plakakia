from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from plakakia import tile_image
from plakakia.onnx import PREFERRED_PROVIDERS, OnnxModel, select_providers

CPU = "CPUExecutionProvider"


def test_select_auto_orders_by_speed() -> None:
    have = [CPU, "OpenVINOExecutionProvider", "CUDAExecutionProvider", "AzureExecutionProvider"]
    assert select_providers("auto", available=have) == [
        "CUDAExecutionProvider",
        "OpenVINOExecutionProvider",
        CPU,
    ]
    assert select_providers(None, available=[CPU]) == [CPU]
    # Providers we don't know about yet are still tried before the CPU.
    assert select_providers(available=[CPU, "NewNpuExecutionProvider"]) == [
        "NewNpuExecutionProvider",
        CPU,
    ]
    assert PREFERRED_PROVIDERS[-1] == CPU


def test_select_explicit_keeps_order_and_options() -> None:
    have = [CPU, "QNNExecutionProvider", "CUDAExecutionProvider"]
    qnn = ("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"})
    assert select_providers([qnn, "CUDAExecutionProvider"], available=have) == [
        qnn,
        "CUDAExecutionProvider",
        CPU,
    ]
    # Unavailable providers are skipped as long as one requested provider exists.
    assert select_providers(["TensorrtExecutionProvider", CPU], available=have) == [CPU]
    with pytest.raises(ValueError, match="None of the requested providers"):
        select_providers(["TensorrtExecutionProvider"], available=have)
    with pytest.raises(ValueError, match="auto"):
        select_providers("cuda", available=have)


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """A tiny NCHW float model: y = mean over channels, and the input unchanged."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    helper, tp = onnx.helper, onnx.TensorProto
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["x"], ["mean"], axes=[1], keepdims=0),
            helper.make_node("Identity", ["x"], ["same"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("x", tp.FLOAT, ["N", 3, "H", "W"])],
        [
            helper.make_tensor_value_info("mean", tp.FLOAT, ["N", "H", "W"]),
            helper.make_tensor_value_info("same", tp.FLOAT, ["N", 3, "H", "W"]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))
    return path


def test_onnx_model_runs_on_tiles(model_path: Path, rng: np.random.Generator) -> None:
    image = rng.integers(0, 255, (64, 80, 3), dtype=np.uint8)
    tiles, _ = tile_image(image, 32, 16)
    model = OnnxModel(model_path, channels_first=True, scale=1 / 255)
    assert model.providers[-1] == CPU
    assert model.input_name == "x"
    assert model.dtype == np.float32
    mean, same = model(tiles)
    expected = tiles.transpose(0, 3, 1, 2).astype(np.float32) / 255
    np.testing.assert_allclose(same, expected, rtol=1e-6)
    np.testing.assert_allclose(mean, expected.mean(axis=1), rtol=1e-5)
    assert "OnnxModel" in repr(model)


def test_onnx_model_options(model_path: Path) -> None:
    model = OnnxModel(model_path.read_bytes(), providers=[CPU], output_names=["mean"])
    (out,) = model(np.ones((2, 3, 4, 4), np.float32))
    assert out.shape == (2, 4, 4)
    with pytest.raises(ValueError, match="no input"):
        OnnxModel(model_path, input_name="nope")
    with pytest.raises(ValueError, match="channels_first"):
        OnnxModel(model_path, channels_first=True)(np.ones((3, 4, 4)))


def test_onnx_model_accepts_torch(model_path: Path) -> None:
    torch: Any = pytest.importorskip("torch")
    (out,) = OnnxModel(model_path, output_names=["same"])(torch.ones(1, 3, 2, 2))
    np.testing.assert_array_equal(out, np.ones((1, 3, 2, 2), np.float32))
