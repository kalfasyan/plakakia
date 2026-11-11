"""Run ONNX models on tiles, on GPUs and NPUs, through ONNX Runtime.

:class:`OnnxModel` wraps an :class:`onnxruntime.InferenceSession` as a plain callable
``model(batch) -> list of outputs``, so it can be fed batches of tiles straight from
:func:`plakakia.tile_image`. With ``providers="auto"`` it picks the fastest execution
provider your ONNX Runtime build offers: TensorRT or CUDA on NVIDIA GPUs, ROCm/MIGraphX
on AMD GPUs, CoreML on Apple silicon (GPU and Neural Engine), DirectML on Windows GPUs
and NPUs, OpenVINO on Intel CPUs, GPUs and NPUs, QNN on Qualcomm NPUs, Vitis AI on
AMD Ryzen AI NPUs, CANN on Huawei Ascend, and the CPU otherwise.

ONNX Runtime is not a dependency of plakakia: install the build for your hardware, e.g.
``pip install onnxruntime-gpu``, ``onnxruntime-openvino``, ``onnxruntime-qnn`` or
``onnxruntime-directml`` (``pip install "plakakia[onnx]"`` installs the CPU build).
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from os import PathLike
from typing import Any, Literal

import numpy as np

from .backend import to_numpy

__all__ = ["PREFERRED_PROVIDERS", "OnnxModel", "available_providers", "select_providers"]

PREFERRED_PROVIDERS: tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "MIGraphXExecutionProvider",
    "ROCMExecutionProvider",
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "QNNExecutionProvider",
    "VitisAIExecutionProvider",
    "CANNExecutionProvider",
    "CPUExecutionProvider",
)
"""Execution providers tried by ``providers="auto"``, fastest first."""

ProviderSpec = str | tuple[str, dict[str, Any]]
"""A provider name, or a ``(name, options)`` pair."""

_CPU = "CPUExecutionProvider"
_NOT_AUTO = {_CPU, "AzureExecutionProvider"}


def _onnxruntime() -> Any:
    try:
        return importlib.import_module("onnxruntime")
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "OnnxModel needs ONNX Runtime. Install the build for your hardware, e.g. "
            "`pip install onnxruntime` (CPU), `onnxruntime-gpu` (NVIDIA), "
            "`onnxruntime-openvino` (Intel), `onnxruntime-qnn` (Qualcomm) or "
            "`onnxruntime-directml` (Windows)."
        ) from err


def available_providers() -> list[str]:
    """Execution providers offered by the installed ONNX Runtime build."""
    return list(_onnxruntime().get_available_providers())


def select_providers(
    providers: Literal["auto"] | Sequence[ProviderSpec] | None = "auto",
    *,
    available: Sequence[str] | None = None,
) -> list[ProviderSpec]:
    """Resolve a provider request into the list ONNX Runtime expects.

    Args:
        providers: ``"auto"`` or ``None`` for every available provider in
            :data:`PREFERRED_PROVIDERS` order, or an explicit list of names (or
            ``(name, options)`` pairs), kept in your order and filtered to those available.
        available: Providers to choose from. Defaults to :func:`available_providers`.

    Returns:
        The providers to pass to :class:`onnxruntime.InferenceSession`, always ending with
        the CPU provider so that unsupported operators can fall back to it.

    >>> select_providers("auto", available=["CPUExecutionProvider", "CUDAExecutionProvider"])
    ['CUDAExecutionProvider', 'CPUExecutionProvider']
    """
    have = list(available_providers() if available is None else available)
    if providers is None or providers == "auto":
        # Known accelerators in speed order, then any provider we don't know yet.
        # Azure (remote inference) is never picked automatically.
        chosen: list[ProviderSpec] = [p for p in PREFERRED_PROVIDERS if p in have]
        chosen += [p for p in have if p not in PREFERRED_PROVIDERS and p not in _NOT_AUTO]
    else:
        if isinstance(providers, str):
            raise ValueError(f'providers must be "auto" or a list of names, got {providers!r}.')
        chosen = []
        missing: list[ProviderSpec] = []
        for spec in providers:
            name = spec if isinstance(spec, str) else spec[0]
            (chosen if name in have else missing).append(spec)
        if missing and not chosen:
            raise ValueError(
                f"None of the requested providers {list(providers)!r} is available; "
                f"this ONNX Runtime build offers {have!r}."
            )
    # The CPU provider goes last, so operators an accelerator lacks can still run.
    chosen = [p for p in chosen if (p if isinstance(p, str) else p[0]) != _CPU]
    return [*chosen, _CPU]


class OnnxModel:
    """An ONNX model as a callable that runs a batch of tiles on the best device.

    Args:
        model: Path to a ``.onnx`` file, or the serialized model as bytes.
        providers: ``"auto"`` (default) to use the fastest available accelerator, or an
            explicit list such as ``["OpenVINOExecutionProvider"]`` or
            ``[("OpenVINOExecutionProvider", {"device_type": "NPU"})]``. See
            :func:`select_providers`.
        input_name: Model input to feed. Defaults to the model's first input.
        output_names: Outputs to return. Defaults to all of them.
        channels_first: Transpose ``(N, H, W, C)`` batches (the layout of
            :func:`plakakia.tile_image` on an ``(H, W, C)`` image) to ``(N, C, H, W)``
            before running the model.
        scale: Multiply the batch by this factor, e.g. ``1 / 255`` for 8-bit images.
        dtype: Cast the batch to this dtype. Defaults to the model input's type.
        session_options: An :class:`onnxruntime.SessionOptions` to create the session with.

    Example:
        >>> from plakakia import tile_image
        >>> from plakakia.onnx import OnnxModel
        >>> model = OnnxModel("detector.onnx", channels_first=True, scale=1 / 255)  # doctest: +SKIP
        >>> tiles, coords = tile_image(image, 640, 512)  # doctest: +SKIP
        >>> outputs = model(tiles[:16])  # doctest: +SKIP
        >>> model.providers  # doctest: +SKIP
        ['CUDAExecutionProvider', 'CPUExecutionProvider']
    """

    def __init__(
        self,
        model: str | PathLike[str] | bytes,
        *,
        providers: Literal["auto"] | Sequence[ProviderSpec] | None = "auto",
        input_name: str | None = None,
        output_names: Sequence[str] | None = None,
        channels_first: bool = False,
        scale: float | None = None,
        dtype: np.dtype[Any] | type | str | None = None,
        session_options: Any = None,
    ) -> None:
        ort = _onnxruntime()
        source = model if isinstance(model, bytes) else str(model)
        self.session = ort.InferenceSession(
            source, sess_options=session_options, providers=select_providers(providers)
        )
        inputs = self.session.get_inputs()
        names = [i.name for i in inputs]
        if input_name is not None and input_name not in names:
            raise ValueError(f"Model has no input {input_name!r}; its inputs are {names!r}.")
        self.input_name: str = names[0] if input_name is None else input_name
        self.output_names: list[str] | None = None if output_names is None else list(output_names)
        self.channels_first = channels_first
        self.scale = scale
        input_type = next(i.type for i in inputs if i.name == self.input_name)
        self.dtype = np.dtype(dtype) if dtype is not None else _numpy_dtype(input_type)

    @property
    def providers(self) -> list[str]:
        """The execution providers the session actually runs on, in priority order."""
        return list(self.session.get_providers())

    def prepare(self, batch: Any) -> np.ndarray:
        """Convert a batch of tiles (NumPy, PyTorch or CuPy) into the model's input."""
        x = to_numpy(batch)
        if self.channels_first:
            if x.ndim != 4:
                raise ValueError(f"channels_first needs (N, H, W, C) batches, got {x.shape}.")
            x = x.transpose(0, 3, 1, 2)
        if self.scale is not None:
            x = x * np.float32(self.scale) if self.dtype.kind == "f" else x * self.scale
        return np.ascontiguousarray(x, dtype=self.dtype)

    def __call__(self, batch: Any) -> list[np.ndarray]:
        """Run the model on ``batch`` and return its outputs as NumPy arrays."""
        outputs = self.session.run(self.output_names, {self.input_name: self.prepare(batch)})
        return [np.asarray(o) for o in outputs]

    def __repr__(self) -> str:
        return f"OnnxModel(input={self.input_name!r}, providers={self.providers!r})"


_ONNX_TYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
    "tensor(uint16)": np.uint16,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(bool)": np.bool_,
}


def _numpy_dtype(onnx_type: str) -> np.dtype[Any]:
    return np.dtype(_ONNX_TYPES.get(onnx_type, np.float32))
