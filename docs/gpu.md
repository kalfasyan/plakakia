# GPU & NPU acceleration

When your model runs on a GPU, the image is usually on the GPU too. Copying it back to the
CPU to cut tiles, and then copying every tile to the GPU again, wastes the time you bought
the GPU for. plakakia tiles **PyTorch tensors and CuPy arrays where they are**: hand it a
tensor on `cuda`, `mps` or `cpu`, and you get tiles on that same device.

Nothing changes in the API. The same `tile_image`, `tile_array` and `iter_tiles` accept
NumPy, PyTorch and CuPy arrays, and plakakia doesn't depend on PyTorch or CuPy: it only uses
them when you pass it one of their arrays.

## Tiling on the GPU

```python
import torch
from torchvision.io import read_image

from plakakia import tile_image

image = read_image("orthomosaic.png").to("cuda")  # (C, H, W) uint8, on the GPU
tiles, coords = tile_image(image, tile_size=640, step_size=512, axes=(1, 2), cover_edges=True)

tiles.device  # device(type='cuda', index=0)
tiles.shape  # (T, 3, 640, 640): ready to batch into your model
coords[:2]  # NumPy (x1, y1, x2, y2) per tile, as always
```

How it works: plakakia builds a **zero-copy sliding-window view** of the tensor
(`Tensor.unfold` in PyTorch, strides in CuPy) and copies the selected windows into one
contiguous batch with a single gather on the device. The image itself is never copied, and
only the tiles you asked for are written.

`iter_tiles` yields views of the tensor, without copying anything, which is handy when a
tile batch would not fit in GPU memory:

```python
from plakakia import iter_tiles

for (y, x), view in iter_tiles(image, (640, 640), (512, 512), axes=(1, 2)):
    detections = model(view[None].float() / 255)  # one tile at a time
```

!!! tip "Channels first or last"
    Tensors from `torchvision` are `(C, H, W)`: pass `axes=(1, 2)`. An `(H, W, C)` tensor
    works with the default `axes=(0, 1)`, and yields `(T, H, W, C)` tiles.

## Merging boxes on the GPU

After prediction, overlapping tiles detect the same object more than once. `nms` removes
the duplicates and runs where the boxes are: through `torchvision.ops.batched_nms` for
PyTorch tensors (on any device), and in NumPy otherwise.

```python
from plakakia import nms

# boxes (N, 4) x1 y1 x2 y2 on the full image, scores (N,), classes (N,): all on cuda
keep = nms(boxes, scores, iou_threshold=0.5, class_ids=classes)
boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
```

With `class_ids`, a box only suppresses boxes of its own class. Without torchvision,
PyTorch tensors take the NumPy path and the indices come back as a tensor on the original
device.

## NPUs and other accelerators with ONNX Runtime

Neural processing units (NPUs) in laptops and edge devices, Apple's Neural Engine, Intel,
AMD and Qualcomm accelerators: ONNX Runtime reaches them all through *execution
providers*. `plakakia.onnx.OnnxModel` loads an exported `.onnx` model and picks the fastest
provider your ONNX Runtime build offers.

```bash
pip install "plakakia[onnx]"    # CPU build; or install the one for your hardware:
pip install onnxruntime-gpu     # NVIDIA: CUDA / TensorRT
pip install onnxruntime-openvino   # Intel CPU, GPU and NPU
pip install onnxruntime-qnn     # Qualcomm Snapdragon NPUs
pip install onnxruntime-directml   # Windows GPUs and NPUs
```

```python
import cv2

from plakakia import tile_image
from plakakia.onnx import OnnxModel

model = OnnxModel("detector.onnx", channels_first=True, scale=1 / 255)
print(model.providers)  # e.g. ['CUDAExecutionProvider', 'CPUExecutionProvider']

image = cv2.imread("trap_042.jpg")  # (H, W, 3)
tiles, coords = tile_image(image, 640, 480, cover_edges=True)
outputs = model(tiles[:16])  # (16, 3, 640, 640) float32 in
```

`OnnxModel(...)` is a plain callable, `model(batch) -> list of outputs`. It converts the
batch (NumPy, PyTorch or CuPy) to NumPy, optionally transposes `(N, H, W, C)` to
`(N, C, H, W)` (`channels_first`), scales it, and casts it to the model's input type.
Decoding the outputs into boxes depends on the model, so that part stays yours.

With `providers="auto"` (the default) the order is:

| Hardware | Execution provider |
|---|---|
| NVIDIA GPU | `TensorrtExecutionProvider`, `CUDAExecutionProvider` |
| AMD GPU | `MIGraphXExecutionProvider`, `ROCMExecutionProvider` |
| Apple silicon (GPU, Neural Engine) | `CoreMLExecutionProvider` |
| Windows GPU / NPU | `DmlExecutionProvider` |
| Intel CPU / GPU / NPU | `OpenVINOExecutionProvider` |
| Qualcomm NPU | `QNNExecutionProvider` |
| AMD Ryzen AI NPU | `VitisAIExecutionProvider` |
| Huawei Ascend | `CANNExecutionProvider` |
| Anything | `CPUExecutionProvider` (always last, as a fallback) |

To choose a device yourself, pass a list, with provider options if needed:

```python
model = OnnxModel(
    "detector.onnx",
    providers=[("OpenVINOExecutionProvider", {"device_type": "NPU"})],
)
```

## How much faster?

`benchmarks/bench_device.py` times tiling and NMS on NumPy against PyTorch on every device it
finds:

```bash
uv run python benchmarks/bench_device.py --size 4096 --tile 640 --step 512
```

A run on a 4-core cloud CPU, without a GPU (4096 × 4096 RGB image, 64 tiles of 640 px,
NMS on 5,000 boxes in 10 classes):

| Operation | NumPy | PyTorch (CPU) |
|---|---|---|
| `tile_image` | 25 ms | 27 ms |
| `nms` | 102 ms | 5 ms (20× faster) |

On the CPU, tiling is one memory copy either way. On a GPU the gain is mostly the copies
you no longer make: the image never leaves the device, and the tiles are ready to batch
into the model. Run the benchmark on your own hardware to see your numbers.
