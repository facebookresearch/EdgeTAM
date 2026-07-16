# Core ML video tracking

This directory adds temporal video tracking to EdgeTAM's Core ML export. A
point or box prompt initializes a track on the first frame. Later frames are
processed without repeating the prompt, using the spatial memories and object
pointers produced by earlier frames.

The exported Core ML models are stateless. The client owns the small,
fixed-shape memory bank, so the same models can be used from Python, Swift, or
another Core ML host.

## Model pipeline

The exporter creates four iOS 18 ML Program packages:

| Model | Responsibility |
| --- | --- |
| `EdgeTAMVideoImageEncoder` | Produces raw, initial, and high-resolution features for each frame. |
| `EdgeTAMVideoInitializer` | Applies the first-frame point or box prompt and returns the seed mask and object pointer. |
| `EdgeTAMVideoMemoryEncoder` | Encodes the prompted mask with EdgeTAM's 2D Spatial Perceiver. |
| `EdgeTAMVideoPropagator` | Conditions the current frame on the explicit memory bank and returns the next mask, pointer, and memory. |

The runtime keeps one conditioning memory, six recent memories, and sixteen
object pointers per tracked object. Validity tensors mask unused slots while
the bank fills.

## Requirements

Install EdgeTAM with its Core ML dependencies:

```bash
pip install -e ".[coreml]"
```

The export targets iOS 18 and requires an EdgeTAM checkpoint. Generated model
packages are build artifacts and are not stored in the repository.

## Export

Run the exporter from the repository root:

```bash
python coreml/video_tracking/export_models.py \
  --config sam2/configs/edgetam.yaml \
  --checkpoint checkpoints/edgetam.pt \
  --output-dir coreml_models/video_tracking
```

`--device` selects the PyTorch device used while tracing. It defaults to
`cpu`; `mps` is also useful on Apple silicon.

## Python inference

`CoreMLVideoPredictor` owns the explicit memory bank for one object:

```bash
PYTHONPATH=coreml/video_tracking python
```

```python
from pathlib import Path

from PIL import Image

from edgetam_coreml_video.predictor import CoreMLVideoPredictor

predictor = CoreMLVideoPredictor.from_directory(
    Path("coreml_models/video_tracking")
)

first_frame = Image.open("frames/00000.jpg")
result = predictor.start_track(
    first_frame,
    points=[[210, 350]],
    labels=[1],
)

next_frame = Image.open("frames/00001.jpg")
result = predictor.track_frame(next_frame)
binary_mask = result.mask
```

Prompt coordinates use the original frame's pixel coordinate system. Point
labels follow EdgeTAM conventions: `1` for a foreground point, `0` for a
background point, and `2`/`3` for the two corners of a box. One to four prompt
tokens are supported. Call `reset()` before starting a different object.

## Numerical validation

The validator runs the official PyTorch video predictor and the Core ML
pipeline on the same ordered JPEG frames. Filenames must have numeric stems,
such as `00000.jpg` and `00001.jpg`.

```bash
PYTHONPATH=coreml/video_tracking \
python coreml/video_tracking/validate_video.py \
  --frames-dir notebooks/videos/bedroom \
  --models-dir coreml_models/video_tracking \
  --checkpoint checkpoints/edgetam.pt \
  --device mps \
  --point 210 350 1 \
  --max-frames 8 \
  --json coreml/video_tracking/validation.json
```

For every frame, the command reports binary-mask IoU, logit cosine similarity,
mean absolute error, and maximum absolute error.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=coreml/video_tracking \
python -m pytest coreml/video_tracking/tests -q
```

The tests cover the fixed model contracts, prompt scaling, explicit memory-bank
updates, masked attention, sequential prediction, and validation metrics.

## Current scope

- Single-object, forward-only tracking.
- One to four prompt tokens on the first frame.
- Fixed 1024-by-1024 model input.
- Fixed temporal memory capacity: one conditioning frame, six recent frames,
  and sixteen pointers.
- No prompt correction after initialization, reverse propagation,
  quantization, or bundled Swift wrapper.
