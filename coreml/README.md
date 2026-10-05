# EdgeTAM CoreML Export

Export EdgeTAM to CoreML format for iOS/macOS deployment.

## Quick Export

```bash
python coreml/export_to_coreml.py \
  --sam2_cfg sam2/configs/edgetam.yaml \
  --sam2_checkpoint checkpoints/edgetam.pt
```

This creates three CoreML models in `./coreml_models/`:
- `edgetam_image_encoder.mlpackage` (9.6MB)
- `edgetam_prompt_encoder.mlpackage` (2.0MB)
- `edgetam_mask_decoder.mlpackage` (9.8MB)

To export at a lower input resolution, pass a smaller config such as `--sam2_cfg sam2/configs/edgetam_512.yaml`; the size is saved as `image_size` in `model_info.json`. `inference_example.py` and `benchmark_coreml.py` still assume a 1024 export.

## Usage Example

```python
import coremltools as ct
from PIL import Image

# Load models
image_encoder = ct.models.MLModel("coreml_models/edgetam_image_encoder.mlpackage")
prompt_encoder = ct.models.MLModel("coreml_models/edgetam_prompt_encoder.mlpackage")
mask_decoder = ct.models.MLModel("coreml_models/edgetam_mask_decoder.mlpackage")

# Segment with point prompt
image = Image.open("image.jpg").resize((1024, 1024))
encoder_out = image_encoder.predict({"image": image})

# Add your point and generate mask
# See inference_example.py for complete video tracking example
```

## Video Tracking Example

The included `inference_example.py` demonstrates real-time video tracking:

```bash
# Demo with default coffee video
python coreml/inference_example.py

# Use your own video
python coreml/inference_example.py --video path/to/your/video.mp4

# Run different examples
python coreml/inference_example.py --example segment  # Single image
python coreml/inference_example.py --example track   # Real-time tracking
python coreml/inference_example.py --example demo    # Video demo (default)
```

## Performance Benchmark

Run benchmark with: `python coreml/benchmark_coreml.py`

Note: This is a limited test on synthetic data. Real-world performance may vary.

### Results

| Metric | PyTorch | CoreML | Difference |
|--------|---------|--------|------------|
| Speed | 40.1ms | 39.2ms | -0.9ms |
| Quality | IoU 0.9897 | IoU 0.9893 | -0.0004 |
| Size | 54MB | 21.4MB | -32.6MB |

## Requirements

- PyTorch
- coremltools
- EdgeTAM checkpoint

The CoreML export maintains identical segmentation quality while being faster and 60% smaller for mobile deployment.