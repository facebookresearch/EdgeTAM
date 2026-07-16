# EdgeTAM Core ML export

EdgeTAM provides separate Core ML pipelines for prompted image segmentation
and temporal video tracking. Install the optional dependencies before using
either exporter:

```bash
pip install -e ".[coreml]"
```

## Image segmentation

Export the image encoder, prompt encoder, and mask decoder:

```bash
python coreml/export_to_coreml.py \
  --sam2_cfg sam2/configs/edgetam.yaml \
  --sam2_checkpoint checkpoints/edgetam.pt \
  --output_dir coreml_models
```

See `inference_example.py` for image prompting and `benchmark_coreml.py` for a
small synthetic benchmark.

## Temporal video tracking

The video export preserves EdgeTAM's temporal memory pipeline instead of
running image segmentation independently on every frame. It produces four
stateless Core ML packages and maintains the fixed-shape memory bank in the
client.

```bash
python coreml/video_tracking/export_models.py \
  --config sam2/configs/edgetam.yaml \
  --checkpoint checkpoints/edgetam.pt \
  --output-dir coreml_models/video_tracking
```

See [video_tracking/README.md](video_tracking/README.md) for the model
architecture, Python predictor, validation command, tests, and integration
constraints.

Generated `.mlpackage` directories belong in `coreml_models/`, which is
excluded from version control.
