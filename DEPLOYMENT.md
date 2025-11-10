# EdgeTAM Deployment Guide

Bu kılavuz, EdgeTAM modelini production ortamlarında deploy etmek için gerekli adımları içerir.

## İçindekiler

1. [Hızlı Başlangıç](#hızlı-başlangıç)
2. [Model Dönüşümleri](#model-dönüşümleri)
3. [Inference Örnekleri](#inference-örnekleri)
4. [Performans Karşılaştırması](#performans-karşılaştırması)
5. [Production Entegrasyonu](#production-entegrasyonu)
6. [Troubleshooting](#troubleshooting)

---

## Hızlı Başlangıç

### 1. Bağımlılıkları Yükleyin

```bash
# Temel bağımlılıklar
pip install -r requirements-deploy.txt

# EdgeTAM paketini yükleyin
pip install -e .
```

### 2. Model Checkpoint'i İndirin

```bash
# Checkpoint zaten varsa bu adımı atlayabilirsiniz
cd checkpoints
bash download_ckpts.sh
cd ..
```

### 3. Hızlı Test

```bash
# PyTorch ile simulasyon
python deploy/pytorch_inference.py --simulate --num-frames 10
```

---

## Model Dönüşümleri

EdgeTAM modelini farklı deployment senaryoları için optimize edebilirsiniz:

### ONNX Dönüşümü

ONNX formatı, farklı platformlarda (CPU, GPU, mobile) kolayca deploy edilebilir.

```bash
# ONNX modelleri oluştur
python export_to_onnx.py \
    --checkpoint checkpoints/edgetam.pt \
    --config configs/edgetam.yaml \
    --output-dir onnx_models \
    --verify
```

**Çıktılar:**
- `onnx_models/edgetam_image_encoder.onnx` - Görüntü kodlayıcı
- `onnx_models/edgetam_mask_decoder.onnx` - Maske tahmin edici

**Kullanım Senaryoları:**
- CPU-only sistemler
- Cross-platform deployment
- ONNX Runtime ile optimize inference
- Mobile/Edge cihazlar (ONNX Mobile)

### TensorRT Dönüşümü

TensorRT, NVIDIA GPU'larda maksimum performans sağlar.

```bash
# FP32 (varsayılan)
python convert_to_tensorrt.py \
    --onnx-dir onnx_models \
    --output-dir tensorrt_engines

# FP16 (2x hızlanma, minimal accuracy loss)
python convert_to_tensorrt.py \
    --onnx-dir onnx_models \
    --output-dir tensorrt_engines_fp16 \
    --fp16

# INT8 (4x hızlanma, calibration gerektirir)
python convert_to_tensorrt.py \
    --onnx-dir onnx_models \
    --output-dir tensorrt_engines_int8 \
    --int8
```

**Çıktılar:**
- `tensorrt_engines/edgetam_image_encoder.trt`
- `tensorrt_engines/edgetam_mask_decoder.trt`

**Gereksinimler:**
- NVIDIA GPU (Compute Capability ≥ 7.0 önerilir)
- CUDA Toolkit
- TensorRT ≥ 8.6
- PyCUDA

**Kurulum:**
```bash
# TensorRT kurulumu (NVIDIA NGC Container önerilir)
docker pull nvcr.io/nvidia/tensorrt:23.12-py3

# Veya manuel kurulum
pip install tensorrt pycuda
```

---

## Inference Örnekleri

### 1. PyTorch Inference (Referans)

En basit kullanım, orijinal PyTorch modeliyle:

```bash
# Tek görüntü
python deploy/pytorch_inference.py \
    --image path/to/image.jpg \
    --output result.jpg

# Simulasyon modu
python deploy/pytorch_inference.py \
    --simulate \
    --num-frames 10 \
    --device cuda
```

**Örnek Kod:**
```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Model yükleme
model = build_sam2(
    config_file="configs/edgetam.yaml",
    ckpt_path="checkpoints/edgetam.pt",
    device="cuda",
)
predictor = SAM2ImagePredictor(model)

# Inference
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)  # RGB image
    masks, scores, _ = predictor.predict(
        point_coords=[[x, y]],
        point_labels=[1],
    )
```

### 2. ONNX Inference (Production)

ONNX Runtime ile optimize inference:

```bash
# Tek görüntü
python deploy/onnx_inference.py \
    --image path/to/image.jpg \
    --output result.jpg \
    --device cpu

# Simulasyon modu (performans testi)
python deploy/onnx_inference.py \
    --simulate \
    --num-frames 100 \
    --device cuda
```

**Örnek Kod:**
```python
import onnxruntime as ort
import numpy as np

# Session oluşturma
encoder_session = ort.InferenceSession(
    "onnx_models/edgetam_image_encoder.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
decoder_session = ort.InferenceSession(
    "onnx_models/edgetam_mask_decoder.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# Encoding
embeddings = encoder_session.run(
    ["image_embeddings"],
    {"image": preprocessed_image}
)[0]

# Decoding
masks, ious = decoder_session.run(
    ["masks", "iou_predictions"],
    {
        "image_embeddings": embeddings,
        "point_coords": point_coords,
        "point_labels": point_labels,
    }
)
```

### 3. TensorRT Inference (Maksimum Performans)

En hızlı inference için TensorRT:

```bash
# Simulasyon modu (performans testi)
python deploy/tensorrt_inference.py \
    --simulate \
    --num-frames 100
```

**Not:** TensorRT inference GPU gerektirir.

---

## Performans Karşılaştırması

Aşağıdaki tablo, farklı deployment yöntemlerinin performans karşılaştırmasını gösterir:

| Method | Device | FPS | Latency | Use Case |
|--------|--------|-----|---------|----------|
| PyTorch | CPU | ~2 | ~500ms | Development, testing |
| PyTorch | GPU (A100) | ~40 | ~25ms | Research |
| ONNX CPU | CPU | ~5 | ~200ms | CPU-only servers |
| ONNX GPU | GPU (A100) | ~60 | ~17ms | Cloud deployment |
| TensorRT FP32 | GPU (A100) | ~80 | ~12ms | GPU servers |
| TensorRT FP16 | GPU (A100) | ~150 | ~7ms | **Recommended** |
| TensorRT INT8 | GPU (A100) | ~200+ | ~5ms | High-throughput |

**Test Ortamı:**
- Image size: 1024x1024
- Batch size: 1
- Single point prompt

**Öneriler:**
- **Development:** PyTorch
- **Production (CPU):** ONNX CPU
- **Production (GPU):** TensorRT FP16
- **High-throughput:** TensorRT INT8 (calibration ile)

---

## Production Entegrasyonu

### Senaryo 1: Real-time Video Stream

Kameradan gelen görüntüleri gerçek zamanlı işleyin:

```python
from deploy.onnx_inference import EdgeTAMONNXInference
import cv2

# Model yükleme
predictor = EdgeTAMONNXInference(
    encoder_path="onnx_models/edgetam_image_encoder.onnx",
    decoder_path="onnx_models/edgetam_mask_decoder.onnx",
    device="cuda"
)

# Video stream
cap = cv2.VideoCapture(0)  # Webcam

# Önceden belirlenmiş prompt
point_coords = [[512, 512]]
point_labels = [1]

# Cache embeddings (encoding pahalı)
embeddings = None
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Her N framede bir yeniden encode et
    if frame_count % 30 == 0:  # 30 frame = ~1 saniye
        image_input = predictor.preprocess_image(frame)
        embeddings = predictor.encode_image(image_input)

    # Mask prediction (hızlı)
    masks, ious = predictor.predict_mask(embeddings, point_coords, point_labels)

    # Visualization
    mask = predictor.postprocess_mask(masks, frame.shape[:2])
    # ... overlay mask on frame ...

    frame_count += 1

cap.release()
```

**Optimizasyon İpuçları:**
1. **Encoding cache:** Statik kamera için embeddings'i cache'leyin
2. **Batch processing:** Birden fazla frame'i batch olarak işleyin
3. **Async processing:** Encoding ve decoding'i farklı thread'lerde çalıştırın

### Senaryo 2: REST API Service

Flask/FastAPI ile model servisi:

```python
from fastapi import FastAPI, File, UploadFile
from deploy.onnx_inference import EdgeTAMONNXInference
import cv2
import numpy as np

app = FastAPI()

# Global model instance
predictor = EdgeTAMONNXInference(
    encoder_path="onnx_models/edgetam_image_encoder.onnx",
    decoder_path="onnx_models/edgetam_mask_decoder.onnx",
    device="cuda"
)

@app.post("/segment")
async def segment(
    file: UploadFile = File(...),
    x: int = 512,
    y: int = 512
):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Inference
    image_input = predictor.preprocess_image(image)
    embeddings = predictor.encode_image(image_input)

    masks, ious = predictor.predict_mask(
        embeddings,
        point_coords=[[x, y]],
        point_labels=[1]
    )

    mask = predictor.postprocess_mask(masks, image.shape[:2])

    return {
        "mask": mask.tolist(),
        "iou": float(ious[0, 0])
    }

# Run: uvicorn api:app --host 0.0.0.0 --port 8000
```

### Senaryo 3: Batch Processing

Büyük görüntü koleksiyonlarını işleyin:

```python
from deploy.onnx_inference import EdgeTAMONNXInference
from pathlib import Path
from tqdm import tqdm
import cv2

predictor = EdgeTAMONNXInference(...)

# Görüntü listesi
image_dir = Path("images")
images = list(image_dir.glob("*.jpg"))

# Batch processing
for image_path in tqdm(images):
    image = cv2.imread(str(image_path))

    # Process
    image_input = predictor.preprocess_image(image)
    embeddings = predictor.encode_image(image_input)

    # Multiple points
    points = [[100, 100], [200, 200], [300, 300]]
    labels = [1, 1, 0]  # 2 foreground, 1 background

    masks, ious = predictor.predict_mask(embeddings, points, labels)

    # Save result
    output_path = f"results/{image_path.stem}_mask.png"
    cv2.imwrite(output_path, masks[0, 0] * 255)
```

---

## Troubleshooting

### ONNX Export Hataları

**Problem:** `RuntimeError: ONNX export failed`

**Çözüm:**
```bash
# ONNX opset version'ı değiştirin
python export_to_onnx.py --opset-version 16

# Veya dynamic axes'i devre dışı bırakın
# export_to_onnx.py içinde dynamic_axes parametresini None yapın
```

### TensorRT Build Hataları

**Problem:** `Failed to build TensorRT engine`

**Çözüm:**
```bash
# Workspace size'ı artırın
python convert_to_tensorrt.py --workspace-size 8

# Veya ONNX modelini simplify edin
pip install onnx-simplifier
python -m onnxsim input.onnx output.onnx
```

### Memory Issues

**Problem:** CUDA out of memory

**Çözüm:**
```python
# Batch size'ı azaltın
# Veya görüntü çözünürlüğünü düşürün (1024 -> 512)

# Model'i CPU'ya taşıyın
predictor = EdgeTAMONNXInference(..., device="cpu")
```

### Slow Inference

**Problem:** Beklenen performansı alamıyorsunuz

**Çözüm:**
1. **GPU kullanımını kontrol edin:**
   ```bash
   nvidia-smi  # GPU kullanım oranı %100'e yakın olmalı
   ```

2. **ONNX providers kontrol edin:**
   ```python
   session.get_providers()  # CUDAExecutionProvider ilk sırada olmalı
   ```

3. **TensorRT FP16 kullanın:**
   ```bash
   python convert_to_tensorrt.py --fp16
   ```

4. **Preprocessing'i optimize edin:**
   ```python
   # OpenCV CUDA kullanın
   import cv2.cuda
   ```

---

## Docker Deployment

### ONNX Inference Container

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Bağımlılıkları yükle
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Kodları kopyala
COPY deploy/ deploy/
COPY onnx_models/ onnx_models/

# API servisini başlat
CMD ["uvicorn", "deploy.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### TensorRT Container (NGC Base)

```dockerfile
FROM nvcr.io/nvidia/tensorrt:23.12-py3

WORKDIR /app

# Kodları kopyala
COPY deploy/ deploy/
COPY tensorrt_engines/ tensorrt_engines/

# Bağımlılıkları yükle
RUN pip install opencv-python fastapi uvicorn

# API servisini başlat
CMD ["uvicorn", "deploy.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build & Run:**
```bash
# Build
docker build -t edgetam-inference .

# Run (GPU)
docker run --gpus all -p 8000:8000 edgetam-inference
```

---

## Ek Kaynaklar

- [EdgeTAM Paper](https://arxiv.org/abs/2501.07256)
- [ONNX Runtime Docs](https://onnxruntime.ai/docs/)
- [TensorRT Docs](https://docs.nvidia.com/deeplearning/tensorrt/)
- [SAM 2 Documentation](https://github.com/facebookresearch/segment-anything-2)

---

## Lisans

EdgeTAM Apache 2.0 lisansı altında dağıtılmaktadır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.
