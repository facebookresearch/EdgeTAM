# EdgeTAM Hızlı Başlangıç Kılavuzu

Bu kılavuz EdgeTAM modelini ONNX formatına export etmek ve çalıştırmak için gereken adımları içerir.

## Gereksinimler

```bash
# Gerekli paketleri yükleyin
pip install torch torchvision onnx onnxruntime opencv-python numpy hydra-core omegaconf timm
```

## Adım 1: ONNX Modellerini Oluşturun

```bash
# EdgeTAM modelini ONNX formatına export edin
python export_to_onnx.py

# Veya doğrulama ile birlikte:
python export_to_onnx.py --verify
```

Bu komut şu dosyaları oluşturacak:
- `onnx_models/edgetam_image_encoder.onnx` (Image Encoder)
- `onnx_models/edgetam_mask_decoder.onnx` (Mask Decoder)

## Adım 2: ONNX İnferens Testini Çalıştırın

### Simülasyon Modu (Sentetik görüntülerle test)

```bash
python deploy/onnx_inference.py --simulate --num-frames 10
```

Bu komut:
- 10 adet sentetik görüntü üzerinde inference yapar
- FPS ve performans istatistiklerini gösterir
- High-resolution features destekleniyorsa bunu otomatik algılar

### Gerçek Görüntü ile Test

```bash
python deploy/onnx_inference.py --image path/to/your/image.jpg --output result.jpg
```

## Performans Karşılaştırması

3 farklı inference modu mevcuttur:

### 1. PyTorch (Referans)
```bash
python deploy/pytorch_inference.py --checkpoint checkpoints/edgetam.pt --simulate --num-frames 10
```

### 2. ONNX (Üretim için önerilen)
```bash
python deploy/onnx_inference.py --simulate --num-frames 10 --device cpu
```

### 3. TensorRT (En yüksek performans - NVIDIA GPU gerekli)
```bash
# Önce TensorRT enginelerini oluşturun
python convert_to_tensorrt.py --onnx-dir onnx_models --output-dir tensorrt_engines

# Inference çalıştırın
python deploy/tensorrt_inference.py --simulate --num-frames 10
```

## Sorun Giderme

### "No module named 'torch'" hatası
```bash
pip install torch torchvision
```

### "No module named 'onnxruntime'" hatası
```bash
pip install onnxruntime
```

### "Config file not found" hatası
Config dosyasının doğru konumda olduğundan emin olun:
```bash
ls sam2/configs/edgetam.yaml
```

### ONNX modelleri bulunamadı
Önce export komutunu çalıştırın:
```bash
python export_to_onnx.py
```

## Detaylı Dokümantasyon

Daha fazla bilgi için `DEPLOYMENT.md` dosyasına bakın.
