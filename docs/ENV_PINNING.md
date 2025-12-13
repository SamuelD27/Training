# Environment Pinning

All versions are pinned to ensure reproducibility per Section 10 of the report.

---

## Base Docker Image

```
nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
```

**Rationale**: CUDA 12.x is stable and widely supported on RunPod GPUs (A6000, H100, B200). The devel image includes nvcc for any compilation needs.

---

## Core Versions

| Component | Pinned Version | Report Requirement |
|-----------|----------------|-------------------|
| Python | `3.10.14` | 3.10 |
| PyTorch | `2.5.1+cu121` | 2.5.x |
| CUDA | `12.1` | 11.8 / 12.x |
| cuDNN | `8.9.x` | (implicit) |

---

## Python Packages (Pinned)

### Training Core
```
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
accelerate==0.25.0
bitsandbytes==0.44.1
transformers==4.46.3
safetensors==0.4.5
diffusers==0.31.0
```

### sd-scripts Dependencies
```
lion-pytorch==0.2.3
prodigyopt==1.0
schedulefree==1.2.6
dadaptation==3.2
pytorch-lightning==2.4.0
tensorboard==2.18.0
```

### Dataset Analysis
```
Pillow==10.4.0
imagehash==4.3.1
opencv-python-headless==4.10.0.84
numpy==1.26.4
```

### Utilities
```
pyyaml==6.0.2
toml==0.10.2
tqdm==4.66.5
huggingface-hub==0.26.2
sentencepiece==0.2.0
ftfy==6.3.1
```

---

## sd-scripts Pinning

### Repository
```
https://github.com/kohya-ss/sd-scripts
```

### Pinned Commit
```
SDSCRIPTS_COMMIT=v0.9.2
```

**Note**: This is the stable FLUX-compatible release. If HEAD breaks, the Dockerfile will checkout this specific tag.

### Clone Command
```bash
git clone https://github.com/kohya-ss/sd-scripts.git third_party/sd-scripts
cd third_party/sd-scripts
git checkout v0.9.2
```

---

## Runtime Outputs

### pip freeze Location
```
/workspace/lora_training/logs/pip_freeze.txt
```

Generated automatically by `docker/start.sh` on container boot.

### Verification Commands
```bash
# Check versions
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
python -c "import bitsandbytes; print(f'BNB: {bitsandbytes.__version__}')"

# Check sd-scripts commit
cd third_party/sd-scripts && git rev-parse HEAD
```

---

## Deviation Log

Any deviations from report recommendations are documented here:

| Component | Report Says | We Use | Reason |
|-----------|-------------|--------|--------|
| CUDA | 11.8 / 12.x | 12.1 | Better H100/B200 support, PyTorch 2.5 native |
| Accelerate | >=0.21 | 0.25.0 | Latest stable with FLUX improvements |
| bitsandbytes | >=0.42 | 0.44.1 | Latest stable with better FP8 support |

---

## Update Policy

To update pinned versions:

1. Test new versions locally first
2. Update this file
3. Rebuild Docker image
4. Run validation suite
5. Record new `pip_freeze.txt`
