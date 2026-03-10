# SingleHDR Service - Installation Guide

## Table of contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Native installation (recommended)](#native-installation)
- [Docker installation](#docker-installation)
- [Vast.ai deployment](#vastai-deployment)
- [Environment variables](#environment-variables)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
┌─────────────────────────┐
│  FastAPI (port 8000)    │
│  - Static files         │
│  - REST API             │
│  - SSE (progress)       │
│  - GPU/MPS inference    │
│  - In-memory storage    │
└─────────────────────────┘
```

The backend uses **PyTorch** with MPS (Apple Silicon), CUDA (NVIDIA), or CPU acceleration.

Data files (PCA basis for CRF estimation) are read from the `vendor/SingleHDR/` submodule at runtime.

Results are kept in memory (no disk writes) and automatically purged after `JOB_TTL_HOURS`.

---

## Prerequisites

### Clone with submodule

```bash
git clone --recurse-submodules <repo-url>
cd singlehdr-service

# Or if already cloned
git submodule update --init
```

### Model checkpoints (native installation only)

> **Docker / Vast.ai**: weights are downloaded and converted automatically. Skip to [Docker installation](#docker-installation) or [Vast.ai deployment](#vastai-deployment).

For native installation, download the pre-trained weights and place them in `ckpt/` at the repo root:

```
singlehdr-service/
  ckpt/
    ckpt_deq/
      model.ckpt.data-00000-of-00001
      model.ckpt.index
      model.ckpt.meta
    ckpt_lin/
      model.ckpt.*
    ckpt_hal/
      model.ckpt.*
    ckpt_deq_lin_hal_ref/
      model.ckpt.*
```

### TF to PyTorch weight conversion

TensorFlow checkpoints must be converted to PyTorch format (`.pt`). This is a **one-time** operation and requires TensorFlow.

```bash
cd service/backend

pip install tensorflow numpy torch

python convert_weights.py --mode both \
  --ckpt_deq ../../ckpt/ckpt_deq/model.ckpt \
  --ckpt_lin ../../ckpt/ckpt_lin/model.ckpt \
  --ckpt_hal ../../ckpt/ckpt_hal/model.ckpt \
  --ckpt_ref ../../ckpt/ckpt_deq_lin_hal_ref/model.ckpt \
  --output_dir ../weights
```

Expected output:

```
service/weights/
  basic/
    dequantization.pt
    linearization.pt
    hallucination.pt
  refinement/
    dequantization.pt
    linearization.pt
    hallucination.pt
    refinement.pt
```

---

## Native installation

> Recommended for Mac Apple Silicon (MPS acceleration).

### 1. Create a Python environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Or with conda:

```bash
conda create -n singlehdr python=3.10
conda activate singlehdr
```

### 2. Install dependencies

```bash
cd service/backend
pip install -r requirements-torch.txt
```

> **Note**: `OpenEXR` may require installing `libopenexr` first:
> ```bash
> brew install openexr
> ```

### 3. Run

```bash
cd service
./run_mac.sh
```

Or manually:

```bash
cd service/backend

export PYTORCH_WEIGHTS_DIR=../weights
export MAX_MEGAPIXELS=50

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The web UI is available at `http://localhost:8000` (FastAPI serves static files directly).

---

## Docker installation

Single command to deploy everything (weight download included):

```bash
cd service
docker compose up -d --build
```

On first build, the image downloads checkpoints (~800 MB) from Google Drive, converts them to PyTorch, and builds the runtime image. Expect ~10 min depending on connection speed.

Subsequent builds use Docker cache and are near-instant.

### Verify

```bash
curl http://localhost/api/health
# {"status":"ok","basic_pipeline":true,"refinement_pipeline":true}
```

The web UI is available at `http://localhost` (port 80, mapped to 8000 inside the container).

### Logs

```bash
docker compose logs -f singlehdr
```

### Stop / Update

```bash
docker compose down
docker compose up -d --build
```

> **Mac note**: Docker Desktop does not support MPS passthrough. Inference will run on CPU, which is significantly slower than native.

---

## Vast.ai deployment

Deploy on a [Vast.ai](https://vast.ai) GPU instance. The deploy script downloads weights and starts the service automatically.

### 1. Create the instance

- Pick a GPU with >= 8 GB VRAM (RTX 3080+ recommended)
- Use a **PyTorch** template image (e.g. `pytorch/pytorch:2.x-cuda12.x-runtime`)
- In **Docker options**, add: `-p 8000:8000`
- Disk space: **10 GB minimum**

> **Important**: port 8000 must be declared when creating the instance. Vast.ai maps it to a random external port on the shared public IP.

### 2. Deploy

```bash
# SSH into the instance (see the "Connect" button on Vast.ai)
ssh -p <PORT> root@<IP>

git clone --recurse-submodules <repo-url>
cd singlehdr-service/service
./deploy_vastai.sh
```

The script will:
1. Install Python dependencies
2. Download checkpoints (~800 MB) and convert them to PyTorch
3. Start uvicorn on port 8000

### 3. Access the service

| Method | How |
|--------|-----|
| **Cloudflare tunnel** | Click the **"Open"** button on the instance card. Auto-generated HTTPS URL (e.g. `https://four-random-words.trycloudflare.com`). |
| **Direct IP** | Open the **"IP Port Info"** popup on the instance card. Use `http://<PUBLIC_IP>:<EXTERNAL_PORT>`. |

> **Note**: the Cloudflare tunnel enables authentication by default. For unauthenticated API access, use the direct IP method.

### 4. Restart after stop

Converted weights are kept in `service/weights/`. Subsequent runs skip the download:

```bash
cd singlehdr-service/service
./deploy_vastai.sh
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTORCH_WEIGHTS_DIR` | `/app/weights` | Directory containing PyTorch weights (.pt) |
| `SINGLEHDR_VENDOR_DIR` | *(auto)* | Path to the SingleHDR submodule (auto-detected, override for Docker) |
| `JOB_TTL_HOURS` | `24` | How long completed results are kept in memory (hours) |
| `MAX_MEGAPIXELS` | `50` | Maximum resolution accepted at upload (megapixels) |
| `MAX_INFERENCE_PIXELS` | `8000000` | GPU inference pixel limit (auto-downscale if exceeded) |

---

## API Reference

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/upload` | Upload SDR image (multipart) |
| `POST` | `/api/generate/{job_id}` | Start HDR generation (queued) |
| `POST` | `/api/cancel/{job_id}` | Cancel a queued or running job |
| `GET` | `/api/status/{job_id}` | SSE progress stream (includes queue position) |
| `GET` | `/api/result/{job_id}` | Result metadata |
| `GET` | `/api/hdr-raw/{job_id}` | Raw HDR data (float32) for client-side tone mapping |
| `GET` | `/api/download/{job_id}` | Download EXR file |

### Example with curl

```bash
# Upload
curl -s -F "file=@photo.jpg" http://localhost:8000/api/upload
# -> {"job_id":"a1b2c3d4e5f6", "width":1920, "height":1080, ...}

# Generate (refinement mode)
curl -s -X POST http://localhost:8000/api/generate/a1b2c3d4e5f6 \
  -H "Content-Type: application/json" \
  -d '{"mode":"refinement"}'

# Progress (SSE)
curl -N http://localhost:8000/api/status/a1b2c3d4e5f6

# Download result
curl -o photo_hdr.exr http://localhost:8000/api/download/a1b2c3d4e5f6
```

### Request body for `/api/generate`

```json
{
  "mode": "basic" | "refinement",
  "hallucination_threshold": 0.12
}
```

- **basic**: 3 networks (Dequantization + Linearization + Hallucination). Faster.
- **refinement**: 4 networks (+ Refinement-Net). Better quality.

---

## Troubleshooting

### PyTorch weights not found

```
Could not load basic pipeline: ...
```

Check that `weights/` contains the `.pt` files. Re-run `convert_weights.py` if needed.

### "Image too large" error

The API rejects images above `MAX_MEGAPIXELS`. Resize the image or increase the variable.

### OpenEXR won't install on Mac

```bash
brew install openexr
pip install OpenEXR
```

If that still fails:

```bash
CFLAGS="-I$(brew --prefix openexr)/include/OpenEXR" \
LDFLAGS="-L$(brew --prefix openexr)/lib" \
pip install OpenEXR
```

### MPS not detected

```python
import torch
print(torch.backends.mps.is_available())  # Should print True
```

If `False`, update PyTorch (`pip install --upgrade torch`). MPS requires macOS 12.3+ and PyTorch 2.0+.

### Performance

| Platform | 1080p | 4K |
|----------|-------|----|
| NVIDIA GPU (RTX 3080) | ~3s | ~12s |
| Apple M2 Pro (MPS) | ~8s | ~30s |
| Apple M2 Pro (CPU) | ~25s | ~90s |

*Approximate times, refinement mode. Basic mode is ~20% faster.*
