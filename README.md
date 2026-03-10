# SingleHDR Service

Web service for SDR-to-HDR image conversion using the [SingleHDR](https://github.com/alex04072000/SingleHDR) neural network.

## Structure

- `service/` — Web application (FastAPI backend + static frontend)
- `vendor/SingleHDR/` — Original SingleHDR repository (git submodule, used at runtime for data files)

## Quick Start (Docker)

```bash
git clone --recurse-submodules <repo-url>
cd singlehdr-service/service
docker compose up -d --build
```

First build downloads and converts model weights automatically (~10 min). Open `http://localhost`.

## Quick Start (Native / Mac)

```bash
git clone --recurse-submodules <repo-url>
cd singlehdr-service

# Install dependencies
cd service/backend
pip install -r requirements-torch.txt

# Convert TF checkpoints to PyTorch weights (one-time, requires TF)
pip install tensorflow
python convert_weights.py --mode both \
  --ckpt_deq ../../ckpt/ckpt_deq/model.ckpt \
  --ckpt_lin ../../ckpt/ckpt_lin/model.ckpt \
  --ckpt_hal ../../ckpt/ckpt_hal/model.ckpt \
  --ckpt_ref ../../ckpt/ckpt_deq_lin_hal_ref/model.ckpt \
  --output_dir ../weights

# Run
cd ..
./run_mac.sh
```

Open `http://localhost:8000`.

## Features

- Basic and Refinement inference modes
- FIFO job queue with real-time progress (SSE)
- Client-side tone mapping (ACES, Reinhard, Linear)
- A/B comparison slider (SDR vs HDR)
- Batch upload and processing
- EXR export
- In-memory storage (no disk writes)
- Auto-downscale for GPU memory safety

See [INSTALL.md](INSTALL.md) for detailed setup, API reference, and troubleshooting.
