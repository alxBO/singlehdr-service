#!/bin/bash
# Deploy SingleHDR on a Vast.ai GPU instance
#
# === Vast.ai instance setup ===
#
# 1. Choose a GPU instance (RTX 3080+ recommended)
# 2. Use a PyTorch template image (e.g. pytorch/pytorch:2.x-cuda12.x-runtime)
# 3. In "Docker options", add:  -p 8000:8000
# 4. Set disk space to at least 10 GB
#
# === On the instance ===
#
# SSH in, then:
#   git clone --recurse-submodules <repo-url>
#   cd singlehdr-service/service
#   ./deploy_vastai.sh
#
# === Access ===
#
# Option A: Click "Open" on the instance card (Cloudflare tunnel, HTTPS)
# Option B: Use direct IP:port from "IP Port Info" popup
#           (or use env var VAST_TCP_PORT_8000 for the external port)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== SingleHDR Vast.ai Deployment ==="
echo ""

# 1. Install Python dependencies
echo "[1/3] Installing dependencies..."
pip install -q -r "$SCRIPT_DIR/backend/requirements-torch.txt"

# 2. Download and convert weights if needed
if [ ! -f "$SCRIPT_DIR/weights/basic/dequantization.pt" ] || [ ! -f "$SCRIPT_DIR/weights/refinement/dequantization.pt" ]; then
    echo "[2/3] Downloading and converting model weights..."
    pip install gdown tensorflow-cpu

    # Download checkpoints from Google Drive
    mkdir -p /tmp/ckpt_dl
    cd /tmp/ckpt_dl
    gdown --id 1e9vP8YPEjGcvXCa0Bfqwxw7qks7dH-VE
    unzip -q -o /tmp/ckpt_dl/*.zip -d /tmp/ckpt_raw
    rm -rf /tmp/ckpt_dl

    # Find the checkpoint root (handles nested or flat zip structures)
    CKPT_ROOT=$(find /tmp/ckpt_raw -name "ckpt_deq" -type d -print -quit | xargs dirname)
    echo "Found checkpoints at: $CKPT_ROOT"

    cd "$SCRIPT_DIR/backend"
    python3 convert_weights.py --mode both \
        --ckpt_deq "$CKPT_ROOT/ckpt_deq/model.ckpt" \
        --ckpt_lin "$CKPT_ROOT/ckpt_lin/model.ckpt" \
        --ckpt_hal "$CKPT_ROOT/ckpt_hal/model.ckpt" \
        --ckpt_ref "$CKPT_ROOT/ckpt_deq_lin_hal_ref/model.ckpt" \
        --output_dir "$SCRIPT_DIR/weights"

    rm -rf /tmp/ckpt_raw
    echo "Weights converted successfully."
else
    echo "[2/3] Weights already present, skipping download."
fi

# 3. Start the service
echo "[3/3] Starting service on port 8000..."
echo ""

# Show access info if running on Vast.ai
if [ -n "$VAST_TCP_PORT_8000" ]; then
    echo "Direct access: http://$(hostname -I | awk '{print $1}'):$VAST_TCP_PORT_8000"
fi
echo "Local: http://0.0.0.0:8000"
echo ""

cd "$SCRIPT_DIR/backend"
export PYTORCH_WEIGHTS_DIR="$SCRIPT_DIR/weights"
export MAX_MEGAPIXELS=50
export JOB_TTL_HOURS=24

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
