#!/bin/bash
# Run SingleHDR service natively on Mac (with MPS GPU acceleration)
#
# Prerequisites:
#   pip install -r backend/requirements-torch.txt
#
# Convert weights first (on a machine with TF installed):
#   cd backend && python convert_weights.py --mode both \
#     --ckpt_deq ../../ckpt/ckpt_deq/model.ckpt \
#     --ckpt_lin ../../ckpt/ckpt_lin/model.ckpt \
#     --ckpt_hal ../../ckpt/ckpt_hal/model.ckpt \
#     --ckpt_ref ../../ckpt/ckpt_deq_lin_hal_ref/model.ckpt \
#     --output_dir ../weights
#
# Usage: ./run_mac.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PYTORCH_WEIGHTS_DIR="$SCRIPT_DIR/weights"
export MAX_MEGAPIXELS=50
export JOB_TTL_HOURS=24

echo "Starting SingleHDR on http://localhost:8000"
echo "Backend: PyTorch (MPS/CPU auto-detect)"
echo "Weights: $PYTORCH_WEIGHTS_DIR"
echo ""

cd "$SCRIPT_DIR/backend"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
