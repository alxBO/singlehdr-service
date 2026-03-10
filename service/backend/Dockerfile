# ==============================================================================
# Stage 1: Download TF checkpoints from Google Drive and convert to PyTorch
# ==============================================================================
FROM python:3.10-slim AS weights-builder

RUN apt-get update && apt-get install -y --no-install-recommends unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir gdown tensorflow-cpu numpy torch --extra-index-url https://download.pytorch.org/whl/cpu

ARG GDRIVE_FILE_ID=1e9vP8YPEjGcvXCa0Bfqwxw7qks7dH-VE

# Download and extract checkpoints
RUN gdown --id ${GDRIVE_FILE_ID} -O /tmp/ckpt.zip \
    && unzip -q /tmp/ckpt.zip -d /tmp/ckpt_raw \
    && rm /tmp/ckpt.zip

COPY service/backend/convert_weights.py /app/convert_weights.py

# Find checkpoint directories and convert
# The zip may contain ckpt/ at root or directly the ckpt_* dirs
RUN CKPT_ROOT=$(find /tmp/ckpt_raw -name "ckpt_deq" -type d -print -quit | xargs dirname) \
    && echo "Found checkpoints at: $CKPT_ROOT" \
    && cd /app && python convert_weights.py --mode both \
        --ckpt_deq "$CKPT_ROOT/ckpt_deq/model.ckpt" \
        --ckpt_lin "$CKPT_ROOT/ckpt_lin/model.ckpt" \
        --ckpt_hal "$CKPT_ROOT/ckpt_hal/model.ckpt" \
        --ckpt_ref "$CKPT_ROOT/ckpt_deq_lin_hal_ref/model.ckpt" \
        --output_dir /weights

# ==============================================================================
# Stage 2: Runtime image (no TensorFlow, no checkpoints)
# ==============================================================================
FROM python:3.10-slim

# System dependencies for OpenEXR and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenexr-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY service/backend/requirements-torch.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy vendor submodule (data files used at runtime)
COPY vendor/SingleHDR/ /app/vendor/SingleHDR/

# Copy network definitions and API code
COPY service/backend/torch_nets.py /app/torch_nets.py
COPY service/backend/app/ /app/app/

# Copy frontend static files (served by FastAPI)
COPY service/frontend/static/ /app/frontend/static/

# Copy converted PyTorch weights from builder stage
COPY --from=weights-builder /weights/ /app/weights/

EXPOSE 8000

ENV PYTORCH_WEIGHTS_DIR=/app/weights
ENV SINGLEHDR_VENDOR_DIR=/app/vendor/SingleHDR

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
