"""PyTorch inference backend for SingleHDR - supports CUDA, MPS, and CPU."""

import logging
import os
import threading
from typing import Callable, Optional

import cv2
import numpy as np
import torch

from torch_nets import (
    DequantizationNet, LinearizationNet, HallucinationNet, RefinementNet,
    apply_rf_torch,
)

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str, float, str], None]]

# Safety limit: auto-downscale images exceeding this before inference.
# 8MP ~= 3264x2448, safe for most GPUs including MPS (shared memory).
MAX_INFERENCE_PIXELS = int(os.environ.get("MAX_INFERENCE_PIXELS", str(8_000_000)))


def _get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _preprocess(img_bytes: bytes, max_pixels: int = 0):
    """Decode image bytes, normalize, pad to 64x multiple + 32px symmetric pad.
    If max_pixels > 0 and image exceeds it, downscale to fit."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Cannot decode image")
    ldr_val = np.flip(img_bgr, -1).astype(np.float32) / 255.0

    # Auto-downscale if image exceeds max_pixels
    h, w = ldr_val.shape[:2]
    if max_pixels > 0 and h * w > max_pixels:
        scale = (max_pixels / (h * w)) ** 0.5
        ldr_val = cv2.resize(ldr_val, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        logger.info("Auto-downscaled from %dx%d to %dx%d for inference",
                     w, h, ldr_val.shape[1], ldr_val.shape[0])

    original_h = ldr_val.shape[0]
    original_w = ldr_val.shape[1]

    was_resized = False
    if original_h % 64 != 0 or original_w % 64 != 0:
        resized_h = int(np.ceil(float(original_h) / 64.0)) * 64
        resized_w = int(np.ceil(float(original_w) / 64.0)) * 64
        ldr_val = cv2.resize(ldr_val, dsize=(resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
        was_resized = True

    padding = 32
    ldr_val = np.pad(ldr_val, ((padding, padding), (padding, padding), (0, 0)), 'symmetric')

    return ldr_val, original_h, original_w, was_resized


def _postprocess(hdr_out: np.ndarray, original_h: int, original_w: int, was_resized: bool):
    """Remove padding, resize back if needed."""
    padding = 32
    hdr_out = hdr_out[padding:-padding, padding:-padding]
    if was_resized:
        hdr_out = cv2.resize(hdr_out, dsize=(original_w, original_h), interpolation=cv2.INTER_CUBIC)
    return hdr_out


class TorchBasicPipeline:
    """PyTorch 3-network pipeline: Dequantization -> Linearization -> Hallucination."""

    def __init__(self, weights_dir: str):
        self.device = _get_device()
        logger.info("PyTorch basic pipeline using device: %s", self.device)

        self.deq_net = DequantizationNet()
        self.lin_net = LinearizationNet()
        self.hal_net = HallucinationNet()

        # Load weights
        self.deq_net.load_state_dict(torch.load(
            os.path.join(weights_dir, "dequantization.pt"), map_location="cpu", weights_only=True))
        self.lin_net.load_state_dict(torch.load(
            os.path.join(weights_dir, "linearization.pt"), map_location="cpu", weights_only=True), strict=False)
        self.hal_net.load_state_dict(torch.load(
            os.path.join(weights_dir, "hallucination.pt"), map_location="cpu", weights_only=True))

        self.deq_net.to(self.device).eval()
        self.lin_net.to(self.device).eval()
        self.hal_net.to(self.device).eval()

        self.lock = threading.Lock()
        logger.info("PyTorch basic pipeline ready.")

    @torch.no_grad()
    def run(self, img_bytes: bytes, progress_cb: ProgressCallback = None, thr: float = 0.12) -> np.ndarray:
        with self.lock:
            if progress_cb:
                progress_cb("preprocessing", 0.05, "Preprocessing image...")

            ldr_val, oh, ow, resized = _preprocess(img_bytes, max_pixels=MAX_INFERENCE_PIXELS)

            # HWC -> BCHW
            x = torch.from_numpy(ldr_val).permute(2, 0, 1).unsqueeze(0).to(self.device)
            del ldr_val

            # Dequantization
            if progress_cb:
                progress_cb("dequantization", 0.10, "Running Dequantization-Net...")
            C_pred = self.deq_net(x).clamp(0, 1)
            del x

            # Linearization
            if progress_cb:
                progress_cb("linearization", 0.30, "Running Linearization-Net...")
            pred_invcrf = self.lin_net(C_pred)        # [1, 1024]
            B_pred = apply_rf_torch(C_pred, pred_invcrf)
            del C_pred, pred_invcrf

            # Alpha mask
            alpha = B_pred.max(dim=1, keepdim=True).values
            alpha = ((alpha - 1.0 + thr) / thr).clamp(0, 1)
            alpha = alpha.expand_as(B_pred)

            # Hallucination (VGG16 — peak memory usage)
            if progress_cb:
                progress_cb("hallucination", 0.50, "Running Hallucination-Net...")
            y_predict = self.hal_net(B_pred)
            y_predict = torch.relu(y_predict)
            A_pred = B_pred + alpha * y_predict
            del y_predict, alpha

            if progress_cb:
                progress_cb("postprocessing", 0.80, "Post-processing...")

            # BCHW -> HWC numpy
            hdr_out = A_pred[0].permute(1, 2, 0).cpu().numpy()
            del A_pred, B_pred
            hdr_out = _postprocess(hdr_out, oh, ow, resized)

            if progress_cb:
                progress_cb("saving", 0.85, "Saving results...")

            return hdr_out

    @staticmethod
    def _clear_device_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def close(self):
        del self.deq_net, self.lin_net, self.hal_net
        self._clear_device_cache()


class TorchRefinementPipeline:
    """PyTorch 4-network pipeline with Refinement-Net."""

    def __init__(self, weights_dir: str):
        self.device = _get_device()
        logger.info("PyTorch refinement pipeline using device: %s", self.device)

        self.deq_net = DequantizationNet()
        self.lin_net = LinearizationNet()
        self.hal_net = HallucinationNet()
        self.ref_net = RefinementNet()

        self.deq_net.load_state_dict(torch.load(
            os.path.join(weights_dir, "dequantization.pt"), map_location="cpu", weights_only=True))
        self.lin_net.load_state_dict(torch.load(
            os.path.join(weights_dir, "linearization.pt"), map_location="cpu", weights_only=True), strict=False)
        self.hal_net.load_state_dict(torch.load(
            os.path.join(weights_dir, "hallucination.pt"), map_location="cpu", weights_only=True))
        self.ref_net.load_state_dict(torch.load(
            os.path.join(weights_dir, "refinement.pt"), map_location="cpu", weights_only=True))

        self.deq_net.to(self.device).eval()
        self.lin_net.to(self.device).eval()
        self.hal_net.to(self.device).eval()
        self.ref_net.to(self.device).eval()

        self.lock = threading.Lock()
        logger.info("PyTorch refinement pipeline ready.")

    @torch.no_grad()
    def run(self, img_bytes: bytes, progress_cb: ProgressCallback = None, thr: float = 0.12) -> np.ndarray:
        with self.lock:
            if progress_cb:
                progress_cb("preprocessing", 0.05, "Preprocessing image...")

            ldr_val, oh, ow, resized = _preprocess(img_bytes, max_pixels=MAX_INFERENCE_PIXELS)
            x = torch.from_numpy(ldr_val).permute(2, 0, 1).unsqueeze(0).to(self.device)
            del ldr_val

            if progress_cb:
                progress_cb("dequantization", 0.10, "Running Dequantization-Net...")
            C_pred = self.deq_net(x).clamp(0, 1)
            del x

            # Linearization
            if progress_cb:
                progress_cb("linearization", 0.30, "Running Linearization-Net...")
            pred_invcrf = self.lin_net(C_pred)
            B_pred = apply_rf_torch(C_pred, pred_invcrf)
            del pred_invcrf

            alpha = B_pred.max(dim=1, keepdim=True).values
            alpha = ((alpha - 1.0 + thr) / thr).clamp(0, 1)
            alpha = alpha.expand_as(B_pred)

            # Hallucination (VGG16 — peak memory usage)
            if progress_cb:
                progress_cb("hallucination", 0.50, "Running Hallucination-Net...")
            y_predict = self.hal_net(B_pred)
            y_predict = torch.relu(y_predict)
            A_pred = B_pred + alpha * y_predict
            del y_predict, alpha

            # Refinement — needs A_pred, B_pred, C_pred
            if progress_cb:
                progress_cb("refinement", 0.70, "Running Refinement-Net...")
            ref_input = torch.cat([A_pred, B_pred, C_pred], dim=1)
            del A_pred, B_pred, C_pred
            refined = torch.relu(self.ref_net(ref_input))
            del ref_input

            if progress_cb:
                progress_cb("postprocessing", 0.80, "Post-processing...")

            hdr_out = refined[0].permute(1, 2, 0).cpu().numpy()
            del refined
            hdr_out = _postprocess(hdr_out, oh, ow, resized)

            if progress_cb:
                progress_cb("saving", 0.85, "Saving results...")

            return hdr_out

    @staticmethod
    def _clear_device_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def close(self):
        del self.deq_net, self.lin_net, self.hal_net, self.ref_net
        self._clear_device_cache()


# --- I/O utilities ---

def load_exr(filepath: str) -> np.ndarray:
    """Load an OpenEXR file as float32 RGB numpy array."""
    import OpenEXR
    import Imath

    exr = OpenEXR.InputFile(filepath)
    header = exr.header()
    dw = header['dataWindow']
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    r = np.frombuffer(exr.channel('R', float_type), dtype=np.float32).reshape(h, w)
    g = np.frombuffer(exr.channel('G', float_type), dtype=np.float32).reshape(h, w)
    b = np.frombuffer(exr.channel('B', float_type), dtype=np.float32).reshape(h, w)
    return np.stack([r, g, b], axis=-1)


def save_exr(filepath: str, img: np.ndarray):
    """Save a float32 RGB image as OpenEXR."""
    import OpenEXR
    import Imath

    h, w, _ = img.shape
    img = img.astype(np.float32)
    header = OpenEXR.Header(w, h)
    float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = {'R': float_chan, 'G': float_chan, 'B': float_chan}
    out = OpenEXR.OutputFile(filepath, header)
    out.writePixels({
        'R': img[:, :, 0].tobytes(),
        'G': img[:, :, 1].tobytes(),
        'B': img[:, :, 2].tobytes(),
    })
    out.close()


def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma curve. gamma=2.4 is sRGB-like."""
    if abs(gamma - 2.4) < 0.01:
        return np.where(img <= 0.0031308,
                        12.92 * img,
                        1.055 * np.power(np.maximum(img, 0.0031308), 1.0 / 2.4) - 0.055)
    return np.power(np.maximum(img, 0.0), 1.0 / gamma)


def tonemap_aces(hdr_img: np.ndarray, gamma: float = 2.4) -> np.ndarray:
    """ACES filmic tone mapping + gamma."""
    x = np.maximum(hdr_img, 0.0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    mapped = (x * (a * x + b)) / (x * (c * x + d) + e)
    mapped = np.clip(mapped, 0.0, 1.0)
    return np.clip(_apply_gamma(mapped, gamma), 0.0, 1.0)


def tonemap_reinhard(hdr_img: np.ndarray, gamma: float = 2.4) -> np.ndarray:
    """Reinhard tone mapping + gamma."""
    x = np.maximum(hdr_img, 0.0)
    mapped = x / (1.0 + x)
    return np.clip(_apply_gamma(mapped, gamma), 0.0, 1.0)


def tonemap_linear(hdr_img: np.ndarray, gamma: float = 2.4) -> np.ndarray:
    """Linear clamp + gamma (no tone mapping)."""
    x = np.clip(hdr_img, 0.0, 1.0)
    return np.clip(_apply_gamma(x, gamma), 0.0, 1.0)


TONEMAPPERS = {
    "aces": tonemap_aces,
    "reinhard": tonemap_reinhard,
    "linear": tonemap_linear,
}


def save_preview_png(filepath: str, hdr_img: np.ndarray,
                     exposure: float = 0.0, tonemap: str = "aces", gamma: float = 2.4):
    """Save a tone-mapped 8-bit PNG preview."""
    img = np.maximum(hdr_img, 0.0)
    if exposure != 0.0:
        img = img * (2.0 ** exposure)
    tonemap_fn = TONEMAPPERS.get(tonemap, tonemap_aces)
    ldr = tonemap_fn(img, gamma)
    ldr_8bit = (ldr * 255.0).astype(np.uint8)
    cv2.imwrite(filepath, np.flip(ldr_8bit, -1))
