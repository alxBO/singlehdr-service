"""Pydantic models for API request/response schemas."""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    width: int
    height: int
    file_size_bytes: int
    format: str
    histogram: Dict[str, List[int]]
    dynamic_range_ev: float
    mean_brightness: float
    median_brightness: float
    clipping_percent: float
    mean_luminance_linear: float = 0.0
    peak_luminance_linear: float = 0.0
    contrast_ratio: float = 0.0


class GenerateRequest(BaseModel):
    mode: Literal["basic", "refinement"] = "refinement"
    hallucination_threshold: float = 0.12  # alpha threshold (0.01 to 0.50)



class ProgressEvent(BaseModel):
    stage: str
    progress: float
    message: str
    queue_position: int = 0


class HdrAnalysis(BaseModel):
    dynamic_range_ev: float
    contrast_ratio: float
    peak_luminance: float
    mean_luminance: float
    luminance_percentiles: Dict[str, float]
    hdr_histogram: dict


class ResultResponse(BaseModel):
    job_id: str
    download_url: str
    analysis: HdrAnalysis
    processing_time_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
