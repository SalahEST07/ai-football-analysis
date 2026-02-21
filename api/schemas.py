"""
Pydantic models for API request / response schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ──────────────────────────────────────────────
# Job lifecycle
# ──────────────────────────────────────────────
class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobCreate(BaseModel):
    max_frames: Optional[int] = Field(
        None,
        description="Stop processing after N frames (null = full video)",
    )


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    progress: Optional[float] = Field(None, description="0-100 percent")
    frames_processed: Optional[int] = None
    total_frames: Optional[int] = None
    error: Optional[str] = None


# ──────────────────────────────────────────────
# Possession
# ──────────────────────────────────────────────
class PossessionStats(BaseModel):
    team_a: float = Field(..., description="Team A possession %")
    team_b: float = Field(..., description="Team B possession %")
    team_a_frames: int
    team_b_frames: int


# ──────────────────────────────────────────────
# Tracking
# ──────────────────────────────────────────────
class PlayerPosition(BaseModel):
    frame: int
    x: int
    y: int
    team: str
    confidence: float


class PlayerTrack(BaseModel):
    team: str
    total_distance_px: float
    positions: list[PlayerPosition]


class BallPosition(BaseModel):
    frame: int
    x: int
    y: int


class TrackingResponse(BaseModel):
    frames_processed: int
    total_video_frames: int
    fps: int
    players_tracked: int
    player_tracking: dict[str, PlayerTrack]
    ball_tracking: list[BallPosition]


# ──────────────────────────────────────────────
# Single-image detection
# ──────────────────────────────────────────────
class Detection(BaseModel):
    label: str
    confidence: float
    bbox: list[int]


class DetectionResponse(BaseModel):
    detections: list[Detection]


# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    active_jobs: int
