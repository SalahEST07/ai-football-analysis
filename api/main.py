"""
Football AI Analytics API
===========================
FastAPI application with:
  • /analyze-video   — async video processing via background tasks
  • /jobs/{id}       — poll job status & progress
  • /jobs/{id}/possession — team possession stats
  • /jobs/{id}/tracking   — full player + ball tracking data
  • /predict-image   — synchronous single-frame detection
"""

import asyncio
import os
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from api.schemas import (
    Detection,
    DetectionResponse,
    HealthResponse,
    JobInfo,
    JobStatus,
    PossessionStats,
    TrackingResponse,
)
from api.services.pipeline import IntegratedPipeline


# ──────────────────────────────────────────────
# App & Model setup
# ──────────────────────────────────────────────
app = FastAPI(
    title="Football AI Analytics API",
    version="2.0.0",
    description="Integrated football analysis: detection, tracking, team classification, and possession stats.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
model = YOLO(MODEL_PATH)

# Thread pool for CPU-heavy video work
executor = ThreadPoolExecutor(max_workers=2)

# In-memory job store  (swap for Redis / DB in production)
jobs: dict[str, dict] = {}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _run_pipeline(job_id: str, video_path: str, max_frames: int | None):
    """Executed in a worker thread — runs the full pipeline."""
    try:
        jobs[job_id]["status"] = JobStatus.PROCESSING

        def _progress(processed, total):
            jobs[job_id]["frames_processed"] = processed
            jobs[job_id]["total_frames"] = total
            if total > 0:
                jobs[job_id]["progress"] = round(100 * processed / total, 1)

        pipeline = IntegratedPipeline(model_path=MODEL_PATH)
        result = pipeline.run(
            video_path=video_path,
            max_frames=max_frames,
            progress_callback=_progress,
        )

        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["result"] = result
        jobs[job_id]["frames_processed"] = result["frames_processed"]
        jobs[job_id]["total_frames"] = result["total_video_frames"]
        jobs[job_id]["progress"] = 100.0
    except Exception as exc:
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = str(exc)
    finally:
        pass


# ──────────────────────────────────────────────
# Shared upload directory
# ──────────────────────────────────────────────
UPLOAD_DIR = "shared_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

# ---- Health ----
@app.get("/", tags=["health"])
def root():
    return {"message": "Football Analytics API v2.0 Running"}


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    active = sum(1 for j in jobs.values() if j["status"] in (JobStatus.QUEUED, JobStatus.PROCESSING))
    return HealthResponse(status="ok", model_loaded=True, active_jobs=active)


@app.get("/debug-model", tags=["debug"])
def debug_model():
    """
    Test the YOLO model on the first frame of the first video in shared_videos.
    Visit http://localhost:8000/debug-model to diagnose detection issues.
    """
    import glob
    videos = (
        glob.glob(f"{UPLOAD_DIR}/*.mp4") +
        glob.glob(f"{UPLOAD_DIR}/*.avi") +
        glob.glob(f"{UPLOAD_DIR}/*.mkv")
    )
    if not videos:
        return {"error": f"No videos found in '{UPLOAD_DIR}'. Upload a video first."}

    video_path = videos[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"OpenCV could not open: {video_path}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"error": f"Could not read first frame from: {video_path}"}

    results = model(frame)[0]
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "label": model.names[cls],
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2],
        })

    return {
        "video":          video_path,
        "frame_shape":    list(frame.shape),
        "total_frames":   total_frames,
        "fps":            fps,
        "model_classes":  model.names,
        "total_detections": len(detections),
        "detections":     detections,
    }

# ---- Async Video Analysis ----
@app.post("/analyze-video", response_model=JobInfo, tags=["analysis"])
async def analyze_video(
    file: UploadFile = File(..., description="Video file (mp4, avi, mkv …)"),
    max_frames: int | None = Query(None, description="Limit frames processed (default: full video)"),
):
    """
    Upload a match video clip.  Processing runs **asynchronously** — you get
    back a `job_id` immediately and poll `/jobs/{job_id}` for progress.
    """
    # Save the file to the shared upload directory
    video_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Debug: Check file existence and size
    size = os.path.getsize(video_path) if os.path.exists(video_path) else -1
    print(f"DEBUG: Saved upload to {video_path} (exists: {os.path.exists(video_path)}, size: {size} bytes)")

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "status": JobStatus.QUEUED,
        "progress": 0.0,
        "frames_processed": 0,
        "total_frames": None,
        "error": None,
        "result": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # fire and forget — runs in thread pool
    loop = asyncio.get_running_loop()
    loop.run_in_executor(executor, _run_pipeline, job_id, video_path, max_frames)

    return JobInfo(
        job_id=job_id,
        status=JobStatus.QUEUED,
        progress=0.0,
    )


# ---- Job Status ----
@app.get("/jobs/{job_id}", response_model=JobInfo, tags=["jobs"])
def get_job(job_id: str):
    """Poll the processing status of an analysis job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    return JobInfo(
        job_id=job_id,
        status=j["status"],
        progress=j.get("progress"),
        frames_processed=j.get("frames_processed"),
        total_frames=j.get("total_frames"),
        error=j.get("error"),
    )


# ---- Possession Stats ----
@app.get("/jobs/{job_id}/possession", response_model=PossessionStats, tags=["results"])
def get_possession(job_id: str):
    """Return team possession statistics once analysis is complete."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j["status"] != JobStatus.COMPLETED:
        raise HTTPException(409, f"Job status is '{j['status']}' — not yet completed")
    return PossessionStats(**j["result"]["possession"])


# ---- Tracking Data ----
@app.get("/jobs/{job_id}/tracking", response_model=TrackingResponse, tags=["results"])
def get_tracking(job_id: str):
    """Return full player + ball tracking data as JSON."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j["status"] != JobStatus.COMPLETED:
        raise HTTPException(409, f"Job status is '{j['status']}' — not yet completed")
    r = j["result"]
    return TrackingResponse(
        frames_processed=r["frames_processed"],
        total_video_frames=r["total_video_frames"],
        fps=r["fps"],
        players_tracked=r["players_tracked"],
        player_tracking=r["player_tracking"],
        ball_tracking=r["ball_tracking"],
    )


# ---- Full Result (raw JSON) ----
@app.get("/jobs/{job_id}/result", tags=["results"])
def get_full_result(job_id: str):
    """Return the complete analysis result as raw JSON."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j["status"] != JobStatus.COMPLETED:
        raise HTTPException(409, f"Job status is '{j['status']}' — not yet completed")
    return j["result"]


# ---- List All Jobs ----
@app.get("/jobs", tags=["jobs"])
def list_jobs():
    """List all jobs with their current status."""
    return [
        JobInfo(
            job_id=jid,
            status=j["status"],
            progress=j.get("progress"),
            frames_processed=j.get("frames_processed"),
            total_frames=j.get("total_frames"),
            error=j.get("error"),
        )
        for jid, j in jobs.items()
    ]


# ---- Synchronous single-image detection (kept from v1) ----
@app.post("/predict-image", response_model=DetectionResponse, tags=["detection"])
async def predict_image(file: UploadFile = File(...)):
    """Run YOLO detection on a single image and return bounding boxes."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "Could not decode image")

    results = model(image)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append(
            Detection(
                label=model.names[cls],
                confidence=round(conf, 4),
                bbox=[x1, y1, x2, y2],
            )
        )

    return DetectionResponse(detections=detections)
