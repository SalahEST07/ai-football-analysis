# ──────────────────────────────────────────────────────────────
# Backend Dockerfile — FastAPI + YOLOv8 + OpenCV + DeepSORT
# ──────────────────────────────────────────────────────────────
FROM python:3.11 AS base

# System deps for OpenCV, video processing, and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies first (layer caching) ──
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application source code ──
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

# ── Environment defaults ──
ENV MODEL_PATH=models/best.pt
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# ── Run with uvicorn ──
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
