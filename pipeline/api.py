#!/usr/bin/env python3

"""FastAPI service for frontend-triggered Earth Engine export jobs."""

from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Literal, Optional
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

try:
    # Works when imported as module: orbit_unops.pipeline.api
    from .main import get_task_status, run_export
except ImportError:
    # Works when run directly from pipeline folder.
    from main import get_task_status, run_export

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class ExportRequest(BaseModel):
    country: str = Field(..., description="Country name matching GAUL ADM0_NAME")
    map_year: int = 2019
    sample_points: int = 25
    sample_scale: int = 30
    embedding_scale: int = 10
    threshold: float = 0.75
    trees: int = 10
    seed: int = 42
    project: Optional[str] = None
    year_start: int = 2020
    year_end: int = 2021
    export_name: Optional[str] = None
    gcs_bucket: str
    gcs_prefix: Optional[str] = None

    @field_validator("country")
    @classmethod
    def validate_country(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("country is required")
        return value.strip()

    @field_validator("gcs_bucket")
    @classmethod
    def validate_gcs_bucket(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("gcs_bucket is required")
        return value.strip()

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("threshold must be between 0 and 1")
        return value

    @field_validator("year_end")
    @classmethod
    def validate_year_range(cls, value: int, info):
        year_start = info.data.get("year_start")
        if year_start is not None and value < year_start:
            raise ValueError("year_end must be >= year_start")
        return value

class ExportStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

app = FastAPI(title="UNOPS Export API", version="1.0.0")

# POC-friendly CORS setup (tighten in production).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = Lock()

def _set_job(job_id: str, updates: Dict[str, Any]) -> None:
    with _jobs_lock:
        if job_id not in _jobs:
            return
        _jobs[job_id].update(updates)
        _jobs[job_id]["updated_at"] = utc_now_iso()

def _run_export_job(job_id: str, request: ExportRequest) -> None:
    _set_job(job_id, {"status": "running"})
    try:
        result = run_export(**request.model_dump())
        _set_job(job_id, {"status": "completed", "result": result})
    except Exception as exc:
        _set_job(job_id, {"status": "failed", "error": str(exc)})

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/exports", response_model=ExportStatusResponse, status_code=202)
def create_export(request: ExportRequest, background_tasks: BackgroundTasks) -> ExportStatusResponse:
    job_id = str(uuid4())
    created_at = utc_now_iso()
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": created_at,
            "updated_at": created_at,
            "result": None,
            "error": None,
        }

    background_tasks.add_task(_run_export_job, job_id, request)
    return ExportStatusResponse(**_jobs[job_id])

@app.get("/exports/{job_id}", response_model=ExportStatusResponse)
def get_export(job_id: str, refresh_task_status: bool = True) -> ExportStatusResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    # Optional refresh from Earth Engine task state for dashboard polling
    if refresh_task_status and job.get("result") and job["result"].get("task_id"):
        try:
            ee_status = get_task_status(task_id=job["result"]["task_id"], project=job["result"].get("project"))
            merged_result = dict(job["result"])
            merged_result["ee_task_status"] = ee_status
            _set_job(job_id, {"result": merged_result})
        except Exception:
            # Keep API resilient even if EE status refresh fails.
            pass

        with _jobs_lock:
            job = _jobs[job_id]

    return ExportStatusResponse(**job)