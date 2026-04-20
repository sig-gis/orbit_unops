#!/usr/bin/env python3

"""FastAPI service for frontend-triggered Earth Engine export jobs."""

from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Dict, Literal, Optional
from uuid import uuid4
import os

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest

try:
    from google.cloud import storage
except Exception:
    storage = None

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
    taskId: Optional[str] = None
    fileId: Optional[str] = None
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class FileStatusResponse(BaseModel):
    ready: bool
    files: list[Dict[str, str]]


class FileDeleteResponse(BaseModel):
    fileId: str
    deleted: int
    files: list[Dict[str, str]]

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
_files: Dict[str, Dict[str, str]] = {}
_jobs_lock = Lock()


def _get_storage_client() -> "storage.Client":
    if storage is None:
        raise RuntimeError("google-cloud-storage is not installed")
    return storage.Client()


def _normalize_gcs_prefix(prefix: Optional[str]) -> str:
    return (prefix or "").strip().strip("/")


def _build_file_scoped_prefix(gcs_prefix: Optional[str], file_id: str) -> str:
    normalized = _normalize_gcs_prefix(gcs_prefix)
    if normalized:
        return f"{normalized}/{file_id}"
    return file_id


def _public_gcs_url(bucket_name: str, blob_name: str) -> str:
    return f"https://storage.googleapis.com/{bucket_name}/{blob_name}"


def _signed_gcs_url(bucket_name: str, blob_name: str, expiration_hours: int = 24) -> str:
    client = _get_storage_client()
    credentials, _ = google.auth.default()
    credentials.refresh(GoogleAuthRequest())

    signing_service_account = os.getenv("SIGNING_SERVICE_ACCOUNT_EMAIL") or getattr(
        credentials, "service_account_email", None
    )
    if not signing_service_account:
        raise RuntimeError(
            "Unable to determine signing service account. Set SIGNING_SERVICE_ACCOUNT_EMAIL env var."
        )

    blob = client.bucket(bucket_name).blob(blob_name)
    return blob.generate_signed_url(
        version="v4",
        method="GET",
        expiration=timedelta(hours=expiration_hours),
        service_account_email=signing_service_account,
        access_token=credentials.token,
    )


def _build_download_url(bucket_name: str, blob_name: str) -> str:
    """Build a download URL based on GCS_URL_MODE.

    Modes:
    - signed: always signed URLs
    - public: always public object URLs
    - auto (default): try signed first, fall back to public URL
    """
    url_mode = os.getenv("GCS_URL_MODE", "auto").strip().lower()

    if url_mode == "public":
        return _public_gcs_url(bucket_name, blob_name)

    if url_mode == "signed":
        return _signed_gcs_url(bucket_name, blob_name)

    if url_mode == "auto":
        try:
            return _signed_gcs_url(bucket_name, blob_name)
        except Exception:
            return _public_gcs_url(bucket_name, blob_name)

    raise RuntimeError("Invalid GCS_URL_MODE. Use one of: signed, public, auto")


def _list_files_for_file_id(file_id: str) -> list[Dict[str, str]]:
    with _jobs_lock:
        file_record = _files.get(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail=f"fileId '{file_id}' not found")

    bucket_name = file_record["bucket"]
    file_prefix = file_record.get("file_prefix")
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)

    matched_files: list[Dict[str, str]] = []
    for blob in bucket.list_blobs(prefix=file_prefix or None):
        if file_id not in blob.name:
            continue
        matched_files.append(
            {
                "name": blob.name,
                "url": _build_download_url(bucket_name, blob.name),
            }
        )

    matched_files.sort(key=lambda item: item["name"])
    return matched_files

def _set_job(job_id: str, updates: Dict[str, Any]) -> None:
    with _jobs_lock:
        if job_id not in _jobs:
            return
        _jobs[job_id].update(updates)
        _jobs[job_id]["updated_at"] = utc_now_iso()

def _run_export_job(job_id: str, request: ExportRequest) -> None:
    _set_job(job_id, {"status": "running"})
    try:
        file_id = _jobs[job_id]["file_id"]
        request_data = request.model_dump()
        request_data["gcs_prefix"] = _build_file_scoped_prefix(request_data.get("gcs_prefix"), file_id)
        result = run_export(**request_data)
        result["fileId"] = file_id
        _set_job(job_id, {"status": "completed", "result": result})
    except Exception as exc:
        _set_job(job_id, {"status": "failed", "error": str(exc)})

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/exports", response_model=ExportStatusResponse, status_code=202)
def create_export(request: ExportRequest, background_tasks: BackgroundTasks) -> ExportStatusResponse:
    job_id = str(uuid4())
    file_id = str(uuid4())
    created_at = utc_now_iso()
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "taskId": job_id,
            "fileId": file_id,
            "task_id": job_id,
            "file_id": file_id,
            "status": "queued",
            "created_at": created_at,
            "updated_at": created_at,
            "result": None,
            "error": None,
        }
        _files[file_id] = {
            "bucket": request.gcs_bucket.strip(),
            "file_prefix": _build_file_scoped_prefix(request.gcs_prefix, file_id),
        }

    background_tasks.add_task(_run_export_job, job_id, request)
    return ExportStatusResponse(**_jobs[job_id])

@app.get("/exports/{job_id}", response_model=ExportStatusResponse)
def get_export(job_id: str, refresh_task_status: bool = True) -> ExportStatusResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    # Optional refresh from Earth Engine task states for dashboard polling
    if refresh_task_status and job.get("result") and job["result"].get("task_ids"):
        try:
            task_ids = job["result"].get("task_ids", {})
            ee_status = {
                key: get_task_status(task_id=task_id, project=job["result"].get("project"))
                for key, task_id in task_ids.items()
                if task_id
            }
            merged_result = dict(job["result"])
            merged_result["ee_task_status"] = ee_status
            _set_job(job_id, {"result": merged_result})
        except Exception:
            # Keep API resilient even if EE status refresh fails.
            pass

    with _jobs_lock:
        job = _jobs[job_id]

    return ExportStatusResponse(**job)


@app.get("/export-status/{fileId}", response_model=FileStatusResponse)
def get_export_status(fileId: str) -> FileStatusResponse:
    files = _list_files_for_file_id(fileId)
    return FileStatusResponse(ready=bool(files), files=files)


@app.get("/download-links/{fileId}", response_model=FileStatusResponse)
def get_download_links(fileId: str) -> FileStatusResponse:
    files = _list_files_for_file_id(fileId)
    return FileStatusResponse(ready=bool(files), files=files)


@app.delete("/export-delete/{fileId}", response_model=FileDeleteResponse)
def delete_export_files(fileId: str) -> FileDeleteResponse:
    files = _list_files_for_file_id(fileId)

    with _jobs_lock:
        file_record = _files.get(fileId)
    if not file_record:
        raise HTTPException(status_code=404, detail=f"fileId '{fileId}' not found")

    bucket_name = file_record["bucket"]
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)

    deleted_files: list[Dict[str, str]] = []
    for file_item in files:
        bucket.blob(file_item["name"]).delete()
        deleted_files.append(file_item)

    with _jobs_lock:
        _files.pop(fileId, None)

    return FileDeleteResponse(fileId=fileId, deleted=len(deleted_files), files=deleted_files)