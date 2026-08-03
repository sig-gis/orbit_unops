#!/usr/bin/env python3

"""FastAPI service for frontend-triggered Earth Engine export jobs."""

from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Dict, Literal, Optional
from uuid import uuid4
import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator, model_validator
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest

try:
    from google.cloud import storage
except Exception:
    storage = None

if __package__:
    # Running as part of the orbit_unops package (e.g. via uvicorn or import).
    from .indicators.sdg_11_03_01.v1.retrieval_method import run_11_03_01
    from .indicators.sdg_15_01_01.v1.retrieval_method import run_15_01_01
    from .indicators.sdg_06_06_01.v1.retrieval_method import run_06_06_01
    from .indicators.sdg_15_04_02.v1.retrieval_method import run_15_04_02
    from .indicators.sdg_15_03_01.v1.retrieval_method import run_15_03_01
    from .indicators.sdg_11_01_01.v1.retrieval_method import run_11_01_01
    from .utils.gee_common import get_task_status, list_saved_models, initialize_ee
else:
    # Running directly from the pipeline/ folder.
    from indicators.sdg_11_03_01.v1.retrieval_method import run_11_03_01
    from indicators.sdg_15_01_01.v1.retrieval_method import run_15_01_01
    from indicators.sdg_06_06_01.v1.retrieval_method import run_06_06_01
    from indicators.sdg_15_04_02.v1.retrieval_method import run_15_04_02
    from indicators.sdg_15_03_01.v1.retrieval_method import run_15_03_01
    from indicators.sdg_11_01_01.v1.retrieval_method import run_11_01_01
    from utils.gee_common import get_task_status, list_saved_models, initialize_ee


_INDICATOR_REGISTRY: Dict[str, Any] = {
    "11.3.1": {"v1": run_11_03_01, "latest": "v1"},
    "15.1.1": {"v1": run_15_01_01, "latest": "v1"},
    "6.6.1": {"v1": run_06_06_01, "latest": "v1"},
    "15.4.2": {"v1": run_15_04_02, "latest": "v1"},
    "15.3.1": {"v1": run_15_03_01, "latest": "v1"},
    "11.1.1": {"v1": run_11_01_01, "latest": "v1"},
}

_INDICATOR_MODEL_PREFIXES: Dict[str, str] = {
    "11.3.1": "urban_extent",
    "15.1.1": "forest",
    "6.6.1": "water",
    "15.4.2": "mountain",
    "15.3.1": "land",
    "11.1.1": "slum",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExportRequest(BaseModel):
    indicator_id: str = Field(
        ..., description="SDG indicator identifier, e.g. '11.3.1' or '15.1.1'"
    )
    version: Optional[str] = Field(
        None, description="Methodology version, defaults to latest"
    )
    model_asset_id: Optional[str] = Field(
        None, description="EE Asset ID of a saved ee.Classifier to use for inference"
    )
    country: Optional[str] = Field(
        None,
        description="Country name matching GAUL ADM0_NAME. "
                    "Required when aoi_geojson is not provided.",
    )
    aoi_geojson: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom Area of Interest as a GeoJSON geometry or "
                    "FeatureCollection. Takes precedence over country when both are supplied.",
    )
    map_year: int = Field(
        ..., description="Year of satellite embedding used for classifier training"
    )
    sample_points: Optional[int] = None
    sample_scale: Optional[int] = None
    embedding_scale: Optional[int] = None
    threshold: Optional[float] = None
    trees: Optional[int] = None
    seed: Optional[int] = None
    project: Optional[str] = None
    year_start: Optional[int] = Field(
        None, description="First year of the multi-year prediction range (inclusive)"
    )
    year_end: Optional[int] = Field(
        None, description="Last year of the multi-year prediction range (inclusive)"
    )
    export_name: Optional[str] = None

   
    start_date: Optional[str] = Field(
        None, description="ISO 8601 or MM/DD/YYYY start date sent by the frontend."
    )
    end_date: Optional[str] = Field(
        None, description="ISO 8601 or MM/DD/YYYY end date sent by the frontend."
    )
    resolution: Optional[str] = None
    priority: Optional[str] = None
    data_sources: Optional[list[str]] = None
    export_formats: Optional[list[str]] = None
    
    population_sources: Optional[list[str]] = Field(
        None, description="List of population sources, e.g. ['GHS_POP', 'WorldPop', 'GPW_v411', 'WorldBank']"
    )
    urban_methods: Optional[list[str]] = Field(
        None, description="List of urban extent methods, e.g. ['RF', 'DW']"
    )
    span_target: Optional[int] = Field(
        5, description="Target span length in years for the LCRPGR ratio"
    )
    wb_population_dict: Optional[Dict[int, int]] = Field(
        None, description="Dictionary mapping years to population totals for WorldBank data"
    )

    @model_validator(mode="after")
    def translate_dates_and_validate(self) -> "ExportRequest":
        """Translate start_date/end_date to year_start/year_end and validate AOI.

        Supported date formats:
          * ISO 8601:   "2025-11-13"
          * US slash:   "11/13/2025"

        If start_date / end_date are supplied they take precedence over
        year_start / year_end provided directly.  After translation both
        year_start and year_end must be valid integers.
        """
        _DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y")

        def _parse_year(date_str: str, field_name: str) -> int:
            for fmt in _DATE_FORMATS:
                try:
                    return datetime.strptime(date_str.strip(), fmt).year
                except ValueError:
                    continue
            raise ValueError(
                f"Cannot parse '{field_name}' value '{date_str}'. "
                "Expected ISO 8601 (YYYY-MM-DD) or MM/DD/YYYY format."
            )

        if self.start_date:
            self.year_start = _parse_year(self.start_date, "start_date")
        if self.end_date:
            self.year_end = _parse_year(self.end_date, "end_date")

        if self.year_start is None:
            raise ValueError(
                "'year_start' is required. Provide it directly or via 'start_date'."
            )
        if self.year_end is None:
            raise ValueError(
                "'year_end' is required. Provide it directly or via 'end_date'."
            )

        # Spatial target check (previously its own validator)
        if not self.country and not self.aoi_geojson:
            raise ValueError(
                "At least one of 'country' or 'aoi_geojson' must be provided."
            )
        return self

   
    @field_validator("indicator_id")
    @classmethod
    def validate_indicator_id(cls, value: str) -> str:
        if value not in _INDICATOR_REGISTRY:
            supported = ", ".join(f"'{k}'" for k in _INDICATOR_REGISTRY)
            raise ValueError(
                f"Unsupported indicator_id '{value}'. Supported: {supported}."
            )
        return value

    @field_validator("country", mode="before")
    @classmethod
    def validate_country(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped if stripped else None



    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, value: Optional[float]) -> Optional[float]:
        """Only validate if a value is explicitly provided; None means use config default."""
        if value is not None and (value < 0 or value > 1):
            raise ValueError("threshold must be between 0 and 1")
        return value

    @field_validator("year_end")
    @classmethod
    def validate_year_range(cls, value: Optional[int], info):
        """Cross-field range check deferred to model_validator when dates are used."""
        year_start = info.data.get("year_start")
        if value is not None and year_start is not None and value < year_start:
            raise ValueError("year_end must be >= year_start")
        return value


class ExportStatusResponse(BaseModel):
    job_id: str
    taskId: Optional[str] = None
    fileId: Optional[str] = None
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
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


@app.get("/", include_in_schema=False)
def root():
    """Redirect backend root to the FastAPI documentation."""
    return RedirectResponse(url="/docs")

# POC-friendly CORS setup (tighten in production).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
import json

_jobs_lock = Lock()

def _get_storage_client() -> "storage.Client":
    if storage is None:
        raise RuntimeError("google-cloud-storage is not installed")
    return storage.Client()

_jobs: Dict[str, Dict[str, Any]] = {}
try:
    _client = _get_storage_client()
    _bucket = _client.bucket(os.getenv("GCS_BUCKET", "unops"))
    _blob = _bucket.blob("orbit_system/jobs.json")
    if _blob.exists():
        _jobs = json.loads(_blob.download_as_text())
except Exception as e:
    print(f"Warning: Could not load jobs from GCS: {e}")

_files: Dict[str, Dict[str, str]] = {}


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


def _save_jobs():
    try:
        # cloud persistence
        client = _get_storage_client()
        bucket = client.bucket(os.getenv("GCS_BUCKET", "unops"))
        blob = bucket.blob("orbit_system/jobs.json")
        blob.upload_from_string(json.dumps(_jobs), content_type="application/json")
    except Exception as e:
        print(f"Cloud persistence error: {e}")

def _set_job(job_id: str, data: Dict[str, Any]) -> None:
    with _jobs_lock:
        if job_id not in _jobs:
            _jobs[job_id] = {}
        _jobs[job_id].update(data)
        _jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
        _save_jobs()


def _run_export_job(job_id: str, request: ExportRequest) -> None:
    """Background task: dispatch to the correct indicator function."""
    _set_job(job_id, {"status": "running"})
    try:
        file_id = _jobs[job_id]["file_id"]
        request_data = request.model_dump()
        request_data["gcs_prefix"] = _build_file_scoped_prefix(
            os.getenv("GCS_PREFIX", "exports/unops"), file_id
        )
        request_data["gcs_bucket"] = os.getenv("GCS_BUCKET", "unops")
        request_data["project"] = os.getenv("GCP_PROJECT", "damage-control-403117")

        # Route to the correct indicator function based on indicator_id and version.
        registry_entry = _INDICATOR_REGISTRY[request.indicator_id]
        version = request.version
        if not version or version == "latest":
            version = registry_entry["latest"]
            
        if version not in registry_entry:
            raise ValueError(f"Version '{version}' not found for indicator '{request.indicator_id}'.")
            
        indicator_fn = registry_entry[version]

        # Strip API-layer-only keys that indicator functions don't accept.
        indicator_data = {k: v for k, v in request_data.items() if k not in ["indicator_id", "version"]}

        print(f"========== [BACKEND: JOB {job_id}] ==========")
        print(f"Indicator: {request.indicator_id} (v{version})")
        print(f"Payload sent to EE script: {indicator_data}")
        print(f"===========================================")

        result = indicator_fn(**indicator_data)
        result["fileId"] = file_id
        _set_job(job_id, {"status": "running", "result": result})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set_job(job_id, {"status": "failed", "error": str(exc)})


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def get_models(project: Optional[str] = None, indicator: Optional[str] = None) -> Dict[str, list[str]]:
    if not project:
        return {"models": []}
        
    prefix_filter = None
    if indicator and indicator in _INDICATOR_MODEL_PREFIXES:
        prefix_filter = _INDICATOR_MODEL_PREFIXES[indicator]
        
    return {"models": list_saved_models(project, prefix_filter=prefix_filter)}


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
            "result": {"country": request.country} if request.country else None,
            "error": None,
        }
        _files[file_id] = {
            "bucket": os.getenv("GCS_BUCKET", "unops").strip(),
            "file_prefix": _build_file_scoped_prefix(os.getenv("GCS_PREFIX", "exports/unops"), file_id),
        }

    background_tasks.add_task(_run_export_job, job_id, request)
    return ExportStatusResponse(**_jobs[job_id])


@app.get("/exports", response_model=list[ExportStatusResponse])
def list_exports():
    """Returns all jobs from the system."""
    with _jobs_lock:
        return [ExportStatusResponse(**job) for job in _jobs.values()]


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

            all_completed = True
            any_failed = False
            for task_key, task_info in ee_status.items():
                if task_key == "classifier_export":
                    continue
                state = task_info.get("state", "UNKNOWN")
                if state in ["READY", "RUNNING"]:
                    all_completed = False
                elif state in ["FAILED", "CANCELLED"]:
                    any_failed = True

            new_status = job["status"]
            if any_failed:
                new_status = "failed"
            elif all_completed and len(ee_status) > 0:
                new_status = "completed"

            _set_job(job_id, {"status": new_status, "result": merged_result})
        except Exception:
            # Keep API resilient even if EE status refresh fails.
            pass

    with _jobs_lock:
        job = _jobs[job_id]

    return ExportStatusResponse(**job)


@app.post("/exports/{job_id}/cancel", response_model=ExportStatusResponse)
def cancel_export(job_id: str) -> ExportStatusResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
            
        if job["status"] in ["completed", "failed", "cancelled"]:
            return ExportStatusResponse(**job)

    try:
        project = job.get("result", {}).get("project")
        initialize_ee(project=project)
        
        task_ids = job.get("result", {}).get("task_ids", {})
        import ee
        for key, task_id in task_ids.items():
            if task_id:
                try:
                    ee.data.cancelTask(task_id)
                except Exception as e:
                    print(f"Warning: Failed to cancel EE task {task_id}: {e}")
                    
        _set_job(job_id, {"status": "cancelled"})
    except Exception as e:
        print(f"Failed to cancel job {job_id}: {e}")
        
    with _jobs_lock:
        return ExportStatusResponse(**_jobs[job_id])


@app.delete("/exports/{job_id}")
def delete_export(job_id: str) -> Dict[str, Any]:
    """Delete a job record without deleting exported GCS files."""
    with _jobs_lock:
        job = _jobs.pop(job_id, None)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        file_id = job.get("fileId") or job.get("file_id")
        if file_id:
            _files.pop(file_id, None)

        _save_jobs()

    return {"deleted": True, "job_id": job_id, "fileId": file_id}


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


@app.get("/indicators")
def get_indicators() -> Dict[str, Any]:
    return {
        "11.3.1": {
            "name": "Urban Expansion (SDG 11.3.1)",
            "description": "Multi-year urban growth analysis.",
            "icon": "building-2",
            "parameters": [
                {"name": "map_year", "type": "number", "label": "Training Year", "required": True, "default": 2020},
                {"name": "span_target", "type": "number", "label": "Target Span (Years)", "required": False, "default": 5},
                {"name": "population_sources", "type": "text", "label": "Pop Sources (comma separated)", "required": False, "default": "GHS_POP"},
                {"name": "urban_methods", "type": "text", "label": "Urban Methods (comma separated)", "required": False, "default": "RF"}
            ]
        },
        "15.1.1": {
            "name": "Forest Area (SDG 15.1.1)",
            "description": "Forest cover analysis and change detection.",
            "icon": "trees",
            "parameters": [
                {"name": "map_year", "type": "number", "label": "Training Year", "required": True, "default": 2020},
                {"name": "threshold", "type": "number", "label": "Forest Threshold (0-1)", "required": False, "default": 0.5}
            ]
        },
        "6.6.1": {
            "name": "Water Ecosystems (SDG 6.6.1)",
            "description": "Spatial extent of water-related ecosystems.",
            "icon": "droplets",
            "parameters": [
                {"name": "map_year", "type": "number", "label": "Training Year", "required": True, "default": 2020}
            ]
        },
        "15.4.2": {
            "name": "Mountain Green Cover (SDG 15.4.2)",
            "description": "Mountain green cover index.",
            "icon": "mountain",
            "parameters": [
                {"name": "map_year", "type": "number", "label": "Training Year", "required": True, "default": 2020}
            ]
        }
    }

import urllib.request
from fastapi import Response

@app.get("/proxy-csv")
def proxy_csv(url: str):
    """Proxy endpoint to bypass CORS when fetching CSVs from GCS in the browser."""
    if not url.startswith("https://storage.googleapis.com/"):
        raise HTTPException(status_code=400, detail="Only storage.googleapis.com URLs are allowed")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            return Response(content=response.read(), media_type="text/csv")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found or failed to fetch: {str(e)}")
