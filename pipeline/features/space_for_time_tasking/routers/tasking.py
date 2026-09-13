import time
import uuid
import traceback
import csv
import io
import ee
from fastapi import APIRouter, BackgroundTasks, File, UploadFile, HTTPException
from datetime import datetime, timezone
from google.cloud import storage

from pipeline.utils.gee_common import initialize_ee
from pipeline.features.space_for_time_tasking.models.schemas import TaskingRunRequest, UploadResponse
from pipeline.features.space_for_time_tasking.services.tasking_engine import run_tasking

router = APIRouter(prefix="/api/tasking", tags=["tasking"])

def _process_tasking_job(job_id: str, request_data: dict):
    from pipeline.api import _jobs, _save_jobs  # Lazy import to avoid circular dependency
    try:
        _jobs[job_id]["status"] = "processing"
        _jobs[job_id]["state"] = "PROCESSING"
        _jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_jobs()

        # Run the heavy spatial cross-validation tasking engine
        result = run_tasking(request_data)

        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["state"] = "COMPLETED"
        _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        _jobs[job_id]["updated_at"] = _jobs[job_id]["completed_at"]
        _jobs[job_id]["result"] = result
        _save_jobs()
        
    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        traceback.print_exc()
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["state"] = "FAILED"
        _jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        _jobs[job_id]["error"] = str(e)
        _save_jobs()


@router.post("/run")
def run_tasking_endpoint(request: TaskingRunRequest, background_tasks: BackgroundTasks):
    from pipeline.api import _jobs, _save_jobs
    import uuid
    job_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    # Pre-register the job in memory
    _jobs[job_id] = {
        "job_id": job_id,
        "id": job_id,
        "status": "queued",
        "state": "QUEUED",
        "created_at": created_at,
        "submitted_at": created_at,
        "updated_at": created_at,
        "indicator_id": "TASKING",
        "aoi_id": request.aoi_name or request.cloud_project, # User-provided country name
        "result": {
            "project": request.cloud_project
        },
        "error": None,
        "request": request.model_dump(),
    }
    _save_jobs()

    # Dispatch to background
    background_tasks.add_task(_process_tasking_job, job_id, request.model_dump())

    return _jobs[job_id]


@router.post("/upload", response_model=UploadResponse)
def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    bucket_name = "orbit-lc"
    # Create a unique path so we don't overwrite
    object_name = f"uploads/{uuid.uuid4()}_{file.filename}"
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        
        # Read the file contents from FastAPI and upload
        contents = file.file.read()
        blob.upload_from_string(contents, content_type="text/csv")
        
        gcs_uri = f"gs://{bucket_name}/{object_name}"
        https_url = f"https://storage.googleapis.com/{bucket_name}/{object_name}"
        
        return UploadResponse(
            gcs_uri=gcs_uri,
            https_url=https_url,
            bucket=bucket_name,
            object_name=object_name,
            filename=file.filename,
            content_type="text/csv"
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/columns/gee")
def get_gee_columns(asset_id: str):
    try:
        initialize_ee()
        # Get the first feature in the collection
        fc = ee.FeatureCollection(asset_id)
        # Exclude system:index to just get user properties
        props = fc.first().propertyNames().remove('system:index').getInfo()
        return {"columns": props}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to fetch GEE Asset columns: {e}")

@router.get("/columns/gcs")
def get_gcs_columns(gcs_uri: str):
    if not gcs_uri.startswith("gs://"):
        raise HTTPException(status_code=400, detail="Invalid GCS URI.")
    
    parts = gcs_uri[5:].split("/", 1)
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail="Invalid GCS URI format.")
        
    bucket_name, object_name = parts
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        
        # Download just enough to get the header line (e.g., 2048 bytes)
        content = blob.download_as_bytes(start=0, end=2048).decode("utf-8")
        
        # Parse the first line
        reader = csv.reader(io.StringIO(content))
        headers = next(reader)
        return {"columns": headers}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to fetch GCS columns: {e}")
