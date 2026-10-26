import traceback
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
import uuid

from pipeline.features.space_for_time_tasking.services.inference_engine import run_inference

router = APIRouter(prefix="/api/tasking", tags=["inference"])

class InferenceRequest(BaseModel):
    model_asset_id: str
    aoi_name: str
    reference_year: int = 2018
    results_bucket: Optional[str] = None
    number_of_embedding_bands: int = 64

def _process_inference_job(job_id: str, request_data: dict):
    from pipeline.api import _jobs, _save_jobs
    try:
        _jobs[job_id]["status"] = "processing"
        _jobs[job_id]["state"] = "PROCESSING"
        _jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_jobs()

        result = run_inference(request_data)

        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["state"] = "COMPLETED"
        _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        _jobs[job_id]["updated_at"] = _jobs[job_id]["completed_at"]
        _jobs[job_id]["result"] = result
        _save_jobs()
        
    except Exception as e:
        print(f"Inference Job {job_id} failed: {e}")
        traceback.print_exc()
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["state"] = "FAILED"
        _jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        _jobs[job_id]["error"] = str(e)
        _save_jobs()

@router.post("/inference")
def run_inference_endpoint(request: InferenceRequest, background_tasks: BackgroundTasks):
    from pipeline.api import _jobs, _save_jobs

    aoi_name = (request.aoi_name or "").strip()
    if not aoi_name or aoi_name.lower() in {"none", "null", "undefined"}:
        raise HTTPException(
            status_code=400,
            detail="A country must be selected before running the inference job."
        )

    job_id = str(uuid.uuid4())
    
    # Register the job
    _jobs[job_id] = {
        "id": job_id,
        "type": "INFERENCE",
        "status": "pending",
        "state": "PENDING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "request": request.model_dump(),
        "module": "space_for_time_inference"
    }
    _save_jobs()
    
    # Dispatch to background
    background_tasks.add_task(_process_inference_job, job_id, request.model_dump())
    return _jobs[job_id]
