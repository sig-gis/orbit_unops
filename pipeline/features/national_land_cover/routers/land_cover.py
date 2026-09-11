import os
import traceback
from datetime import datetime
from uuid import uuid4
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pipeline.features.national_land_cover.models.schemas import UnifiedLandCoverRequest
from pipeline.features.national_land_cover.services.retrieval_method import run_national_land_cover

router = APIRouter()

def _run_nlc_job(job_id: str, request_obj: UnifiedLandCoverRequest):
    # Lazy import to access the global job database
    from pipeline.api import _jobs_lock, _jobs, _set_job, _save_jobs
    
    _set_job(job_id, {"status": "running"})
    try:
        # Run the real pipeline
        output_data = run_national_land_cover(request_obj.model_dump())
        
        # Merge outputs into result
        with _jobs_lock:
            if job_id in _jobs:
                current_result = _jobs[job_id].get("result", {})
                
                # Combine what we got from retrieval_method
                current_result["task_ids"] = output_data.get("task_ids", {})
                current_result["metrics"] = output_data.get("metrics", {})
                current_result["html_chart"] = output_data.get("html_chart")
                current_result["json_report"] = output_data.get("json_report")
                
                _jobs[job_id]["result"] = current_result
                # We don't mark as completed here. We let `_poll_ee_tasks_daemon` in api.py
                # poll the `task_ids`. When it sees they are done, it marks the job "completed".
                _save_jobs()
                
    except Exception as e:
        traceback.print_exc()
        _set_job(job_id, {"status": "failed", "error": str(e)})


@router.post("/run")
async def run_land_cover_job(request: UnifiedLandCoverRequest, background_tasks: BackgroundTasks):
    """
    Accepts the unified JSON payload from the frontend (Demo or Custom run),
    and orchestrates the Earth Engine land cover classification pipeline via the global Job Queue.
    """
    from pipeline.api import _jobs_lock, _jobs, utc_now_iso
    
    job_id = str(uuid4())
    created_at = utc_now_iso()
    
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": created_at,
            "updated_at": created_at,
            "indicator_id": "NLC",
            "aoi_id": request.cloud_project, # arbitrary label for UI
            "result": {
                "project": "damage-control-403117"
            },
            "error": None,
            "request": request.model_dump(),
        }

    # Dispatch to background task
    background_tasks.add_task(_run_nlc_job, job_id, request)
    
    with _jobs_lock:
        return _jobs[job_id]
