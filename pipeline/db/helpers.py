import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from .models import Client, User, Region, Job, JobResult

DEFAULT_CLIENT_NAME = "UNOPS Global Operations"
DEFAULT_TEST_EMAIL = "test.scientist@unops.org"

def get_or_create_default_tenant(db: Session) -> Tuple[Client, User]:
    """
    Ensure the canonical default Client and Test User exist for development & testing.
    This allows jobs to have true relational ownership without blocking on full auth.
    """
    client = db.query(Client).filter(Client.name == DEFAULT_CLIENT_NAME).first()
    if not client:
        client = Client(
            id=uuid.uuid4(),
            name=DEFAULT_CLIENT_NAME,
            billing_email="billing@unops.org",
            subscription_tier="Enterprise",
        )
        db.add(client)
        db.flush()

    user = db.query(User).filter(User.email == DEFAULT_TEST_EMAIL).first()
    if not user:
        user = User(
            id=uuid.uuid4(),
            client_id=client.id,
            email=DEFAULT_TEST_EMAIL,
            role="lead_scientist",
        )
        db.add(user)
        db.flush()

    return client, user

def get_or_create_region(db: Session, region_name: Optional[str]) -> Optional[Region]:
    """
    Lookup or create a Region record by name (e.g. Ireland, Jamaica).
    """
    if not region_name or region_name.strip().lower() in {"none", "null", ""}:
        return None

    clean_name = region_name.strip()
    # Normalize variants like "Ireland Custom Run" -> "Ireland"
    if "ireland" in clean_name.lower():
        lookup_name = "Ireland"
    elif "jamaica" in clean_name.lower():
        lookup_name = "Jamaica"
    else:
        lookup_name = clean_name

    region = db.query(Region).filter(Region.name.ilike(lookup_name)).first()
    if not region:
        region = Region(
            id=uuid.uuid4(),
            name=lookup_name,
            metadata_json={"source": "auto_registered", "created_at": datetime.now(timezone.utc).isoformat()}
        )
        db.add(region)
        db.flush()
    return region

def sync_job_to_db(
    db: Session,
    job_id: str,
    status: str,
    module_name: Optional[str] = None,
    aoi_name: Optional[str] = None,
    parameters: Optional[dict] = None,
    result: Optional[dict] = None,
    error: Optional[str] = None,
    credits_used: float = 0.0,
) -> Job:
    """
    Upsert job into PostgreSQL, automatically assigning default client, test user,
    and region, while populating granular job_results.
    """
    default_client, default_user = get_or_create_default_tenant(db)
    region = get_or_create_region(db, aoi_name)

    job = db.query(Job).filter(Job.id == str(job_id)).first()
    now = datetime.now(timezone.utc)

    if not job:
        job = Job(
            id=str(job_id),
            client_id=default_client.id,
            user_id=default_user.id,
            region_id=region.id if region else None,
            module_name=module_name or "pipeline_task",
            status=status.lower(),
            credits_used=credits_used,
            started_at=now,
            created_at=now,
            updated_at=now,
            parameters=parameters or {},
            result=result or {},
            error=error,
        )
        db.add(job)
    else:
        # Update existing
        job.status = status.lower()
        if not job.client_id:
            job.client_id = default_client.id
        if not job.user_id:
            job.user_id = default_user.id
        if not job.region_id and region:
            job.region_id = region.id
        if module_name and not job.module_name:
            job.module_name = module_name
        if parameters:
            job.parameters = parameters
        if result:
            job.result = result
        if error is not None:
            job.error = error
        if status.lower() in {"completed", "success"} and not job.completed_at:
            job.completed_at = now
        job.updated_at = now

    # Also record granular job_results if available
    if result and isinstance(result, dict):
        _record_granular_results(db, str(job_id), result, now)

    db.commit()
    return job

def _record_granular_results(db: Session, job_id: str, result: dict, timestamp: datetime):
    """Break down the result into searchable job_results rows."""
    # Check what already exists to prevent duplicate rows
    existing_types = {r.result_type for r in db.query(JobResult).filter(JobResult.job_id == job_id).all()}

    # 1. Model metrics (ML Random Forest / AUC / Accuracy)
    metrics_data = (
        result.get("metrics")
        or result.get("results", {}).get("earth_engine")
        or result.get("earth_engine")
    )
    if metrics_data and "model_metrics" not in existing_types:
        db.add(JobResult(
            job_id=job_id,
            result_type="model_metrics",
            data=metrics_data,
            created_at=timestamp,
        ))

    # 2. Inference / Raster Output
    inference_data = (
        result.get("inference")
        or result.get("results", {}).get("inference")
    )
    if inference_data and "inference" not in existing_types:
        db.add(JobResult(
            job_id=job_id,
            result_type="inference",
            data=inference_data,
            created_at=timestamp,
        ))

    # 3. Report
    report_data = (
        result.get("report")
        or result.get("results", {}).get("summary")
        or result.get("summary")
    )
    if report_data and "report" not in existing_types:
        db.add(JobResult(
            job_id=job_id,
            result_type="report",
            data=report_data,
            created_at=timestamp,
        ))
