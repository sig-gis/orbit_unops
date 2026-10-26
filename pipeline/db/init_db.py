import os
import json
from datetime import datetime, timezone
from dateutil import parser
from sqlalchemy import text
from google.cloud import storage

from .session import engine, SessionLocal, Base
from .models import Client, User, Region, Job, JobResult

def parse_iso(dt_str):
    if not dt_str:
        return None
    try:
        return parser.isoparse(dt_str)
    except Exception:
        return datetime.now(timezone.utc)

def init_database(migrate_historical: bool = True):
    print("=== Initializing PostgreSQL Database ===")
    
    # 1. Enable PostGIS
    with engine.connect() as conn:
        print("Ensuring PostGIS extension is enabled...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        conn.commit()

    # 2. Create Tables
    print("Creating ERD tables (clients, users, regions, jobs, job_results)...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully!")

    if not migrate_historical:
        return

    # 3. Migrate historical jobs from GCS / local jobs.json
    print("\n=== Checking for Historical Jobs to Migrate ===")
    loaded_jobs = {}
    gcs_bucket_name = os.getenv("GCS_BUCKET")
    
    # Try fetching from GCS first
    if gcs_bucket_name:
        try:
            client = storage.Client()
            bucket = client.bucket(gcs_bucket_name)
            blob = bucket.blob("orbit_system/jobs.json")
            if blob.exists():
                print(f"Found orbit_system/jobs.json in gs://{gcs_bucket_name}!")
                loaded_jobs = json.loads(blob.download_as_text())
                print(f"Loaded {len(loaded_jobs)} job(s) from GCS.")
        except Exception as e:
            print(f"Note: Could not read directly from GCS ({e}). Checking local files...")

    # Fallback to local files if GCS wasn't read
    for local_fname in ["jobs_live_backup.json", "jobs.json", "jobs_example.json"]:
        if not loaded_jobs and os.path.exists(local_fname):
            try:
                with open(local_fname, "r") as f:
                    raw_content = json.load(f)
                if isinstance(raw_content, list):
                    loaded_jobs = {
                        str(j.get("job_id") or j.get("id") or i): j
                        for i, j in enumerate(raw_content)
                    }
                elif isinstance(raw_content, dict):
                    loaded_jobs = raw_content
                print(f"Loaded {len(loaded_jobs)} job(s) from {local_fname}.")
            except Exception as e:
                print(f"{local_fname} read error: {e}")


    if not loaded_jobs:
        print("No historical jobs found to migrate.")
        return

    # Insert into PostgreSQL
    db = SessionLocal()
    migrated_count = 0
    try:
        for raw_id, job_dict in loaded_jobs.items():
            if not isinstance(job_dict, dict):
                continue

            job_id = str(job_dict.get("job_id") or job_dict.get("id") or raw_id)
            
            # Check if job already exists in DB
            existing = db.query(Job).filter(Job.id == job_id).first()
            if existing:
                continue

            # Determine module name
            module_name = job_dict.get("module_name")
            if not module_name:
                res = job_dict.get("result") or {}
                if "report" in res or "inference" in res or "loi_pct" in str(job_dict).lower():
                    module_name = "national landcover"
                elif "sdg_11_3_1" in str(job_dict).lower():
                    module_name = "sdg_11_3_1"
                else:
                    module_name = "pipeline_task"

            status = str(job_dict.get("status") or job_dict.get("state") or "completed").lower()
            if status in {"complete", "succeeded", "success"}:
                status = "completed"

            created_at = parse_iso(job_dict.get("created_at") or job_dict.get("submitted_at"))
            updated_at = parse_iso(job_dict.get("updated_at") or job_dict.get("completed_at") or job_dict.get("created_at"))
            completed_at = parse_iso(job_dict.get("completed_at")) if status == "completed" else None

            # Preserve all results and parameters
            result_payload = job_dict.get("result") or {}
            params_payload = job_dict.get("parameters") or job_dict.get("request") or {}

            db_job = Job(
                id=job_id,
                module_name=module_name,
                status=status,
                credits_used=float(job_dict.get("credits_used", 0.0) or 0.0),
                started_at=created_at,
                completed_at=completed_at,
                created_at=created_at,
                updated_at=updated_at,
                parameters=params_payload,
                result=result_payload,
                error=job_dict.get("error"),
            )
            db.add(db_job)

            # Also create structured JobResult records if metrics or inference exist
            if isinstance(result_payload, dict):
                if result_payload.get("metrics"):
                    db.add(JobResult(
                        job_id=job_id,
                        result_type="model_metrics",
                        data=result_payload["metrics"],
                        created_at=completed_at or updated_at,
                    ))
                if result_payload.get("inference"):
                    db.add(JobResult(
                        job_id=job_id,
                        result_type="inference",
                        data=result_payload["inference"],
                        created_at=completed_at or updated_at,
                    ))
                if result_payload.get("report"):
                    db.add(JobResult(
                        job_id=job_id,
                        result_type="report",
                        data=result_payload["report"],
                        created_at=completed_at or updated_at,
                    ))

            migrated_count += 1

        db.commit()
        print(f"\nSUCCESS: Migrated {migrated_count} historical job(s) and results into PostgreSQL!")
    except Exception as e:
        db.rollback()
        print(f"Error during job migration: {e}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    init_database()
