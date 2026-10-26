from pipeline.db import (
    SessionLocal,
    Job,
    get_or_create_default_tenant,
    get_or_create_region,
    sync_job_to_db,
)

def link_all_existing_jobs():
    print("=== Linking Existing Jobs to Tenant, Test User & Regions ===")
    db = SessionLocal()
    try:
        client, user = get_or_create_default_tenant(db)
        print(f"Default Client: {client.name} (ID: {client.id})")
        print(f"Default Test User: {user.email} (ID: {user.id})")

        jobs = db.query(Job).all()
        print(f"Found {len(jobs)} jobs in database.")

        linked_count = 0
        for job in jobs:
            # Extract AOI name from parameters or result
            aoi_name = (
                job.parameters.get("aoi_name")
                or job.result.get("aoi_name")
                or job.parameters.get("country")
                or job.result.get("country")
                or job.parameters.get("aoi_id")
            )

            # Also check nested inference
            if not aoi_name and isinstance(job.result, dict):
                aoi_name = (
                    job.result.get("results", {}).get("inference", {}).get("aoi_name")
                    or job.result.get("inference", {}).get("aoi_name")
                )

            # Assign tenant and user
            job.client_id = client.id
            job.user_id = user.id

            # Assign region
            if aoi_name:
                region = get_or_create_region(db, aoi_name)
                if region:
                    job.region_id = region.id

            # Sync granular results if missing
            if job.result:
                from pipeline.db.helpers import _record_granular_results
                _record_granular_results(db, job.id, job.result, job.completed_at or job.updated_at)

            linked_count += 1
            region_name = job.region.name if job.region else "Global/Custom"
            print(f" - Linked Job {job.id[:8]}... | User: {user.email} | Region: {region_name} | Module: {job.module_name}")

        db.commit()
        print(f"\nSUCCESS: Successfully linked all {linked_count} historical jobs to test user and real regions!")

    except Exception as e:
        db.rollback()
        print(f"Error linking jobs: {e}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    link_all_existing_jobs()
