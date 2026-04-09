# orbit_unops
Repo for Google partner project with UNOPS.

## Frontend-triggered export backend

This repo now includes a reusable Earth Engine pipeline and a FastAPI backend endpoint that a frontend dashboard can call.

### Files
- `pipeline/main.py`: reusable `run_export(...)` pipeline + CLI wrapper
- `pipeline/api.py`: FastAPI service with async job execution

### Start API
From the repo root:

```bash
python -m uvicorn orbit_unops.pipeline.api:app --reload
```

### Endpoints
- `GET /health`
- `POST /exports` → starts export job (returns `job_id`)
- `GET /exports/{job_id}` → job status + Earth Engine task status (if available)

### Example request

```json
{
  "country": "Colombia",
  "map_year": 2019,
  "year_start": 2020,
  "year_end": 2021,
  "gcs_bucket": "your-gcs-bucket",
  "gcs_prefix": "exports/unops"
}
```

### Notes
- For production backend usage, use Earth Engine service account authentication.
- Earth Engine exports are asynchronous; frontend should poll `GET /exports/{job_id}`.
- Backend export target is now **GCS only** (`toCloudStorage`).

## Quick frontend (POC)

1. Start backend API:

```bash
python -m uvicorn orbit_unops.pipeline.api:app --reload
```

2. In a second terminal, serve the frontend:

```bash
python -m http.server 5500 --directory orbit_unops/frontend
```

3. Open:
- `http://127.0.0.1:5500`

The page lets you submit export requests and polls job status automatically.
