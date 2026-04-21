# orbit_unops

Repo for the Google partner project with UNOPS.

This project currently includes:
- An Earth Engine pipeline that trains a classifier and starts **CSV table exports** to GCS.
- A FastAPI backend that manages export jobs and file discovery.
- A simple frontend POC that starts exports, polls file readiness, downloads directly from GCS, and then deletes exported files.

---

## Current architecture (accurate flow)

1. Frontend calls `POST /exports`.
2. Backend creates a job and returns identifiers including `taskId` and `fileId`.
3. Backend runs `run_export(...)` asynchronously and starts Earth Engine table exports to GCS.
4. Frontend polls `GET /export-status/{fileId}`.
5. Backend lists matching GCS objects and returns:

```json
{
  "ready": true,
  "files": [
    {
      "name": "...",
      "url": "..."
    }
  ]
}
```

6. Frontend stores `files[0].url` as `downloadUrl`.
7. On download click, frontend triggers a hidden `<a>` download (`target="_blank"`, `download=""`).
8. Frontend calls `DELETE /export-delete/{fileId}` to clean up files.

> Important: backend does **not** stream file bytes; browser downloads directly from GCS using URL returned by backend.

---

## What the pipeline exports

`pipeline/main.py` runs a classification workflow and starts two Earth Engine table exports (`ee.batch.Export.table.toCloudStorage`) with `fileFormat="CSV"`:

1. `*_prediction_stats.csv`
2. `*_yearly_urban_area.csv`

So yes, this implementation is currently **CSV-only**.

---

## Key files

- `pipeline/main.py`
  - `run_export(...)` reusable pipeline function
  - CLI entrypoint for manual runs
- `pipeline/api.py`
  - FastAPI app
  - async job creation and status tracking
  - GCS file discovery, download URL generation, and cleanup endpoints
- `frontend/index.html`
  - simple dashboard for starting export + polling + direct download + cleanup

---

## API reference

### Health
- `GET /health`

### Export job lifecycle
- `POST /exports`
  - Starts async export job.
  - Returns job status + identifiers (including `taskId` and `fileId`).

- `GET /exports/{job_id}`
  - Returns backend job record.
  - Can include Earth Engine task status refresh when available.

### File availability + download handoff
- `GET /export-status/{fileId}`
  - Returns `{ ready, files }` for matching GCS objects.

- `GET /download-links/{fileId}`
  - Same `{ ready, files }` response shape.

- `DELETE /export-delete/{fileId}`
  - Deletes matching files in bucket and returns deletion summary.

---

## Download URL behavior

`pipeline/api.py` supports `GCS_URL_MODE`:

- `signed`: always return signed URLs (requires signing configuration)
- `public`: always return `https://storage.googleapis.com/{bucket}/{object}`
- `auto` (default): try signed URL, fall back to public URL

### Signed URL requirements

If using `signed` mode (or `auto` with private bucket), set:

- `SIGNING_SERVICE_ACCOUNT_EMAIL=<service-account-email>`

And grant required IAM (e.g. token creator + object read path, depending on runtime identity model).

---

## Run locally

From repo root:

### 1) Start API

```bash
python -m uvicorn orbit_unops.pipeline.api:app --reload
```

Optional environment examples:

```bash
# Use direct public object URLs
set GCS_URL_MODE=public

# Or force signed URLs
set GCS_URL_MODE=signed
set SIGNING_SERVICE_ACCOUNT_EMAIL=your-sa@your-project.iam.gserviceaccount.com
```

### 2) Serve frontend POC

```bash
python -m http.server 5500 --directory orbit_unops/frontend
```

### 3) Open frontend

- `http://127.0.0.1:5500`

---

## Operational notes

- Earth Engine exports are asynchronous; delays in `Submitted to server` are usually EE queue/compute behavior.
- Current `run_export(...)` includes diagnostics and multiple `.getInfo()` calls before/around task startup, which can add latency.
- If you see `AccessDenied` on direct GCS URL downloads, either:
  - use signed URLs correctly, or
  - use public URL mode + public read access (if allowed by your org policy).
