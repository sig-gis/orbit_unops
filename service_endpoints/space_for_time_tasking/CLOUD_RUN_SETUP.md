# Cloud Run setup

This service is deployed from a subdirectory of the `orbit_unops` repository.
The Cloud Run service is public, but every `/task` request must supply the
caller's Google OAuth bearer token. Earth Engine work and assets use the
caller's account and the `cloud_project` in the request. The Cloud Run service
account is used only to write `results.json` and `viewer.html` to the service's
results bucket.

## 1. Create the results bucket

Create a Cloud Storage bucket in the Cloud Run project and region.

For publicly readable result and viewer links:

- Use uniform bucket-level access.
- Leave public access prevention off.
- Grant `allUsers` the `Storage Object Viewer` role.

Do not grant public write access.

## 2. Connect the repository

In **Cloud Run**, click **Connect repository** and select **Cloud Build**.
Connect the GitHub repository and select the deployment branch.

Use:

```text
Build type: Dockerfile
Source location: /service_endpoints/space_for_time_tasking/Dockerfile
```

The source location is critical in this monorepo. `/Dockerfile` builds the
unrelated application at the repository root.

If the organization repository cannot authorize the Google Cloud Build GitHub
app, connect a personal fork for testing.

## 3. Configure the service

```text
Service name: space-for-time-tasking
Region: us-west1
Authentication: Allow public access
Container port: 8080
Memory: 1 GiB
Request timeout: 3600 seconds
Maximum concurrent requests per instance: 1
```

The 3600-second timeout is Cloud Run's maximum. The endpoint is synchronous,
so a run must finish within that request window.

Add the environment variable:

```text
RESULTS_BUCKET=YOUR_BUCKET_NAME
```

Under **Security**, select the Cloud Run runtime service account. Grant that
service account `Storage Object Creator` on the results bucket. It does not
need Earth Engine authority because the caller supplies that authority.

## 4. Deploy and verify the build

Create the service and open the Cloud Build log. A correct build shows:

```text
FROM python:3.11-slim
COPY app ./app
COPY viewer ./viewer
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
```

If the log shows `pipeline.api`, port `8000`, or Python 3.10, the trigger used
the repository-root Dockerfile. Disable that trigger and recreate it with the
nested source location above.

Check the deployed service:

```bash
curl https://YOUR_SERVICE_URL/health
```

Expected response:

```json
{"status":"healthy"}
```

## 5. Test a task

The caller must have Earth Engine access and permission to use and create assets
in the requested Earth Engine Cloud project.

```bash
gcloud auth login USER_EMAIL

ACCESS_TOKEN="$(gcloud auth print-access-token --account=USER_EMAIL)"

curl --fail-with-body --max-time 3600 \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{
    "csv_url": "PUBLIC_CSV_URL",
    "cloud_project": "EARTH_ENGINE_PROJECT_ID",
    "run_name": "UNIQUE_RUN_NAME"
  }' \
  https://YOUR_SERVICE_URL/task
```

The CSV URL must be directly downloadable from Cloud Run. Use a stable HTTPS
object URL. Each run name must be unique because it becomes part of the Earth
Engine asset IDs and results-bucket object names.

A successful response includes:

```json
{
  "status": "success",
  "run_name": "...",
  "results_uri": "gs://.../results.json",
  "results_url": "https://storage.googleapis.com/.../results.json",
  "viewer_url": "https://storage.googleapis.com/.../viewer.html",
  "assets": {
    "points": "projects/.../assets/..._points",
    "reference_samples": "projects/.../assets/..._reference_samples",
    "classifier": "projects/.../assets/..._random_forest"
  }
}
```

## 6. Continuous deployment

The repository connection creates a Cloud Build trigger. A push to the selected
branch rebuilds and deploys a new Cloud Run revision automatically. Confirm the
trigger still targets:

```text
/service_endpoints/space_for_time_tasking/Dockerfile
```

before relying on automatic deployments.

## Working test deployment

```text
Service: https://space-for-time-tasking-384104229795.us-west1.run.app
Project: ee-seismosmsr-gis
Region: us-west1
Results bucket: ee-seismosmsr-gis-space-for-time-results
```

Successful viewer:

https://storage.googleapis.com/ee-seismosmsr-gis-space-for-time-results/peatlands-service-viewer-20260826b/viewer.html
