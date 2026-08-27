# Space-for-time tasking service

This folder contains the Cloud Run version of the workflow in
`docs/studies/04_service_endpoints/space_for_time_tasking.ipynb`.

## TODO

- [ ] Make spatial blocking accept latitude and longitude only. Use `pyproj`
  internally to create metre-based blocks from those coordinates, remove the
  public `block_x_column` and `block_y_column` inputs and the `X_ITM`/`Y_ITM`
  defaults, and validate the blocking function locally without rerunning the
  Earth Engine workflow.

The service accepts a public CSV URL and a caller's Google OAuth access token,
runs the sampling and training workflow in the caller's Earth Engine project,
saves the Earth Engine assets there, and returns the complete results JSON. It
also writes the same JSON to the service-owned results bucket.

## Working deployment

- Service: `space-for-time-tasking`
- Region: `us-west1`
- URL: `https://space-for-time-tasking-384104229795.us-west1.run.app`
- Health: `GET /health`
- Task: `POST /task`
- Results bucket: `ee-seismosmsr-gis-space-for-time-results`
- Test source: `aboettcher-sig/orbit_unops`
- Test branch: `feature/space-for-time-tasking-service`
- Dockerfile: `/service_endpoints/space_for_time_tasking/Dockerfile`

Successful test outputs:

- Viewer: https://storage.googleapis.com/ee-seismosmsr-gis-space-for-time-results/peatlands-service-viewer-20260826b/viewer.html
- JSON: https://storage.googleapis.com/ee-seismosmsr-gis-space-for-time-results/peatlands-service-viewer-20260826b/results.json

## Deploy from the Google Cloud console

1. Open Cloud Run and click **Connect repository**.
2. Select **Cloud Build** and connect the GitHub repository.
3. Select the deployment branch.
4. Choose **Dockerfile** as the build type.
5. Set **Source location** to:

   ```text
   /service_endpoints/space_for_time_tasking/Dockerfile
   ```

   This field matters in this monorepo. `/Dockerfile` builds the unrelated
   application at the repository root. A correct build starts with
   `FROM python:3.11-slim` and has a build context of roughly 23 KB.

6. Configure the Cloud Run service:

   ```text
   Service name: space-for-time-tasking
   Region: us-west1
   Authentication: Allow public access
   Container port: 8080
   Memory: 1 GiB
   Request timeout: 3600 seconds
   Maximum concurrent requests per instance: 1
   ```

7. Add the runtime environment variable:

   ```text
   RESULTS_BUCKET=ee-seismosmsr-gis-space-for-time-results
   ```

8. Select the Cloud Run runtime service account. Grant that account
   `Storage Object Creator` on the results bucket.
9. The current bucket uses uniform bucket-level access, public access
   prevention is off, and `allUsers` has `Storage Object Viewer`. This makes
   returned `results_url` links readable without granting public write access.
10. Create the service and check the Cloud Build log before using it. The
    container command must be `app.main:app` on port `8080`.

The Cloud Build GitHub app must have access to the selected repository. A
personal fork can be used for testing without installing the app on the
organization-owned repository.

## Call the service

The Cloud Run URL is public, but `/task` requires the caller's temporary Google
OAuth bearer token. Earth Engine initializes with that token and the
`cloud_project` in the request. The Cloud Run service account is used only to
write the results JSON to the service-owned bucket.

Authenticate as the intended Earth Engine user:

```bash
gcloud auth login USER_EMAIL
```

Then call the endpoint. Keep the token in the shell; do not paste it into source
code or documentation.

```bash
ACCESS_TOKEN="$(gcloud auth print-access-token --account=USER_EMAIL)"

curl --fail-with-body --max-time 3600 \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{
    "csv_url": "https://storage.googleapis.com/ee-seismosmsr-gis-space-for-time-results/test-inputs/peatlands.csv",
    "cloud_project": "EARTH_ENGINE_PROJECT_ID",
    "run_name": "UNIQUE_RUN_NAME"
  }' \
  https://space-for-time-tasking-384104229795.us-west1.run.app/task
```

The Earth Engine project must be registered for Earth Engine, available for
compute, and writable by the caller. Each `run_name` must be unique because it
is used in the three Earth Engine asset names.

The CSV URL must be directly downloadable from Cloud Run. The original Catbox
test URL closed connections from Google's backend; the identical CSV in the
public results bucket worked.

## Request defaults

The peatlands CSV works with only `csv_url`, `cloud_project`, and an optional
`run_name`. The exposed defaults are:

```text
longitude_column: lon
latitude_column: lat
block_x_column: X_ITM
block_y_column: Y_ITM
target_column: LOI_PCT
target_threshold: 30
reference_year: 2018
block_size_m: 10000
test_block_fraction: 0.20
points_per_block: [1, 2, 5, 10, 20, "all"]
block_fractions: [0.10, 0.25, 0.50, 0.75, 1.00]
auc_tolerance: 0.01
number_of_trees: 100
number_of_embedding_bands: 64
sampling_scale_m: 10
seed: 42
```

## Outputs

The response includes the full report under `results`, plus:

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

Cloud Run requests have a 60-minute maximum. The current endpoint is
synchronous and has been tested successfully with a roughly ten-minute run.
Earth Engine runtimes can vary.

## Viewer

Each successful run saves `viewer.html` beside `results.json`. The viewer loads
that adjacent JSON and provides switchable Leaflet layers for the spatial
split, sklearn outcomes, Earth Engine outcomes, and model disagreements. The
background map is optional and off by default.

Each point record includes truth, split, both model probabilities and predicted
classes, both truth-relative outcome categories, and whether the models
disagree.

The repository copy defaults to the successful test result above. Serve it
locally from this folder:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/viewer/`. The viewer reads the public JSON URL
directly.
