import os
import time
import uuid
from datetime import datetime, timezone

import ee
from google.cloud import storage

from pipeline.features.space_for_time_tasking.services.tasking_engine import load_tasking_config

# Defaults loaded from config.yaml (with fallbacks)
_CONFIG = load_tasking_config()
EMBEDDINGS = _CONFIG["datasets"]["embeddings"]
GAUL_DATASET = _CONFIG["datasets"]["countries"]

def _wait_for_task(task, poll_interval=None):
    """Wait for an Earth Engine task to complete."""
    if poll_interval is None:
        poll_interval = load_tasking_config().get("runtime", {}).get("task_poll_interval_s", 10)
    while task.active():
        time.sleep(poll_interval)
    status = task.status()
    if status["state"] != "COMPLETED":
        error_msg = status.get("error_message", str(status))
        raise RuntimeError(f"Earth Engine Export Failed: {error_msg}")

def run_inference(config: dict) -> dict:
    """
    Expert Inference Module:
    Loads a trained Random Forest model, extracts the exact country boundary,
    classifies the embeddings, and exports a high-quality clipped GeoTIFF.
    """
    cfg = load_tasking_config()
    datasets_cfg = cfg.get("datasets", {})
    params_cfg = cfg.get("parameters", {})

    model_asset_id = config["model_asset_id"]
    aoi_name = config.get("aoi_name")
    reference_year = int(config.get("reference_year", params_cfg.get("reference_year", 2018)))
    results_bucket = (
        config.get("results_bucket")
        or config.get("gcs_bucket")
        or os.environ.get("RESULTS_BUCKET")
        or os.environ.get("GCS_BUCKET")
    )
    if not results_bucket:
        raise ValueError(
            "A target GCS bucket must be specified via results_bucket parameter, or RESULTS_BUCKET/GCS_BUCKET environment variable."
        )
    n_bands = int(config.get("number_of_embedding_bands", params_cfg.get("number_of_embedding_bands", 64)))
    
    if not aoi_name:
        raise ValueError("aoi_name (country name) is required for inference clipping.")
        
    # Generate unique run ID for the output file
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"inference-{stamp}-{uuid.uuid4().hex[:8]}"
    
    # 1. Fetch exact country boundary from FAO GAUL dataset (configured in config.yaml)
    country_dataset = config.get("country_dataset") or datasets_cfg.get("countries", GAUL_DATASET)
    countries = ee.FeatureCollection(country_dataset)
    country_roi = countries.filter(ee.Filter.eq("ADM0_NAME", aoi_name)).geometry()
    
    # 2. Build the satellite composite for the reference year
    embeddings_col = config.get("embedding_collection") or datasets_cfg.get("embeddings", EMBEDDINGS)
    bands = [f"A{i:02d}" for i in range(n_bands)]
    composite = (
        ee.ImageCollection(embeddings_col)
        .filterDate(f"{reference_year}-01-01", f"{reference_year + 1}-01-01")
        .mosaic()
        .select(bands)
    )
    
    # 3. Load the Trained Random Forest Classifier from Step 1
    classifier = ee.Classifier.load(model_asset_id)
    
    # 4. Classify the composite and clip to country boundary
    # We clip it perfectly to the borders so there is no ugly square box
    classified_image = composite.classify(classifier).clip(country_roi)
    
    # 5. Export high-quality Cloud Optimized GeoTIFF (COG) to Google Cloud Storage
    export_prefix = f"space_for_time_tasking/inference/{run_name}"
    
    task = ee.batch.Export.image.toCloudStorage(
        image=classified_image,
        description=f"Export_{run_name}",
        bucket=results_bucket,
        fileNamePrefix=export_prefix,
        region=country_roi,
        scale=200, # 200m resolution to match the tasking engine logic
        crs="EPSG:4326",
        maxPixels=1e13,
        formatOptions={"cloudOptimized": True} # Expert standard for web viewing
    )
    
    print(f"[Inference] Starting GeoTIFF export to gs://{results_bucket}/{export_prefix}.tif ...")
    task.start()
    
    # Wait for the export to complete
    _wait_for_task(task)
    
    final_gcs_uri = f"gs://{results_bucket}/{export_prefix}.tif"
    final_https_url = f"https://storage.googleapis.com/{results_bucket}/{export_prefix}.tif"
    
    print(f"[Inference] Export successful: {final_gcs_uri}")
    
    return {
        "status": "success",
        "run_name": run_name,
        "aoi_name": aoi_name,
        "results_uri": final_gcs_uri,
        "results_url": final_https_url
    }
