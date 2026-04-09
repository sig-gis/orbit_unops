#!/usr/bin/env python3

import argparse
import json
import sys
from typing import Any, Dict, Optional

import ee

def initialize_ee(project: Optional[str] = None) -> None:
    """Initialize Earth Engine for server or CLI execution."""
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize Earth Engine. "
            "Authenticate first with service account credentials or "
            "'earthengine authenticate'."
        ) from exc

def embeddings_by_year(embeddings_ic: ee.ImageCollection, boundary: ee.FeatureCollection, year: int) -> ee.Image:
    """Return annual embedding mosaic for a given year and boundary."""
    return (
        embeddings_ic
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(boundary)
        .mosaic()
    )

def _validate_inputs(
    country: str,
    year_start: int,
    year_end: int,
    threshold: float,
    gcs_bucket: str,
) -> None:
    if not country or not country.strip():
        raise ValueError("'country' is required.")
    if year_end < year_start:
        raise ValueError("'year_end' must be >= 'year_start'.")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("'threshold' must be between 0 and 1.")
    if not gcs_bucket or not gcs_bucket.strip():
        raise ValueError("'gcs_bucket' is required.")

def run_export(
    country: str,
    map_year: int = 2019,
    sample_points: int = 25,
    sample_scale: int = 30,
    embedding_scale: int = 10,
    threshold: float = 0.75,
    trees: int = 10,
    seed: int = 42,
    project: Optional[str] = None,
    year_start: int = 2020,
    year_end: int = 2021,
    export_name: Optional[str] = None,
    gcs_bucket: str = "",
    gcs_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the classification pipeline and start a GCS export task."""
    _validate_inputs(country, year_start, year_end, threshold, gcs_bucket)
    initialize_ee(project=project)

    # Datasets
    countries = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
    wsf2019 = ee.ImageCollection("projects/sat-io/open-datasets/WSF/WSF_2019")
    embeddings = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    # Country boundary and bbox
    boundary = countries.filter(ee.Filter.eq("ADM0_NAME", country))
    boundary_size = boundary.size().getInfo()
    if boundary_size == 0:
        raise ValueError(f'No country found with ADM0_NAME="{country}"')

    bbox = boundary.bounds()

    # Ground truth from WSF 2019
    filtered_wsf = wsf2019.filterBounds(bbox).mosaic().gt(0).unmask()

    sample = (
        filtered_wsf.stratifiedSample(
            numPoints=sample_points,
            region=bbox,
            scale=sample_scale,
            geometries=True,
        )
        .filterBounds(boundary)
    )

    # Join labels with embedding vectors
    map_year_embeddings = embeddings_by_year(embeddings, boundary, map_year)
    labels_and_vectors = map_year_embeddings.sampleRegions(
        collection=sample,
        properties=["b1"],
        scale=embedding_scale,
    )

    # Train/validation split
    collection_with_random = labels_and_vectors.randomColumn(columnName="random", seed=seed)
    training_set = collection_with_random.filter(ee.Filter.lt("random", 0.7))
    validation_set = collection_with_random.filter(ee.Filter.gte("random", 0.7))

    binary_filter = ee.Filter.inList("b1", [0, 1])
    filtered_collection = training_set.filter(binary_filter)

    # Use selected map-year embedding band names as model inputs
    input_properties = map_year_embeddings.bandNames()

    classifier = (
        ee.Classifier.smileRandomForest(numberOfTrees=trees)
        .train(
            features=filtered_collection,
            classProperty="b1",
            inputProperties=input_properties,
        )
        .setOutputMode("PROBABILITY")
    )

    # Multi-year stack
    year_list = list(range(year_start, year_end + 1))
    year_names = [f"Y{year}" for year in year_list]
    yearly_images = [
        embeddings_by_year(embeddings, boundary, year).classify(classifier)
        for year in year_list
    ]
    all_year_results = ee.ImageCollection(yearly_images).toBands().rename(year_names)
    output_image = all_year_results.gte(threshold).toByte()

    safe_country = country.lower().replace(" ", "_").replace("-", "_")
    default_export_name = f"urban_extent_{safe_country}_{year_start}-{year_end}"
    final_export_name = export_name or default_export_name
    normalized_prefix = (gcs_prefix or "").strip().strip("/")
    file_name_prefix = f"{normalized_prefix}/{final_export_name}" if normalized_prefix else final_export_name

    result: Dict[str, Any] = {
        "country": country,
        "project": project,
        "training_year": map_year,
        "year_start": year_start,
        "year_end": year_end,
        "threshold": threshold,
        "export_name": final_export_name,
        "export_started": False,
        "export_target": "gcs",
        "gcs_bucket": gcs_bucket,
        "gcs_prefix": normalized_prefix or None,
        "file_name_prefix": file_name_prefix,
    }

    # Optional diagnostics (best-effort)
    try:
        result["training_samples"] = training_set.size().getInfo()
        result["validation_samples"] = validation_set.size().getInfo()
        result["filtered_training_samples"] = filtered_collection.size().getInfo()
    except Exception:
        pass

    task = ee.batch.Export.image.toCloudStorage(
        image=output_image.clip(boundary),
        description=final_export_name,
        bucket=gcs_bucket,
        fileNamePrefix=file_name_prefix,
        region=bbox,
        scale=10,
        maxPixels=1e13,
    )
    task.start()
    status = task.status()

    result.update(
        {
            "export_started": True,
            "task_id": status.get("id"),
            "task_state": status.get("state"),
            "task_description": status.get("description"),
        }
    )

    return result

def get_task_status(task_id: str, project: Optional[str] = None) -> Dict[str, Any]:
    """Return Earth Engine task status by task id."""
    if not task_id:
        raise ValueError("task_id is required")

    initialize_ee(project=project)
    status_list = ee.data.getTaskStatus([task_id])
    if not status_list:
        raise ValueError(f"No Earth Engine task found for id '{task_id}'")
    return status_list[0]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an urban extent classifier for a country and optionally start an EE export task."
    )
    parser.add_argument("country", type=str, help='Country name matching GAUL ADM0_NAME, e.g. "Colombia"')
    parser.add_argument("--map-year", type=int, default=2019, help="Reference year used for training. Default: 2019")
    parser.add_argument("--sample-points", type=int, default=25, help="Number of stratified sample points. Default: 25")
    parser.add_argument("--sample-scale", type=int, default=30, help="Scale for sampling WSF labels. Default: 30")
    parser.add_argument("--embedding-scale", type=int, default=10, help="Scale for sampling embeddings. Default: 10")
    parser.add_argument("--threshold", type=float, default=0.75, help="Probability threshold (0-1). Default: 0.75")
    parser.add_argument("--trees", type=int, default=10, help="Number of trees for Random Forest. Default: 10")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/validation split. Default: 42")
    parser.add_argument("--project", type=str, default=None, help="Optional GCP project for ee.Initialize(project=...)" )
    parser.add_argument("--year-start", type=int, default=2020, help="Start year for output stack. Default: 2020")
    parser.add_argument("--year-end", type=int, default=2021, help="End year for output stack. Default: 2021")
    parser.add_argument("--export-name", type=str, default=None, help="Optional export name. Default: auto-generated")
    parser.add_argument("--gcs-bucket", type=str, required=True, help="Required Google Cloud Storage bucket name")
    parser.add_argument("--gcs-prefix", type=str, default=None, help="Optional GCS prefix/folder path")
    return parser

def main() -> None:
    args = build_parser().parse_args()
    result = run_export(
        country=args.country,
        map_year=args.map_year,
        sample_points=args.sample_points,
        sample_scale=args.sample_scale,
        embedding_scale=args.embedding_scale,
        threshold=args.threshold,
        trees=args.trees,
        seed=args.seed,
        project=args.project,
        year_start=args.year_start,
        year_end=args.year_end,
        export_name=args.export_name,
        gcs_bucket=args.gcs_bucket,
        gcs_prefix=args.gcs_prefix,
    )
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)