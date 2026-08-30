"""Common Google Earth Engine utilities shared across all indicators."""

# NOTE: Keep this module free of indicator-specific logic.

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


def _validate_inputs(
    country: Optional[str],
    year_start: int,
    year_end: int,
    threshold: float,
    gcs_bucket: str,
) -> None:
    if country is not None and not country.strip():
        raise ValueError("'country' must not be blank when provided.")
    if year_end < year_start:
        raise ValueError("'year_end' must be >= 'year_start'.")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("'threshold' must be between 0 and 1.")
    if not gcs_bucket or not gcs_bucket.strip():
        raise ValueError("'gcs_bucket' is required.")


def get_task_status(task_id: str, project: Optional[str] = None) -> Dict[str, Any]:
    """Return Earth Engine task status by task id."""
    if not task_id:
        raise ValueError("task_id is required")

    initialize_ee(project=project)
    status_list = ee.data.getTaskStatus([task_id])
    if not status_list:
        raise ValueError(f"No Earth Engine task found for id '{task_id}'")
    return status_list[0]


def list_saved_models(project_path: str, prefix_filter: Optional[str] = None) -> list[str]:
    """Retrieve saved Earth Engine classifier assets from a given project path."""
    initialize_ee()
    try:
        models = []
        page_token = None
        while True:
            params = {'parent': project_path}
            if page_token:
                params['pageToken'] = page_token
                
            assets = ee.data.listAssets(params)
            
            for asset in assets.get('assets', []):
                asset_type = asset.get('type', '')
                asset_id = asset.get('name', asset.get('id', ''))
                if asset_type == 'CLASSIFIER' or '_classifier' in asset_id:
                    if prefix_filter and prefix_filter not in asset_id:
                        continue
                    models.append(asset_id)
                    
            page_token = assets.get('nextPageToken')
            if not page_token:
                break
                
        return models
    except Exception as exc:
        print(f"Warning: Failed to list models for {project_path}: {exc}")
        return []




def export_table_to_gcs(
    collection: "ee.FeatureCollection",
    description: str,
    bucket: str,
    filename_prefix: str,
    file_format: str = "CSV",
) -> "ee.batch.Task":
    """Wrap ee.batch.Export.table.toCloudStorage and start the task.

    Returns the started task so callers can query ``task.status()``.
    """
    task = ee.batch.Export.table.toCloudStorage(
        collection=collection,
        description=description,
        bucket=bucket,
        fileNamePrefix=filename_prefix,
        fileFormat=file_format,
    )
    task.start()
    return task


def export_image_to_gcs(
    image: "ee.Image",
    description: str,
    bucket: str,
    filename_prefix: str,
    scale: int,
    region: Optional["ee.Geometry"] = None,
    crs: str = "EPSG:4326",
) -> "ee.batch.Task":
    """Export an ee.Image to Google Cloud Storage as a GeoTIFF.

    Args:
        image: The Earth Engine image to export.
        description: Human-readable task description (used as EE task name).
        bucket: Destination GCS bucket name.
        filename_prefix: GCS object prefix / filename (without extension).
        scale: Pixel resolution in metres.
        region: Optional export geometry. If ``None`` the image footprint is used.
        crs: Coordinate reference system (default ``"EPSG:4326"``).

    Returns:
        The started ``ee.batch.Task`` object so callers can monitor
        ``task.status()``.
    """
    kwargs: Dict[str, Any] = dict(
        image=image,
        description=description,
        bucket=bucket,
        fileNamePrefix=filename_prefix,
        scale=scale,
        crs=crs,
        fileFormat="GeoTIFF",
        formatOptions={'cloudOptimized': True},
        maxPixels=1e13
    )
    if region is not None:
        kwargs["region"] = region

    task = ee.batch.Export.image.toCloudStorage(**kwargs)
    task.start()
    return task


def export_vector_to_gcs(
    collection: "ee.FeatureCollection",
    description: str,
    bucket: str,
    filename_prefix: str,
    file_format: str = "GeoJSON",
) -> "ee.batch.Task":
    """Export an ee.FeatureCollection to Google Cloud Storage as GeoJSON.

    Args:
        collection: The Earth Engine feature collection to export.
        description: Human-readable task description (used as EE task name).
        bucket: Destination GCS bucket name.
        filename_prefix: GCS object prefix / filename (without extension).
        file_format: Export format passed to EE (default ``"GeoJSON"``).

    Returns:
        The started ``ee.batch.Task`` object so callers can monitor
        ``task.status()``.
    """
    task = ee.batch.Export.table.toCloudStorage(
        collection=collection,
        description=description,
        bucket=bucket,
        fileNamePrefix=filename_prefix,
        fileFormat=file_format,
    )
    task.start()
    return task


def aggregate_regional_stats(
    image: "ee.Image",
    geometry: "ee.Geometry",
    scale: int,
    reducer: "ee.Reducer" = None,
    max_pixels: float = 1e13,
) -> "ee.Number":
    """Run reduceRegion and return the first (or only) result as an ee.Number.

    Args:
        image: The image whose pixels are reduced. Should have exactly one band
               unless the reducer produces a single output key.
        geometry: The region over which to reduce.
        scale: Nominal scale in metres.
        reducer: Defaults to ee.Reducer.sum().
        max_pixels: Safety cap on pixel count (default 1e13).

    Returns:
        The reduced value as an ``ee.Number``.
    """
    if reducer is None:
        reducer = ee.Reducer.sum()

    result_dict = image.reduceRegion(
        reducer=reducer,
        geometry=geometry,
        scale=scale,
        maxPixels=max_pixels,
        tileScale=16,
    )
    # Return the first value from the dictionary (works for single-band images).
    key = image.bandNames().get(0)
    return ee.Number(result_dict.get(key))


def get_missing_years(collection: "ee.ImageCollection", years: list[int], boundary: "ee.Geometry") -> list[int]:
    """Check an Earth Engine ImageCollection and return a list of years that have no imagery.

    Args:
        collection: The Earth Engine ImageCollection to check (assumes annual/continuous data).
        years: A list of years to verify.
        boundary: The geometry to filter bounds against.

    Returns:
        A list of integers representing the years that have no matching images.
    """
    required_years = list(set(years))
    if not required_years:
        return []

    def check_year(y):
        y_num = ee.Number(y)
        d_start = ee.Date.fromYMD(y_num, 1, 1)
        d_end = d_start.advance(1, "year")
        count = collection.filterDate(d_start, d_end).filterBounds(boundary).size()
        return ee.Feature(None, {"year": y_num, "count": count})

    counts_fc = ee.FeatureCollection(ee.List(required_years).map(check_year))
    
    try:
        counts_list = counts_fc.reduceColumns(ee.Reducer.toList(2), ["year", "count"]).get("list").getInfo()
        return [int(item[0]) for item in counts_list if item[1] == 0]
    except Exception as exc:
        print(f"Warning: Failed to validate missing years: {exc}")
        # If the check fails for EE reasons, we return empty to avoid blocking the pipeline erroneously
        return []
