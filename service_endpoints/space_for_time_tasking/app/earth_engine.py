from __future__ import annotations

from typing import Any, Optional

import ee


def start_table_ingestion(
    gcs_uri: str,
    asset_id: str,
    file_format: str,
    description: Optional[str] = None,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
) -> dict[str, Any]:
    normalized_format = file_format.upper()
    if normalized_format not in {"CSV", "GEOJSON", "GEO_JSON", "JSON", "SHP", "SHAPEFILE", "ZIPPED_SHAPEFILE"}:
        raise ValueError(f"Unsupported table ingestion format: {file_format}")

    request_id = ee.data.newTaskId()[0]
    manifest = {
        "name": asset_id,
        "sources": [{"uris": [gcs_uri]}],
        "properties": {"description": description or "space-for-time tasking input table"},
    }
    if normalized_format == "CSV":
        if bool(x_column) != bool(y_column):
            raise ValueError("CSV table ingestion requires both x_column and y_column when either is provided.")
        if x_column and y_column:
            manifest["sources"][0]["x_column"] = x_column
            manifest["sources"][0]["y_column"] = y_column

    operation = ee.data.startTableIngestion(request_id, manifest)
    return {"task_id": request_id, "asset_id": asset_id, "operation": operation}
