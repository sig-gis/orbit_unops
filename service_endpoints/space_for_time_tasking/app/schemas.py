from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class TableFileFormat(str, Enum):
    csv = "CSV"
    geo_json = "GEO_JSON"
    json = "JSON"
    shp = "SHP"


class UploadResponse(BaseModel):
    gcs_uri: str
    https_url: str
    bucket: str
    object_name: str
    filename: str
    content_type: Optional[str] = None


class IngestInputRequest(BaseModel):
    gcs_uri: str
    asset_id: str
    file_format: TableFileFormat
    cloud_project: str
    description: Optional[str] = None
    x_column: Optional[str] = None
    y_column: Optional[str] = None


class IngestInputResponse(BaseModel):
    task_id: str
    asset_id: str
    operation: dict[str, Any]
