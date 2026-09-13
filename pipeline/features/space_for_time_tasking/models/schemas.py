from __future__ import annotations
from enum import Enum
from typing import Any, Optional, Union
from pydantic import BaseModel, Field

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

class TaskingRunRequest(BaseModel):
    input_asset_id: Optional[str] = None
    csv_url: Optional[str] = None
    cloud_project: str
    run_name: Optional[str] = None
    aoi_name: Optional[str] = None

    longitude_column: str = "lon"
    latitude_column: str = "lat"
    block_x_column: Optional[str] = None
    block_y_column: Optional[str] = None
    block_crs: str = "EPSG:6933"
    target_column: str = "LOI_PCT"
    target_threshold: float = 30.0
    reference_year: int = 2018

    block_size_m: int = 10_000
    test_block_fraction: float = 0.20
    points_per_block: list[Union[int, str]] = Field(
        default_factory=lambda: [1, 2, 5, 10, 20, "all"]
    )
    block_fractions: list[float] = Field(
        default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 1.00]
    )
    auc_tolerance: float = 0.01
    number_of_trees: int = 100
    number_of_embedding_bands: int = 64
    sampling_scale_m: int = 10
    seed: int = 42

    asset_root: Optional[str] = None
    sample_asset_id: Optional[str] = None
    model_asset_id: Optional[str] = None
    results_bucket: Optional[str] = None
    results_prefix: Optional[str] = None
