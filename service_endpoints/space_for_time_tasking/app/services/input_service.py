from __future__ import annotations

from app.earth_engine import start_table_ingestion
from app.schemas import IngestInputRequest


def ingest_input(request: IngestInputRequest):
    return start_table_ingestion(
        gcs_uri=request.gcs_uri,
        asset_id=request.asset_id,
        file_format=request.file_format.value,
        description=request.description,
        x_column=request.x_column,
        y_column=request.y_column,
    )
