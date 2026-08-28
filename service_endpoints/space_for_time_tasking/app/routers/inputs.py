from __future__ import annotations

import ee
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.gcs import upload_fileobj_to_gcs
from app.schemas import IngestInputRequest, IngestInputResponse, UploadResponse
from app.services.input_service import ingest_input

router = APIRouter(prefix="/inputs", tags=["inputs"])

ALLOWED_EXTENSIONS = {".csv", ".geojson", ".json", ".zip"}


def _extension(filename: str) -> str:
    lowered = filename.lower()
    for ext in ALLOWED_EXTENSIONS:
        if lowered.endswith(ext):
            return ext
    return ""


@router.post("/upload", response_model=UploadResponse)
def upload_input_file(
    file: UploadFile = File(...),
    bucket_name: str = Form(...),
    gcs_prefix: str = Form(default="space-for-time-tasking/inputs"),
):
    ext = _extension(file.filename or "")
    if not ext:
        raise HTTPException(status_code=400, detail="Unsupported input file. Use CSV, GeoJSON/JSON, or zipped shapefile.")
    uploaded = upload_fileobj_to_gcs(file.file, bucket_name, gcs_prefix, file.filename, file.content_type)
    return UploadResponse(filename=file.filename, **uploaded)


@router.post("/ingest", response_model=IngestInputResponse)
def ingest_input_table(request: IngestInputRequest):
    try:
        ee.Initialize(project=request.cloud_project)
        return ingest_input(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

