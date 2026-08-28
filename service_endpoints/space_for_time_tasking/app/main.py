from threading import Lock
from typing import Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.endpoint_functions import run_tasking
from app.routers import inputs


app = FastAPI(
    title="Space-for-time Model Creation API",
    description="API for uploading reference data, ingesting it to Earth Engine, and creating Earth Engine model assets from space-for-time tasking experiments.",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)
app.include_router(inputs.router)

# Earth Engine's Python client stores credentials globally. Serialize requests
# so credentials from two users cannot overlap inside one container instance.
ee_lock = Lock()


class TaskingRunRequest(BaseModel):
    input_asset_id: str
    cloud_project: str
    run_name: Optional[str] = None

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

@app.post("/tasking/run")
def tasking_run_endpoint(request: TaskingRunRequest):
    try:
        with ee_lock:
            return run_tasking(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health_check():
    return {"status": "healthy"}
