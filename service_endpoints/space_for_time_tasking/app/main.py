from threading import Lock
from typing import Optional, Union

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.endpoint_functions import run_tasking


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Earth Engine's Python client stores credentials globally. Serialize requests
# so credentials from two users cannot overlap inside one container instance.
ee_lock = Lock()


class TaskingRequest(BaseModel):
    csv_url: str
    cloud_project: str
    run_name: Optional[str] = None

    longitude_column: str = "lon"
    latitude_column: str = "lat"
    block_x_column: str = "X_ITM"
    block_y_column: str = "Y_ITM"
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


@app.post("/task")
def tasking_endpoint(
    request: TaskingRequest,
    authorization: Optional[str] = Header(default=None),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")

    access_token = authorization.removeprefix("Bearer ").strip()
    if not access_token:
        raise HTTPException(status_code=401, detail="Bearer token required")

    try:
        with ee_lock:
            return run_tasking(request.model_dump(), access_token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health_check():
    return {"status": "healthy"}
