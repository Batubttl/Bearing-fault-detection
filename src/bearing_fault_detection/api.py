"""REST API: POST a 2048-sample vibration window, get the predicted bearing condition."""
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi.exceptions import RequestValidationError
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .inference import WINDOW_SIZE, BearingClassifier

MAX_BODY_BYTES = 256_000  # a 2048-float JSON body is ~40 KB; reject anything far larger

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class PredictRequest(BaseModel):
    signal: list[FiniteFloat] = Field(
        min_length=WINDOW_SIZE, max_length=WINDOW_SIZE,
        description=f"Drive-end acceleration, exactly {WINDOW_SIZE} samples at 12 kHz",
    )


class PredictResponse(BaseModel):
    class_id: int
    label: str
    probabilities: dict[str, float]


state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["clf"] = BearingClassifier()   # load the model once at startup, not per request
    yield
    state.clear()


app = FastAPI(title="Bearing Fault Detection API", version="0.1.0", lifespan=lifespan)
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # Don't echo the input back: it can be huge, and NaN/inf can't be encoded as JSON
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    length = request.headers.get("content-length")
    if length is not None and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
        return JSONResponse(status_code=413, content={"detail": "Request body too large"})
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return state["clf"].predict(np.array(req.signal))