"""FastAPI application for online fraud scoring."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated, cast
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from fraud_detection import __version__
from fraud_detection.model import FraudModel, ModelArtifactError, load_model

MODEL_PATH_ENVIRONMENT_VARIABLE = "FRAUD_MODEL_PATH"
REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time-Ms"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Bounded batch of numeric transaction feature mappings."""

    model_config = ConfigDict(extra="forbid")

    transactions: Annotated[
        list[dict[str, float]],
        Field(min_length=1, max_length=1_000),
    ]


class PredictionResult(BaseModel):
    """Fraud score and thresholded decision for one transaction."""

    fraud_probability: float
    is_fraud: bool


class PredictionResponse(BaseModel):
    """Ordered batch prediction response."""

    model_version: str
    threshold: float
    predictions: list[PredictionResult]


class HealthResponse(BaseModel):
    """Readiness and loaded model information."""

    status: str
    service_version: str
    model_created_at: str
    feature_count: int
    threshold: float


def create_app(
    *,
    model: FraudModel | None = None,
    model_path: Path | str | None = None,
) -> FastAPI:
    """Create an application using an injected model or a trusted artifact path."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        loaded_model = model
        if loaded_model is None:
            configured_path = model_path or os.getenv(MODEL_PATH_ENVIRONMENT_VARIABLE)
            if configured_path is None:
                raise RuntimeError(
                    f"No model configured. Set {MODEL_PATH_ENVIRONMENT_VARIABLE} "
                    "or pass model_path to create_app()."
                )
            loaded_model = load_model(configured_path)
        application.state.model = loaded_model
        yield

    application = FastAPI(
        title="Credit Card Fraud Detection API",
        summary="Low-latency scoring with a validation-tuned fraud threshold.",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @application.middleware("http")
    async def request_context(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        supplied_request_id = request.headers.get(REQUEST_ID_HEADER)
        request_id = (
            supplied_request_id
            if supplied_request_id is not None
            and _REQUEST_ID_PATTERN.fullmatch(supplied_request_id)
            else uuid4().hex
        )
        request.state.request_id = request_id
        started_at = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (perf_counter() - started_at) * 1_000
            logger.exception(
                "request_failed method=%s path=%s duration_ms=%.3f request_id=%s",
                request.method,
                request.url.path,
                duration_ms,
                request_id,
            )
            raise

        duration_ms = (perf_counter() - started_at) * 1_000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[PROCESS_TIME_HEADER] = f"{duration_ms:.3f}"
        logger.info(
            "request_completed method=%s path=%s status_code=%d duration_ms=%.3f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
        return response

    @application.exception_handler(ModelArtifactError)
    async def model_artifact_error_handler(
        _request: Request,
        exception: ModelArtifactError,
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exception)})

    @application.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health(request: Request) -> HealthResponse:
        loaded = _model_from_request(request)
        return HealthResponse(
            status="ready",
            service_version=__version__,
            model_created_at=str(loaded.metadata["created_at"]),
            feature_count=len(loaded.feature_names),
            threshold=loaded.threshold,
        )

    @application.post(
        "/v1/predict",
        response_model=PredictionResponse,
        tags=["predictions"],
    )
    async def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        loaded = _model_from_request(request)
        frame = pd.DataFrame(payload.transactions)
        probabilities = loaded.predict_probabilities(frame)
        decisions = probabilities >= loaded.threshold
        results = [
            PredictionResult(fraud_probability=float(probability), is_fraud=bool(decision))
            for probability, decision in zip(probabilities, decisions, strict=True)
        ]
        return PredictionResponse(
            model_version=str(loaded.metadata["dataset_fingerprint"])[:12],
            threshold=loaded.threshold,
            predictions=results,
        )

    return application


def _model_from_request(request: Request) -> FraudModel:
    return cast(FraudModel, request.app.state.model)


def app_from_environment() -> FastAPI:
    """Uvicorn factory that resolves the model path during application startup."""
    return create_app()
