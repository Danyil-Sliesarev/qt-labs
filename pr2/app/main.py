"""FastAPI application for local YOLO object detection."""

from __future__ import annotations

import asyncio
import logging
import warnings
from contextlib import asynccontextmanager
from io import BytesIO
from math import isfinite
from pathlib import Path
from typing import Annotated, Any, Callable

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.inference import DEFAULT_CONFIDENCE, detect, load_model

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
SUPPORTED_MEDIA_TYPES = {
    "image/jpeg": {"JPEG"},
    "image/png": {"PNG"},
    "image/webp": {"WEBP"},
    "image/gif": {"GIF"},
    "image/bmp": {"BMP"},
}
BASE_DIR = Path(__file__).resolve().parent


def _decode_image(content: bytes, content_type: str) -> Image.Image:
    """Validate and decode an upload into an RGB image."""
    if content_type not in SUPPORTED_MEDIA_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image media type.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                if image.format not in SUPPORTED_MEDIA_TYPES[content_type]:
                    raise HTTPException(
                        status_code=415,
                        detail="The image content does not match its media type.",
                    )
                width, height = image.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=413,
                        detail="Image dimensions exceed the 25-megapixel limit.",
                    )
                image.load()
                return image.convert("RGB").copy()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(status_code=400, detail="The uploaded image is invalid.") from error
    except Image.DecompressionBombWarning as error:
        raise HTTPException(status_code=413, detail="Image dimensions are too large.") from error


def create_app(model_loader: Callable[[], Any] = load_model) -> FastAPI:
    """Create the application; the injected loader is called once at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.inference_lock = asyncio.Lock()
        try:
            app.state.model = model_loader()
        except Exception:
            app.state.model = None
            logger.exception("YOLO model initialization failed")
        yield

    app = FastAPI(title="Local AI Object Detection", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=BASE_DIR / "templates")

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "index.html")

    @app.get("/health")
    async def health(request: Request):
        ready = getattr(request.app.state, "model", None) is not None
        if not ready:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "model_ready": False, "inference": "local"},
            )
        return {"status": "ok", "model_ready": True, "inference": "local"}

    @app.post("/detect")
    async def detect_image(
        request: Request,
        file: Annotated[UploadFile, File(...)],
        confidence: Annotated[float, Form(ge=0.0, le=1.0)] = DEFAULT_CONFIDENCE,
    ):
        if not isfinite(confidence):
            raise HTTPException(status_code=422, detail="Confidence must be a finite number.")

        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")
        image = _decode_image(content, file.content_type or "")

        model = getattr(request.app.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail="The local YOLO model is not ready.")
        try:
            async with request.app.state.inference_lock:
                return await run_in_threadpool(detect, model, image, confidence)
        except Exception as error:
            logger.exception("Local YOLO inference failed")
            raise HTTPException(status_code=500, detail="Object detection failed.") from error

    return app


app = create_app()
