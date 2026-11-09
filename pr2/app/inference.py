"""Local YOLO model loading and object detection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent.parent / "yolov8n.pt"
MODEL_NAME = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25


def load_model() -> YOLO:
    """Load pretrained YOLO weights once, using CUDA when it is available."""
    model = YOLO(str(MODEL_PATH if MODEL_PATH.exists() else MODEL_NAME))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model


def detect(
    model: YOLO,
    image: Image.Image,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Run local inference and return JSON-compatible detections and timing."""
    with torch.inference_mode():
        started_at = time.perf_counter()
        results = model.predict(
            source=image,
            conf=confidence_threshold,
            verbose=False,
        )
        inference_time_ms = (time.perf_counter() - started_at) * 1000

    detections: list[dict[str, Any]] = []
    if results:
        result = results[0]
        boxes = result.boxes
        if boxes is not None:
            names = result.names
            for box in boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                if confidence < confidence_threshold:
                    continue
                xyxy = box.xyxy[0].tolist()
                class_name = names[class_id]
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": str(class_name),
                        "confidence": round(confidence, 6),
                        "bbox": {
                            "x1": int(round(float(xyxy[0]))),
                            "y1": int(round(float(xyxy[1]))),
                            "x2": int(round(float(xyxy[2]))),
                            "y2": int(round(float(xyxy[3]))),
                        },
                    }
                )

    return {
        "detections": detections,
        "count": len(detections),
        "inference_time_ms": round(inference_time_ms, 3),
        "confidence_threshold": confidence_threshold,
    }
