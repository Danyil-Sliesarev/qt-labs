from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

import app.inference as inference
from app.inference import detect
from app.main import create_app


def image_bytes(color: str = "white") -> bytes:
    image = Image.new("RGB", (32, 24), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def sample_result():
    return {
        "detections": [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.92,
                "bbox": {"x1": 1, "y1": 2, "x2": 20, "y2": 22},
            }
        ],
        "count": 1,
        "inference_time_ms": 12.5,
        "confidence_threshold": 0.25,
    }


def test_startup_loads_model_once_and_health_is_ready(monkeypatch):
    load_count = 0
    used_models = []

    def loader():
        nonlocal load_count
        load_count += 1
        return object()

    def fake_detect(model, *args):
        used_models.append(model)
        return sample_result()

    monkeypatch.setattr("app.main.detect", fake_detect)
    with TestClient(create_app(loader)) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "model_ready": True,
            "inference": "local",
        }
        for _ in range(2):
            response = client.post(
                "/detect",
                files={"file": ("sample.png", image_bytes(), "image/png")},
            )
            assert response.status_code == 200
    assert load_count == 1
    assert len(used_models) == 2
    assert used_models[0] is used_models[1]


def test_health_reports_model_initialization_failure():
    with TestClient(create_app(lambda: (_ for _ in ()).throw(RuntimeError("load failed")))) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["model_ready"] is False
    assert "load failed" not in response.text


def test_home_serves_interface_and_static_assets():
    with TestClient(create_app(lambda: object())) as client:
        page = client.get("/")
        stylesheet = client.get("/static/style.css")
    assert page.status_code == 200
    assert "Local AI Object Detection" in page.text
    assert stylesheet.status_code == 200


def test_valid_upload_returns_structured_detection(monkeypatch):
    monkeypatch.setattr("app.main.detect", lambda *args: sample_result())
    with TestClient(create_app(lambda: object())) as client:
        response = client.post(
            "/detect",
            files={"file": ("sample.png", image_bytes(), "image/png")},
            data={"confidence": "0.25"},
        )
    assert response.status_code == 200
    assert response.json() == sample_result()


def test_unsupported_media_type_returns_415():
    with TestClient(create_app(lambda: object())) as client:
        response = client.post(
            "/detect",
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )
    assert response.status_code == 415


def test_empty_file_returns_400():
    with TestClient(create_app(lambda: object())) as client:
        response = client.post(
            "/detect",
            files={"file": ("empty.png", b"", "image/png")},
        )
    assert response.status_code == 400


def test_corrupt_image_returns_400():
    with TestClient(create_app(lambda: object())) as client:
        response = client.post(
            "/detect",
            files={"file": ("broken.png", b"not a PNG", "image/png")},
        )
    assert response.status_code == 400


def test_oversized_upload_returns_413():
    from app.main import MAX_UPLOAD_BYTES

    with TestClient(create_app(lambda: object())) as client:
        response = client.post(
            "/detect",
            files={"file": ("large.png", b"x" * (MAX_UPLOAD_BYTES + 1), "image/png")},
        )
    assert response.status_code == 413


def test_oversized_image_dimensions_return_413(monkeypatch):
    monkeypatch.setattr("app.main.MAX_IMAGE_PIXELS", 100)
    with TestClient(create_app(lambda: object())) as client:
        response = client.post(
            "/detect",
            files={"file": ("sample.png", image_bytes(), "image/png")},
        )
    assert response.status_code == 413


def test_mismatched_image_media_type_returns_415():
    with TestClient(create_app(lambda: object())) as client:
        response = client.post(
            "/detect",
            files={"file": ("sample.jpg", image_bytes(), "image/jpeg")},
        )
    assert response.status_code == 415


def test_invalid_confidence_returns_422():
    with TestClient(create_app(lambda: object())) as client:
        response = client.post(
            "/detect",
            files={"file": ("sample.png", image_bytes(), "image/png")},
            data={"confidence": "1.1"},
        )
    assert response.status_code == 422


def test_empty_detections_are_reported(monkeypatch):
    result = {
        "detections": [],
        "count": 0,
        "inference_time_ms": 3.2,
        "confidence_threshold": 0.25,
    }
    monkeypatch.setattr("app.main.detect", lambda *args: result)
    with TestClient(create_app(lambda: object())) as client:
        response = client.post(
            "/detect",
            files={"file": ("sample.png", image_bytes(), "image/png")},
        )
    assert response.status_code == 200
    assert response.json() == result


def test_inference_failure_does_not_expose_exception(monkeypatch):
    def fail(*args):
        raise RuntimeError("private internal details")

    monkeypatch.setattr("app.main.detect", fail)
    with TestClient(create_app(lambda: object())) as client:
        response = client.post(
            "/detect",
            files={"file": ("sample.png", image_bytes(), "image/png")},
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "Object detection failed."}
    assert "private internal details" not in response.text


class FakeBox:
    def __init__(self, class_id: int, confidence: float, coordinates: list[float]):
        self.cls = [FakeScalar(class_id)]
        self.conf = [FakeScalar(confidence)]
        self.xyxy = [FakeCoordinates(coordinates)]


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeCoordinates(list):
    def tolist(self):
        return list(self)


class FakeModel:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [type("Result", (), {"boxes": self.boxes, "names": {0: "person"}})()]


def test_detect_filters_threshold_and_returns_python_values():
    result = detect(
        FakeModel(
            [
                FakeBox(0, 0.9, [1.4, 2.2, 20.8, 22.1]),
                FakeBox(0, 0.2, [0, 0, 5, 5]),
            ]
        ),
        Image.new("RGB", (32, 24)),
        0.25,
    )
    assert result["count"] == 1
    detection = result["detections"][0]
    assert detection == {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.9,
        "bbox": {"x1": 1, "y1": 2, "x2": 21, "y2": 22},
    }
    assert isinstance(result["inference_time_ms"], float)


def test_detect_passes_confidence_threshold_to_model():
    model = FakeModel([])
    detect(model, Image.new("RGB", (32, 24)), 0.6)
    assert len(model.calls) == 1
    assert model.calls[0]["conf"] == 0.6


def test_detect_returns_empty_list_when_model_finds_nothing():
    result = detect(FakeModel([]), Image.new("RGB", (32, 24)), 0.25)
    assert result["detections"] == []
    assert result["count"] == 0


def test_load_model_uses_local_weights_and_cpu_when_cuda_unavailable(monkeypatch, tmp_path):
    weights = tmp_path / "yolov8n.pt"
    weights.write_bytes(b"test weights")
    model = type("FakeYOLO", (), {"to": lambda self, device: setattr(self, "device", device)})()
    received_paths = []
    monkeypatch.setattr(inference, "MODEL_PATH", weights)
    monkeypatch.setattr(inference, "YOLO", lambda path: received_paths.append(path) or model)
    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: False)

    loaded = inference.load_model()

    assert loaded is model
    assert received_paths == [str(weights)]
    assert model.device == "cpu"
