"""Check the Python environment and run one local YOLO inference."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REQUIRED_MODULES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "ultralytics": "ultralytics",
    "Pillow": "PIL",
    "python-multipart": "multipart",
    "Jinja2": "jinja2",
    "pytest": "pytest",
    "httpx": "httpx",
    "PyTorch": "torch",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="optional image to use for the inference check")
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("Python 3.10 or newer is required.")
        return 1

    missing = []
    for package, module in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module)
            print(f"{package}: installed")
        except ImportError:
            print(f"{package}: MISSING")
            missing.append(package)
    if missing:
        print("Install the dependencies with: python -m pip install -r requirements.txt")
        return 1

    import torch
    from PIL import Image

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}")

    from app.inference import detect, load_model

    image = Image.open(args.image).convert("RGB") if args.image else Image.new("RGB", (640, 480), "white")
    model = load_model()
    result = detect(model, image, 0.25)
    print(
        "Inference succeeded: "
        f"{result['count']} detections in {result['inference_time_ms']:.2f} ms "
        f"(threshold {result['confidence_threshold']:.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
