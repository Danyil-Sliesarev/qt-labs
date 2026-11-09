"""Compare YOLO predictions with manually verified class-count annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from app.inference import detect, load_model

THRESHOLDS = (0.25, 0.50, 0.75)
EVAL_DIR = Path(__file__).resolve().parent


def box_iou(left: dict[str, int], right: dict[str, int]) -> float:
    x1 = max(left["x1"], right["x1"])
    y1 = max(left["y1"], right["y1"])
    x2 = min(left["x2"], right["x2"])
    y2 = min(left["y2"], right["y2"])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, left["x2"] - left["x1"]) * max(0, left["y2"] - left["y1"])
    right_area = max(0, right["x2"] - right["x1"]) * max(0, right["y2"] - right["y1"])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def analyze_image(
    expected: dict,
    detections: list[dict],
    iou_threshold: float = 0.5,
) -> dict[str, list[dict]]:
    ground_truth = expected["expected_objects"]
    matched_truth: set[int] = set()
    true_positives = []
    false_positives = []
    misclassifications = []
    duplicates = []

    ordered_detections = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    for detection in ordered_detections:
        overlaps = [
            (index, box_iou(detection["bbox"], target["bbox"]))
            for index, target in enumerate(ground_truth)
        ]
        relevant = [
            (index, iou)
            for index, iou in overlaps
            if iou >= iou_threshold
        ]
        matching = [
            (index, iou)
            for index, iou in relevant
            if ground_truth[index]["class_name"] == detection["class_name"]
            and index not in matched_truth
        ]
        if matching:
            best_index, _ = max(matching, key=lambda pair: pair[1])
            matched_truth.add(best_index)
            true_positives.append(detection)
            continue

        unmatched = [
            (index, iou)
            for index, iou in relevant
            if index not in matched_truth
        ]
        if unmatched:
            best_index, best_iou = max(unmatched, key=lambda pair: pair[1])
            target = ground_truth[best_index]
            misclassifications.append(
                {
                    "expected": target["class_name"],
                    "predicted": detection["class_name"],
                    "confidence": detection["confidence"],
                    "iou": round(best_iou, 3),
                }
            )
            continue

        duplicate = [
            (index, iou)
            for index, iou in relevant
            if index in matched_truth
            and ground_truth[index]["class_name"] == detection["class_name"]
        ]
        if duplicate:
            duplicates.append(detection)
        else:
            false_positives.append(detection)

    missed = [
        item
        for index, item in enumerate(ground_truth)
        if index not in matched_truth
    ]
    return {
        "true_positives": true_positives,
        "missed": missed,
        "false_positives": false_positives,
        "misclassifications": misclassifications,
        "duplicates": duplicates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=EVAL_DIR / "expected.json")
    args = parser.parse_args()
    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    if data.get("status") != "complete" or len(data.get("images", [])) < 5:
        parser.error("expected.json must contain at least five real, manually annotated images.")

    model = load_model()
    for threshold in THRESHOLDS:
        print(f"\nConfidence threshold: {threshold:.2f}")
        for expected in data["images"]:
            image_path = EVAL_DIR / "images" / expected["file"]
            with Image.open(image_path) as source:
                result = detect(model, source.convert("RGB"), threshold)
            analysis = analyze_image(expected, result["detections"])
            print(
                f"{expected['file']}: {result['count']} predicted, "
                f"{len(analysis['true_positives'])} matched, "
                f"{result['inference_time_ms']:.1f} ms; "
                f"missed={len(analysis['missed'])}; "
                f"false positives={len(analysis['false_positives'])}; "
                f"misclassifications={len(analysis['misclassifications'])}; "
                f"duplicates={len(analysis['duplicates'])}"
            )
            for issue in analysis["misclassifications"]:
                print(
                    f"  class error: expected {issue['expected']}, "
                    f"predicted {issue['predicted']} (IoU {issue['iou']:.2f})"
                )
            for issue in analysis["false_positives"]:
                print(
                    f"  false positive: {issue['class_name']} "
                    f"({issue['confidence']:.3f})"
                )
            for issue in analysis["duplicates"]:
                print(
                    f"  duplicate: {issue['class_name']} "
                    f"({issue['confidence']:.3f})"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
