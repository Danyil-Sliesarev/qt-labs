from eval.evaluate import analyze_image, box_iou


def box(x1, y1, x2, y2):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def detection(name, confidence, coordinates):
    return {
        "class_name": name,
        "confidence": confidence,
        "bbox": coordinates,
    }


def test_box_iou_for_identical_boxes():
    assert box_iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) == 1


def test_evaluation_distinguishes_misses_false_positives_and_class_errors():
    expected = {
        "expected_objects": [
            {"class_name": "dog", "bbox": box(0, 0, 20, 20)},
            {"class_name": "cat", "bbox": box(30, 30, 50, 50)},
        ]
    }
    predictions = [
        detection("bear", 0.9, box(0, 0, 20, 20)),
        detection("person", 0.8, box(70, 70, 90, 90)),
    ]

    report = analyze_image(expected, predictions)

    assert [item["class_name"] for item in report["missed"]] == ["dog", "cat"]
    assert report["misclassifications"][0]["predicted"] == "bear"
    assert report["false_positives"][0]["class_name"] == "person"


def test_evaluation_marks_second_match_to_same_object_as_duplicate():
    expected = {"expected_objects": [{"class_name": "cat", "bbox": box(0, 0, 20, 20)}]}
    predictions = [
        detection("cat", 0.9, box(0, 0, 20, 20)),
        detection("cat", 0.8, box(1, 1, 19, 19)),
    ]
    report = analyze_image(expected, predictions)
    assert len(report["true_positives"]) == 1
    assert len(report["duplicates"]) == 1
