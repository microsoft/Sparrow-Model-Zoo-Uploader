from sparrow_uploader.parity import compare_classification, compare_detections, norm_label


def test_norm_label_ignores_case_and_separators():
    assert norm_label("red_deer") == norm_label("Red deer") == norm_label(" red-deer ")
    assert norm_label("red deer") != norm_label("roe deer")


def test_classification_matches_across_label_spelling():
    cmp, errs = compare_classification({"red_deer": 0.9, "roe_deer": 0.1}, {"Red deer": 0.9, "roe deer": 0.1})
    assert not errs and cmp["top1_match"] and cmp["max_prob_delta"] == 0


def test_classification_keeps_labels_that_would_collide():
    cmp, _ = compare_classification({"a_b": 0.6, "a b": 0.4}, {"a_b": 0.6, "a b": 0.4})
    assert cmp["max_prob_delta"] == 0 and cmp["top1_reference"] == "a_b"


def test_detections_match_across_label_spelling():
    box = [0.1, 0.1, 0.5, 0.5]
    out = compare_detections(
        [{"label": "Animal", "confidence": 0.9, "bbox": box}],
        [{"label": "animal", "confidence": 0.9, "bbox": box}],
        threshold=0.2,
        boundary=0.05,
    )
    assert out["matched"] == 1 and not out["unmatched_blocking"]
