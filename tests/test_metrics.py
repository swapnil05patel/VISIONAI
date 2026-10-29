from evaluation.mot_io import xyxy_to_mot_row
from evaluation.mot_metrics import compute_all_metrics


def test_perfect_identity_scores():
    gt = [
        xyxy_to_mot_row(1, 1, [10, 10, 40, 80]),
        xyxy_to_mot_row(2, 1, [12, 10, 42, 80]),
        xyxy_to_mot_row(1, 2, [100, 20, 140, 90]),
        xyxy_to_mot_row(2, 2, [102, 20, 142, 90]),
    ]
    pred = [dict(row) for row in gt]
    metrics = compute_all_metrics(gt, pred)
    assert metrics["MOTA"] == 1.0
    assert metrics["IDF1"] == 1.0
    assert metrics["IDSW"] == 0
    assert metrics["HOTA"] > 0.99


def test_id_switch_is_counted():
    gt = [
        xyxy_to_mot_row(1, 1, [10, 10, 50, 90]),
        xyxy_to_mot_row(2, 1, [12, 10, 52, 90]),
    ]
    pred = [
        xyxy_to_mot_row(1, 7, [10, 10, 50, 90]),
        xyxy_to_mot_row(2, 8, [12, 10, 52, 90]),
    ]
    metrics = compute_all_metrics(gt, pred)
    assert metrics["IDSW"] == 1
    assert metrics["MOTA"] < 1.0
