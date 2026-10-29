"""
CLEAR MOT, IDF1, fragmentation, and HOTA computed from MOT-format rows.

These implementations follow the standard matching rules (IoU Hungarian
assignment per frame). They are intended for local evaluation when TrackEval
is not installed. If TrackEval is available, evaluate.py prefers it.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker.utils import iou


def _frame_range(gt_by_frame: dict, pred_by_frame: dict) -> range:
    frames = set(gt_by_frame) | set(pred_by_frame)
    if not frames:
        return range(0)
    return range(min(frames), max(frames) + 1)


def _match_frame(gt_rows: List[dict], pred_rows: List[dict], iou_thr: float) -> List[Tuple[int, int, float]]:
    if not gt_rows or not pred_rows:
        return []
    cost = np.ones((len(gt_rows), len(pred_rows)), dtype=np.float32)
    for i, gt in enumerate(gt_rows):
        for j, pred in enumerate(pred_rows):
            cost[i, j] = 1.0 - iou(gt["box"], pred["box"])
    rows, cols = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(rows, cols):
        score = 1.0 - float(cost[r, c])
        if score >= iou_thr:
            matches.append((r, c, score))
    return matches


def compute_clear_mot(
    gt_rows: List[dict],
    pred_rows: List[dict],
    iou_thr: float = 0.5,
) -> dict:
    from evaluation.mot_io import index_by_frame

    gt_by_frame = index_by_frame(gt_rows)
    pred_by_frame = index_by_frame(pred_rows)

    last_pred_for_gt: Dict[int, int] = {}
    gt_seen_match: Dict[int, bool] = defaultdict(bool)
    idsw = 0
    frag = 0
    tp = 0
    fp = 0
    fn = 0
    num_gt = 0

    for frame in _frame_range(gt_by_frame, pred_by_frame):
        gts = gt_by_frame.get(frame, [])
        preds = pred_by_frame.get(frame, [])
        num_gt += len(gts)
        matches = _match_frame(gts, preds, iou_thr)
        matched_gt = {r for r, _, _ in matches}
        matched_pred = {c for _, c, _ in matches}
        tp += len(matches)
        fn += len(gts) - len(matched_gt)
        fp += len(preds) - len(matched_pred)

        currently_matched = set()
        for r, c, _ in matches:
            gt_id = gts[r]["id"]
            pred_id = preds[c]["id"]
            currently_matched.add(gt_id)
            prev = last_pred_for_gt.get(gt_id)
            if prev is not None and prev != pred_id:
                idsw += 1
            if prev is None and gt_seen_match[gt_id]:
                frag += 1
            last_pred_for_gt[gt_id] = pred_id
            gt_seen_match[gt_id] = True

        for gt_id in list(last_pred_for_gt):
            if gt_id not in currently_matched:
                last_pred_for_gt.pop(gt_id, None)

    mota = 1.0 - (fn + fp + idsw) / num_gt if num_gt else 0.0
    motp = None
    return {
        "MOTA": mota,
        "IDSW": idsw,
        "FRAG": frag,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "GT": num_gt,
        "MOTP": motp,
    }


def compute_idf1(gt_rows: List[dict], pred_rows: List[dict], iou_thr: float = 0.5) -> dict:
    from evaluation.mot_io import index_by_frame

    gt_by_frame = index_by_frame(gt_rows)
    pred_by_frame = index_by_frame(pred_rows)

    pair_hits = defaultdict(int)
    gt_len = defaultdict(int)
    pred_len = defaultdict(int)

    for row in gt_rows:
        gt_len[row["id"]] += 1
    for row in pred_rows:
        pred_len[row["id"]] += 1

    for frame in _frame_range(gt_by_frame, pred_by_frame):
        gts = gt_by_frame.get(frame, [])
        preds = pred_by_frame.get(frame, [])
        for r, c, _ in _match_frame(gts, preds, iou_thr):
            pair_hits[(gts[r]["id"], preds[c]["id"])] += 1

    gt_ids = list(gt_len)
    pred_ids = list(pred_len)
    if not gt_ids or not pred_ids:
        return {"IDF1": 0.0, "IDTP": 0, "IDFP": sum(pred_len.values()), "IDFN": sum(gt_len.values())}

    cost = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for i, gid in enumerate(gt_ids):
        for j, pid in enumerate(pred_ids):
            cost[i, j] = -float(pair_hits.get((gid, pid), 0))
    rows, cols = linear_sum_assignment(cost)
    assigned_gt = set()
    assigned_pred = set()
    idtp = 0
    for r, c in zip(rows, cols):
        hits = pair_hits.get((gt_ids[r], pred_ids[c]), 0)
        if hits <= 0:
            continue
        idtp += hits
        assigned_gt.add(gt_ids[r])
        assigned_pred.add(pred_ids[c])

    idfn = sum(gt_len[g] for g in gt_ids) - idtp
    idfp = sum(pred_len[p] for p in pred_ids) - idtp
    idf1 = (2 * idtp) / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else 0.0
    return {"IDF1": idf1, "IDTP": idtp, "IDFP": idfp, "IDFN": idfn}


def compute_hota(gt_rows: List[dict], pred_rows: List[dict]) -> dict:
    """HOTA averaged over IoU alphas 0.05..0.95 as in Luiten et al."""
    from evaluation.mot_io import index_by_frame

    gt_by_frame = index_by_frame(gt_rows)
    pred_by_frame = index_by_frame(pred_rows)
    alphas = [i / 100.0 for i in range(5, 100, 5)]
    hota_vals = []

    for alpha in alphas:
        tp = 0
        fp = 0
        fn = 0
        pair_tp = defaultdict(int)
        gt_dets = defaultdict(int)
        pred_dets = defaultdict(int)

        for frame in _frame_range(gt_by_frame, pred_by_frame):
            gts = gt_by_frame.get(frame, [])
            preds = pred_by_frame.get(frame, [])
            for g in gts:
                gt_dets[g["id"]] += 1
            for p in preds:
                pred_dets[p["id"]] += 1
            matches = _match_frame(gts, preds, alpha)
            tp += len(matches)
            fn += len(gts) - len(matches)
            fp += len(preds) - len(matches)
            for r, c, _ in matches:
                pair_tp[(gts[r]["id"], preds[c]["id"])] += 1

        if tp == 0:
            hota_vals.append(0.0)
            continue

        assa_num = 0.0
        for (gid, pid), tpa in pair_tp.items():
            fna = gt_dets[gid] - tpa
            fpa = pred_dets[pid] - tpa
            assa_num += tpa * (tpa / (tpa + fna + fpa))
        assa = assa_num / tp
        deta = tp / (tp + fn + fp)
        hota_vals.append(float(np.sqrt(max(0.0, deta * assa))))

    return {"HOTA": float(np.mean(hota_vals)) if hota_vals else 0.0}


def compute_all_metrics(gt_rows: List[dict], pred_rows: List[dict], iou_thr: float = 0.5) -> dict:
    clear = compute_clear_mot(gt_rows, pred_rows, iou_thr=iou_thr)
    ident = compute_idf1(gt_rows, pred_rows, iou_thr=iou_thr)
    hota = compute_hota(gt_rows, pred_rows)
    out = {}
    out.update(clear)
    out.update(ident)
    out.update(hota)
    return out
