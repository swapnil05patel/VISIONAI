# Evaluation results

These numbers were produced by `evaluation/evaluate.py`. Missing datasets are reported as skipped; metrics are never invented.

- Device: `cpu`

- Default backbone: `osnet_x0_25`

- Part embeddings: `False`

- Gallery enabled: `True`

## Sequence `synthetic_two_boxes`

- Frames loaded: 60
- GT detections: 120
- Detection source: ground-truth boxes (tracker-only evaluation, not detector MOTA)

### Config `default` (backbone=osnet_x0_25, parts=False, gallery=True)

- Pred file: `/home/swapnil/Fixed bug verification/VisionTrace AI/person-reid-multicamera/evaluation/preds/synthetic_two_boxes_default.txt`
- Unique predicted IDs: 2
- Feature dim: 512
- Runtime: 2.656s

- HOTA: 0.7521
- IDF1: 0.8571
- MOTA: 0.7167
- IDSW: 0
- FRAG: 1
- TP: 102
- FP: 16
- FN: 18
- FPS: 22.593

TrackEval was not installed; metrics above come from `evaluation/mot_metrics.py` (CLEAR MOT / IDF1 / HOTA).
