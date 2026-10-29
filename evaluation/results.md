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
- Runtime: 2.689s

- HOTA: 0.7521
- IDF1: 0.8571
- MOTA: 0.7167
- IDSW: 0
- FRAG: 1
- TP: 102
- FP: 16
- FN: 18
- FPS: 22.315

TrackEval was not installed; metrics above come from `evaluation/mot_metrics.py` (CLEAR MOT / IDF1 / HOTA).

### Config `no_gallery` (backbone=osnet_x0_25, parts=False, gallery=False)

- Pred file: `/home/swapnil/Fixed bug verification/VisionTrace AI/person-reid-multicamera/evaluation/preds/synthetic_two_boxes_no_gallery.txt`
- Unique predicted IDs: 2
- Feature dim: 512
- Runtime: 4.267s

- HOTA: 0.7521
- IDF1: 0.8571
- MOTA: 0.7167
- IDSW: 0
- FRAG: 1
- TP: 102
- FP: 16
- FN: 18
- FPS: 14.061

TrackEval was not installed; metrics above come from `evaluation/mot_metrics.py` (CLEAR MOT / IDF1 / HOTA).

### Config `parts_on` (backbone=osnet_x0_25, parts=True, gallery=True)

- Pred file: `/home/swapnil/Fixed bug verification/VisionTrace AI/person-reid-multicamera/evaluation/preds/synthetic_two_boxes_parts_on.txt`
- Unique predicted IDs: 2
- Feature dim: 512
- Runtime: 5.738s

- HOTA: 0.7521
- IDF1: 0.8571
- MOTA: 0.7167
- IDSW: 0
- FRAG: 1
- TP: 102
- FP: 16
- FN: 18
- FPS: 10.457

TrackEval was not installed; metrics above come from `evaluation/mot_metrics.py` (CLEAR MOT / IDF1 / HOTA).

### Config `resnet50` (backbone=resnet50, parts=False, gallery=True)

- Pred file: `/home/swapnil/Fixed bug verification/VisionTrace AI/person-reid-multicamera/evaluation/preds/synthetic_two_boxes_resnet50.txt`
- Unique predicted IDs: 2
- Feature dim: 2048
- Runtime: 8.077s

- HOTA: 0.7521
- IDF1: 0.8571
- MOTA: 0.7167
- IDSW: 0
- FRAG: 1
- TP: 102
- FP: 16
- FN: 18
- FPS: 7.428

TrackEval was not installed; metrics above come from `evaluation/mot_metrics.py` (CLEAR MOT / IDF1 / HOTA).
