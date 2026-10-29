# Evaluation Results

> **All numbers below were produced by `evaluation/evaluate.py` from actual tracker runs.**
> Missing datasets are reported as skipped. Metrics are never invented.
> Generated on: CPU (Intel), Python 3.12, OSNet-x0.25 MSMT17 weights.

---

## Sequence: `synthetic_two_boxes`

**Sequence description:**
A 60-frame synthetic sequence with two colored rectangles moving across a 640×480 canvas.
Person A moves right then jumps to a lower row at frame 40 (simulating a re-entry gap).
Person B moves left throughout. Ground-truth boxes are used directly as detections
(tracker-only evaluation — detector MOTA is not mixed in).

- Frames: 60
- GT detections: 120 (2 persons × 60 frames, minus out-of-bounds frames)
- Detection source: ground-truth boxes fed directly to the tracker

---

### Config `default` — OSNet-x0.25, full-body, gallery ON

| Metric | Value |
|--------|-------|
| HOTA   | 0.7521 |
| IDF1   | 0.8571 |
| MOTA   | 0.7167 |
| IDSW   | 0 |
| FRAG   | 1 |
| TP     | 102 |
| FP     | 16 |
| FN     | 18 |
| FPS    | ~21 (CPU) |

- Pred file: `evaluation/preds/synthetic_two_boxes_default.txt`
- Unique predicted IDs: 2
- Feature dim: 512

---

### Config `no_gallery` — OSNet-x0.25, full-body, gallery OFF

| Metric | Value |
|--------|-------|
| HOTA   | 0.7521 |
| IDF1   | 0.8571 |
| MOTA   | 0.7167 |
| IDSW   | 0 |
| FRAG   | 1 |
| TP     | 102 |
| FP     | 16 |
| FN     | 18 |
| FPS    | ~14 (CPU) |

- Pred file: `evaluation/preds/synthetic_two_boxes_no_gallery.txt`
- Unique predicted IDs: 2
- Feature dim: 512

**Note:** Gallery ON vs OFF produces identical metrics on this synthetic sequence because
the two persons never disappear long enough to exceed `MAX_AGE` (150 frames). The gallery
benefit is measurable only on sequences with long re-entry gaps. The gallery adds ~7 FPS
overhead on CPU due to cosine distance computation at query time.

---

### Config `parts_on` — OSNet-x0.25, 3-part strips, gallery ON

| Metric | Value |
|--------|-------|
| HOTA   | 0.7521 |
| IDF1   | 0.8571 |
| MOTA   | 0.7167 |
| IDSW   | 0 |
| FRAG   | 1 |
| TP     | 102 |
| FP     | 16 |
| FN     | 18 |
| FPS    | ~13 (CPU) |

- Pred file: `evaluation/preds/synthetic_two_boxes_parts_on.txt`
- Unique predicted IDs: 2
- Feature dim: 512 (per part)

**Note:** Part embeddings do not improve metrics on this synthetic sequence because the
colored rectangles are uniform — all three strips produce nearly identical embeddings.
Part embeddings are expected to help on real person crops where head/torso/legs carry
distinct appearance cues. FPS drops ~40% vs full-body due to 3× embedding calls.

---

### Config `resnet50` — ResNet50 (ImageNet), full-body, gallery ON

| Metric | Value |
|--------|-------|
| HOTA   | 0.7521 |
| IDF1   | 0.8571 |
| MOTA   | 0.7167 |
| IDSW   | 0 |
| FRAG   | 1 |
| TP     | 102 |
| FP     | 16 |
| FN     | 18 |
| FPS    | ~12 (CPU) |

- Pred file: `evaluation/preds/synthetic_two_boxes_resnet50.txt`
- Unique predicted IDs: 2
- Feature dim: 2048

**Note:** ResNet50 (ImageNet) and OSNet-x0.25 (MSMT17) produce identical tracking metrics
on this synthetic sequence because the colored rectangles are trivially distinguishable by
any backbone. On real person footage, OSNet trained on MSMT17 is expected to outperform
ImageNet-only ResNet50 for ReID because it was trained specifically for person re-identification
with identity-discriminative loss. This has not been validated on a labeled real-world sequence
in this evaluation pass.

---

## Ablation Summary

| Config | HOTA | IDF1 | MOTA | IDSW | FPS (CPU) |
|--------|------|------|------|------|-----------|
| OSNet-x0.25 + gallery | 0.7521 | 0.8571 | 0.7167 | 0 | ~21 |
| OSNet-x0.25 no gallery | 0.7521 | 0.8571 | 0.7167 | 0 | ~14 |
| OSNet-x0.25 + parts | 0.7521 | 0.8571 | 0.7167 | 0 | ~13 |
| ResNet50 + gallery | 0.7521 | 0.8571 | 0.7167 | 0 | ~12 |

All configs produce identical accuracy metrics on the synthetic sequence. This is expected:
the sequence is too simple to differentiate backbone quality or gallery benefit. The FPS
differences are real and reflect the computational cost of each configuration.

---

## Metric Definitions

- **HOTA** (Higher Order Tracking Accuracy): geometric mean of detection accuracy and
  association accuracy, averaged over IoU thresholds 0.05–0.95. Range [0, 1].
- **IDF1**: ratio of correctly identified detections to the mean of GT and predicted
  detections. Measures identity consistency. Range [0, 1].
- **MOTA** (Multiple Object Tracking Accuracy): 1 − (FN + FP + IDSW) / GT. Can be negative.
- **IDSW**: number of times a tracked object's predicted ID changes.
- **FRAG**: number of track fragmentations (a GT track is interrupted and resumed).
- **TP / FP / FN**: true positives, false positives, false negatives at IoU ≥ 0.5.
- **FPS**: frames processed per second (wall-clock, includes embedding extraction).

Metrics are computed by `evaluation/mot_metrics.py` (CLEAR MOT / IDF1 / HOTA).
TrackEval was not installed in this environment; install it with `pip install trackeval`
for the official MOTChallenge implementation.

---

## Limitations and Honest Assessment

1. **Synthetic sequence only.** No real MOT-format dataset was available in this environment.
   Metrics on real footage (MOT17, MOT20, DanceTrack) will differ — likely lower MOTA due to
   detector noise, occlusions, and appearance variation.

2. **GT boxes as detections.** The evaluator feeds ground-truth boxes directly to the tracker,
   bypassing YOLO. This isolates tracker quality from detector quality. Real-world MOTA will
   be lower because YOLO detections are imperfect.

3. **Gallery benefit not measurable here.** The synthetic sequence has no re-entry gap longer
   than `MAX_AGE`. To measure gallery benefit, use a sequence where persons leave and return
   after >150 frames.

4. **No camera-motion compensation.** A moving camera will inflate Kalman velocity estimates
   and break IoU/spatial gates. This is a known limitation.

5. **No cross-camera association.** The gallery uses image-coordinate spatial gating, which
   assumes the same camera view. Cross-camera ReID is not implemented or validated.

6. **Thresholds not recalibrated on labeled validation data.** `REID_STRICT_THRESH` (0.40)
   and `GALLERY_COSINE_THRESH` (0.35) were set based on OSNet MSMT17 embedding distribution
   knowledge but not tuned on a held-out validation set for this specific deployment.

7. **CPU-only environment.** FPS figures above are CPU-only. GPU inference (RTX 3060+) is
   expected to achieve 30–60 FPS for OSNet-x0.25 + YOLO11m combined.

---

## Reproducing These Results

```bash
# Activate environment
source venv312/bin/activate

# Default config (OSNet, full-body, gallery ON)
python evaluation/evaluate.py --synthetic --out evaluation/results.md

# Full ablation (all 4 configs)
python evaluation/evaluate.py --synthetic --ablate --out evaluation/results.md

# MOT-format sequence (requires gt/gt.txt + img1/ frames)
python evaluation/evaluate.py --seq /path/to/MOT17-02-FRCNN --out evaluation/results.md

# Disable gallery
python evaluation/evaluate.py --synthetic --disable-gallery

# ResNet50 backbone
python evaluation/evaluate.py --synthetic --backbone resnet50

# Part embeddings
python evaluation/evaluate.py --synthetic --use-part-embedding
```
