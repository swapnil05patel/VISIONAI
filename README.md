# Person Detection, Tracking, and Re-Identification

A modular person tracking system built with YOLO11, pretrained OSNet ReID embeddings (ResNet50 baseline), Kalman filtering, Hungarian assignment, optional ByteTrack-style low-confidence association, and a persistent Track Gallery. Deployed as an interactive Streamlit web application.

This is a research/engineering codebase. It is **not** claimed production-ready. Single-camera long-term re-identification is implemented; **cross-camera ReID is not implemented or validated**.

**Live demo:**  https://person-reid-multicamera.streamlit.app/


---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tracking Pipeline](#tracking-pipeline)
- [Track Gallery](#track-gallery)
- [Project Structure](#project-structure)
- [Quick Start from GitHub](#quick-start-from-github)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Outputs](#outputs)
- [Hardware Requirements](#hardware-requirements)

---

## Overview

Standard object trackers fail under real-world surveillance conditions. Occlusions, re-entries after long absences, crossing paths, and appearance changes all cause identity switches and fragmented tracks. This project implements a full multi-stage tracking pipeline to address these challenges.

**Core capabilities:**
- Consistent person IDs throughout the video, even across occlusions
- Re-identification of returning persons in the **same camera/view** using a persistent Track Gallery
- **Fast-motion handling:** Dynamic search radiuses and momentum-based direction consistency
- **Static object filtering:** Automatically detects and ignores non-moving human-like objects (e.g. posters/mannequins)
- All parameters tunable live via a Streamlit sidebar — no code changes needed
- Modular codebase: tracking logic and UI are fully separated

---

## Architecture

The system is divided into two packages with a strict dependency direction:

```
app.py  ->  ui/*  ->  tracker/*
```

`tracker/` contains all detection, tracking, and re-identification logic. It has no dependency on Streamlit or any UI library. `ui/` contains all Streamlit rendering code and calls into `tracker/`. This makes the core tracking engine independently testable and reusable.

### Component Overview

| Component | Role |
|---|---|
| `tracker/config.py` | Single source of truth for all tunable constants |
| `tracker/utils.py` | Pure stateless math utilities (IoU, box conversions, cosine distance) |
| `tracker/embedder.py` | OSNet / ResNet50 appearance embedder (full-body or part strips) |
| `tracker/osnet.py` | Omni-Scale Network definition and MSMT17 weight loading |
| `tracker/kalman.py` | Constant-velocity Kalman filter |
| `tracker/track.py` | Per-track state manager (Kalman state, embedding cache, lifecycle) |
| `tracker/gallery.py` | Persistent appearance gallery with gated retrieval |
| `tracker/reid_tracker.py` | Pipeline orchestrator — runs all steps per frame |
| `ui/sidebar.py` | Sidebar controls, returns settings dict |
| `ui/uploader.py` | Upload panel, preview, process button |
| `ui/results.py` | Results panel (4 tabs) |
| `ui/pipeline.py` | Video processing loop, YOLO inference, CSV export |

---

## Tracking Pipeline

Every frame is processed through the following 11 steps in order:

### Step 1 — YOLO11 Detection
YOLO11m scans the frame for all persons (class 0). Two confidence thresholds are used:
- **Low threshold** (default 0.12): passed to YOLO as the prediction floor.
- **High threshold** (default 0.45): gate applied after YOLO returns — only detections above this are passed to the tracker.

### Step 2 — Appearance Feature Extraction
The default backbone is **OSNet-x0.25** trained on MSMT17 (`512-D` L2-normalized embeddings). **ResNet50** (ImageNet, `2048-D`) remains available as a baseline via `REID_BACKBONE=resnet50`.

When `USE_PART_EMBEDDING=True`, each box is split into `NUM_PARTS` horizontal strips and each strip is embedded separately. The default is full-body (`USE_PART_EMBEDDING=False`). Invalid crops, missing weights, and non-finite features return no embedding; random fallbacks are not used.

### Step 3 — Kalman Prediction
Each active track runs a **Kalman filter** with an 8-dimensional state vector:
```
state = [cx, cy, w, h, vx, vy, vw, vh]
```
Position is predicted from previous position + estimated velocity. This produces a predicted bounding box for each track before any detections are matched.

### Step 4 — Cost Matrix Construction
An N × M cost matrix is built for all (track, detection) pairs:
```
cost[i, j] = 0.3 × (1 - IoU) + 0.5 × ReID_cosine_distance + 0.2 × Direction_penalty
```
- **IoU** measures spatial overlap between predicted track box and detection box.
- **ReID cosine distance** is the minimum cosine distance between the track's average embedding and the detection's part embeddings.
- **Direction penalty** heavily penalizes detections that contradict the predicted momentum path of fast-moving tracks (shutting down ID-swaps during crowded occlusions).

### Step 5 — Hungarian Assignment
The **Hungarian algorithm** (via `scipy.optimize.linear_sum_assignment`) solves the cost matrix globally. Each track and detection is assigned at most once. Global optimization means the system won't greedily assign a track to its locally best detection if doing so results in a worse total assignment.

### Step 6 — Acceptance Gating
Each proposed assignment from Step 5 is accepted only if one of these conditions is true:
- `IoU >= iou_thr` (direct spatial match), **OR**
- All of: ReID cosine distance below strict threshold + center distance within limits set by **Velocity-Based Dynamic Gating** (search radius scales up if the object is moving fast and was lost for multiple frames) + area ratio within limit + track was lost recently enough

This two-path acceptance means IoU handles normal continuous tracking, while ReID (paired with physics-based prediction) handles recovery from missed detections.

### Step 7 — Track Update
Accepted matches trigger a **Kalman correction** (measurement update) on the track. The bounding box is further refined with **adaptive exponential smoothing**:
- Slow-moving tracks use high alpha (0.70) for stable, jitter-free boxes.
- Fast-moving tracks use low alpha (0.18) for responsive, accurate boxes.

### Step 8 — ByteTrack stage-2 then fallback IoU
Unmatched tracks are first associated with **low-confidence** detections by IoU (`ENABLE_BYTETRACK_STAGE2`). Remaining unmatched **high-confidence** detections are then compared against lost tracks using IoU on the smoothed box (`FALLBACK_IOU_FOR_NEW`). Low-confidence boxes never spawn new IDs.

### Step 9 — Gallery Query
Still-unmatched detections query the **Track Gallery** (see below). If a retired track matches by appearance and all spatial gates pass, it is **revived with its original ID**. This is what makes long-term re-identification possible.

### Step 10 — New Track Creation
Detections that survived all previous steps become new tracks. A detection that overlaps a confirmed active track by more than 0.45 IoU is ignored to prevent ghost tracks.

### Step 11 — Track Retirement
Tracks exceeding `MAX_AGE` (default 150) frames without a match are retired. Before deletion, they are saved to the Track Gallery. Boundary-exit retirement (when the predicted center is consistently outside the frame) is handled separately in Step 3. **Static tracks** (posters/signs detected via Kalman velocity and positional displacement) are silently deleted here and never enter the gallery or receive confirmed IDs.

---

## Track Gallery

The Track Gallery enables re-identification of persons returning after long absences (up to 10 minutes by default).

### Storage (on track retirement)

1. Tracks with fewer than `GALLERY_MIN_HITS` detections are not saved — they are too unreliable.
2. Appearance is stored according to `GALLERY_REPRESENTATION`: mean, EMA, or a small set of exemplars (`512-D` for OSNet, `2048-D` for ResNet50).
3. Metadata stored with the vector(s): last known center, last box area, retirement timestamp, hit count.
4. If the gallery is at capacity (`GALLERY_MAX_SIZE`), the oldest unlocked entry is evicted first.

### Retrieval (4-gate pipeline)

When a new detection cannot be matched to any active track, the gallery is queried. Cheap gates run first:

| Gate | Check | Cost |
|---|---|---|
| 1. TTL | Entry older than `GALLERY_TTL_SECONDS`? Skip. | O(1) |
| 2. Spatial | Optional (`GALLERY_USE_SPATIAL_GATE`). Last center farther than `GALLERY_SPATIAL_MAX_DIST`? Skip. | O(1) |
| 3. Area ratio | Box area differs by more than `GALLERY_AREA_RATIO_MAX`? Skip. | O(1) |
| 4. Cosine | Distance below `GALLERY_COSINE_THRESH`. Mean/EMA or min-exemplar distance. | O(D) |

In typical scenes, gates 1–3 eliminate 90%+ of gallery entries. Cosine runs on only the small remaining set.

### Re-attachment

On a gallery match, a track is revived with the **original track ID**. It is confirmed only after `REVIVAL_CONFIRMATION_FRAMES` consecutive matches (the gallery entry stays locked until then). Spatial gating assumes the person reappears near the last location; disable `GALLERY_USE_SPATIAL_GATE` only if you accept matches anywhere in the frame. That is still **not** multi-camera association.

---

## Project Structure

```
Real-Time Multi-Camera Person Detection & Re-Identification/
├── app.py                   Entry point — initialises session state and renders layout
├── requirements.txt         Python dependencies
├── README.md
├── .gitignore
│
├── tracker/
│   ├── __init__.py
│   ├── config.py            All tunable constants — edit here to change behaviour
│   ├── utils.py             Pure utility functions: iou, box ops, cosine distance
│   ├── osnet.py             OSNet architecture + MSMT17 checkpoint loading
│   ├── embedder.py          PartEmbedder: OSNet or ResNet50 features
│   ├── kalman.py            KalmanBox: constant-velocity Kalman filter
│   ├── track.py             Track: per-track state, embedding cache, smoothing
│   ├── gallery.py           TrackGallery: save, gated query, TTL eviction
│   └── reid_tracker.py      PersonReIDTracker: 11-step per-frame pipeline
│
├── evaluation/
│   ├── evaluate.py          MOT / synthetic evaluation entry point
│   ├── mot_metrics.py       IDF1, HOTA, MOTA, IDSW, fragmentation
│   └── results.md           Written by evaluate.py from actual runs
│
├── tests/
│   └── test_*.py            Unit / integration / metric tests
│
└── ui/
    ├── __init__.py
    ├── sidebar.py           render_sidebar() — returns settings dict
    ├── uploader.py          render_uploader() — upload panel and process button
    ├── results.py           render_results() — 4-tab results panel
    └── pipeline.py          run_tracking_pipeline() — YOLO + tracker video loop
```

---

## Quick Start from GitHub

### 1. Clone the repository

```bash
git clone https://github.com/prxsnn/person-reid-multicamera.git
cd person-reid-multicamera
```

### 2. Create and activate a virtual environment

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

Because PyTorch installation is heavily dependent on hardware (CPU vs GPU), you **must** install it first before running the requirements script.

#### Option A: GPU Environment (Recommended for speed)

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

#### Option B: CPU-Only Environment (For local testing without GPU)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. Download the YOLO model

Download `yolo11m.pt` from the [Ultralytics releases](https://github.com/ultralytics/assets/releases) and place it in the project root directory.

Alternatively, the model will be downloaded automatically on first run if you have an internet connection.

### 5. Run the application

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. Upload a video, adjust settings in the sidebar if needed, and click **Start Tracking**.

---

## Configuration

All parameters are defined in `tracker/config.py` and exposed as live controls in the Streamlit sidebar. Changes take effect on the next processing run — no code edits needed.

### Detection

| Parameter | Default | Description |
|---|---|---|
| `HIGH_CONF_THR` | 0.45 | Minimum YOLO confidence to pass a detection to the tracker |
| `LOW_CONF_THR` | 0.12 | YOLO prediction confidence floor |

### ReID backbone

| Parameter | Default | Description |
|---|---|---|
| `REID_BACKBONE` | `osnet_x0_25` | `osnet_x0_25`, `osnet_x1_0`, or `resnet50` |
| `USE_PART_EMBEDDING` | `False` | `True` = horizontal part strips; `False` = full-body crop |
| `NUM_PARTS` | 3 | Strip count when part embeddings are enabled |

Place OSNet checkpoints in `weights/` (for example `weights/osnet_x0_25_msmt17.pth`). Missing files are downloaded from HuggingFace on first use, or initialization fails.

### Matching

| Parameter | Default | Description |
|---|---|---|
| `IOU_THR` | 0.30 | Minimum IoU for direct track-detection match |
| `MATCH_WEIGHT_IOU` | 0.3 | Weight of spatial cost in assignment matrix |
| `MATCH_WEIGHT_REID` | 0.5 | Weight of appearance cost in assignment matrix |
| `MATCH_WEIGHT_DIRECTION` | 0.2 | Weight of motion trajectory penalty in assignment matrix |
| `DIRECTION_SPEED_THRESH` | 3.0 | Minimum pixels/frame to trigger direction penalty |
| `VELOCITY_GATE_MULTIPLIER`| 1.5 | How much speed scales the reattachment search area |
| `REID_CENTER_MAX_DIST_BASE`| 160 | Base pixel distance for ReID spatial gate |
| `ENABLE_BYTETRACK_STAGE2` | `True` | Associate leftover tracks with low-confidence boxes by IoU |
| `STAGE2_IOU_THR` | 0.40 | IoU threshold for that second stage |

Similarity thresholds (`REID_STRICT_THRESH`, `GALLERY_COSINE_THRESH`) should be changed only with validation evidence. Defaults were not retuned in this pass.

### Track Lifecycle

| Parameter | Default | Description |
|---|---|---|
| `MIN_HITS_TO_CONFIRM` | 2 | Detections before a track is confirmed |
| `MAX_AGE` | 150 | Frames without match before retirement |
| `FALLBACK_IOU_FOR_NEW` | 0.18 | IoU threshold for fallback attach (prevents duplicate IDs) |

### Track Gallery

| Parameter | Default | Description |
|---|---|---|
| `ENABLE_GALLERY` | `True` | Master switch for long-term gallery recovery |
| `GALLERY_COSINE_THRESH` | 0.35 | Max cosine distance for gallery re-attachment |
| `GALLERY_TTL_SECONDS` | 600 | Gallery entry expires after this many seconds |
| `GALLERY_MAX_SIZE` | 1000 | Maximum gallery entries |
| `GALLERY_MIN_HITS` | 2 | Minimum hits for a track to be saved to gallery |
| `GALLERY_USE_SPATIAL_GATE` | `True` | Require last position near the query box |
| `GALLERY_REPRESENTATION` | `mean` | `mean`, `ema`, or `exemplars` |
| `REVIVAL_CONFIRMATION_FRAMES` | 2 | Consecutive matches before a revived ID is confirmed |
| `ENABLE_STATIC_PRUNING` | `True` | Drop tracks that never move |

## Evaluation

```bash
# Built-in synthetic sequence (GT boxes + real embeddings from generated frames)
python evaluation/evaluate.py --synthetic --out evaluation/results.md

# Full ablation (all 4 configs on the same sequence)
python evaluation/evaluate.py --synthetic --ablate --out evaluation/results.md

# MOT-format sequence (folder with img1/ and gt/gt.txt)
python evaluation/evaluate.py --seq /path/to/MOT17-02-FRCNN

# Disable gallery
python evaluation/evaluate.py --synthetic --disable-gallery

# ResNet50 baseline
python evaluation/evaluate.py --synthetic --backbone resnet50
```

The evaluator writes MOT-format predictions under `evaluation/preds/` and metrics to `evaluation/results.md`. Detection quality is not mixed in when GT boxes are used as the detection source (tracker-only). TrackEval is used only if it is installed (`pip install trackeval`); otherwise `evaluation/mot_metrics.py` reports IDF1, HOTA, MOTA, ID switches, fragmentation, and FPS.

### Measured results (synthetic sequence, CPU, GT boxes as detections)

| Config | HOTA | IDF1 | MOTA | IDSW | FPS |
|--------|------|------|------|------|-----|
| OSNet-x0.25 + gallery | 0.7521 | 0.8571 | 0.7167 | 0 | ~21 |
| OSNet-x0.25 no gallery | 0.7521 | 0.8571 | 0.7167 | 0 | ~14 |
| OSNet-x0.25 + parts | 0.7521 | 0.8571 | 0.7167 | 0 | ~13 |
| ResNet50 + gallery | 0.7521 | 0.8571 | 0.7167 | 0 | ~12 |

All configs produce identical accuracy on the synthetic sequence (expected — the sequence is too simple to differentiate backbone quality). FPS differences are real and reflect computational cost. See `evaluation/results.md` for full details and interpretation.

```bash
pytest tests -q
```

## Limitations

- **No camera-motion compensation.** A moving camera will inflate Kalman motion and can break IoU/spatial gates.
- **No cross-camera association.** There is no camera ID, no shared global gallery across views, and no calibrated homography. Spatial gallery gating assumes the same image coordinates.
- **No production SLO.** Latency, identity stability, and detector quality depend on hardware, video, and thresholds.
- **Thresholds.** ReID cosine thresholds were not recalibrated on a labeled validation set in this work.

---

## Outputs

### Annotated Video (MP4)

- Inline **HTML5 video playback** directly in the Streamlit browser UI (automatically transcoded to H.264).
- Unique color-coded bounding box per confirmed track (bright, high-contrast)
- Track ID label above each box
- Horizontal lines dividing body parts (showing ReID feature regions, visible only on medium-to-large objects)
- Top-left overlay: frame count, live confirmed track count, current gallery size

### Tracking Log (CSV)

| Column | Description |
|---|---|
| `ID` | Track identifier (e.g. `ID_001`) |
| `Entry_Time` | Wall-clock timestamp of first detection |
| `Exit_Time` | Wall-clock timestamp of retirement, or `Active` |
| `Duration` | Time in scene in seconds, or `Active` |
| `Status` | `Active` or `Completed` |

---

## Hardware Requirements

| Component | Recommended | Minimum |
|---|---|---|
| GPU | NVIDIA RTX 3060 8GB or higher | NVIDIA RTX 2060 6GB |
| RAM | 16 GB | 8 GB |
| Storage | SSD | HDD |

Processing is supported on CPU but will run significantly slower (often under 5 fps) and is not suitable for real-time use. The GPU is used by YOLO11 detection and the ReID backbone (OSNet by default).

**Processing time estimate:**
```
Total Processing Time (seconds) = Total Video Frames / Inference FPS
```
Example: 5-minute video at 30 fps = 9000 frames. At 45 fps inference = ~200 seconds.
