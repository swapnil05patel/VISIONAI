"""
scripts/run_inference.py
------------------------
Headless (no Streamlit) inference entry point.

Usage
-----
  # Video: detect + track, write annotated output and CSV
  python scripts/run_inference.py --input path/to/video.mp4

  # Image: detect + embed, print results
  python scripts/run_inference.py --input path/to/person.jpg --mode image

  # Image with gallery comparison (Re-ID ranking)
  python scripts/run_inference.py --input path/to/query.jpg --mode image \
      --gallery test_data/images/gallery

  # Video with custom thresholds
  python scripts/run_inference.py --input test_data/videos/sample.mp4 \
      --high-conf 0.50 --low-conf 0.15 --iou-thr 0.30 --reid-thr 0.45 \
      --max-age 150 --min-hits 2
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from tracker.config import (
    HIGH_CONF_THR,
    LOW_CONF_THR,
    IOU_THR,
    REID_STRICT_THRESH,
    MAX_AGE,
    MIN_HITS_TO_CONFIRM,
    NUM_PARTS,
)
from tracker.embedder import PartEmbedder
from tracker.reid_tracker import PersonReIDTracker
from tracker.utils import l2_normalize_vec, cosine_distance_vec

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _detect_mode(input_path, explicit_mode):
    if explicit_mode in ("video", "image"):
        return explicit_mode
    ext = input_path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(
        f"Cannot infer mode from extension '{ext}'. "
        "Use --mode video or --mode image explicitly."
    )


def _load_yolo(device):
    weights_path = _PROJECT_ROOT / "yolo11m.pt"
    model_path = str(weights_path) if weights_path.exists() else "yolo11m.pt"
    print(f"[YOLO] Loading weights: {model_path}")
    model = YOLO(model_path)
    if device == "cuda":
        model.to(device)
    return model


def _auto_device(requested):
    if requested in ("cuda", "cpu"):
        if requested == "cuda" and not torch.cuda.is_available():
            print("[WARNING] CUDA requested but not available — falling back to CPU.")
            return "cpu"
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ #
# VIDEO MODE                                                           #
# ------------------------------------------------------------------ #

def run_video(args):
    device = _auto_device(args.device)
    input_path = Path(args.input).resolve()

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    out_video_path = output_dir / f"tracked_{stem}.mp4"
    out_csv_path   = output_dir / f"tracking_{stem}.csv"

    print(f"[INFO] Device      : {device.upper()}")
    print(f"[INFO] Input       : {input_path}")
    print(f"[INFO] Output video: {out_video_path}")
    print(f"[INFO] Output CSV  : {out_csv_path}")

    yolo = _load_yolo(device)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return 1

    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Resolution  : {width}x{height}")
    print(f"[INFO] FPS         : {fps:.1f}")
    print(f"[INFO] Frames      : {total_frames}")

    tracker = PersonReIDTracker(
        device             = device,
        frame_size         = (width, height),
        fps                = fps,
        iou_thr            = args.iou_thr,
        reid_strict_thresh = args.reid_thr,
        max_age            = args.max_age,
        min_hits           = args.min_hits,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        print("[ERROR] Could not open VideoWriter.")
        cap.release()
        return 1

    frame_count = 0
    start_time  = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = yolo.predict(
            source  = frame,
            classes = [0],
            device  = device,
            verbose = False,
            conf    = args.low_conf,
            iou     = 0.7,
            half    = (device == "cuda"),
            imgsz   = 640,
        )

        all_boxes = results[0].boxes.data.cpu().numpy() if len(results[0].boxes.data) > 0 else np.empty((0, 6))
        high_mask = (all_boxes[:, 4] >= args.high_conf) if len(all_boxes) > 0 else np.array([], dtype=bool)
        high_boxes = all_boxes[high_mask][:, :4] if high_mask.any() else np.empty((0, 4))

        detections  = []
        parts_feats = []
        for bb in high_boxes:
            box   = [float(b) for b in bb]
            parts = tracker.embedder.extract(frame, box)
            if parts:
                detections.append(box)
                parts_feats.append(parts)

        tracks = tracker.step(detections, parts_feats, frame_count)

        out_frame = frame.copy()
        live_confirmed = 0

        for tid, tr in tracks.items():
            if tr.retired:
                continue
            r_c = int((tid * 37) % 205) + 50
            g_c = int((tid * 97) % 205) + 50
            b_c = int((tid * 61) % 205) + 50
            color = (r_c, g_c, b_c)
            x1, y1, x2, y2 = [int(round(v)) for v in tr.smoothed_box]

            if tr.confirmed:
                live_confirmed += 1
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out_frame, f"ID:{tid}", (x1, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                box_h = y2 - y1
                if box_h >= 40 and NUM_PARTS > 1:
                    part_h = box_h // NUM_PARTS
                    lc = (min(255, r_c + 80), min(255, g_c + 80), min(255, b_c + 80))
                    for i in range(1, NUM_PARTS):
                        ly = y1 + i * part_h
                        cv2.line(out_frame, (x1 + 1, ly), (x2 - 1, ly), (0, 0, 0), 3)
                        cv2.line(out_frame, (x1 + 1, ly), (x2 - 1, ly), lc, 2)
            else:
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), (180, 180, 180), 1)

        cv2.putText(
            out_frame,
            f"Frame {frame_count}/{total_frames}  Confirmed:{live_confirmed}  Gallery:{len(tracker.gallery)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 210), 2,
        )
        writer.write(out_frame)

        if args.verbose or (frame_count % 50 == 0):
            elapsed = time.time() - start_time
            spd = frame_count / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_count:4d}/{total_frames}  |  Speed: {spd:.1f} fps  |  "
                  f"Confirmed: {live_confirmed}  |  Gallery: {len(tracker.gallery)}")

    cap.release()
    writer.release()

    elapsed_total = time.time() - start_time
    avg_fps = frame_count / elapsed_total if elapsed_total > 0 else 0

    print(f"\n[DONE] {frame_count} frames in {elapsed_total:.1f}s ({avg_fps:.1f} fps avg)")
    print(f"[INFO] Unique track IDs seen : {tracker.next_id - 1}")
    print(f"[INFO] Final gallery size    : {len(tracker.gallery)}")

    _write_tracking_csv(tracker, out_csv_path)
    print(f"[INFO] CSV saved   : {out_csv_path}")
    print(f"[INFO] Video saved : {out_video_path}")
    return 0


def _write_tracking_csv(tracker, out_path):
    fieldnames = ["ID", "Entry_Time", "Exit_Time", "Duration", "Status", "Hits"]
    rows = []
    for tid, tr in tracker.tracks.items():
        entry_str = tr.entry_time.strftime("%Y-%m-%d %H:%M:%S") if tr.entry_time else ""
        if tr.exit_time is not None:
            exit_str = tr.exit_time.strftime("%Y-%m-%d %H:%M:%S")
            duration = f"{(tr.exit_time - tr.entry_time).total_seconds():.1f}s"
            status   = "Completed"
        else:
            exit_str = "Active"
            duration = "Active"
            status   = "Active"
        rows.append({
            "ID": f"ID_{tid:03d}", "Entry_Time": entry_str,
            "Exit_Time": exit_str, "Duration": duration,
            "Status": status, "Hits": tr.hits,
        })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ #
# IMAGE MODE                                                           #
# ------------------------------------------------------------------ #

def run_image(args):
    device = _auto_device(args.device)
    input_path = Path(args.input).resolve()

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return 1

    print(f"[INFO] Device : {device.upper()}")
    print(f"[INFO] Input  : {input_path}")

    yolo     = _load_yolo(device)
    embedder = PartEmbedder(device=device, num_parts=NUM_PARTS)

    if not embedder.available:
        print("[WARNING] PartEmbedder failed to load ResNet50 — using random fallback embeddings.")

    frame = cv2.imread(str(input_path))
    if frame is None:
        print(f"[ERROR] OpenCV could not read image: {input_path}")
        return 1

    h, w = frame.shape[:2]
    print(f"[INFO] Image size: {w}x{h}")

    results = yolo.predict(
        source=frame, classes=[0], device=device,
        verbose=False, conf=args.low_conf, iou=0.7,
        half=(device == "cuda"), imgsz=640,
    )

    all_boxes = results[0].boxes.data.cpu().numpy() if len(results[0].boxes.data) > 0 else np.empty((0, 6))
    print(f"\n[DETECTION] Found {len(all_boxes)} person candidate(s) with conf >= {args.low_conf:.2f}")

    high_mask  = (all_boxes[:, 4] >= args.high_conf) if len(all_boxes) > 0 else np.array([], dtype=bool)
    high_boxes = all_boxes[high_mask] if high_mask.any() else np.empty((0, 6))
    print(f"[DETECTION] High-confidence (>= {args.high_conf:.2f}): {len(high_boxes)}")

    query_embeddings = []

    for idx, bb in enumerate(high_boxes):
        x1, y1, x2, y2, conf, cls = bb
        box = [float(x1), float(y1), float(x2), float(y2)]
        print(f"\n  Detection {idx + 1}:")
        print(f"    Box    : [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        print(f"    Conf   : {conf:.4f}")
        print(f"    Size   : {(x2 - x1):.0f} x {(y2 - y1):.0f} px")

        parts = embedder.extract(frame, box)
        print(f"    Parts  : {len(parts)} embeddings (each 2048D)")
        for pi, p in enumerate(parts):
            norm = float(np.linalg.norm(p))
            print(f"      Part {pi + 1}: norm={norm:.6f}  (should be ~1.0)")

        arr = np.stack(parts, axis=0)
        mean_embed = l2_normalize_vec(np.mean(arr, axis=0))
        query_embeddings.append((idx + 1, mean_embed, box, float(conf)))

    if not query_embeddings:
        print("\n[INFO] No high-confidence detections — nothing to compare against gallery.")

    if args.gallery and query_embeddings:
        _run_gallery_comparison(args, device, embedder, query_embeddings)

    return 0


def _run_gallery_comparison(args, device, embedder, queries):
    gallery_dir = Path(args.gallery).resolve()
    if not gallery_dir.exists():
        print(f"\n[WARNING] Gallery directory not found: {gallery_dir}")
        return

    gallery_files = [
        p for p in sorted(gallery_dir.iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not gallery_files:
        print(f"\n[INFO] No images found in gallery: {gallery_dir}")
        return

    print(f"\n[GALLERY] Loading {len(gallery_files)} gallery image(s)...")
    yolo_g = _load_yolo(device)

    gallery_embeddings = []
    for gpath in gallery_files:
        gframe = cv2.imread(str(gpath))
        if gframe is None:
            continue
        results = yolo_g.predict(
            source=gframe, classes=[0], device=device,
            verbose=False, conf=args.low_conf, iou=0.7,
            half=(device == "cuda"), imgsz=640,
        )
        gboxes = results[0].boxes.data.cpu().numpy() if len(results[0].boxes.data) > 0 else np.empty((0, 6))
        if len(gboxes) == 0:
            gh, gw = gframe.shape[:2]
            best_box = [0.0, 0.0, float(gw), float(gh)]
        else:
            best_idx = int(np.argmax(gboxes[:, 4]))
            best_box = [float(v) for v in gboxes[best_idx, :4]]

        gparts = embedder.extract(gframe, best_box)
        garr   = np.stack(gparts, axis=0)
        gembed = l2_normalize_vec(np.mean(garr, axis=0))
        gallery_embeddings.append((gpath.name, gembed))

    print(f"[GALLERY] Embedded {len(gallery_embeddings)} gallery image(s).")

    for qidx, qembed, qbox, qconf in queries:
        print(f"\n[RANKING] Query detection {qidx} (conf={qconf:.3f}) — top-{args.top_k} matches:")
        distances = sorted(
            [(cosine_distance_vec(qembed, gembed), gname) for gname, gembed in gallery_embeddings]
        )
        print(f"  {'Rank':<5} {'Distance':<12} {'Gallery image'}")
        print(f"  {'-'*5} {'-'*12} {'-'*30}")
        for rank, (dist, gname) in enumerate(distances[: args.top_k], start=1):
            print(f"  {rank:<5} {dist:<12.6f} {gname}")


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def _build_parser():
    p = argparse.ArgumentParser(
        description="Headless inference for person-reid-multicamera",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      required=True)
    p.add_argument("--mode",       default="auto", choices=["auto", "video", "image"])
    p.add_argument("--output-dir", default=".",    dest="output_dir")
    p.add_argument("--gallery",    default=None)
    p.add_argument("--top-k",      type=int,   default=5,           dest="top_k")
    p.add_argument("--high-conf",  type=float, default=HIGH_CONF_THR, dest="high_conf")
    p.add_argument("--low-conf",   type=float, default=LOW_CONF_THR,  dest="low_conf")
    p.add_argument("--iou-thr",    type=float, default=IOU_THR,            dest="iou_thr")
    p.add_argument("--reid-thr",   type=float, default=REID_STRICT_THRESH, dest="reid_thr")
    p.add_argument("--max-age",    type=int,   default=MAX_AGE,             dest="max_age")
    p.add_argument("--min-hits",   type=int,   default=MIN_HITS_TO_CONFIRM, dest="min_hits")
    p.add_argument("--device",     default=None, choices=["cuda", "cpu"])
    p.add_argument("--verbose",    action="store_true")
    return p


def main():
    args  = _build_parser().parse_args()
    mode  = _detect_mode(Path(args.input), args.mode)
    print("=" * 60)
    print(f"  person-reid-multicamera  |  mode={mode.upper()}")
    print("=" * 60)
    sys.exit(run_video(args) if mode == "video" else run_image(args))


if __name__ == "__main__":
    main()
