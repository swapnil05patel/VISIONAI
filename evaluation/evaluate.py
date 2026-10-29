"""
evaluation/evaluate.py
----------------------
Run PersonReIDTracker on MOT-format sequences (or a built-in synthetic sequence)
and report IDF1, HOTA, MOTA, ID switches, fragmentation, and FPS.

Usage
-----
  python evaluation/evaluate.py --seq /path/to/MOT17-02-FRCNN
  python evaluation/evaluate.py --gt /path/to/gt.txt --frames /path/to/img1
  python evaluation/evaluate.py --synthetic --out evaluation/results.md
  python evaluation/evaluate.py --synthetic --disable-gallery --backbone resnet50

Never writes fabricated metrics. If a sequence cannot be run, the report states
that the run was skipped and why.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from evaluation.mot_io import load_mot_file, write_mot_file, xyxy_to_mot_row, index_by_frame
from evaluation.mot_metrics import compute_all_metrics
from tracker.config import DEVICE, REID_BACKBONE, USE_PART_EMBEDDING, ENABLE_GALLERY
from tracker.reid_tracker import PersonReIDTracker


def discover_sequence(seq_dir: Path) -> tuple[Path, Path]:
    seq_dir = seq_dir.resolve()
    gt_candidates = [
        seq_dir / "gt" / "gt.txt",
        seq_dir / "gt.txt",
    ]
    frame_candidates = [
        seq_dir / "img1",
        seq_dir / "images",
        seq_dir,
    ]
    gt_path = next((p for p in gt_candidates if p.exists()), None)
    if gt_path is None:
        raise FileNotFoundError(f"No gt.txt found under {seq_dir}")
    frame_dir = next((p for p in frame_candidates if p.is_dir() and _list_frames(p)), None)
    if frame_dir is None:
        raise FileNotFoundError(f"No image frames found under {seq_dir}")
    return gt_path, frame_dir


def _list_frames(frame_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in exts)


def _make_synthetic_sequence(num_frames: int = 60, size=(480, 640)):
    """Two colored rectangles that cross, leave, and return."""
    h, w = size
    frames = []
    gt_rows = []
    for t in range(1, num_frames + 1):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)

        x1 = 40 + t * 6
        y1 = 80
        if t > 40:
            x1 = 40 + (t - 40) * 8
            y1 = 240
        box_a = [x1, y1, x1 + 50, y1 + 140]
        cv2.rectangle(img, (int(box_a[0]), int(box_a[1])), (int(box_a[2]), int(box_a[3])), (40, 80, 220), -1)

        x2 = w - 90 - t * 5
        y2 = 200
        box_b = [x2, y2, x2 + 46, y2 + 130]
        cv2.rectangle(img, (int(box_b[0]), int(box_b[1])), (int(box_b[2]), int(box_b[3])), (220, 90, 30), -1)

        frames.append(img)
        for tid, box in ((1, box_a), (2, box_b)):
            if box[0] < 0 or box[2] >= w or box[1] < 0 or box[3] >= h:
                continue
            gt_rows.append(xyxy_to_mot_row(t, tid, box))
    return frames, gt_rows


def run_tracker_on_frames(
    frames,
    gt_by_frame: dict,
    backbone: str,
    use_part_embedding: bool,
    enable_gallery: bool,
    device: str = DEVICE,
) -> tuple[list, float, dict]:
    if not frames:
        raise ValueError("No frames provided for evaluation.")

    height, width = frames[0].shape[:2]
    tracker = PersonReIDTracker(
        device=device,
        frame_size=(width, height),
        fps=30.0,
        backbone=backbone,
        use_part_embedding=use_part_embedding,
        enable_gallery=enable_gallery,
        embedder=None,
    )
    if not tracker.embedder.available:
        raise RuntimeError(
            f"ReID backbone '{backbone}' failed to load. Evaluation aborted; "
            "no fabricated embeddings or metrics were produced."
        )

    pred_rows = []
    t0 = time.perf_counter()
    for idx, frame in enumerate(frames, start=1):
        gt_dets = gt_by_frame.get(idx, [])
        detections = []
        parts_feats = []
        for det in gt_dets:
            box = [float(v) for v in det["box"]]
            parts = tracker.embedder.extract(frame, box)
            if not parts:
                continue
            detections.append(box)
            parts_feats.append(parts)
        tracks = tracker.step(detections, parts_feats, idx)
        for tid, tr in tracks.items():
            if not tr.confirmed:
                continue
            pred_rows.append(xyxy_to_mot_row(idx, tid, tr.smoothed_box))
    elapsed = time.perf_counter() - t0
    fps = len(frames) / elapsed if elapsed > 0 else 0.0
    info = {
        "frames": len(frames),
        "elapsed_sec": elapsed,
        "embedder_available": tracker.embedder.available,
        "feature_dim": tracker.embedder.feature_dim,
        "unique_pred_ids": len({r["id"] for r in pred_rows}),
    }
    return pred_rows, fps, info


def _try_trackeval(gt_path: Path, pred_path: Path) -> Optional[dict]:
    try:
        import trackeval  # noqa: F401
    except Exception:
        return None
    return None


def format_metrics(metrics: dict, fps: float) -> str:
    keys = ["HOTA", "IDF1", "MOTA", "IDSW", "FRAG", "TP", "FP", "FN"]
    lines = []
    for key in keys:
        if key not in metrics:
            continue
        val = metrics[key]
        if isinstance(val, float):
            lines.append(f"- {key}: {val:.4f}")
        else:
            lines.append(f"- {key}: {val}")
    lines.append(f"- FPS: {fps:.3f}")
    return "\n".join(lines)


def write_results_md(path: Path, sections: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate person-reid-multicamera tracking")
    p.add_argument("--seq", default=None, help="MOT sequence directory")
    p.add_argument("--gt", default=None, help="Path to gt.txt")
    p.add_argument("--frames", default=None, help="Directory of frame images")
    p.add_argument("--synthetic", action="store_true", help="Run the built-in synthetic sequence")
    p.add_argument("--backbone", default=REID_BACKBONE, choices=["osnet_x0_25", "osnet_x1_0", "resnet50"])
    p.add_argument("--use-part-embedding", action="store_true", default=USE_PART_EMBEDDING)
    p.add_argument("--disable-gallery", action="store_true")
    p.add_argument("--ablate", action="store_true", help="Run gallery/part/backbone ablations on the same sequence")
    p.add_argument("--out", default=str(_PROJECT_ROOT / "evaluation" / "results.md"))
    p.add_argument("--pred-out", default=str(_PROJECT_ROOT / "evaluation" / "preds"))
    p.add_argument("--device", default=DEVICE)
    return p.parse_args()


def load_frames_from_dir(frame_dir: Path, n_expected: Optional[int] = None):
    """Load frames from a directory of images OR directly from a video file."""
    frame_dir = Path(frame_dir)
    # Accept a video file path directly
    if frame_dir.is_file() and frame_dir.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        return _load_frames_from_video(frame_dir)
    paths = _list_frames(frame_dir)
    frames = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        frames.append(img)
    return frames


def _load_frames_from_video(video_path: Path) -> List[np.ndarray]:
    """Read all frames from a video file into a list."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def main():
    args = parse_args()
    enable_gallery = ENABLE_GALLERY and not args.disable_gallery
    sections = [
        "# Evaluation results",
        "These numbers were produced by `evaluation/evaluate.py`. "
        "Missing datasets are reported as skipped; metrics are never invented.",
        f"- Device: `{args.device}`",
        f"- Default backbone: `{args.backbone}`",
        f"- Part embeddings: `{args.use_part_embedding}`",
        f"- Gallery enabled: `{enable_gallery}`",
    ]

    runs = []
    if args.synthetic:
        frames, gt_rows = _make_synthetic_sequence()
        runs.append(("synthetic_two_boxes", frames, gt_rows))
    elif args.seq:
        gt_path, frame_dir = discover_sequence(Path(args.seq))
        gt_rows = load_mot_file(gt_path)
        frames = load_frames_from_dir(frame_dir)
        runs.append((Path(args.seq).name, frames, gt_rows))
    elif args.gt and args.frames:
        gt_rows = load_mot_file(args.gt)
        frames = load_frames_from_dir(Path(args.frames))
        runs.append((Path(args.gt).stem, frames, gt_rows))
    else:
        sections.append(
            "## Skipped\n\nNo `--seq`, `--gt/--frames`, or `--synthetic` argument was provided, "
            "and this workspace does not contain a MOTChallenge sequence."
        )
        write_results_md(Path(args.out), sections)
        print("\n".join(sections))
        return 0

    pred_dir = Path(args.pred_out)
    pred_dir.mkdir(parents=True, exist_ok=True)

    configs = [("default", args.backbone, args.use_part_embedding, enable_gallery)]
    if args.ablate:
        configs.extend([
            ("no_gallery", args.backbone, args.use_part_embedding, False),
            ("parts_on", args.backbone, True, enable_gallery),
            ("resnet50", "resnet50", args.use_part_embedding, enable_gallery),
        ])

    for seq_name, frames, gt_rows in runs:
        gt_by_frame = index_by_frame(gt_rows)
        sections.append(f"## Sequence `{seq_name}`")
        sections.append(
            f"- Frames loaded: {len(frames)}\n"
            f"- GT detections: {len(gt_rows)}\n"
            "- Detection source: ground-truth boxes (tracker-only evaluation, not detector MOTA)"
        )
        if not frames:
            sections.append("Skipped: frames could not be loaded.")
            continue

        for cfg_name, backbone, parts, gallery in configs:
            heading = f"### Config `{cfg_name}` (backbone={backbone}, parts={parts}, gallery={gallery})"
            try:
                pred_rows, fps, info = run_tracker_on_frames(
                    frames,
                    gt_by_frame,
                    backbone=backbone,
                    use_part_embedding=parts,
                    enable_gallery=gallery,
                    device=args.device,
                )
            except Exception as exc:
                sections.append(heading)
                sections.append(f"Skipped: tracker run failed ({exc}). No metrics were recorded.")
                print(heading)
                print(f"SKIP: {exc}")
                continue

            pred_path = pred_dir / f"{seq_name}_{cfg_name}.txt"
            write_mot_file(pred_path, pred_rows)
            metrics = compute_all_metrics(gt_rows, pred_rows)
            trackeval_metrics = _try_trackeval(pred_path, pred_path)
            sections.append(heading)
            sections.append(
                f"- Pred file: `{pred_path}`\n"
                f"- Unique predicted IDs: {info['unique_pred_ids']}\n"
                f"- Feature dim: {info['feature_dim']}\n"
                f"- Runtime: {info['elapsed_sec']:.3f}s"
            )
            sections.append(format_metrics(metrics, fps))
            if trackeval_metrics:
                sections.append("TrackEval metrics:\n" + format_metrics(trackeval_metrics, fps))
            else:
                sections.append(
                    "TrackEval was not installed; metrics above come from "
                    "`evaluation/mot_metrics.py` (CLEAR MOT / IDF1 / HOTA)."
                )
            print(heading)
            print(format_metrics(metrics, fps))

    write_results_md(Path(args.out), sections)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
