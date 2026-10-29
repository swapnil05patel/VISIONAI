"""
scripts/generate_test_assets.py
--------------------------------
Generates a complete set of synthetic test videos and images for
person-reid-multicamera without requiring any external dataset.

All assets are written to test_data/ and are immediately usable by:
  - scripts/run_inference.py  (headless tracking)
  - evaluation/evaluate.py    (MOT-format evaluation)
  - streamlit run app.py      (upload the videos)

Scenarios generated
-------------------
Videos (test_data/videos/):
  1. single_person.mp4          — one person walking left to right
  2. two_persons_cross.mp4      — two persons crossing paths (occlusion test)
  3. reentry_gap.mp4            — person leaves frame, re-enters (gallery test)
  4. crowd_five.mp4             — five persons moving simultaneously
  5. static_object.mp4          — one moving person + one static mannequin
  6. fast_motion.mp4            — fast-moving person (Kalman/velocity gate test)

Images (test_data/images/query/ and gallery/):
  query_person_A.png            — query crop of person A
  query_person_B.png            — query crop of person B
  gallery_person_A_1.png        — gallery image of person A (same appearance)
  gallery_person_A_2.png        — gallery image of person A (slightly different pose)
  gallery_person_B_1.png        — gallery image of person B
  gallery_person_C_1.png        — gallery image of person C (distractor)

Annotations (test_data/annotations/):
  <video_name>.txt              — MOT-format ground-truth for each video

Usage
-----
  python scripts/generate_test_assets.py
  python scripts/generate_test_assets.py --out-dir test_data --fps 30 --size 640x480
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── types ──────────────────────────────────────────────────────────────────
Color = Tuple[int, int, int]
Box   = Tuple[float, float, float, float]   # x1 y1 x2 y2

FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


# ── drawing helpers ────────────────────────────────────────────────────────

def _blank(h: int, w: int, bg: Color = (45, 45, 45)) -> np.ndarray:
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    return img


def _draw_person(
    img: np.ndarray,
    box: Box,
    body_color: Color,
    head_color: Color,
    label: str = "",
) -> None:
    """Draw a stylised person: filled rectangle body + circle head."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1] - 1, x2); y2 = min(img.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return

    bw = x2 - x1
    bh = y2 - y1
    head_r = max(4, bw // 3)
    body_y1 = y1 + head_r * 2

    # body
    cv2.rectangle(img, (x1, body_y1), (x2, y2), body_color, -1)
    # head
    cx = (x1 + x2) // 2
    cy = y1 + head_r
    cv2.circle(img, (cx, cy), head_r, head_color, -1)
    # outline
    cv2.rectangle(img, (x1, body_y1), (x2, y2), (0, 0, 0), 1)
    cv2.circle(img, (cx, cy), head_r, (0, 0, 0), 1)

    if label:
        cv2.putText(img, label, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def _draw_background(img: np.ndarray, frame_idx: int) -> None:
    """Add subtle floor line and frame counter."""
    h, w = img.shape[:2]
    cv2.line(img, (0, int(h * 0.85)), (w, int(h * 0.85)), (80, 80, 80), 1)
    cv2.putText(img, f"f{frame_idx:03d}", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)


def _clamp_box(box: Box, w: int, h: int) -> Box:
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(x1, w - 2)),
        max(0.0, min(y1, h - 2)),
        max(1.0, min(x2, float(w))),
        max(1.0, min(y2, float(h))),
    )


def _box_visible(box: Box, w: int, h: int, margin: int = 4) -> bool:
    x1, y1, x2, y2 = box
    return x2 > margin and y2 > margin and x1 < w - margin and y1 < h - margin


def _mot_row(frame: int, tid: int, box: Box) -> str:
    x1, y1, x2, y2 = box
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return f"{frame},{tid},{x1:.1f},{y1:.1f},{bw:.1f},{bh:.1f},1,1,1\n"


def _write_video(path: Path, frames: List[np.ndarray], fps: int) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), FOURCC, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {path}")
    for f in frames:
        writer.write(f)
    writer.release()
    kb = path.stat().st_size // 1024
    print(f"  wrote {path.name}  ({len(frames)} frames, {kb} KB)")


def _write_gt(path: Path, rows: List[str]) -> None:
    path.write_text("".join(rows))
    print(f"  wrote {path.name}  ({len(rows)} GT rows)")


# ── scenario generators ────────────────────────────────────────────────────

def gen_single_person(h: int, w: int, fps: int, n_frames: int = 90):
    """One person walking left → right across the frame."""
    frames, gt = [], []
    body = (60, 120, 220)
    head = (220, 180, 140)
    bw, bh = 50, 130

    for t in range(1, n_frames + 1):
        img = _blank(h, w)
        _draw_background(img, t)
        x1 = -bw + t * (w + bw) / n_frames
        y1 = h * 0.35
        box: Box = _clamp_box((x1, y1, x1 + bw, y1 + bh), w, h)
        if _box_visible(box, w, h):
            _draw_person(img, box, body, head, "P1")
            gt.append(_mot_row(t, 1, box))
        frames.append(img)
    return frames, gt


def gen_two_persons_cross(h: int, w: int, fps: int, n_frames: int = 90):
    """Two persons crossing paths — tests ID preservation through occlusion."""
    frames, gt = [], []
    body_a, head_a = (60, 120, 220), (220, 180, 140)   # blue
    body_b, head_b = (220, 80,  40), (240, 200, 160)   # red
    bw, bh = 48, 125

    for t in range(1, n_frames + 1):
        img = _blank(h, w)
        _draw_background(img, t)

        # A: left → right, upper lane
        xa = -bw + t * (w + bw) / n_frames
        ya = h * 0.25
        box_a: Box = _clamp_box((xa, ya, xa + bw, ya + bh), w, h)

        # B: right → left, lower lane (crosses A around frame 45)
        xb = w + bw - t * (w + bw) / n_frames
        yb = h * 0.45
        box_b: Box = _clamp_box((xb, yb, xb + bw, yb + bh), w, h)

        # draw back-to-front by x position so overlap looks natural
        if xa < xb:
            if _box_visible(box_a, w, h):
                _draw_person(img, box_a, body_a, head_a, "A")
            if _box_visible(box_b, w, h):
                _draw_person(img, box_b, body_b, head_b, "B")
        else:
            if _box_visible(box_b, w, h):
                _draw_person(img, box_b, body_b, head_b, "B")
            if _box_visible(box_a, w, h):
                _draw_person(img, box_a, body_a, head_a, "A")

        if _box_visible(box_a, w, h):
            gt.append(_mot_row(t, 1, box_a))
        if _box_visible(box_b, w, h):
            gt.append(_mot_row(t, 2, box_b))
        frames.append(img)
    return frames, gt


def gen_reentry_gap(h: int, w: int, fps: int, n_frames: int = 120):
    """
    Person walks in, exits right, re-enters from right after a gap.
    Tests gallery recovery: the revived track should keep the original ID.
    Timeline:
      frames  1-40  : person walks left → right (visible)
      frames 41-70  : person off-screen (gap — gallery should store them)
      frames 71-120 : person re-enters from right, walks left
    """
    frames, gt = [], []
    body = (40, 180, 80)
    head = (200, 220, 160)
    bw, bh = 52, 135

    for t in range(1, n_frames + 1):
        img = _blank(h, w)
        _draw_background(img, t)

        if t <= 40:
            # walk right
            x1 = 20 + (t - 1) * (w - 20) / 40
            y1 = h * 0.30
            box: Box = _clamp_box((x1, y1, x1 + bw, y1 + bh), w, h)
            if _box_visible(box, w, h):
                _draw_person(img, box, body, head, "P1")
                gt.append(_mot_row(t, 1, box))
        elif t <= 70:
            # off-screen — no detection, no GT row
            pass
        else:
            # re-enter from right, walk left
            progress = (t - 71) / (n_frames - 71 + 1)
            x1 = w - bw - progress * (w - 40)
            y1 = h * 0.30
            box = _clamp_box((x1, y1, x1 + bw, y1 + bh), w, h)
            if _box_visible(box, w, h):
                _draw_person(img, box, body, head, "P1")
                gt.append(_mot_row(t, 1, box))

        frames.append(img)
    return frames, gt


def gen_crowd_five(h: int, w: int, fps: int, n_frames: int = 90):
    """Five persons moving simultaneously — tests multi-person assignment."""
    frames, gt = [], []

    persons = [
        # (body_color, head_color, start_x_frac, y_frac, direction, bw, bh)
        ((60,  120, 220), (220, 180, 140), 0.0,  0.15, +1, 46, 120),
        ((220,  80,  40), (240, 200, 160), 1.0,  0.30, -1, 50, 130),
        ((40,  180,  80), (200, 220, 160), 0.0,  0.45, +1, 44, 115),
        ((200, 160,  40), (240, 220, 180), 1.0,  0.60, -1, 48, 125),
        ((160,  40, 200), (220, 180, 240), 0.0,  0.72, +1, 52, 135),
    ]

    for t in range(1, n_frames + 1):
        img = _blank(h, w)
        _draw_background(img, t)
        progress = (t - 1) / (n_frames - 1)

        for pid, (body, head_c, sx, yf, direction, bw, bh) in enumerate(persons, start=1):
            if direction == +1:
                x1 = -bw + progress * (w + bw)
            else:
                x1 = w + bw - progress * (w + bw)
            y1 = h * yf
            box: Box = _clamp_box((x1, y1, x1 + bw, y1 + bh), w, h)
            if _box_visible(box, w, h):
                _draw_person(img, box, body, head_c, f"P{pid}")
                gt.append(_mot_row(t, pid, box))
        frames.append(img)
    return frames, gt


def gen_static_object(h: int, w: int, fps: int, n_frames: int = 90):
    """
    One moving person + one static mannequin/poster.
    Tests static-track pruning: the static object should be suppressed.
    """
    frames, gt = [], []
    body_m, head_m = (60, 120, 220), (220, 180, 140)   # moving person
    body_s, head_s = (160, 160, 160), (180, 180, 180)  # static object (gray)
    bw, bh = 50, 130

    # static object fixed position
    sx1, sy1 = w * 0.65, h * 0.30
    static_box: Box = (sx1, sy1, sx1 + bw, sy1 + bh)

    for t in range(1, n_frames + 1):
        img = _blank(h, w)
        _draw_background(img, t)

        # static object — always at same position, GT id=2
        _draw_person(img, static_box, body_s, head_s, "STATIC")
        gt.append(_mot_row(t, 2, static_box))

        # moving person — GT id=1
        x1 = -bw + t * (w + bw) / n_frames
        y1 = h * 0.35
        box: Box = _clamp_box((x1, y1, x1 + bw, y1 + bh), w, h)
        if _box_visible(box, w, h):
            _draw_person(img, box, body_m, head_m, "P1")
            gt.append(_mot_row(t, 1, box))

        frames.append(img)
    return frames, gt


def gen_fast_motion(h: int, w: int, fps: int, n_frames: int = 60):
    """
    Person moving very fast — tests velocity-gated dynamic search radius.
    Speed is ~3× normal so the Kalman velocity gate must expand the search area.
    """
    frames, gt = [], []
    body = (220, 140, 40)
    head = (240, 200, 160)
    bw, bh = 50, 130

    for t in range(1, n_frames + 1):
        img = _blank(h, w)
        _draw_background(img, t)
        # 3× speed: covers full width in 1/3 of frames, then bounces
        phase = (t - 1) % (n_frames // 2)
        direction = 1 if ((t - 1) // (n_frames // 2)) % 2 == 0 else -1
        x1 = (w // 2 - bw // 2) + direction * phase * (w // (n_frames // 2))
        y1 = h * 0.30
        box: Box = _clamp_box((x1, y1, x1 + bw, y1 + bh), w, h)
        if _box_visible(box, w, h):
            _draw_person(img, box, body, head, "FAST")
            gt.append(_mot_row(t, 1, box))
        frames.append(img)
    return frames, gt


# ── image generators ───────────────────────────────────────────────────────

def _person_crop(
    body_color: Color,
    head_color: Color,
    size: Tuple[int, int] = (128, 256),
    bg: Color = (50, 50, 50),
    noise: float = 0.0,
) -> np.ndarray:
    """Render a single person crop at the given size."""
    w, h = size
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    box: Box = (4.0, 4.0, float(w - 4), float(h - 4))
    _draw_person(img, box, body_color, head_color)
    if noise > 0:
        n = (np.random.randn(h, w, 3) * noise * 255).astype(np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    return img


def gen_reid_images(images_dir: Path) -> None:
    """Generate query and gallery person crops for image-mode ReID demo."""
    query_dir   = images_dir / "query"
    gallery_dir = images_dir / "gallery"
    query_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    persons = {
        "A": {"body": (60,  120, 220), "head": (220, 180, 140)},  # blue
        "B": {"body": (220,  80,  40), "head": (240, 200, 160)},  # red
        "C": {"body": (40,  180,  80), "head": (200, 220, 160)},  # green
        "D": {"body": (200, 160,  40), "head": (240, 220, 180)},  # yellow
    }

    for pid, colors in persons.items():
        body, head = colors["body"], colors["head"]

        # query: clean crop
        q = _person_crop(body, head, size=(128, 256), noise=0.0)
        qpath = query_dir / f"query_person_{pid}.png"
        cv2.imwrite(str(qpath), q)
        print(f"  wrote {qpath.name}")

        # gallery image 1: same appearance, slight background change
        g1 = _person_crop(body, head, size=(128, 256),
                          bg=(60, 55, 50), noise=0.02)
        g1path = gallery_dir / f"gallery_person_{pid}_1.png"
        cv2.imwrite(str(g1path), g1)
        print(f"  wrote {g1path.name}")

        # gallery image 2: same person, more noise (simulates different lighting)
        g2 = _person_crop(body, head, size=(128, 256),
                          bg=(40, 40, 45), noise=0.05)
        g2path = gallery_dir / f"gallery_person_{pid}_2.png"
        cv2.imwrite(str(g2path), g2)
        print(f"  wrote {g2path.name}")

    print(f"  {len(persons)} persons × 3 images = {len(persons)*3} total ReID images")


# ── main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic test videos and images for person-reid-multicamera"
    )
    p.add_argument("--out-dir", default="test_data",
                   help="Root output directory (default: test_data)")
    p.add_argument("--fps",  type=int, default=30,
                   help="Video frame rate (default: 30)")
    p.add_argument("--size", default="640x480",
                   help="Video resolution WxH (default: 640x480)")
    return p.parse_args()


def main():
    args = parse_args()
    w_str, h_str = args.size.split("x")
    W, H, FPS = int(w_str), int(h_str), args.fps

    root        = _PROJECT_ROOT / args.out_dir
    videos_dir  = root / "videos"
    annot_dir   = root / "annotations"
    images_dir  = root / "images"

    for d in (videos_dir, annot_dir, images_dir):
        d.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ("single_person",      gen_single_person,    90),
        ("two_persons_cross",  gen_two_persons_cross, 90),
        ("reentry_gap",        gen_reentry_gap,      120),
        ("crowd_five",         gen_crowd_five,        90),
        ("static_object",      gen_static_object,     90),
        ("fast_motion",        gen_fast_motion,       60),
    ]

    print(f"\nGenerating {len(scenarios)} test videos  ({W}x{H} @ {FPS}fps)")
    print("=" * 60)

    total_frames = 0
    for name, fn, n_frames in scenarios:
        print(f"\n[{name}]")
        frames, gt_rows = fn(H, W, FPS, n_frames)
        _write_video(videos_dir / f"{name}.mp4", frames, FPS)
        _write_gt(annot_dir / f"{name}.txt", gt_rows)
        total_frames += len(frames)

    print(f"\nGenerating ReID images")
    print("=" * 60)
    gen_reid_images(images_dir)

    print("\n" + "=" * 60)
    print(f"Done.  {total_frames} video frames + ReID images written to {root}/")
    print()
    print("Quick-start commands:")
    print(f"  # Track a video (headless)")
    print(f"  python scripts/run_inference.py --input {root}/videos/two_persons_cross.mp4")
    print()
    print(f"  # Evaluate with MOT metrics")
    print(f"  python evaluation/evaluate.py \\")
    print(f"      --gt {root}/annotations/reentry_gap.txt \\")
    print(f"      --frames {root}/videos/reentry_gap.mp4 \\")
    print(f"      --out evaluation/results.md")
    print()
    print(f"  # Image ReID ranking")
    print(f"  python scripts/run_inference.py \\")
    print(f"      --input {root}/images/query/query_person_A.png \\")
    print(f"      --mode image \\")
    print(f"      --gallery {root}/images/gallery/")
    print()
    print(f"  # Upload any video in the Streamlit app")
    print(f"  streamlit run app.py")


if __name__ == "__main__":
    main()
