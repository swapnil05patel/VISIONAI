"""MOT-format loaders and writers (frame,id,x,y,w,h,conf,class,vis)."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


MotFrame = Dict[int, List[dict]]


def _parse_row(parts: List[str]) -> dict:
    frame = int(float(parts[0]))
    tid = int(float(parts[1]))
    x = float(parts[2])
    y = float(parts[3])
    w = float(parts[4])
    h = float(parts[5])
    conf = float(parts[6]) if len(parts) > 6 else 1.0
    cls = int(float(parts[7])) if len(parts) > 7 else 1
    vis = float(parts[8]) if len(parts) > 8 else 1.0
    return {
        "frame": frame,
        "id": tid,
        "box": [x, y, x + w, y + h],
        "xywh": [x, y, w, h],
        "conf": conf,
        "cls": cls,
        "vis": vis,
    }


def load_mot_file(path, ignore_zero_conf: bool = True) -> List[dict]:
    """Load a MOT gt.txt / pred.txt file into a list of detection dicts."""
    path = Path(path)
    rows: List[dict] = []
    if not path.exists():
        raise FileNotFoundError(f"MOT file not found: {path}")
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            row = _parse_row(parts)
            if ignore_zero_conf and row["conf"] == 0:
                continue
            if row["id"] < 0:
                continue
            rows.append(row)
    return rows


def index_by_frame(rows: Iterable[dict]) -> MotFrame:
    by_frame: MotFrame = defaultdict(list)
    for row in rows:
        by_frame[int(row["frame"])].append(row)
    return dict(by_frame)


def write_mot_file(path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            x, y, w, h = row["xywh"]
            conf = row.get("conf", 1.0)
            cls = row.get("cls", 1)
            vis = row.get("vis", 1.0)
            handle.write(
                f"{int(row['frame'])},{int(row['id'])},"
                f"{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                f"{conf:.4f},{cls},{vis:.4f}\n"
            )


def xyxy_to_mot_row(frame_idx: int, tid: int, box: List[float], conf: float = 1.0) -> dict:
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return {
        "frame": int(frame_idx),
        "id": int(tid),
        "box": [x1, y1, x2, y2],
        "xywh": [x1, y1, w, h],
        "conf": float(conf),
        "cls": 1,
        "vis": 1.0,
    }
