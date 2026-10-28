import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ui.pipeline import run_tracking_pipeline, cleanup_temp_files, cleanup_old_tracked_videos
from ui.sidebar import render_sidebar
from ui.uploader import render_uploader
from ui.results import render_results

__all__ = [
    "run_tracking_pipeline",
    "cleanup_temp_files",
    "cleanup_old_tracked_videos",
    "render_sidebar",
    "render_uploader",
    "render_results",
]
