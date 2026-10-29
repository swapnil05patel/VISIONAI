"""Search the sample videos for a person supplied as a reference image."""

import hashlib
import sys
from pathlib import Path
from typing import Callable, Optional, Protocol

import cv2
import numpy as np
import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tracker.config import REID_BACKBONE, REID_STRICT_THRESH, USE_PART_EMBEDDING
from tracker.embedder import PartEmbedder
from tracker.utils import cosine_distance_vec
from ui.pipeline import load_yolo_model


class SearchEmbedder(Protocol):
    available: bool

    def extract(self, frame: np.ndarray, box: list[float]) -> list[np.ndarray]: ...

    def compute_weighted_mean(
        self,
        embeds: list[np.ndarray],
        weights: Optional[list[float]] = None,
    ) -> Optional[np.ndarray]: ...


@st.cache_resource
def load_search_embedder(device: str) -> PartEmbedder:
    return PartEmbedder(
        device=device,
        backbone=REID_BACKBONE,
        use_part_embedding=USE_PART_EMBEDDING,
    )


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:05.2f}"


def _new_segment(
    video_name: str,
    frame_number: int,
    distance: float,
    frame: np.ndarray,
    box: np.ndarray,
    confidence: float,
    fps: float,
) -> dict:
    timestamp = _format_timestamp(frame_number / fps)
    return {
        "video": video_name,
        "start_frame": frame_number,
        "end_frame": frame_number,
        "start_time": timestamp,
        "end_time": timestamp,
        "matched_frames": 1,
        "best_similarity": 1.0 - distance,
        "best_frame_number": frame_number,
        "best_frame": frame.copy(),
        "best_box": [int(round(value)) for value in box],
        "best_detection_confidence": confidence,
    }


def search_video_set(
    query_image: np.ndarray,
    video_paths: list[Path],
    yolo_model,
    embedder: SearchEmbedder,
    similarity_distance_threshold: float,
    confidence_threshold: float,
    device: str,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
) -> list[dict]:
    """Scan every frame in each video and group nearby matches into appearances."""
    if query_image is None or query_image.size == 0:
        raise ValueError("The uploaded image could not be decoded.")
    if not embedder.available:
        raise RuntimeError("The person re-identification model could not be loaded.")

    image_height, image_width = query_image.shape[:2]
    query_parts = embedder.extract(
        query_image,
        [0, 0, image_width, image_height],
    )
    query_embedding = embedder.compute_weighted_mean(query_parts)
    if query_embedding is None:
        raise ValueError(
            "Could not create an embedding from this image. Upload a clear image "
            "with the person visible."
        )

    results = []
    for video_index, video_path in enumerate(video_paths):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open video: {video_path.name}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 1.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        gap_tolerance = max(1, int(round(fps * 0.5)))
        current_segment = None
        video_segments = []
        frame_number = 0
        progress_interval = max(1, total_frames // 100)

        try:
            while True:
                has_frame, frame = capture.read()
                if not has_frame:
                    break
                frame_number += 1

                detections = yolo_model.predict(
                    source=frame,
                    classes=[0],
                    conf=confidence_threshold,
                    device=device,
                    verbose=False,
                    half=(device == "cuda"),
                    imgsz=640,
                )
                boxes = detections[0].boxes.data.cpu().numpy()

                best_match = None
                for detection in boxes:
                    box = detection[:4].astype(np.float32)
                    parts = embedder.extract(frame, box.tolist())
                    embedding = embedder.compute_weighted_mean(parts)
                    if embedding is None:
                        continue

                    distance = cosine_distance_vec(query_embedding, embedding)
                    if distance <= similarity_distance_threshold and (
                        best_match is None or distance < best_match[0]
                    ):
                        best_match = (distance, box, float(detection[4]))

                if best_match is not None:
                    distance, box, confidence = best_match
                    if (
                        current_segment is None
                        or frame_number - current_segment["end_frame"] > gap_tolerance
                    ):
                        if current_segment is not None:
                            video_segments.append(current_segment)
                        current_segment = _new_segment(
                            video_path.name,
                            frame_number,
                            distance,
                            frame,
                            box,
                            confidence,
                            fps,
                        )
                    else:
                        current_segment["end_frame"] = frame_number
                        current_segment["end_time"] = _format_timestamp(frame_number / fps)
                        current_segment["matched_frames"] += 1
                        similarity = 1.0 - distance
                        if similarity > current_segment["best_similarity"]:
                            current_segment["best_similarity"] = similarity
                            current_segment["best_frame_number"] = frame_number
                            current_segment["best_frame"] = frame.copy()
                            current_segment["best_box"] = [
                                int(round(value)) for value in box
                            ]
                            current_segment["best_detection_confidence"] = confidence

                if progress_callback and (
                    frame_number % progress_interval == 0
                    or (total_frames > 0 and frame_number == total_frames)
                ):
                    progress_callback(
                        video_index,
                        frame_number,
                        total_frames,
                        video_path.name,
                    )

            if current_segment is not None:
                video_segments.append(current_segment)
        finally:
            capture.release()

        if video_segments:
            results.extend(video_segments)
        else:
            results.append({"video": video_path.name, "not_found": True})

    return results


def _render_search_results(results: list[dict]) -> None:
    if not results:
        st.info("No videos were searched.")
        return

    display_rows = []
    for result in results:
        if result.get("not_found"):
            display_rows.append({
                "Video": result["video"],
                "Status": "No match found",
                "First seen": "",
                "Last seen": "",
                "Matched frames": 0,
                "Best similarity": "",
                "Detector confidence": "",
            })
        else:
            display_rows.append({
                "Video": result["video"],
                "Status": "Match found",
                "First seen": result["start_time"],
                "Last seen": result["end_time"],
                "Matched frames": result["matched_frames"],
                "Best similarity": f'{result["best_similarity"]:.3f}',
                "Detector confidence": f'{result["best_detection_confidence"]:.3f}',
            })
    st.dataframe(pd.DataFrame(display_rows), hide_index=True, width="stretch")
    st.download_button(
        "Download search results",
        data=pd.DataFrame(display_rows).to_csv(index=False),
        file_name="person_search_results.csv",
        mime="text/csv",
        width="stretch",
    )

    for result in results:
        if result.get("not_found"):
            continue
        caption = (
            f'{result["video"]} — {result["start_time"]} to {result["end_time"]}; '
            f'best match at frame {result["best_frame_number"]}, '
            f'similarity {result["best_similarity"]:.3f}'
        )
        with st.expander(caption):
            preview = result["best_frame"].copy()
            x1, y1, x2, y2 = result["best_box"]
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            st.image(preview, channels="BGR", caption=caption)


def render_person_search() -> None:
    st.header("Find a Person in Test Videos")
    st.caption(
        "Upload a clear image of the person. The app scans every video in "
        "`test_data/` and groups matching frames into appearances."
    )
    uploaded_image = st.file_uploader(
        "Choose a reference image",
        type=["jpg", "jpeg", "png", "webp"],
        help="A clear, full-body image usually gives the best person match.",
        key="person_search_image",
    )

    if uploaded_image is None:
        st.session_state.pop("person_search_results", None)
        st.session_state.pop("person_search_digest", None)
        return

    image_bytes = uploaded_image.getvalue()
    query_digest = hashlib.sha256(image_bytes).hexdigest()
    if st.session_state.get("person_search_digest") != query_digest:
        st.session_state.pop("person_search_results", None)
        st.session_state["person_search_digest"] = query_digest

    st.image(image_bytes, caption=f"Reference image: {uploaded_image.name}", width=240)
    similarity_distance_threshold = st.slider(
        "Maximum cosine distance",
        min_value=0.10,
        max_value=0.80,
        value=float(REID_STRICT_THRESH),
        step=0.01,
        help="Lower values require a closer appearance match; higher values can find "
        "more matches but may include false positives.",
        key="person_search_threshold",
    )

    if st.button("Search all test videos", type="primary", key="search_test_videos"):
        video_root = _PROJECT_ROOT / "test_data"
        supported_extensions = {".mp4", ".mov", ".avi", ".mkv"}
        video_paths = sorted(
            path for path in video_root.rglob("*")
            if path.is_file() and path.suffix.lower() in supported_extensions
        )
        if not video_paths:
            st.error(f"No supported videos were found under {video_root}.")
            return

        query_image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if query_image is None:
            st.error("Could not read the uploaded image. Try another image file.")
            return
        yolo_model, device = load_yolo_model()
        if yolo_model is None or device is None:
            st.error("Could not load the YOLO person detector.")
            return

        embedder = load_search_embedder(device)
        if not embedder.available:
            st.error("Could not load the person re-identification model.")
            return

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(
            video_index: int,
            frame_number: int,
            total_frames: int,
            video_name: str,
        ) -> None:
            fraction = frame_number / total_frames if total_frames > 0 else 0.0
            progress_bar.progress((video_index + fraction) / len(video_paths))
            status_text.text(
                f"Scanning {video_name}: frame {frame_number}/{total_frames}"
            )

        try:
            search_results = search_video_set(
                query_image=query_image,
                video_paths=video_paths,
                yolo_model=yolo_model,
                embedder=embedder,
                similarity_distance_threshold=similarity_distance_threshold,
                confidence_threshold=0.45,
                device=device,
                progress_callback=update_progress,
            )
        except (RuntimeError, ValueError) as error:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Person search failed: {error}")
            return

        progress_bar.empty()
        status_text.empty()
        st.session_state["person_search_results"] = search_results

    search_results = st.session_state.get("person_search_results")
    if (
        search_results is not None
        and st.session_state.get("person_search_digest") == query_digest
    ):
        _render_search_results(search_results)
