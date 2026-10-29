import numpy as np
import pytest

from ui import person_search


class _FakeData:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.values, dtype=np.float32).reshape(-1, 6)


class _FakeBoxes:
    def __init__(self, values):
        self.data = _FakeData(values)


class _FakeResult:
    def __init__(self, values):
        self.boxes = _FakeBoxes(values)


class _FakeModel:
    def predict(
        self,
        source,
        classes=None,
        conf=0.25,
        device="cpu",
        verbose=False,
        half=False,
        imgsz=640,
    ):
        _ = (classes, conf, device, verbose, half, imgsz)
        frame_marker = int(source[0, 0, 0])
        values = [[10, 10, 30, 35, 0.9, 0]] if frame_marker in {1, 2, 5} else []
        return [_FakeResult(values)]


class _FakeEmbedder:
    available = True

    def extract(self, frame: np.ndarray, box: list[float]) -> list[np.ndarray]:
        if frame.size == 0 or len(box) != 4:
            return []
        return [np.array([1.0, 0.0], dtype=np.float32)]

    def compute_weighted_mean(
        self,
        embeds: list[np.ndarray],
        weights: list[float] | None = None,
    ) -> np.ndarray | None:
        if not embeds:
            return None
        vector = (
            np.average(embeds, axis=0, weights=weights)
            if weights is not None and len(weights) == len(embeds)
            else np.mean(embeds, axis=0)
        )
        return vector / np.linalg.norm(vector)


class _FakeCapture:
    def __init__(self, frames):
        self.frames = frames

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == person_search.cv2.CAP_PROP_FPS:
            return 2.0
        if prop == person_search.cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        return 0

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        pass


def test_search_video_set_groups_matches_into_appearances(monkeypatch, tmp_path):
    frames = [
        np.full((40, 40, 3), marker, dtype=np.uint8)
        for marker in range(1, 7)
    ]
    monkeypatch.setattr(
        person_search.cv2,
        "VideoCapture",
        lambda _: _FakeCapture(frames.copy()),
    )

    results = person_search.search_video_set(
        query_image=np.zeros((20, 20, 3), dtype=np.uint8),
        video_paths=[tmp_path / "sample.mp4"],
        yolo_model=_FakeModel(),
        embedder=_FakeEmbedder(),
        similarity_distance_threshold=0.4,
        confidence_threshold=0.45,
        device="cpu",
    )

    assert len(results) == 2
    assert (results[0]["start_frame"], results[0]["end_frame"]) == (1, 2)
    assert results[0]["matched_frames"] == 2
    assert results[0]["start_time"] == "00:00.50"
    assert results[1]["start_frame"] == results[1]["end_frame"] == 5
    assert results[1]["best_similarity"] == pytest.approx(1.0)
    assert results[1]["best_detection_confidence"] == pytest.approx(0.9)
    assert results[1]["best_frame"].shape == (40, 40, 3)


def test_search_video_set_rejects_an_unusable_query_image(tmp_path):
    with pytest.raises(ValueError, match="could not be decoded"):
        person_search.search_video_set(
            query_image=np.empty((0, 0, 3), dtype=np.uint8),
            video_paths=[tmp_path / "sample.mp4"],
            yolo_model=_FakeModel(),
            embedder=_FakeEmbedder(),
            similarity_distance_threshold=0.4,
            confidence_threshold=0.45,
            device="cpu",
        )


def test_search_video_set_reports_videos_without_matches(monkeypatch, tmp_path):
    frames = [
        np.full((40, 40, 3), marker, dtype=np.uint8)
        for marker in (7, 8)
    ]
    monkeypatch.setattr(
        person_search.cv2,
        "VideoCapture",
        lambda _: _FakeCapture(frames.copy()),
    )

    results = person_search.search_video_set(
        query_image=np.zeros((20, 20, 3), dtype=np.uint8),
        video_paths=[tmp_path / "no_match.mp4"],
        yolo_model=_FakeModel(),
        embedder=_FakeEmbedder(),
        similarity_distance_threshold=0.4,
        confidence_threshold=0.45,
        device="cpu",
    )

    assert results == [{"video": "no_match.mp4", "not_found": True}]
