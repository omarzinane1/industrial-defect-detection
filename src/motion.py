"""Estimation simple du mouvement par Lucas-Kanade pour le flux camera."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class MotionEstimate:
    """Resume lisible de l'etat du mouvement entre deux frames."""

    status: str
    motion_score: float
    mean_displacement: float
    max_displacement: float
    tracked_points: int
    detected_points: int
    capture_quality: str
    message: str
    flow_available: bool
    should_block_prediction: bool = False
    reinitialized: bool = False


def prepare_motion_frame(frame: np.ndarray) -> np.ndarray:
    """Prepare une frame grayscale stable pour l'optical flow."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    gray = np.asarray(gray, dtype=np.uint8)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def initialize_tracking_points(
    frame_gray: np.ndarray,
    max_corners: int = 120,
    quality_level: float = 0.01,
    min_distance: int = 10,
    block_size: int = 7,
) -> np.ndarray | None:
    """Detecte des points interessants a suivre avec Shi-Tomasi."""
    points = cv2.goodFeaturesToTrack(
        frame_gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
    )
    return points


def _count_points(points: np.ndarray | None) -> int:
    if points is None:
        return 0
    return int(len(points))


def classify_motion(
    motion_score: float,
    stable_threshold: float = 0.8,
    unstable_threshold: float = 2.2,
) -> tuple[str, str, str, bool]:
    """Traduit un score de mouvement en etat interpretable."""
    if motion_score <= stable_threshold:
        return "Stable", "Prete", "Scene stable pour l'analyse.", False
    if motion_score <= unstable_threshold:
        return "En mouvement", "A surveiller", "Mouvement detecte. Stabilisez la piece pour une meilleure capture.", False
    return "Instable", "Faible", "Stabilisez la piece avant l'analyse.", True


def estimate_lucas_kanade_motion(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_points: np.ndarray | None,
    lk_params: dict[str, Any] | None = None,
    min_points: int = 8,
    stable_threshold: float = 0.8,
    unstable_threshold: float = 2.2,
) -> tuple[MotionEstimate, np.ndarray | None]:
    """Estime le mouvement entre deux frames avec Lucas-Kanade pyramidal."""
    detected_points = _count_points(previous_points)
    if previous_points is None or detected_points < min_points:
        estimate = MotionEstimate(
            status="Indisponible",
            motion_score=0.0,
            mean_displacement=0.0,
            max_displacement=0.0,
            tracked_points=0,
            detected_points=detected_points,
            capture_quality="A verifier",
            message="Mouvement impossible a calculer : points de suivi insuffisants.",
            flow_available=False,
        )
        return estimate, None

    params = lk_params or {
        "winSize": (21, 21),
        "maxLevel": 2,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    }

    next_points, status, _error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        None,
        **params,
    )

    if next_points is None or status is None:
        estimate = MotionEstimate(
            status="Indisponible",
            motion_score=0.0,
            mean_displacement=0.0,
            max_displacement=0.0,
            tracked_points=0,
            detected_points=detected_points,
            capture_quality="A verifier",
            message="Mouvement impossible a calculer sur cette frame.",
            flow_available=False,
        )
        return estimate, None

    valid_mask = status.reshape(-1) == 1
    good_old = previous_points.reshape(-1, 2)[valid_mask]
    good_new = next_points.reshape(-1, 2)[valid_mask]
    tracked_points = int(len(good_new))

    if tracked_points < min_points:
        estimate = MotionEstimate(
            status="Indisponible",
            motion_score=0.0,
            mean_displacement=0.0,
            max_displacement=0.0,
            tracked_points=tracked_points,
            detected_points=detected_points,
            capture_quality="A verifier",
            message="Points suivis insuffisants. Repositionnez la piece ou enrichissez la scene.",
            flow_available=False,
        )
        return estimate, None

    displacements = np.linalg.norm(good_new - good_old, axis=1)
    mean_displacement = float(np.mean(displacements))
    median_displacement = float(np.median(displacements))
    max_displacement = float(np.max(displacements))
    motion_score = float(0.7 * mean_displacement + 0.3 * median_displacement)
    status_name, capture_quality, message, should_block = classify_motion(
        motion_score,
        stable_threshold=stable_threshold,
        unstable_threshold=unstable_threshold,
    )

    estimate = MotionEstimate(
        status=status_name,
        motion_score=motion_score,
        mean_displacement=mean_displacement,
        max_displacement=max_displacement,
        tracked_points=tracked_points,
        detected_points=detected_points,
        capture_quality=capture_quality,
        message=message,
        flow_available=True,
        should_block_prediction=should_block,
    )
    return estimate, good_new.reshape(-1, 1, 2)


class MotionEstimator:
    """Gestionnaire simple de suivi Lucas-Kanade frame par frame."""

    def __init__(
        self,
        max_corners: int = 120,
        min_points: int = 8,
        redetect_threshold: int = 20,
        stable_threshold: float = 0.8,
        unstable_threshold: float = 2.2,
    ) -> None:
        self.max_corners = max_corners
        self.min_points = min_points
        self.redetect_threshold = redetect_threshold
        self.stable_threshold = stable_threshold
        self.unstable_threshold = unstable_threshold
        self.previous_gray: np.ndarray | None = None
        self.previous_points: np.ndarray | None = None

    def reset(self) -> None:
        """Reinitialise l'etat du suivi."""
        self.previous_gray = None
        self.previous_points = None

    def update(self, frame: np.ndarray) -> MotionEstimate:
        """Met a jour l'estimation a partir d'une nouvelle frame."""
        current_gray = prepare_motion_frame(frame)

        if self.previous_gray is None:
            self.previous_gray = current_gray
            self.previous_points = initialize_tracking_points(current_gray, max_corners=self.max_corners)
            detected_points = _count_points(self.previous_points)
            return MotionEstimate(
                status="Initialisation",
                motion_score=0.0,
                mean_displacement=0.0,
                max_displacement=0.0,
                tracked_points=0,
                detected_points=detected_points,
                capture_quality="A verifier",
                message="Initialisation du suivi Lucas-Kanade en cours.",
                flow_available=False,
            )

        if self.previous_points is None or _count_points(self.previous_points) < self.min_points:
            self.previous_points = initialize_tracking_points(self.previous_gray, max_corners=self.max_corners)

        estimate, next_points = estimate_lucas_kanade_motion(
            self.previous_gray,
            current_gray,
            self.previous_points,
            min_points=self.min_points,
            stable_threshold=self.stable_threshold,
            unstable_threshold=self.unstable_threshold,
        )

        if not estimate.flow_available or next_points is None:
            self.previous_gray = current_gray
            self.previous_points = initialize_tracking_points(current_gray, max_corners=self.max_corners)
            estimate.detected_points = _count_points(self.previous_points)
            estimate.reinitialized = True
            if estimate.detected_points >= self.min_points:
                estimate.message = f"{estimate.message} Suivi reinitialise automatiquement."
            return estimate

        if estimate.tracked_points < self.redetect_threshold:
            refreshed_points = initialize_tracking_points(current_gray, max_corners=self.max_corners)
            if refreshed_points is not None and _count_points(refreshed_points) >= self.min_points:
                self.previous_points = refreshed_points
                estimate.detected_points = _count_points(refreshed_points)
                estimate.reinitialized = True
            else:
                self.previous_points = next_points
                estimate.detected_points = estimate.tracked_points
        else:
            self.previous_points = next_points
            estimate.detected_points = estimate.tracked_points

        self.previous_gray = current_gray
        return estimate
