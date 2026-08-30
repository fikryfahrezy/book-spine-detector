from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


Orientation = Literal["auto", "vertical", "horizontal"]


@dataclass(frozen=True)
class CountResult:
    count: int
    orientation: Literal["vertical", "horizontal"]
    boundary_positions: tuple[int, ...]
    confidence: float
    profile: np.ndarray


def _normalized_profile(
    probability: np.ndarray,
    orientation: Literal["vertical", "horizontal"],
    smoothing_window: int,
) -> np.ndarray:
    # Vertical boundaries are projected over rows and located on the x-axis.
    axis = 0 if orientation == "vertical" else 1
    profile = probability.mean(axis=axis).astype(np.float32)
    window = max(1, int(smoothing_window))
    if window > 1:
        profile = uniform_filter1d(profile, size=window, mode="nearest")
    low = float(np.percentile(profile, 10))
    high = float(np.percentile(profile, 99.5))
    if high <= low + 1e-8:
        return np.zeros_like(profile)
    return np.clip((profile - low) / (high - low), 0.0, 1.0)


def _find_profile_peaks(
    profile: np.ndarray,
    *,
    peak_height: float,
    peak_prominence: float,
    minimum_peak_distance: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    return find_peaks(
        profile,
        height=peak_height,
        prominence=peak_prominence,
        distance=max(1, int(minimum_peak_distance)),
    )


def count_from_probability(
    probability: np.ndarray,
    *,
    orientation: Orientation = "auto",
    pixel_threshold: float = 0.45,
    peak_height: float = 0.30,
    peak_prominence: float = 0.08,
    smoothing_window: int = 9,
    minimum_peak_distance: int = 8,
) -> CountResult:
    if probability.ndim != 2:
        raise ValueError(f"Expected a 2-D probability map, received {probability.shape}")

    filtered = np.where(probability >= pixel_threshold, probability, 0.0)
    candidates: dict[str, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = {}
    orientations = ("vertical", "horizontal") if orientation == "auto" else (orientation,)

    for candidate in orientations:
        profile = _normalized_profile(filtered, candidate, smoothing_window)
        peaks, properties = _find_profile_peaks(
            profile,
            peak_height=peak_height,
            peak_prominence=peak_prominence,
            minimum_peak_distance=minimum_peak_distance,
        )
        candidates[candidate] = (profile, peaks, properties)

    def candidate_score(item):
        _, (_, peaks, properties) = item
        prominences = properties.get("prominences", np.empty(0))
        if not len(peaks):
            return 0.0
        return float(prominences.mean()) * min(len(peaks), 8)

    selected_orientation, (profile, peaks, properties) = max(
        candidates.items(), key=candidate_score
    )
    prominences = properties.get("prominences", np.empty(0))
    peak_heights = properties.get("peak_heights", np.empty(0))
    confidence = 0.0
    if len(peaks):
        confidence = float(
            np.clip(0.55 * peak_heights.mean() + 0.45 * prominences.mean(), 0.0, 1.0)
        )

    # Masks contain both outer edges. N books therefore produce N+1 unique boundaries.
    count = max(0, len(peaks) - 1)
    return CountResult(
        count=count,
        orientation=selected_orientation,
        boundary_positions=tuple(int(position) for position in peaks),
        confidence=confidence,
        profile=profile,
    )

