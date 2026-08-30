import numpy as np

from book_spine_ml.postprocess import count_from_probability


def test_counts_vertical_boundaries():
    probability = np.zeros((120, 220), dtype=np.float32)
    for x in (15, 50, 85, 120, 155, 190):
        probability[:, x - 1 : x + 2] = 0.95

    result = count_from_probability(
        probability,
        orientation="auto",
        pixel_threshold=0.4,
        peak_height=0.2,
        peak_prominence=0.1,
        smoothing_window=3,
        minimum_peak_distance=12,
    )

    assert result.orientation == "vertical"
    assert result.count == 5
    assert len(result.boundary_positions) == 6


def test_counts_horizontal_boundaries():
    probability = np.zeros((180, 120), dtype=np.float32)
    for y in (20, 60, 100, 140):
        probability[y - 1 : y + 2, :] = 0.95

    result = count_from_probability(
        probability,
        orientation="auto",
        pixel_threshold=0.4,
        peak_height=0.2,
        peak_prominence=0.1,
        smoothing_window=3,
        minimum_peak_distance=12,
    )

    assert result.orientation == "horizontal"
    assert result.count == 3

