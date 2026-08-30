import numpy as np

from book_spine_ml.prepare_yolo import longest_boundary_edges, parse_yolo_shape


def test_parses_standard_yolo_box():
    polygon = parse_yolo_shape("0 0.5 0.5 0.2 0.8", width=200, height=100)
    assert polygon == [(80.0, 10.0), (120.0, 10.0), (120.0, 90.0), (80.0, 90.0)]


def test_extracts_two_long_edges():
    polygon = [(80.0, 10.0), (120.0, 10.0), (120.0, 90.0), (80.0, 90.0)]
    edges = longest_boundary_edges(polygon)
    assert len(edges) == 2
    lengths = [np.linalg.norm(np.subtract(end, start)) for start, end in edges]
    assert all(abs(length - 80.0) < 1e-6 for length in lengths)

