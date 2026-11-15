import numpy as np

from bearing_fault_detection.preprocessing import make_windows, normalize


def test_make_windows_overlapping_toy_example():
    w = make_windows(np.arange(10), window_size=4, step=2)
    assert w.tolist() == [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]]


def test_make_windows_drops_leftover_samples():
    w = make_windows(np.arange(10), window_size=4, step=4)
    assert w.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_make_windows_signal_shorter_than_window():
    assert make_windows(np.arange(3), window_size=4).shape == (0, 4)


def test_normalize_zero_mean_unit_std_per_window():
    x = normalize(np.random.default_rng(0).normal(5, 3, size=(8, 2048)))
    assert np.allclose(x.mean(axis=1), 0, atol=1e-5)
    assert np.allclose(x.std(axis=1), 1, atol=1e-3)


def test_normalize_constant_window_has_no_nan():
    assert np.isfinite(normalize(np.ones(2048))).all()