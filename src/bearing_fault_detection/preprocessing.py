"""Windowing and per-window normalization (shared by training and inference)."""
import numpy as np

WINDOW_SIZE = 2048


def make_windows(signal, window_size=WINDOW_SIZE, step=None):
    """Split a 1-D signal into windows of `window_size`; leftover samples at the end are dropped."""
    step = step or window_size
    starts = range(0, len(signal) - window_size + 1, step)
    if len(starts) == 0:
        return np.empty((0, window_size))
    return np.stack([signal[s:s + window_size] for s in starts])


def normalize(windows):
    """Zero mean, unit std per window: removes absolute vibration level, keeps shape/rhythm."""
    x = np.asarray(windows, dtype=np.float32)
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)