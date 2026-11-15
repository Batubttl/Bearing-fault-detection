"""Hand-crafted features for the Random Forest baseline."""
import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis

FS = 12_000
RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
# fault frequency / shaft frequency for the SKF 6205 drive-end bearing
ORDERS = {"bpfi": 5.4152, "bsf": 4.7135, "bpfo": 3.5848}


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def time_features(windows):
    """RMS, kurtosis, peak, crest factor for each window."""
    feats = []
    for w in windows:
        r, peak = rms(w), np.max(np.abs(w))
        feats.append([r, kurtosis(w), peak, peak / r])
    return np.array(feats)


def envelope_features(window, rpm, fs=FS):
    """Envelope-spectrum peak at BPFI/BSF/BPFO (1x and 2x), normalized by the spectrum median."""
    env = np.abs(hilbert(window - window.mean()))
    env -= env.mean()
    spec = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(len(env), 1 / fs)
    ref = np.median(spec) + 1e-12
    feats = []
    for order in ORDERS.values():
        for h in (1, 2):
            f0 = h * order * rpm / 60
            band = (freqs > f0 - 6) & (freqs < f0 + 6)
            feats.append(spec[band].max() / ref)
    return feats