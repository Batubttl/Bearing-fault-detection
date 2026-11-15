"""CWRU bearing dataset: file catalog and downloader (12k drive-end subset)."""
from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = "https://engineering.case.edu/sites/default/files/{}.mat"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

CLASS_NAMES = ["Normal", "Inner race", "Ball", "Outer race"]

# fault size -> first file number for (Inner race, Ball, Outer race @6:00) at 0 hp;
# loads 1-3 hp are the next consecutive numbers
_FAULT_START = {
    0.007: (105, 118, 130),
    0.014: (169, 185, 197),
    0.021: (209, 222, 234),
}


def build_catalog():
    """Return {file_number: (class_label, fault_size_inch, load_hp)}."""
    catalog = {97 + load: (0, 0.0, load) for load in range(4)}  # Normal 97-100
    for size, starts in _FAULT_START.items():
        for label, first in enumerate(starts, start=1):
            for load in range(4):
                catalog[first + load] = (label, size, load)
    return catalog


CATALOG = build_catalog()


import time
from urllib.error import HTTPError, URLError


def _fetch(url, path, retries=5, wait=5):
    """Download with retries; write to a .part file first so a failed download leaves no broken file."""
    tmp = path.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            urlretrieve(url, tmp)
            tmp.rename(path)
            return
        except (HTTPError, URLError) as e:
            print(f"  attempt {attempt}/{retries} failed ({e}), retrying in {wait}s")
            time.sleep(wait)
            wait *= 2  # wait longer each time: 5s, 10s, 20s...
    raise RuntimeError(f"Could not download {url}")


def download(force=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for num in sorted(CATALOG):
        path = DATA_DIR / f"{num}.mat"
        if path.exists() and not force:
            continue  # already downloaded
        print(f"Downloading {num}.mat ...")
        _fetch(BASE_URL.format(num), path)
        time.sleep(1)  # be polite to the server
    print(f"Done: {len(list(DATA_DIR.glob('*.mat')))} files in {DATA_DIR}")


if __name__ == "__main__":
    download()