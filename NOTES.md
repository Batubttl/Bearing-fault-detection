## Session 0 – 2026-11-12
- Env: macOS, Apple M2 (8 GB), VS Code, Python 3.12 via uv (system 3.14 untouched)
- Fixed ~/.local ownership (root → user), installed uv and gh (GitHub CLI auth)
- Installed numpy, scipy, pandas, matplotlib, scikit-learn, torch 2.15 (MPS available), ipykernel
- Repo skeleton: data/ (gitignored), notebooks/, src/, tests/
- First notebook: plotted a 500 Hz sine (fs = 12 kHz) → frequency = cycles per second
- Pushed to GitHub
- Next: Step 1 – download CWRU dataset, load .mat files, plot raw signals per class
## Session 1 – 2026-11-12
- Downloaded 4 CWRU files (12k DE, 0.007", 0 hp): 97 Normal, 105 IR, 118 Ball, 130 OR@6
- Loaded .mat with scipy.io.loadmat → dict; signal key: X<num>_DE_time
- 97.mat: 243,938 samples ≈ 20 s at 12 kHz
- Plotted first 0.2 s per class:
  - Normal: calm, low amplitude
  - IR / OR: periodic impulses (OR ≈ every 9 ms ≈ 107 Hz)
  - Ball: hard to distinguish from Normal by eye → expect the model to struggle here
- Next: Step 2 – windowing + leakage-free train/test split
## Session 2 – 2026-11-15
- Wrote make_windows(signal, window_size, step) using slicing
- Window = 2048 samples (~0.17 s, ~5 shaft rotations), step = 2048 (no overlap)
- Leakage-free split: split each recording by time FIRST (80% train / 20% test), THEN window
  - Why: windowing + random shuffle puts near-identical/overlapping windows in both sets → inflated scores
  - Limitation: train/test still come from the same recording → later test on unseen motor load
- Result: X_train (236, 2048), X_test (56, 2048), labels 0=Normal 1=IR 2=Ball 3=OR
- Class imbalance: train [95 47 47 47] → Normal recording is 2x longer (explains bigger 97.mat)
  → don't rely on accuracy alone; use confusion matrix / per-class metrics
- Next: Step 3 – hand-crafted features (RMS, kurtosis, FFT) + Random Forest baseline
## Session 3 – 2026-11-15
- Features: RMS, kurtosis, peak, crest factor → classes separate perfectly on 0.007"/0 hp
- RF results:
  - same-recording split: 100% | cross-load (train 0 hp, test 1-3 hp): 100%
  - BUT CWRU reuses the same physical bearing across loads → cross-load split still leaks
    (Hendriks et al. 2022, Rosa et al. 2024)
  - fault-size split (train 0.007"+0.021", test 0.014"): acc 0.44, macro-F1 0.35
    → time features measure vibration strength (depends on fault size), not fault type
- Added src/bearing_fault_detection/cwru.py: file catalog + downloader with retries (40 files)
- FFT: faults excite a ~2.5–3.7 kHz resonance (the "bell"), fault type is in the impact rhythm
- Envelope analysis: clear peaks at theoretical BPFI (~162 Hz) and BPFO (~107 Hz); Ball unclear
- Envelope features (normalized by spectrum median, rpm-aware) + time features: acc 0.53, macro-F1 0.49
  - IR recall: 0 → 131/236
  - OR 0.014" predicted as Normal: its envelope spectrum has no BPFO peak
    → consistent with Smith & Randall (2015): records 198-200 not/only partly diagnosable
- Next: Step 4 – 1D-CNN in PyTorch on raw windows, same fault-size split