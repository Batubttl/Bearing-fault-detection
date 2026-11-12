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