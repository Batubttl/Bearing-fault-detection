## Session 0 – 2026-11-12
- Env: macOS, Apple M2 (8 GB), VS Code, Python 3.12 via uv (system 3.14 untouched)
- Fixed ~/.local ownership (root → user), installed uv and gh (GitHub CLI auth)
- Installed numpy, scipy, pandas, matplotlib, scikit-learn, torch 2.15 (MPS available), ipykernel
- Repo skeleton: data/ (gitignored), notebooks/, src/, tests/
- First notebook: plotted a 500 Hz sine (fs = 12 kHz) → frequency = cycles per second
- Pushed to GitHub
- Next: Step 1 – download CWRU dataset, load .mat files, plot raw signals per class