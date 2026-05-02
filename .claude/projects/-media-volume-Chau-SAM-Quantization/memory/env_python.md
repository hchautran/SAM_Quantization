---
name: SAM project Python env
description: Use the sam conda env (Python 3.12) for any script in this repo — the system default `python` resolves to the wrong env (autoresearch venv).
type: project
---

The runnable Python is `/media/volume/Chau/miniconda3/envs/sam/bin/python`. The shell's default `python` and the project's `.venv/bin/python` are both **wrong** — they lack `accelerate`, the right torch build, etc.

**Why:** Pre-flight on 2026-05-02 found `python` → `/media/volume/Chau/autoresearch/.venv/bin/python` and `source .venv/bin/activate` did not change it. The `sam` conda env at `/media/volume/Chau/miniconda3/envs/sam` has the working stack (accelerate 1.13, torch 2.5.1+cu121, segment_anything installed editable from `sam-hq/`).

**How to apply:** Always invoke scripts with the absolute interpreter path AND `cd /media/volume/Chau/SAM_Quantization` first (sam_engine and PiToMe are imported relative to repo root). Skip `conda activate` — it errors with "Run 'conda init' before 'conda activate'".
