# SARM: Stage-aware reward modeling

This repository provides training and evaluation scripts for **SARM (our method)** and **ReWiND (baseline)** on both the LeRobot dataset and raw robot trajectories.

---

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.  

### 1. Install `uv`
```bash
pip install uv
```

### 2. Sync environment
```bash
uv sync
```

### 3. Activate environment
```bash
source .venv/bin/activate
```

---

## Training

### Train with SARM (our method)
```bash
python train.py --config-name sarm
```

### Train with ReWiND (baseline)
```bash
python train.py --config-name rewind
```

---

## Evaluation

### Evaluate on LeRobot dataset validation set
```bash
python eval.py --config-name sarm
```

### Evaluate on raw robot trajectory
```bash
python eval.py --config-name sarm --mode raw_datt
```

---

## Notes
- Replace `sarm` with `rewind` in the commands above if you want to run the baseline method.  
- All configs are stored under the `config/` directory.
