# GDCH (Graph-Diffused Competing Hazards)

This project implements a Graph-Diffused Competing Hazards (GDCH) model for unmarked multivariate temporal point processes using PyTorch and torchdiffeq. It learns node-wise conditional intensities from event sequences with continuous-time latent dynamics, graph diffusion, and event jumps.

## Setup

Install dependencies:

```bash
pip install torch torchdiffeq numpy pandas matplotlib
```

## Data Preprocessing

Convert the provided `dataframes/cleaned_data.csv` into the GDCH event format and build the distance matrix:

```bash
python scripts/process_data.py
```

If your raw data has repeated timestamps, you can enforce a minimum spacing (in days) to avoid identical event times:

```bash
python scripts/process_data.py --min-time-delta 1e-6
```

This writes:
- `data/events.csv`
- `data/metadata.json`
- `data/opo_metadata.csv`
- `data/distance.npy`

You can also call the module directly:

```bash
python -m gdch.data --input dataframes/cleaned_data.csv --output-events data/events.csv --output-metadata data/metadata.json --output-opo-metadata data/opo_metadata.csv --output-distance data/distance.npy
```

## Training

```bash
python -m gdch.train --config gdch/configs/base.json
```

Training artifacts (checkpoints, logs, config) are saved under `artifacts/` with a timestamped run directory.

## Evaluation

```bash
python -m gdch.eval --config gdch/configs/base.json --checkpoint artifacts/<run_dir>/checkpoint_best.pt --split val
```

## Simulation

```bash
python -m gdch.simulate --config gdch/configs/base.json --checkpoint artifacts/<run_dir>/checkpoint_best.pt --horizon 30 --max-events 100 --output simulated_events.csv
```

## Inference Plots (Next-k Arrivals)

Generate plots comparing predicted vs true next-k arrival times using the latest run's best checkpoint:

```bash
python scripts/plot_inference.py --k 200 --time-mode median --opo-mode argmax
```

Outputs are saved under `artifacts/<run_dir>/plots/`.

Notes:
- `--time-mode median` (default) is deterministic but slower; use smaller `--k` if needed.
- `--time-mode sample` is faster but noisier because it samples event times/locations.

## Toy Example

Generate a synthetic dataset and run a short training loop:

```bash
python scripts/run_toy.py
```

## Notebook

A minimal end-to-end notebook is available at:

- `notebooks/GDCH_Training.ipynb`

It preprocesses data and launches training using `gdch/train.py`.

## Tests

Run the unit tests:

```bash
python -m unittest discover -s tests
```

## Project Structure

```
.
├── dataframes/cleaned_data.csv
├── data/
├── gdch/
│   ├── configs/
│   ├── data.py
│   ├── eval.py
│   ├── features.py
│   ├── graph.py
│   ├── losses.py
│   ├── model.py
│   ├── simulate.py
│   ├── train.py
│   └── utils.py
├── notebooks/
├── scripts/
├── tests/
└── README.md
```
