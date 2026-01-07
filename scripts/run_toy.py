import json
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gdch.train import train_from_config
from gdch.utils import load_config


def generate_toy_events(
    num_nodes: int = 4,
    num_events: int = 200,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    rates = rng.uniform(0.05, 0.2, size=num_nodes).astype(np.float64)
    total_rate = float(rates.sum())

    times = []
    nodes = []
    t = 0.0
    for _ in range(num_events):
        t += rng.exponential(1.0 / total_rate)
        s = rng.choice(num_nodes, p=rates / total_rate)
        times.append(t)
        nodes.append(s)

    return np.array(times, dtype=np.float32), np.array(nodes, dtype=np.int64)


def main() -> None:
    os.makedirs("data", exist_ok=True)
    times, nodes = generate_toy_events()

    df = pd.DataFrame({"t": times, "opo_id": nodes})
    df.to_csv("data/toy_events.csv", index=False)

    metadata = {
        "num_nodes": int(nodes.max() + 1),
        "t0_days": 0.0,
        "duration_days": float(times.max() - times.min()),
    }
    with open("data/toy_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    config = load_config("gdch/configs/toy.json")
    train_from_config(config)


if __name__ == "__main__":
    main()
