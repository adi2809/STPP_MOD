import argparse
import json
import os
from typing import Dict, Optional, Tuple

import math
import numpy as np
import pandas as pd
import torch

from gdch.graph import compute_distance_matrix


def _parse_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def _enforce_min_delta(times: np.ndarray, min_delta: float) -> np.ndarray:
    if min_delta <= 0 or len(times) < 2:
        return times
    t = times.astype(np.float64, copy=True)
    for i in range(1, len(t)):
        if t[i] - t[i - 1] < min_delta:
            t[i] = t[i - 1] + min_delta
    return t


def process_raw_csv(
    input_path: str,
    output_events_path: str,
    output_metadata_path: str,
    output_opo_metadata_path: Optional[str] = None,
    output_distance_path: Optional[str] = None,
    time_col: str = "event_datetime_local",
    opo_col: str = "OPO_ENTIRE_NAME_CLEAN",
    lat_col: str = "OPO_LAT_ZIP",
    lon_col: str = "OPO_LON_ZIP",
    min_time_delta: float = 1e-6,
) -> Dict[str, object]:
    df = pd.read_csv(input_path)
    opo_map = None

    if "t" in df.columns and "opo_id" in df.columns:
        df = df.copy().sort_values("t", kind="mergesort")
        t = df["t"].astype(float).to_numpy()
        t = _enforce_min_delta(t, min_time_delta)
        df["t"] = t
        opo_id = df["opo_id"].astype(int).to_numpy()
        t0_days = float(df.get("t0_days", pd.Series([0.0])).iloc[0]) if "t0_days" in df else 0.0
        num_nodes = int(np.max(opo_id) + 1)
        timestamp = df["timestamp"] if "timestamp" in df.columns else None
        if opo_col in df.columns:
            name_by_id = df.groupby("opo_id")[opo_col].first().to_dict()
            opo_map = {str(name): int(idx) for idx, name in name_by_id.items()}
    else:
        if time_col not in df.columns:
            raise ValueError(f"Missing time column: {time_col}")
        if opo_col not in df.columns:
            raise ValueError(f"Missing OPO column: {opo_col}")

        df = df.copy()
        df["timestamp"] = _parse_timestamp(df[time_col])
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
        t0 = df["timestamp"].min()
        df["t"] = (df["timestamp"] - t0).dt.total_seconds() / 86400.0

        opo_values = sorted(df[opo_col].astype(str).unique().tolist())
        opo_map = {name: idx for idx, name in enumerate(opo_values)}
        df["opo_id"] = df[opo_col].astype(str).map(opo_map).astype(int)

        df = df.sort_values("t", kind="mergesort")
        t = df["t"].astype(float).to_numpy()
        t = _enforce_min_delta(t, min_time_delta)
        df["t"] = t
        opo_id = df["opo_id"].astype(int).to_numpy()
        t0_days = t0.timestamp() / 86400.0
        num_nodes = len(opo_values)
        timestamp = df["timestamp"]

    os.makedirs(os.path.dirname(output_events_path), exist_ok=True)
    out_df = pd.DataFrame({"t": t.astype(np.float32), "opo_id": opo_id.astype(int)})
    if timestamp is not None:
        out_df["timestamp"] = timestamp.astype(str).to_numpy()
    out_df.to_csv(output_events_path, index=False)

    metadata = {
        "num_nodes": int(num_nodes),
        "t0_days": float(t0_days),
        "duration_days": float(t.max() - t.min()) if len(t) > 1 else 0.0,
    }
    if opo_map is not None:
        metadata["opo_map"] = opo_map

    if opo_col in df.columns:
        if opo_map is not None:
            opo_meta = pd.DataFrame(
                {"opo_id": list(opo_map.values()), "opo_name": list(opo_map.keys())}
            ).sort_values("opo_id")
        else:
            opo_meta = df[["opo_id", opo_col]].drop_duplicates().rename(columns={opo_col: "opo_name"})

        if lat_col in df.columns and lon_col in df.columns:
            coords = df[["opo_id", lat_col, lon_col]].dropna().groupby("opo_id").mean()
            coords = coords.reindex(range(num_nodes))
            coords = coords.fillna(coords.mean(numeric_only=True))
            coords = coords.reset_index().rename(columns={lat_col: "lat", lon_col: "lon"})

            if output_distance_path:
                coord_array = coords[["lat", "lon"]].to_numpy()
                dmat = compute_distance_matrix(coord_array)
                np.save(output_distance_path, dmat)

            if output_opo_metadata_path:
                merged = opo_meta.merge(coords, on="opo_id", how="left")
                merged.to_csv(output_opo_metadata_path, index=False)
        else:
            if output_opo_metadata_path:
                opo_meta.to_csv(output_opo_metadata_path, index=False)

    with open(output_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def load_processed_events(events_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(events_path)
    times = torch.tensor(df["t"].to_numpy(dtype=np.float32))
    nodes = torch.tensor(df["opo_id"].to_numpy(dtype=np.int64))
    return times, nodes


class ChunkDataset(torch.utils.data.IterableDataset):
    def __init__(self, num_events: int, chunk_size: int) -> None:
        super().__init__()
        self.num_events = int(num_events)
        self.chunk_size = int(chunk_size)

    def __iter__(self):
        if self.num_events <= 0:
            return iter(())
        for start in range(0, self.num_events, self.chunk_size):
            end = min(start + self.chunk_size - 1, self.num_events - 1)
            yield {"start": start, "end": end}

    def __len__(self) -> int:
        if self.num_events <= 0:
            return 0
        return int(math.ceil(self.num_events / self.chunk_size))


def make_chunk_loader(
    num_events: int,
    chunk_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> torch.utils.data.DataLoader:
    dataset = ChunkDataset(num_events, chunk_size)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )


def load_metadata(metadata_path: str) -> Dict[str, object]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_by_time(times: np.ndarray, train_frac: float, val_frac: float) -> Dict[str, np.ndarray]:
    t_min = float(times.min())
    t_max = float(times.max())
    t_train = t_min + train_frac * (t_max - t_min)
    t_val = t_min + (train_frac + val_frac) * (t_max - t_min)

    train_idx = np.where(times <= t_train)[0]
    val_idx = np.where((times > t_train) & (times <= t_val))[0]
    test_idx = np.where(times > t_val)[0]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def main() -> None:
    parser = argparse.ArgumentParser(description="Process raw OPO events into GDCH format.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output-events", required=True, help="Output events CSV path")
    parser.add_argument("--output-metadata", required=True, help="Output metadata JSON path")
    parser.add_argument("--output-opo-metadata", default=None, help="Output OPO metadata CSV path")
    parser.add_argument("--output-distance", default=None, help="Output distance matrix .npy path")
    parser.add_argument("--time-col", default="event_datetime_local")
    parser.add_argument("--opo-col", default="OPO_ENTIRE_NAME_CLEAN")
    parser.add_argument("--lat-col", default="OPO_LAT_ZIP")
    parser.add_argument("--lon-col", default="OPO_LON_ZIP")
    parser.add_argument("--min-time-delta", type=float, default=1e-6)
    args = parser.parse_args()

    process_raw_csv(
        input_path=args.input,
        output_events_path=args.output_events,
        output_metadata_path=args.output_metadata,
        output_opo_metadata_path=args.output_opo_metadata,
        output_distance_path=args.output_distance,
        time_col=args.time_col,
        opo_col=args.opo_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        min_time_delta=args.min_time_delta,
    )


if __name__ == "__main__":
    main()
