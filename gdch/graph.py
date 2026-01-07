import math
from typing import Optional, Tuple

import numpy as np
import torch


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    dmat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        lat1, lon1 = coords[i]
        lat2 = coords[:, 0]
        lon2 = coords[:, 1]
        dmat[i] = haversine_km(lat1, lon1, lat2, lon2)
    return dmat


def load_distance_matrix(path: str) -> torch.Tensor:
    if path.endswith(".npy"):
        dmat = np.load(path)
        return torch.tensor(dmat, dtype=torch.float32)
    if path.endswith(".pt") or path.endswith(".pth"):
        return torch.load(path, map_location="cpu")
    raise ValueError(f"Unsupported distance matrix format: {path}")


def build_weight_matrix(dmat: torch.Tensor, sigma: float, knn: Optional[int] = None) -> torch.Tensor:
    d2 = dmat ** 2
    w = torch.exp(-d2 / (2.0 * sigma ** 2))
    w = w.clone()
    w.fill_diagonal_(0.0)

    if knn is not None and knn > 0:
        n = w.shape[0]
        mask = torch.zeros_like(w, dtype=torch.bool)
        for i in range(n):
            drow = dmat[i].clone()
            drow[i] = float("inf")
            _, idx = torch.topk(-drow, k=knn)
            mask[i, idx] = True
        w = torch.where(mask, w, torch.zeros_like(w))
    return w


def build_laplacian(w: torch.Tensor, use_sparse: bool = False) -> torch.Tensor:
    deg = torch.sum(w, dim=1)
    l = torch.diag(deg) - w
    if use_sparse:
        idx = l.nonzero(as_tuple=False).t()
        vals = l[idx[0], idx[1]]
        return torch.sparse_coo_tensor(idx, vals, size=l.shape)
    return l


def build_jump_kernel(dmat: torch.Tensor, sigma_jump: float) -> torch.Tensor:
    d2 = dmat ** 2
    k = torch.exp(-d2 / (2.0 * sigma_jump ** 2))
    col_sum = torch.sum(k, dim=0, keepdim=True).clamp_min(1e-8)
    k = k / col_sum
    return k


def build_graph_from_distance(
    dmat: torch.Tensor,
    sigma: float,
    knn: Optional[int] = None,
    use_sparse_if_knn: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    w = build_weight_matrix(dmat, sigma=sigma, knn=knn)
    use_sparse = use_sparse_if_knn and knn is not None and knn > 0
    l = build_laplacian(w, use_sparse=use_sparse)
    return l, w
