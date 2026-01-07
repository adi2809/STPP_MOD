import argparse
import math
import os
from typing import Dict, Optional, Tuple

import torch
from torchdiffeq import odeint

from gdch import data as data_utils
from gdch import graph as graph_utils
from gdch.model import GDCH
from gdch.utils import apply_overrides, get_device, load_config


def _build_graph(distance_path: str, graph_cfg: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if not distance_path or not os.path.exists(distance_path):
        return None, None
    dmat = graph_utils.load_distance_matrix(distance_path)
    sigma = float(graph_cfg.get("sigma", 1.0))
    sigma_jump = float(graph_cfg.get("sigma_jump", sigma))
    knn = graph_cfg.get("knn", None)

    l, _w = graph_utils.build_graph_from_distance(dmat, sigma=sigma, knn=knn)
    k = graph_utils.build_jump_kernel(dmat, sigma_jump=sigma_jump)

    return l.to(device=device), k.to(device=device)


def _build_model(config: Dict, metadata: Dict, device: torch.device, laplacian, jump_kernel) -> GDCH:
    model_cfg = config["model"]
    model = GDCH(
        num_nodes=int(metadata["num_nodes"]),
        latent_dim=int(model_cfg.get("latent_dim", 16)),
        time_embed_dim=int(model_cfg.get("time_embed_dim", 16)),
        node_embed_dim=int(model_cfg.get("node_embed_dim", 8)),
        mlp_hidden_dim=model_cfg.get("mlp_hidden_dim", None),
        time_hidden_dim=model_cfg.get("time_hidden_dim", None),
        alpha_init=float(model_cfg.get("alpha_init", 0.1)),
        beta_init=float(model_cfg.get("beta_init", 0.1)),
        jump_eta=float(model_cfg.get("jump_eta", 0.0)),
        jump_tanh=bool(model_cfg.get("jump_tanh", False)),
        jump_scale=float(model_cfg.get("jump_scale", 1.0)),
        gate_use_z=bool(model_cfg.get("gate_use_z", False)),
        intensity_use_z=bool(model_cfg.get("intensity_use_z", False)),
        eps=float(model_cfg.get("eps", 1e-8)),
        t0_days=float(metadata.get("t0_days", 0.0)),
        total_time=float(metadata.get("duration_days", 1.0)),
        baseline_init=float(model_cfg.get("baseline_init", 0.0)),
        per_node_w=bool(model_cfg.get("per_node_w", False)),
        laplacian=laplacian,
        jump_kernel=jump_kernel,
    )
    return model.to(device)


def _run_sequence(
    model: GDCH,
    times: torch.Tensor,
    nodes: torch.Tensor,
    solver_config: Dict,
    chunk_size: int,
    state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[float, int, Tuple[torch.Tensor, torch.Tensor]]:
    total_loss = 0.0
    total_events = 0
    state_local = state

    with torch.no_grad():
        for start in range(0, len(times), chunk_size):
            end = min(start + chunk_size - 1, len(times) - 1)
            loss, state_local, _ = model.nll_chunk(
                times,
                nodes,
                start_idx=start,
                end_idx=end,
                state=state_local,
                solver_config=solver_config,
                collect_reg=False,
            )
            total_loss += loss.item()
            total_events += end - start + 1
            state_local = (state_local[0].detach(), state_local[1].detach())

    return total_loss, total_events, state_local


def evaluate_nll(
    model: GDCH,
    times: torch.Tensor,
    nodes: torch.Tensor,
    solver_config: Dict,
    chunk_size: int,
    warmup: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> float:
    state = None
    if warmup is not None and len(warmup[0]) > 0:
        _loss, _events, state = _run_sequence(
            model, warmup[0], warmup[1], solver_config, chunk_size, state=None
        )
        if len(times) > 0:
            t_last = warmup[0][-1]
            t_next = times[0]
            if t_next > t_last:
                z, a = state
                z, a = model.integrate(t_last, t_next, z, a, solver_config)
                state = (z, a)
    loss, events, _ = _run_sequence(model, times, nodes, solver_config, chunk_size, state=state)
    return loss / max(events, 1)


def expected_distance_metric(
    model: GDCH,
    times: torch.Tensor,
    nodes: torch.Tensor,
    dmat: torch.Tensor,
    solver_config: Dict,
    warmup: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> float:
    state = None
    if warmup is not None and len(warmup[0]) > 0:
        _loss, _events, state = _run_sequence(
            model, warmup[0], warmup[1], solver_config, chunk_size=len(warmup[0]), state=None
        )
        if len(times) > 0:
            t_last = warmup[0][-1]
            t_next = times[0]
            if t_next > t_last:
                z, a = state
                z, a = model.integrate(t_last, t_next, z, a, solver_config)
                state = (z, a)
    if state is None:
        z = model.Z0
        a = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    else:
        z, a = state

    total = 0.0
    count = 0
    with torch.no_grad():
        for n in range(len(times)):
            t_n = times[n]
            s_n = nodes[n]
            g = model._time_embedding(t_n)
            lam, lam_sum = model._compute_intensity(g, z)
            p = lam / lam_sum
            s_idx = int(s_n.item()) if torch.is_tensor(s_n) else int(s_n)
            total += (p * dmat[:, s_idx]).sum().item()
            count += 1

            z_post, _delta = model.apply_jump(z, s_n, t_n, g=g)
            if n < len(times) - 1:
                t_next = times[n + 1]
                if t_next > t_n:
                    z, a = model.integrate(t_n, t_next, z_post, a, solver_config)
                else:
                    z = z_post
            else:
                z = z_post

    return total / max(count, 1)


def compute_intensity(model: GDCH, t: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return model.intensity(t, z)


def next_event_time_quantiles(
    model: GDCH,
    t0: float,
    z0: torch.Tensor,
    quantiles,
    solver_config: Dict,
    horizon: float,
    steps: int = 200,
) -> torch.Tensor:
    q = torch.tensor(quantiles, device=z0.device, dtype=z0.dtype)
    q_targets = -torch.log1p(-q)

    t0_t = torch.tensor(t0, device=z0.device, dtype=z0.dtype)
    t_grid = torch.linspace(t0_t, t0_t + horizon, steps=steps, device=z0.device, dtype=z0.dtype)

    y0 = model._flatten_state(z0, torch.tensor(0.0, device=z0.device, dtype=z0.dtype))
    method = solver_config.get("method", "dopri5")
    rtol = solver_config.get("rtol", 1e-4)
    atol = solver_config.get("atol", 1e-6)
    max_num_steps = solver_config.get("max_num_steps", 1000)
    options = {"max_num_steps": int(max_num_steps)} if max_num_steps is not None else None

    sol = odeint(model.ode_func, y0, t_grid, rtol=rtol, atol=atol, method=method, options=options)
    a_grid = sol[:, -1]

    out_times = []
    for q_target in q_targets:
        idx = torch.nonzero(a_grid >= q_target, as_tuple=False)
        if len(idx) == 0:
            out_times.append(t_grid[-1])
            continue
        i = int(idx[0].item())
        if i == 0:
            out_times.append(t_grid[0])
            continue
        t_left, t_right = t_grid[i - 1], t_grid[i]
        a_left, a_right = a_grid[i - 1], a_grid[i]
        denom = (a_right - a_left).clamp_min(1e-8)
        frac = (q_target - a_left) / denom
        out_times.append(t_left + frac * (t_right - t_left))

    return torch.stack(out_times)


def next_event_location_probs(model: GDCH, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    lam, lam_sum = model.intensity(t, z)
    return lam / lam_sum


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GDCH model")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--override", action="append", default=[], help="Override config keys")
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    data_cfg = config["data"]
    solver_cfg = config.get("solver", {})
    device = get_device(config.get("training", {}).get("device", "cuda_if_available"))

    times, nodes = data_utils.load_processed_events(data_cfg["events_path"])
    metadata = data_utils.load_metadata(data_cfg["metadata_path"])

    split_idx = data_utils.split_by_time(
        times.numpy(),
        train_frac=float(data_cfg.get("train_split", 0.8)),
        val_frac=float(data_cfg.get("val_split", 0.1)),
    )

    train_idx = split_idx["train"]
    val_idx = split_idx["val"]
    test_idx = split_idx["test"]

    times_train = times[train_idx].to(device=device)
    nodes_train = nodes[train_idx].to(device=device)
    times_val = times[val_idx].to(device=device)
    nodes_val = nodes[val_idx].to(device=device)
    times_test = times[test_idx].to(device=device)
    nodes_test = nodes[test_idx].to(device=device)

    laplacian, jump_kernel = _build_graph(data_cfg.get("distance_matrix_path", ""), config.get("graph", {}), device)
    model = _build_model(config, metadata, device, laplacian, jump_kernel)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.split == "train":
        eval_times, eval_nodes = times_train, nodes_train
        warmup = None
    elif args.split == "val":
        eval_times, eval_nodes = times_val, nodes_val
        warmup = (times_train, nodes_train)
    else:
        eval_times, eval_nodes = times_test, nodes_test
        warmup = (torch.cat([times_train, times_val]), torch.cat([nodes_train, nodes_val]))

    nll = evaluate_nll(
        model,
        eval_times,
        eval_nodes,
        solver_config=solver_cfg,
        chunk_size=int(config.get("training", {}).get("eval_chunk_size", 512)),
        warmup=warmup,
    )

    print(f"{args.split} nll_per_event: {nll:.6f}")

    dpath = data_cfg.get("distance_matrix_path", "")
    if dpath and os.path.exists(dpath):
        dmat = graph_utils.load_distance_matrix(dpath).to(device=device)
        exp_dist = expected_distance_metric(model, eval_times, eval_nodes, dmat, solver_cfg, warmup=warmup)
        print(f"{args.split} expected_distance: {exp_dist:.6f}")


if __name__ == "__main__":
    main()
