import argparse
import os
from typing import Dict, Tuple

import torch
from torchdiffeq import odeint, odeint_event

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


def sample_next_event(
    model: GDCH,
    t0: float,
    z0: torch.Tensor,
    solver_config: Dict,
    max_steps: int = 1000,
    tol: float = 1e-6,
) -> Tuple[float, int, torch.Tensor]:
    method = solver_config.get("method", "dopri5")
    rtol = solver_config.get("rtol", 1e-4)
    atol = solver_config.get("atol", 1e-6)
    max_num_steps = solver_config.get("max_num_steps", 1000)
    event_max_steps = solver_config.get("event_max_num_steps", max_num_steps)
    min_step = solver_config.get("min_step", None)
    max_step = solver_config.get("max_step", None)
    first_step = solver_config.get("first_step", None)
    step_size = solver_config.get("step_size", None)

    options = {}
    if event_max_steps is not None:
        options["max_num_steps"] = int(event_max_steps)
    if min_step is not None:
        options["min_step"] = float(min_step)
    if max_step is not None:
        options["max_step"] = float(max_step)
    if first_step is not None:
        options["first_step"] = float(first_step)
    if step_size is not None:
        options["step_size"] = float(step_size)
    if not options:
        options = None

    device = z0.device
    dtype = z0.dtype
    time_dtype_name = str(solver_config.get("time_dtype", "float64")).lower()
    time_dtype = torch.float64 if time_dtype_name in ("float64", "double", "fp64") else dtype
    remaining = -torch.log(torch.rand((), device=device, dtype=dtype))
    t_cur = torch.tensor(t0, device=device, dtype=time_dtype)
    z = z0

    for _ in range(max_steps):
        y0 = model._flatten_state(z, torch.tensor(0.0, device=device, dtype=dtype))

        def event_fn(t, y):
            return y[-1] - remaining

        try:
            t_event, sol = odeint_event(
                model.ode_func,
                y0,
                t_cur,
                event_fn=event_fn,
                rtol=rtol,
                atol=atol,
                method=method,
                options=options,
            )
            y_event = sol[-1]
            z_event, a_event = model._unflatten_state(y_event)
        except AssertionError:
            fallback_steps = int(solver_config.get("event_fallback_steps", 200))
            t_grid = torch.linspace(
                t_cur, t_cur + 1.0, steps=fallback_steps, device=device, dtype=time_dtype
            )
            sol = odeint(
                model.ode_func,
                y0,
                t_grid,
                rtol=rtol,
                atol=atol,
                method=method,
                options=options,
            )
            a_grid = sol[:, -1]
            idx = torch.nonzero(a_grid >= remaining, as_tuple=False)
            if len(idx) == 0:
                y_event = sol[-1]
                z_event, a_event = model._unflatten_state(y_event)
                t_event = t_grid[-1]
            else:
                i = int(idx[0].item())
                if i == 0:
                    y_event = sol[0]
                    z_event, a_event = model._unflatten_state(y_event)
                    t_event = t_grid[0]
                else:
                    t_left = t_grid[i - 1]
                    t_right = t_grid[i]
                    a_left = a_grid[i - 1]
                    a_right = a_grid[i]
                    denom = (a_right - a_left).clamp_min(1e-8)
                    frac = (remaining - a_left) / denom
                    t_event = t_left + frac * (t_right - t_left)
                    y_left = sol[i - 1]
                    y_right = sol[i]
                    y_event = y_left + frac * (y_right - y_left)
                    z_event, a_event = model._unflatten_state(y_event)

        if a_event >= remaining - tol:
            lam, lam_sum = model.intensity(t_event, z_event)
            probs = lam / lam_sum
            s = int(torch.multinomial(probs, 1).item())
            z_post, _delta = model.apply_jump(z_event, s, t_event)
            return float(t_event.item()), s, z_post

        remaining = (remaining - a_event).detach()
        t_cur = t_cur + 1.0
        z = z_event

    raise RuntimeError("Event not found within max_steps")


def predict_next_event_time(
    model: GDCH,
    t0: float,
    z0: torch.Tensor,
    solver_config: Dict,
    quantile: float = 0.5,
    max_steps: int = 1000,
    tol: float = 1e-6,
) -> Tuple[float, torch.Tensor]:
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0, 1)")

    method = solver_config.get("method", "dopri5")
    rtol = solver_config.get("rtol", 1e-4)
    atol = solver_config.get("atol", 1e-6)
    max_num_steps = solver_config.get("max_num_steps", 1000)
    event_max_steps = solver_config.get("event_max_num_steps", max_num_steps)
    min_step = solver_config.get("min_step", None)
    max_step = solver_config.get("max_step", None)
    first_step = solver_config.get("first_step", None)
    step_size = solver_config.get("step_size", None)

    options = {}
    if event_max_steps is not None:
        options["max_num_steps"] = int(event_max_steps)
    if min_step is not None:
        options["min_step"] = float(min_step)
    if max_step is not None:
        options["max_step"] = float(max_step)
    if first_step is not None:
        options["first_step"] = float(first_step)
    if step_size is not None:
        options["step_size"] = float(step_size)
    if not options:
        options = None

    device = z0.device
    dtype = z0.dtype
    time_dtype_name = str(solver_config.get("time_dtype", "float64")).lower()
    time_dtype = torch.float64 if time_dtype_name in ("float64", "double", "fp64") else dtype
    remaining = -torch.log1p(torch.tensor(1.0 - quantile, device=device, dtype=dtype))
    t_cur = torch.tensor(t0, device=device, dtype=time_dtype)
    z = z0

    for _ in range(max_steps):
        y0 = model._flatten_state(z, torch.tensor(0.0, device=device, dtype=dtype))

        def event_fn(t, y):
            return y[-1] - remaining

        try:
            t_event, sol = odeint_event(
                model.ode_func,
                y0,
                t_cur,
                event_fn=event_fn,
                rtol=rtol,
                atol=atol,
                method=method,
                options=options,
            )
            y_event = sol[-1]
            z_event, a_event = model._unflatten_state(y_event)
        except AssertionError:
            fallback_steps = int(solver_config.get("event_fallback_steps", 200))
            t_grid = torch.linspace(
                t_cur, t_cur + 1.0, steps=fallback_steps, device=device, dtype=time_dtype
            )
            sol = odeint(
                model.ode_func,
                y0,
                t_grid,
                rtol=rtol,
                atol=atol,
                method=method,
                options=options,
            )
            a_grid = sol[:, -1]
            idx = torch.nonzero(a_grid >= remaining, as_tuple=False)
            if len(idx) == 0:
                y_event = sol[-1]
                z_event, a_event = model._unflatten_state(y_event)
                t_event = t_grid[-1]
            else:
                i = int(idx[0].item())
                if i == 0:
                    y_event = sol[0]
                    z_event, a_event = model._unflatten_state(y_event)
                    t_event = t_grid[0]
                else:
                    t_left = t_grid[i - 1]
                    t_right = t_grid[i]
                    a_left = a_grid[i - 1]
                    a_right = a_grid[i]
                    denom = (a_right - a_left).clamp_min(1e-8)
                    frac = (remaining - a_left) / denom
                    t_event = t_left + frac * (t_right - t_left)
                    y_left = sol[i - 1]
                    y_right = sol[i]
                    y_event = y_left + frac * (y_right - y_left)
                    z_event, a_event = model._unflatten_state(y_event)

        if a_event >= remaining - tol:
            return float(t_event.item()), z_event

        remaining = (remaining - a_event).detach()
        t_cur = t_cur + 1.0
        z = z_event

    raise RuntimeError("Quantile not reached within max_steps")


def simulate_horizon(
    model: GDCH,
    t0: float,
    z0: torch.Tensor,
    horizon: float,
    max_events: int,
    solver_config: Dict,
) -> Tuple[list, torch.Tensor]:
    events = []
    t = t0
    z = z0
    for _ in range(max_events):
        t_next, s_next, z = sample_next_event(model, t, z, solver_config)
        if t_next > t0 + horizon:
            break
        events.append((t_next, s_next))
        t = t_next
    return events, z


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate events with GDCH")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--horizon", type=float, default=30.0)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--output", default="simulated_events.csv")
    parser.add_argument("--override", action="append", default=[], help="Override config keys")
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    data_cfg = config["data"]
    solver_cfg = config.get("solver", {})
    device = get_device(config.get("training", {}).get("device", "cuda_if_available"))

    metadata = data_utils.load_metadata(data_cfg["metadata_path"])
    laplacian, jump_kernel = _build_graph(data_cfg.get("distance_matrix_path", ""), config.get("graph", {}), device)
    model = _build_model(config, metadata, device, laplacian, jump_kernel)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    z0 = model.Z0
    events, _ = simulate_horizon(model, args.t0, z0, args.horizon, args.max_events, solver_cfg)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("t,opo_id\n")
        for t, s in events:
            f.write(f"{t},{s}\n")

    print(f"Wrote {len(events)} events to {args.output}")


if __name__ == "__main__":
    main()
