import argparse
import math
import os
import time
from typing import Dict, Tuple

import torch
import torch.optim as optim

from gdch import data as data_utils
from gdch import graph as graph_utils
from gdch.eval import evaluate_nll
from gdch.losses import add_regularization
from gdch.model import GDCH
from gdch.utils import apply_overrides, detach_state, get_device, load_config, make_run_dir, save_config, set_seed, setup_logger


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
        mlp_layers=model_cfg.get("mlp_layers", None),
        time_mlp_layers=model_cfg.get("time_mlp_layers", None),
        mlp_dropout=float(model_cfg.get("mlp_dropout", 0.0)),
        mlp_layer_norm=bool(model_cfg.get("mlp_layer_norm", False)),
        time_mlp_dropout=float(model_cfg.get("time_mlp_dropout", 0.0)),
        time_mlp_layer_norm=bool(model_cfg.get("time_mlp_layer_norm", False)),
        intensity_activation=model_cfg.get("intensity_activation", "softplus"),
        intensity_beta=float(model_cfg.get("intensity_beta", 1.0)),
        intensity_min=float(model_cfg.get("intensity_min", 0.0)),
        intensity_neg_slope=float(model_cfg.get("intensity_neg_slope", 0.01)),
        gate_activation=model_cfg.get("gate_activation", "sigmoid"),
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


def _maybe_adjust_min_dt(solver_cfg: Dict, min_gap: float, logger) -> None:
    if min_gap <= 0:
        return
    current_min_dt = float(solver_cfg.get("min_dt", 0.0))
    if current_min_dt <= 0:
        solver_cfg["min_dt"] = min_gap / 2.0
        logger.info("min_dt was unset; setting min_dt=%.6g based on min gap %.6g", solver_cfg["min_dt"], min_gap)
        return
    if current_min_dt >= min_gap:
        solver_cfg["min_dt"] = min_gap / 2.0
        logger.info(
            "min_dt=%.6g >= min gap %.6g; lowering min_dt to %.6g to avoid skipping dynamics",
            current_min_dt,
            min_gap,
            solver_cfg["min_dt"],
        )


def _log_graph_warnings(data_cfg: Dict, model_cfg: Dict, laplacian, jump_kernel, logger) -> None:
    distance_path = data_cfg.get("distance_matrix_path", "")
    if not distance_path:
        logger.warning(
            "distance_matrix_path is empty; spatial diffusion and jump spillover will be disabled"
        )
        return
    if laplacian is None:
        logger.warning(
            "distance_matrix_path=%s not found or invalid; spatial diffusion will be disabled",
            distance_path,
        )
    if jump_kernel is None:
        logger.warning(
            "distance_matrix_path=%s not found or invalid; jump spillover will be disabled",
            distance_path,
        )
    jump_eta = float(model_cfg.get("jump_eta", 0.0))
    if jump_kernel is not None and jump_eta == 0.0:
        logger.warning(
            "jump_eta=0.0 with a valid distance matrix; set model.jump_eta to enable spillover"
        )


def train_from_config(config: Dict) -> str:
    train_cfg = config["training"]
    data_cfg = config["data"]
    solver_cfg = config.get("solver", {})
    reg_cfg = config.get("regularization", {})

    set_seed(int(train_cfg.get("seed", 42)))
    device = get_device(train_cfg.get("device", "cuda_if_available"))
    use_amp = bool(train_cfg.get("use_amp", False)) and device.type == "cuda"
    use_compile = bool(train_cfg.get("torch_compile", False)) and hasattr(torch, "compile")

    times, nodes = data_utils.load_processed_events(data_cfg["events_path"])
    metadata = data_utils.load_metadata(data_cfg["metadata_path"])

    split_idx = data_utils.split_by_time(
        times.numpy(),
        train_frac=float(data_cfg.get("train_split", 0.8)),
        val_frac=float(data_cfg.get("val_split", 0.1)),
    )

    train_idx = split_idx["train"]
    val_idx = split_idx["val"]

    times_train = times[train_idx].to(device=device)
    nodes_train = nodes[train_idx].to(device=device)
    times_val = times[val_idx].to(device=device)
    nodes_val = nodes[val_idx].to(device=device)

    laplacian, jump_kernel = _build_graph(data_cfg.get("distance_matrix_path", ""), config.get("graph", {}), device)
    model = _build_model(config, metadata, device, laplacian, jump_kernel)
    if use_compile:
        model = torch.compile(model)

    model_cfg = config["model"]
    if bool(model_cfg.get("baseline_from_data", False)) and len(times_train) > 1:
        duration = (times_train[-1] - times_train[0]).clamp_min(1e-6)
        counts = torch.bincount(nodes_train, minlength=int(metadata["num_nodes"])).to(times_train.device)
        rates = (counts.float() / duration).clamp_min(1e-8)
        b_init = torch.log(torch.expm1(rates))
        b_init = torch.where(torch.isfinite(b_init), b_init, torch.log(rates))
        with torch.no_grad():
            model.b.copy_(b_init)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler_cfg = train_cfg.get("lr_scheduler", None)
    scheduler = None
    scheduler_type = None
    if isinstance(scheduler_cfg, dict):
        scheduler_type = str(scheduler_cfg.get("type", "plateau")).lower()
        if scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(scheduler_cfg.get("factor", 0.5)),
                patience=int(scheduler_cfg.get("patience", 2)),
                threshold=float(scheduler_cfg.get("threshold", 1e-3)),
                min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
            )
        elif scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(scheduler_cfg.get("t_max", 10)),
                eta_min=float(scheduler_cfg.get("min_lr", 1e-6)),
            )

    run_dir = make_run_dir(train_cfg.get("artifacts_dir", "artifacts"), train_cfg.get("run_name", "gdch"))
    logger = setup_logger(os.path.join(run_dir, "train.log"))
    save_config(config, os.path.join(run_dir, "config.json"))

    _log_graph_warnings(data_cfg, model_cfg, laplacian, jump_kernel, logger)

    if len(times_train) > 1:
        min_gap = float((times_train[1:] - times_train[:-1]).min().item())
        _maybe_adjust_min_dt(solver_cfg, min_gap, logger)

    chunk_size = int(train_cfg.get("chunk_size", 256))
    use_dataloader = bool(train_cfg.get("use_dataloader", False))
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", False))
    eval_chunk_size = int(train_cfg.get("eval_chunk_size", chunk_size))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    log_every = int(train_cfg.get("log_every", 50))
    log_every_seconds = float(train_cfg.get("log_every_seconds", 0.0))
    eval_every = int(train_cfg.get("eval_every", 1))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    base_lr = float(train_cfg.get("lr", 1e-3))

    best_val = math.inf
    metrics = []

    global_step = 0
    for epoch in range(1, int(train_cfg.get("epochs", 10)) + 1):
        model.train()
        state = None
        nll_sum = 0.0
        event_count = 0

        total_chunks = max(1, math.ceil(len(times_train) / chunk_size))
        chunk_iter = None
        if use_dataloader:
            chunk_iter = data_utils.make_chunk_loader(
                len(times_train), chunk_size, num_workers=num_workers, pin_memory=pin_memory
            )
        chunk_counter = 0
        window_nll = 0.0
        window_events = 0
        last_log_time = time.time()
        logger.info("epoch=%d/%d start", epoch, int(train_cfg.get("epochs", 10)))

        if chunk_iter is None:
            chunk_iter = range(0, len(times_train), chunk_size)
        for batch in chunk_iter:
            if isinstance(batch, int):
                start = batch
                end = min(start + chunk_size - 1, len(times_train) - 1)
            else:
                start = batch["start"]
                end = batch["end"]
                if torch.is_tensor(start):
                    start = int(start.flatten()[0].item())
                if torch.is_tensor(end):
                    end = int(end.flatten()[0].item())
            chunk_counter += 1
            with torch.cuda.amp.autocast(enabled=use_amp):
                nll, state, reg_terms = model.nll_chunk(
                    times_train,
                    nodes_train,
                    start_idx=start,
                    end_idx=end,
                    state=state,
                    solver_config=solver_cfg,
                    collect_reg=True,
                )
                num_events = end - start + 1
                nll_avg = nll / max(num_events, 1)
                loss, reg_info = add_regularization(nll_avg, reg_terms, model, reg_cfg)
            nll, state, reg_terms = model.nll_chunk(
                times_train,
                nodes_train,
                start_idx=start,
                end_idx=end,
                state=state,
                solver_config=solver_cfg,
                collect_reg=True,
            )
            num_events = end - start + 1
            nll_avg = nll / max(num_events, 1)
            loss, reg_info = add_regularization(nll_avg, reg_terms, model, reg_cfg)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if warmup_steps > 0 and global_step < warmup_steps:
                warmup_lr = base_lr * float(global_step + 1) / float(warmup_steps)
                optimizer.param_groups[0]["lr"] = warmup_lr
            scaler.step(optimizer)
            scaler.update()
            if warmup_steps > 0 and global_step < warmup_steps:
                optimizer.param_groups[0]["lr"] = base_lr
            global_step += 1

            state = detach_state(state)
            nll_sum += nll.item()
            event_count += end - start + 1

            window_nll += nll.item()
            window_events += end - start + 1
            should_log_chunk = log_every > 0 and (
                chunk_counter % log_every == 0 or chunk_counter == total_chunks
            )
            should_log_time = log_every_seconds > 0 and (time.time() - last_log_time) >= log_every_seconds
            if should_log_chunk or should_log_time:
                avg_nll = window_nll / max(window_events, 1)
                logger.info(
                    "epoch=%d chunk=%d/%d avg_nll=%.6f",
                    epoch,
                    chunk_counter,
                    total_chunks,
                    avg_nll,
                )
                window_nll = 0.0
                window_events = 0
                last_log_time = time.time()

        train_nll = nll_sum / max(event_count, 1)

        if eval_every > 0 and epoch % eval_every == 0 and len(times_val) > 0:
            val_nll = evaluate_nll(
                model,
                times_val,
                nodes_val,
                solver_config=solver_cfg,
                chunk_size=eval_chunk_size,
                warmup=(times_train, nodes_train),
            )
        else:
            val_nll = float("nan")

        if scheduler is not None:
            metric = val_nll if not math.isnan(val_nll) else train_nll
            if scheduler_type == "plateau":
                scheduler.step(metric)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        metrics.append(
            {"epoch": epoch, "train_nll": train_nll, "val_nll": val_nll, "lr": current_lr}
        )
        logger.info(
            "epoch=%d train_nll=%.6f val_nll=%.6f lr=%.6g",
            epoch,
            train_nll,
            val_nll,
            current_lr,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_nll": train_nll,
            "val_nll": val_nll,
        }
        torch.save(checkpoint, os.path.join(run_dir, "checkpoint_last.pt"))

        if not math.isnan(val_nll) and val_nll < best_val:
            best_val = val_nll
            torch.save(checkpoint, os.path.join(run_dir, "checkpoint_best.pt"))

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        import json

        json.dump(metrics, f, indent=2)

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GDCH model")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--override", action="append", default=[], help="Override config keys, e.g. training.epochs=5")
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    train_from_config(config)


if __name__ == "__main__":
    main()
