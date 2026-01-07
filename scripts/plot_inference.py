import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gdch.data import load_metadata, load_processed_events, split_by_time
from gdch.simulate import predict_next_event_time, sample_next_event
from gdch.train import _build_graph, _build_model
from gdch.utils import get_device, load_config


def find_latest_run(artifacts_dir: str) -> Optional[str]:
    if not os.path.exists(artifacts_dir):
        return None
    candidates = []
    for name in os.listdir(artifacts_dir):
        path = os.path.join(artifacts_dir, name)
        if not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, "checkpoint_best.pt")):
            candidates.append((os.path.getmtime(path), path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def load_run_config(run_dir: str, fallback_config: str) -> Dict:
    cfg_path = os.path.join(run_dir, "config.json") if run_dir else None
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return load_config(fallback_config)


def run_warmup(
    model,
    times: torch.Tensor,
    nodes: torch.Tensor,
    solver_config: Dict,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    state = None
    with torch.no_grad():
        for start in range(0, len(times), chunk_size):
            end = min(start + chunk_size - 1, len(times) - 1)
            _loss, state, _ = model.nll_chunk(
                times,
                nodes,
                start_idx=start,
                end_idx=end,
                state=state,
                solver_config=solver_config,
                collect_reg=False,
            )
            state = (state[0].detach(), state[1].detach())
    return state


def evaluate_opo_on_true_times(
    model,
    times: torch.Tensor,
    nodes: torch.Tensor,
    z0: torch.Tensor,
    t0: float,
    solver_config: Dict,
    top_k: int,
) -> Tuple[float, Optional[float], Optional[float], np.ndarray]:
    device = z0.device
    z = z0
    a = torch.tensor(0.0, device=device, dtype=z0.dtype)

    if len(times) == 0:
        return float("nan"), None, None, None

    t_first = float(times[0].item())
    if t_first > t0:
        z, a = model.integrate(t0, times[0], z, a, solver_config)

    correct = 0
    topk_hits = 0
    true_probs = []
    num_nodes = int(model.N)
    conf = np.zeros((num_nodes, num_nodes), dtype=np.int64)

    with torch.no_grad():
        for i in range(len(times)):
            t_n = times[i]
            s_true = int(nodes[i].item())
            g = model._time_embedding(t_n)
            lam, lam_sum = model._compute_intensity(g, z)
            probs = (lam / lam_sum).detach().cpu().numpy()
            pred = int(np.argmax(probs))
            conf[s_true, pred] += 1
            correct += int(pred == s_true)
            true_probs.append(float(probs[s_true]))

            topk = min(max(1, int(top_k)), len(probs))
            top_idx = np.argpartition(-probs, topk - 1)[:topk]
            topk_hits += int(s_true in top_idx)

            z_post, _delta = model.apply_jump(z, s_true, t_n, g=g)
            if i < len(times) - 1:
                t_next = times[i + 1]
                if t_next > t_n:
                    z, a = model.integrate(t_n, t_next, z_post, a, solver_config)
                else:
                    z = z_post
            else:
                z = z_post

    opo_acc = correct / len(times)
    opo_topk = topk_hits / len(times)
    mean_true_prob = float(np.mean(true_probs)) if true_probs else None
    return opo_acc, opo_topk, mean_true_prob, conf


def predict_sequence(
    model,
    t0: float,
    z0: torch.Tensor,
    k: int,
    solver_config: Dict,
    seed: int,
    time_mode: str,
    quantile: float,
    opo_mode: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], torch.Tensor]:
    torch.manual_seed(seed)
    pred_times = []
    pred_nodes = []
    pred_probs = []
    t = t0
    z = z0
    with torch.no_grad():
        for _ in range(k):
            if time_mode == "sample":
                t, s, z = sample_next_event(model, t, z, solver_config)
                pred_times.append(t)
                pred_nodes.append(s)
                pred_probs.append(None)
                continue

            t, z_pre = predict_next_event_time(
                model, t, z, solver_config, quantile=quantile
            )
            t_tensor = torch.tensor(t, device=z_pre.device, dtype=z_pre.dtype)
            lam, lam_sum = model.intensity(t_tensor, z_pre)
            probs = (lam / lam_sum).detach().cpu().numpy()
            if opo_mode == "argmax":
                s = int(np.argmax(probs))
            else:
                s = int(np.random.choice(len(probs), p=probs))

            z, _delta = model.apply_jump(z_pre, s, t_tensor)
            pred_times.append(t)
            pred_nodes.append(s)
            pred_probs.append(probs)

    pred_probs_arr = None
    if time_mode != "sample":
        pred_probs_arr = np.stack(pred_probs, axis=0)
    return (
        np.array(pred_times, dtype=np.float64),
        np.array(pred_nodes, dtype=np.int64),
        pred_probs_arr,
        z,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GDCH inference diagnostics for next-k arrivals")
    parser.add_argument("--config", default="gdch/configs/base.json")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--run-dir", default=None, help="Specific run directory under artifacts")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--time-mode", choices=["median", "quantile", "sample"], default="median")
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--opo-mode", choices=["argmax", "sample"], default="argmax")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run(args.artifacts_dir)
    if run_dir is None:
        raise SystemExit("No run directory found under artifacts")

    config = load_run_config(run_dir, args.config)
    data_cfg = config["data"]
    solver_cfg = config.get("solver", {})

    device = get_device(config.get("training", {}).get("device", "cuda_if_available"))
    metadata = load_metadata(data_cfg["metadata_path"])
    times, nodes = load_processed_events(data_cfg["events_path"])

    split_idx = split_by_time(
        times.numpy(),
        train_frac=float(data_cfg.get("train_split", 0.8)),
        val_frac=float(data_cfg.get("val_split", 0.1)),
    )

    train_idx = split_idx["train"]
    val_idx = split_idx["val"]
    test_idx = split_idx["test"]

    if args.split == "val":
        warmup_idx = train_idx
        target_idx = val_idx
    else:
        warmup_idx = np.concatenate([train_idx, val_idx])
        target_idx = test_idx

    times_warmup = times[warmup_idx].to(device=device)
    nodes_warmup = nodes[warmup_idx].to(device=device)
    times_target = times[target_idx].to(device=device)
    nodes_target = nodes[target_idx].to(device=device)

    if len(times_target) == 0:
        raise SystemExit("Target split is empty")

    laplacian, jump_kernel = _build_graph(data_cfg.get("distance_matrix_path", ""), config.get("graph", {}), device)
    model = _build_model(config, metadata, device, laplacian, jump_kernel)

    checkpoint_path = os.path.join(run_dir, "checkpoint_best.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if len(times_warmup) > 0:
        state = run_warmup(model, times_warmup, nodes_warmup, solver_cfg, args.chunk_size)
        z0 = state[0]
        t0 = float(times_warmup[-1].item())
    else:
        z0 = model.Z0
        t0 = float(times_target[0].item())

    k = min(args.k, len(times_target))
    true_times = times_target[:k].cpu().numpy()
    true_nodes = nodes_target[:k].cpu().numpy()

    time_mode = args.time_mode
    quantile = args.quantile if time_mode != "median" else 0.5
    pred_times, pred_nodes, pred_probs, _ = predict_sequence(
        model,
        t0,
        z0,
        k,
        solver_cfg,
        args.seed,
        time_mode=time_mode,
        quantile=quantile,
        opo_mode=args.opo_mode,
    )

    true_rel = true_times - t0
    pred_rel = pred_times - t0

    true_inter = np.diff(np.concatenate([[t0], true_times]))
    pred_inter = np.diff(np.concatenate([[t0], pred_times]))

    time_error = pred_times - true_times
    abs_error = np.abs(time_error)
    opo_match = pred_nodes == true_nodes
    opo_accuracy = float(opo_match.mean())
    true_node_prob = None
    topk_accuracy = None
    if pred_probs is not None:
        true_node_prob = pred_probs[np.arange(k), true_nodes]
        topk = max(1, int(args.top_k))
        topk_hits = 0
        for i in range(k):
            top_idx = np.argpartition(-pred_probs[i], topk - 1)[:topk]
            if true_nodes[i] in top_idx:
                topk_hits += 1
        topk_accuracy = float(topk_hits / k)

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    idx = np.arange(1, k + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(idx, true_rel, label="true", linewidth=2)
    plt.plot(idx, pred_rel, label="pred", linewidth=2)
    plt.xlabel("Event index")
    plt.ylabel("Time since t0 (days)")
    plt.title("Next-k arrival times")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "next_k_times.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(idx, true_inter, label="true inter-arrival", linewidth=2)
    plt.plot(idx, pred_inter, label="pred inter-arrival", linewidth=2)
    plt.xlabel("Event index")
    plt.ylabel("Inter-arrival (days)")
    plt.title("Inter-arrival times")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "interarrival.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(idx, time_error, label="pred - true", linewidth=2)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Event index")
    plt.ylabel("Error (days)")
    plt.title("Temporal error")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "temporal_error.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(abs_error, bins=30)
    plt.xlabel("Absolute error (days)")
    plt.ylabel("Count")
    plt.title("Absolute temporal error distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "temporal_error_hist.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    cum_acc = np.cumsum(opo_match) / np.arange(1, k + 1)
    plt.plot(idx, cum_acc, linewidth=2)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Event index")
    plt.ylabel("Cumulative accuracy")
    plt.title("OPO prediction accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "opo_accuracy.png"))
    plt.close()

    num_nodes = int(metadata.get("num_nodes", max(true_nodes.max(), pred_nodes.max()) + 1))
    conf = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    for t_node, p_node in zip(true_nodes, pred_nodes):
        conf[int(t_node), int(p_node)] += 1
    plt.figure(figsize=(6, 5))
    plt.imshow(conf, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Count")
    plt.xlabel("Predicted OPO")
    plt.ylabel("True OPO")
    plt.title("OPO confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "opo_confusion.png"))
    plt.close()

    opo_true_acc, opo_true_topk, opo_true_prob, conf_true = evaluate_opo_on_true_times(
        model,
        times_target[:k],
        nodes_target[:k],
        z0,
        t0,
        solver_cfg,
        args.top_k,
    )
    if conf_true is not None:
        plt.figure(figsize=(6, 5))
        plt.imshow(conf_true, aspect="auto", interpolation="nearest")
        plt.colorbar(label="Count")
        plt.xlabel("Predicted OPO")
        plt.ylabel("True OPO")
        plt.title("OPO confusion matrix (true times)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "opo_confusion_true_time.png"))
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.bar(["top1", f"top{args.top_k}"], [opo_true_acc, opo_true_topk])
        plt.ylim(0.0, 1.0)
        plt.ylabel("Accuracy")
        plt.title("OPO accuracy (true times)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "opo_accuracy_true_time.png"))
        plt.close()

    df = pd.DataFrame(
        {
            "event_idx": idx,
            "true_time": true_times,
            "pred_time": pred_times,
            "true_node": true_nodes,
            "pred_node": pred_nodes,
            "true_node_prob": true_node_prob if true_node_prob is not None else np.nan,
            "true_interarrival": true_inter,
            "pred_interarrival": pred_inter,
            "time_error": time_error,
            "abs_error": abs_error,
        }
    )
    df.to_csv(os.path.join(plots_dir, "predictions.csv"), index=False)

    summary = {
        "k": int(k),
        "mean_abs_error_days": float(abs_error.mean()),
        "median_abs_error_days": float(np.median(abs_error)),
        "mean_error_days": float(time_error.mean()),
        "opo_accuracy": float(opo_accuracy),
        "opo_topk_accuracy": None if topk_accuracy is None else float(topk_accuracy),
        "mean_true_node_prob": None if true_node_prob is None else float(true_node_prob.mean()),
        "opo_accuracy_true_time": float(opo_true_acc) if opo_true_acc is not None else None,
        "opo_topk_accuracy_true_time": float(opo_true_topk) if opo_true_topk is not None else None,
        "mean_true_node_prob_true_time": float(opo_true_prob) if opo_true_prob is not None else None,
    }
    with open(os.path.join(plots_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
