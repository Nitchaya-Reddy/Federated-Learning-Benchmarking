"""
run_simulation.py
-----------------
Runs the full FL pipeline in "simulation mode" on a single machine.

Why simulation mode?
  NVFLARE's full multi-process server/client setup needs network ports
  and persistent processes.  For HiPerGator single-node SLURM jobs it's
  cleaner to simulate the federation loop in one process: we instantiate
  N client models, do local training, aggregate on the same process.

  This is functionally identical to the real NVFLARE multi-process run
  for our research purposes (same math, same data splits).

Supports:
  --algo     fedavg | fednova | quant_fedavg | quant_fednova
  --partition iid | noniid
  --num_clients  2 | 5 | 10 | 20 | 50 | 100
  --rounds    number of FL communication rounds
  --quant_bits  2 | 4 | 8 | 16 | 32  (only for quant_ algos)
  --top_k_ratio  fraction of top gradient elements to keep (0.0 = disabled)

Output:
  results/<algo>_<partition>_<num_clients>clients/metrics.json
  results/<algo>_<partition>_<num_clients>clients/model.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import (
    get_datasets, iid_partition, dirichlet_partition,
    make_client_loader, make_test_loader
)
from model import get_model
from quantized_trainer import uniform_quantize, dequantize, compute_bytes


# ──────────────────────────────────────────────────────────────────
# Local training helpers
# ──────────────────────────────────────────────────────────────────

def local_train(model, loader, device, local_epochs, lr, momentum, wd,
                lr_schedule="none"):
    """
    Runs `local_epochs` of SGD on `loader`.
    Returns the updated model (in-place) and number of steps taken.

    lr_schedule:
      'none'   -- constant LR throughout local training
      'cosine' -- CosineAnnealingLR over all local steps (eta_min = lr/100)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=wd)
    if lr_schedule == "cosine":
        total_steps = local_epochs * len(loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps, 1), eta_min=lr * 0.01
        )
    model.train()
    steps = 0
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
            if lr_schedule == "cosine":
                scheduler.step()
            steps += 1
    return steps


@torch.no_grad()
def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x)
        total_loss += criterion(out, y).item() * x.size(0)
        correct    += out.argmax(1).eq(y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────

def fedavg_aggregate(global_weights, client_updates, client_sizes):
    """Weighted average of client weight dicts."""
    total = sum(client_sizes)
    new_weights = {}
    for key in global_weights:
        new_weights[key] = sum(
            (n / total) * upd[key]
            for upd, n in zip(client_updates, client_sizes)
        )
    return new_weights


def fednova_aggregate(global_weights, norm_grads, client_sizes, taus):
    """
    FedNova aggregation:
      delta = sum_i (n_i/N) * tau_i * d_i
      w_new = w_old - delta
    """
    total = sum(client_sizes)
    delta = {}
    for key in global_weights:
        delta[key] = sum(
            (n / total) * tau * grad[key]
            for grad, n, tau in zip(norm_grads, client_sizes, taus)
        )
    return {k: global_weights[k] - delta[k] for k in global_weights}


def apply_quantization(delta_dict, quant_bits):
    """
    Quantize a dict of numpy arrays.
    Returns (dequantized_dict, total_bytes_across_all_layers).
    """
    out   = {}
    total = 0
    for k, arr in delta_dict.items():
        q_int, scale, zp = uniform_quantize(arr.astype(np.float32), quant_bits)
        total           += compute_bytes(q_int, scale, quant_bits)
        out[k]           = dequantize(q_int, scale, zp, quant_bits)
    return out, total


def magnitude_prune(delta_dict, prune_ratio):
    """
    Unstructured magnitude pruning of weight deltas.
    Zeros out the smallest-magnitude `prune_ratio` fraction of elements.
    Returns (pruned_dict, total_bytes_as_sparse_representation).
    Sparse cost: (index int32 + value float32) per non-zero element.
    """
    if prune_ratio <= 0.0:
        total = sum(v.size * 4 for v in delta_dict.values())
        return delta_dict, total

    out = {}
    total_bytes = 0
    for k, arr in delta_dict.items():
        flat_abs = np.abs(arr.flatten())
        threshold_idx = min(int(len(flat_abs) * prune_ratio), len(flat_abs) - 1)
        threshold = np.sort(flat_abs)[threshold_idx]
        mask = np.abs(arr) > threshold
        out[k] = arr * mask
        nnz = int(mask.sum())
        total_bytes += nnz * (4 + 4)  # int32 index + float32 value
    return out, total_bytes


# ──────────────────────────────────────────────────────────────────
# NEW: Error Feedback + Top-k Sparsification
# ──────────────────────────────────────────────────────────────────

def topk_sparsify(delta_dict, ratio):
    """
    Keep only the top-k% largest elements (by absolute value).
    Returns (sparsified_dict, residual_dict).
    The residual is what gets saved as error feedback for the next round.
    ratio=0.1 means keep the largest 10% of values.
    """
    sparsified = {}
    residual   = {}
    for k, arr in delta_dict.items():
        flat = arr.flatten()
        k_count = max(1, int(len(flat) * ratio))
        threshold_idx = np.argpartition(np.abs(flat), -k_count)[-k_count:]
        mask = np.zeros(flat.shape, dtype=bool)
        mask[threshold_idx] = True
        sparse = np.where(mask, flat, 0.0).reshape(arr.shape).astype(np.float32)
        sparsified[k] = sparse
        residual[k]   = (arr - sparse).astype(np.float32)
    return sparsified, residual


def apply_quantization_with_ef_topk(delta_dict, quant_bits,
                                     error_buffer, top_k_ratio):
    """
    Combined Error Feedback + Top-k Sparsification + Quantization.

    Steps:
      1. Add accumulated error from previous round (Error Feedback)
      2. Keep only top-k% elements (Top-k Sparsification)
      3. Save the residual back to error_buffer for next round
      4. Quantize the sparse delta

    Returns (dequantized_dict, total_bytes, updated_error_buffer).
    """
    # Step 1: add error feedback
    corrected = {}
    for k, arr in delta_dict.items():
        ef = error_buffer.get(k, np.zeros_like(arr))
        corrected[k] = arr.astype(np.float32) + ef

    # Step 2 & 3: top-k sparsification + save residual
    sparse, residual = topk_sparsify(corrected, top_k_ratio)

    # Update error buffer with residual
    new_error_buffer = residual

    # Step 4: quantize the sparse delta
    out   = {}
    total = 0
    for k, arr in sparse.items():
        q_int, scale, zp = uniform_quantize(arr, quant_bits)
        total           += compute_bytes(q_int, scale, quant_bits)
        out[k]           = dequantize(q_int, scale, zp, quant_bits)

    return out, total, new_error_buffer


# ──────────────────────────────────────────────────────────────────
# FL round
# ──────────────────────────────────────────────────────────────────

def fl_round(global_weights, client_loaders, client_sizes, algo,
             device, local_epochs, lr, momentum, wd, quant_bits,
             prune_ratio=0.0, model_arch="cnn", lr_schedule="none",
             top_k_ratio=0.0, error_buffers=None):
    """
    Runs one FL communication round.
    Returns (new_global_weights, round_bytes_sent, updated_error_buffers).
    """
    client_updates = []
    client_taus    = []
    total_bytes    = 0

    # Initialize error buffers if not provided
    if error_buffers is None:
        error_buffers = [{} for _ in client_loaders]

    new_error_buffers = []

    for i, loader in enumerate(client_loaders):
        # Load global model into a fresh client model
        model = get_model(model_arch).to(device)
        model.load_state_dict({k: torch.tensor(v) for k, v in global_weights.items()})

        if algo in ("fedavg", "quant_fedavg", "prune_fedavg"):
            local_train(model, loader, device, local_epochs, lr, momentum, wd,
                        lr_schedule=lr_schedule)
            w_after = {k: v.cpu().numpy().astype(np.float32)
                       for k, v in model.state_dict().items()}

            if algo == "quant_fedavg":
                delta = {k: global_weights[k] - w_after[k] for k in global_weights}

                # Use Error Feedback + Top-k if enabled, else plain quantization
                if top_k_ratio > 0.0:
                    q_delta, b, new_buf = apply_quantization_with_ef_topk(
                        delta, quant_bits, error_buffers[i], top_k_ratio
                    )
                    new_error_buffers.append(new_buf)
                else:
                    q_delta, b = apply_quantization(delta, quant_bits)
                    new_error_buffers.append({})

                total_bytes += b
                w_after = {k: global_weights[k] - q_delta[k] for k in global_weights}

            elif algo == "prune_fedavg":
                delta = {k: global_weights[k] - w_after[k] for k in global_weights}
                p_delta, b = magnitude_prune(delta, prune_ratio)
                total_bytes += b
                w_after = {k: global_weights[k] - p_delta[k] for k in global_weights}
                new_error_buffers.append({})
            else:
                total_bytes += sum(v.size * 4 for v in w_after.values())
                new_error_buffers.append({})

            client_updates.append(w_after)

        else:  # fednova, quant_fednova, prune_fednova
            w0   = {k: v.cpu().numpy().astype(np.float32)
                    for k, v in model.state_dict().items()}
            tau  = local_train(model, loader, device, local_epochs, lr, momentum, wd,
                               lr_schedule=lr_schedule)
            w_i  = {k: v.cpu().numpy().astype(np.float32)
                    for k, v in model.state_dict().items()}
            norm_grad = {k: (w0[k] - w_i[k]) / tau for k in w0}

            if algo == "quant_fednova":
                # Use Error Feedback + Top-k if enabled, else plain quantization
                if top_k_ratio > 0.0:
                    norm_grad, b, new_buf = apply_quantization_with_ef_topk(
                        norm_grad, quant_bits, error_buffers[i], top_k_ratio
                    )
                    new_error_buffers.append(new_buf)
                else:
                    norm_grad, b = apply_quantization(norm_grad, quant_bits)
                    new_error_buffers.append({})
                total_bytes += b

            elif algo == "prune_fednova":
                norm_grad, b = magnitude_prune(norm_grad, prune_ratio)
                total_bytes += b
                new_error_buffers.append({})
            else:
                total_bytes += sum(v.size * 4 for v in norm_grad.values())
                new_error_buffers.append({})

            client_updates.append(norm_grad)
            client_taus.append(tau)

    # Aggregate
    if algo in ("fedavg", "quant_fedavg", "prune_fedavg"):
        new_weights = fedavg_aggregate(global_weights, client_updates, client_sizes)
    else:
        new_weights = fednova_aggregate(global_weights, client_updates,
                                        client_sizes, client_taus)

    return new_weights, total_bytes, new_error_buffers


# ──────────────────────────────────────────────────────────────────
# Main experiment loop
# ──────────────────────────────────────────────────────────────────

def run_experiment(args):
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f" Algorithm   : {args.algo.upper()}")
    print(f" Model       : {args.model}")
    print(f" LR schedule : {args.lr_schedule}")
    print(f" Partition   : {args.partition}")
    print(f" Clients     : {args.num_clients}")
    print(f" Rounds      : {args.rounds}")
    print(f" Local epochs: {args.local_epochs}")
    if "quant" in args.algo:
        print(f" Quant bits  : {args.quant_bits}")
        print(f" Top-k ratio : {args.top_k_ratio}")
    if "prune" in args.algo:
        print(f" Prune ratio : {args.prune_ratio}")
    print(f" Device      : {device}")
    print(f"{'='*60}\n")

    # Data
    train_ds, test_ds = get_datasets(args.data_dir)
    test_loader       = make_test_loader(test_ds)

    if args.partition == "iid":
        partitions = iid_partition(train_ds, args.num_clients, args.seed)
    else:
        partitions = dirichlet_partition(train_ds, args.num_clients,
                                         args.alpha, args.seed)

    client_loaders = [make_client_loader(train_ds, p, args.batch_size)
                      for p in partitions]
    client_sizes   = [len(p) for p in partitions]

    # Initial global model
    global_model   = get_model(args.model).to(device)
    global_weights = {k: v.cpu().numpy().astype(np.float32)
                      for k, v in global_model.state_dict().items()}

    history        = []
    total_comm     = 0
    t_start        = time.time()

    # Initialize per-client error buffers for Error Feedback
    error_buffers = [{} for _ in client_loaders]

    for rnd in range(1, args.rounds + 1):
        t_rnd = time.time()

        global_weights, round_bytes, error_buffers = fl_round(
            global_weights, client_loaders, client_sizes,
            algo=args.algo, device=device,
            local_epochs=args.local_epochs,
            lr=args.lr, momentum=args.momentum, wd=args.weight_decay,
            quant_bits=args.quant_bits,
            prune_ratio=args.prune_ratio,
            model_arch=args.model,
            lr_schedule=args.lr_schedule,
            top_k_ratio=args.top_k_ratio,
            error_buffers=error_buffers,
        )
        total_comm += round_bytes

        # Evaluate global model on test set
        global_model.load_state_dict(
            {k: torch.tensor(v) for k, v in global_weights.items()}
        )
        te_loss, te_acc = evaluate(global_model, test_loader, device)

        # Per-client accuracy on each client's local data (fairness tracking)
        client_accs = []
        for c_loader in client_loaders:
            _, c_acc = evaluate(global_model, c_loader, device)
            client_accs.append(round(c_acc, 4))
        fairness_std = round(float(np.std(client_accs)), 4)

        row = dict(
            round=rnd,
            test_acc=round(te_acc, 4),
            test_loss=round(te_loss, 4),
            round_bytes=round_bytes,
            cumulative_bytes=total_comm,
            round_time_s=round(time.time() - t_rnd, 2),
            client_accs=client_accs,
            fairness_std=fairness_std,
        )
        history.append(row)

        print(f"Round {rnd:3d}/{args.rounds}  "
              f"acc={te_acc:.4f}  loss={te_loss:.4f}  "
              f"fairness_std={fairness_std:.4f}  "
              f"comm={round_bytes/1e6:.2f}MB  "
              f"total={total_comm/1e9:.3f}GB")

    elapsed = time.time() - t_start

    # Compute round-to-round variance
    accs = [r["test_acc"] for r in history]
    diffs = [accs[i] - accs[i - 1] for i in range(1, len(accs))]
    rtr_var = round(float(np.var(diffs)), 6) if diffs else 0.0

    results = dict(
        algo=args.algo,
        model=args.model,
        lr_schedule=args.lr_schedule,
        partition=args.partition,
        num_clients=args.num_clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        quant_bits=args.quant_bits if "quant" in args.algo else 32,
        prune_ratio=args.prune_ratio if "prune" in args.algo else 0.0,
        top_k_ratio=args.top_k_ratio,                          # NEW
        final_test_acc=history[-1]["test_acc"],
        final_test_loss=history[-1]["test_loss"],
        total_comm_bytes=total_comm,
        total_comm_GB=round(total_comm / 1e9, 4),
        total_time_s=round(elapsed, 1),
        round_to_round_var=rtr_var,
        avg_fairness_std=round(float(np.mean([r["fairness_std"] for r in history])), 4),
        final_fairness_std=history[-1]["fairness_std"],
        history=history,
    )

    out_path = os.path.join(args.out_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    torch.save(global_model.state_dict(),
               os.path.join(args.out_dir, "model.pt"))

    print(f"\nDone. Final acc={results['final_test_acc']:.4f}  "
          f"Total comm={results['total_comm_GB']:.4f} GB  "
          f"RTR-var={rtr_var:.6f}  "
          f"Fairness-std={results['final_fairness_std']:.4f}")
    print(f"Results saved to {args.out_dir}")
    return results


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FL Simulation Runner")
    p.add_argument("--algo",        default="fedavg",
                   choices=["fedavg", "fednova", "quant_fedavg", "quant_fednova",
                            "prune_fedavg", "prune_fednova"])
    p.add_argument("--partition",   default="iid", choices=["iid", "noniid"])
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--rounds",      type=int, default=50)
    p.add_argument("--local_epochs",type=int, default=5)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--lr",          type=float, default=0.01)
    p.add_argument("--momentum",    type=float, default=0.9)
    p.add_argument("--weight_decay",type=float, default=5e-4)
    p.add_argument("--alpha",       type=float, default=0.5,
                   help="Dirichlet alpha for non-IID partition")
    p.add_argument("--quant_bits",  type=int, default=8,
                   choices=[2, 4, 8, 16, 32])
    p.add_argument("--prune_ratio", type=float, default=0.0,
                   help="Fraction of smallest-magnitude elements to prune (0.0 = no pruning)")
    p.add_argument("--top_k_ratio", type=float, default=0.0,      # NEW
                   help="Top-k sparsification ratio (0.0 = disabled, 0.1 = keep top 10%%)")
    p.add_argument("--model",       default="cnn", choices=["cnn", "resnet18"],
                   help="Model architecture: 'cnn' (~1.3M params) or 'resnet18' (~11.2M params)")
    p.add_argument("--lr_schedule", default="none", choices=["none", "cosine"],
                   help="LR schedule within each local training step")
    p.add_argument("--data_dir",    default="./data")
    p.add_argument("--out_dir",     default=None,
                   help="If not set, auto-generated from experiment params")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.out_dir is None:
        tag = f"{args.algo}_{args.partition}_{args.num_clients}clients"
        if "quant" in args.algo:
            tag += f"_{args.quant_bits}bit"
            if args.top_k_ratio > 0.0:
                tag += f"_top{int(args.top_k_ratio * 100)}k_ef"  # NEW
        elif "prune" in args.algo:
            tag += f"_{int(args.prune_ratio * 100)}pct"
        if args.local_epochs != 5:
            tag += f"_{args.local_epochs}ep"
        if args.model != "cnn":
            tag += f"_{args.model}"
        if args.lr_schedule != "none":
            tag += f"_{args.lr_schedule}"
        args.out_dir = f"./results/{tag}"
    run_experiment(args)
