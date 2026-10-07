"""
plot_results.py
---------------
Generates all figures for the report after experiments are done.

Run once all JSON result files are in ./results/:

  python plot_results.py --results_dir ./results --out_dir ./figures

Produces:
  figures/convergence_iid.png       -- FedAvg vs FedNova, IID, varying clients
  figures/convergence_noniid.png    -- same but non-IID
  figures/client_count_bar.png      -- final accuracy vs num clients
  figures/quant_tradeoff.png        -- accuracy vs comm cost for quant experiments
  figures/quant_robustness.png      -- FedAvg vs FedNova under quantization
  figures/comm_cost_comparison.png  -- total GB sent per strategy
"""

from __future__ import annotations

import argparse
import json
import os
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "fedavg":           "#2196F3",
    "fednova":          "#FF5722",
    "quant_fedavg":     "#9C27B0",
    "quant_fednova":    "#4CAF50",
    "quant_fedavg_ef":  "#CE93D8",
    "quant_fednova_ef": "#A5D6A7",
    "prune_fedavg":     "#FF9800",
    "prune_fednova":    "#00BCD4",
    "centralized":      "#607D8B",
}
MARKERS = {
    "fedavg": "o", "fednova": "s",
    "quant_fedavg": "^", "quant_fednova": "D",
    "prune_fedavg": "P", "prune_fednova": "X",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_results(results_dir, algo, partition, num_clients,
                 quant_bits=None, prune_ratio=None, local_epochs=None):
    tag = f"{algo}_{partition}_{num_clients}clients"
    if quant_bits is not None:
        tag += f"_{quant_bits}bit"
    if prune_ratio is not None:
        tag += f"_{int(prune_ratio * 100)}pct"
    if local_epochs is not None:
        tag += f"_{local_epochs}ep"
    path = os.path.join(results_dir, tag, "metrics.json")
    if not os.path.exists(path):
        return None
    return load_json(path)


# ──────────────────────────────────────────────────────────────────
# Figure 1 & 2: Convergence curves
# ──────────────────────────────────────────────────────────────────

def plot_convergence(results_dir, out_dir, partition, client_counts=(2, 5, 10)):
    fig, axes = plt.subplots(1, len(client_counts), figsize=(5 * len(client_counts), 4),
                              sharey=True)
    if len(client_counts) == 1:
        axes = [axes]

    for ax, nc in zip(axes, client_counts):
        for algo in ("fedavg", "fednova"):
            data = find_results(results_dir, algo, partition, nc)
            if data is None:
                continue
            rounds = [r["round"]    for r in data["history"]]
            accs   = [r["test_acc"] for r in data["history"]]
            ax.plot(rounds, accs, label=algo.upper(),
                    color=COLORS[algo], marker=MARKERS[algo],
                    markevery=max(1, len(rounds)//10), linewidth=1.8, markersize=5)

        ax.set_title(f"{nc} clients", fontsize=11)
        ax.set_xlabel("Round")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Test Accuracy")
    fig.suptitle(f"Convergence — {partition.upper()} partition", fontsize=13, y=1.02)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"convergence_{partition}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────
# Figure 3: Final accuracy vs client count bar chart
# ──────────────────────────────────────────────────────────────────

def plot_client_count_bar(results_dir, out_dir, client_counts=(2, 5, 10)):
    for partition in ("iid", "noniid"):
        fig, ax = plt.subplots(figsize=(8, 4))
        x     = np.arange(len(client_counts))
        width = 0.35

        for i, algo in enumerate(("fedavg", "fednova")):
            accs = []
            for nc in client_counts:
                data = find_results(results_dir, algo, partition, nc)
                accs.append(data["final_test_acc"] if data else 0)
            ax.bar(x + i * width, accs, width, label=algo.upper(),
                   color=COLORS[algo], alpha=0.85)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([str(nc) for nc in client_counts])
        ax.set_xlabel("Number of Clients")
        ax.set_ylabel("Final Test Accuracy")
        ax.set_title(f"Final Accuracy vs Client Count — {partition.upper()}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"client_count_bar_{partition}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────
# Figure 4: Quantization tradeoff (accuracy vs comm cost)
# ──────────────────────────────────────────────────────────────────

def find_results_ef(results_dir, algo, partition, num_clients, quant_bits):
    """Find EF+Top-k results (top50k_ef suffix)."""
    tag  = f"{algo}_{partition}_{num_clients}clients_{quant_bits}bit_top50k_ef"
    path = os.path.join(results_dir, tag, "metrics.json")
    if not os.path.exists(path):
        return None
    return load_json(path)


def plot_quant_tradeoff(results_dir, out_dir, num_clients=10, partition="noniid"):
    bits_list = [2, 4, 8, 16, 32]
    fig, ax   = plt.subplots(figsize=(7, 4))

    # Baseline quantization
    for algo in ("quant_fedavg", "quant_fednova"):
        accs  = []
        comms = []
        for bits in bits_list:
            data = find_results(results_dir, algo, partition, num_clients, bits)
            if data is None:
                accs.append(None); comms.append(None)
                continue
            accs.append(data["final_test_acc"])
            comms.append(data["total_comm_GB"])

        valid = [(c, a) for c, a in zip(comms, accs) if c is not None]
        if not valid:
            continue
        cs, as_ = zip(*valid)
        ax.plot(cs, as_, "o-", label=algo.upper(),
                color=COLORS[algo], linewidth=1.8, markersize=7)

        for bits, c, a in zip(bits_list, comms, accs):
            if c is not None:
                ax.annotate(f"{bits}b", (c, a), textcoords="offset points",
                            xytext=(4, 4), fontsize=8)

    # EF + Top-k sparsification
    for algo in ("quant_fedavg", "quant_fednova"):
        accs  = []
        comms = []
        for bits in bits_list:
            data = find_results_ef(results_dir, algo, partition, num_clients, bits)
            if data is None:
                accs.append(None); comms.append(None)
                continue
            accs.append(data["final_test_acc"])
            comms.append(data["total_comm_GB"])

        valid = [(c, a) for c, a in zip(comms, accs) if c is not None]
        if not valid:
            continue
        cs, as_ = zip(*valid)
        ef_color = COLORS[algo + "_ef"]
        label = algo.upper().replace("QUANT_", "") + "+EF+TopK"
        ax.plot(cs, as_, "s--", label=label,
                color=ef_color, linewidth=1.8, markersize=7)

    ax.set_xlabel("Total Communication Cost (GB)")
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title(f"Accuracy vs Communication Cost ({partition.upper()}, {num_clients} clients)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"quant_tradeoff_{partition}_{num_clients}c.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────
# Figure 5: FedNova robustness to quantization noise
# ──────────────────────────────────────────────────────────────────

def plot_quant_robustness(results_dir, out_dir, num_clients=10):
    bits_list = [2, 4, 8, 16, 32]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, partition in zip(axes, ("iid", "noniid")):
        for algo in ("quant_fedavg", "quant_fednova"):
            accs = []
            for bits in bits_list:
                data = find_results(results_dir, algo, partition, num_clients, bits)
                accs.append(data["final_test_acc"] if data else None)

            valid_bits = [b for b, a in zip(bits_list, accs) if a is not None]
            valid_accs = [a for a in accs if a is not None]
            ax.plot(valid_bits, valid_accs, "o-", label=algo.upper(),
                    color=COLORS[algo], linewidth=1.8, markersize=7)

        ax.set_xlabel("Quantization Bits")
        ax.set_title(f"{partition.upper()} partition")
        ax.set_xticks(bits_list)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Final Test Accuracy")
    fig.suptitle(f"Robustness to Quantization Noise ({num_clients} clients)", fontsize=12)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"quant_robustness_{num_clients}c.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────
# Figure 6: Communication cost comparison bar
# ──────────────────────────────────────────────────────────────────

def plot_comm_cost(results_dir, out_dir, num_clients=10, partition="noniid"):
    strategies = [
        ("fedavg",       None),
        ("fednova",      None),
        ("quant_fedavg", 8),
        ("quant_fednova",8),
        ("quant_fedavg", 4),
        ("quant_fednova",4),
    ]
    labels, costs = [], []
    for algo, bits in strategies:
        data = find_results(results_dir, algo, partition, num_clients, bits)
        if data is None:
            continue
        label = algo.upper() + (f" {bits}b" if bits else "")
        labels.append(label)
        costs.append(data["total_comm_GB"])

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = [COLORS.get(algo, "#888") for algo, _ in strategies
               if find_results(results_dir, algo, partition, num_clients, _) is not None]
    bars    = ax.bar(labels, costs, color=colors, alpha=0.85)

    ax.bar_label(bars, fmt="%.2f GB", padding=3, fontsize=9)
    ax.set_ylabel("Total Communication (GB)")
    ax.set_title(f"Communication Cost — {partition.upper()}, {num_clients} clients")
    plt.xticks(rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"comm_cost_{partition}_{num_clients}c.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────
# Figure 7: Round-to-round variance (stability metric)
# ──────────────────────────────────────────────────────────────────

def plot_variance(results_dir, out_dir, partition="noniid", num_clients=10, window=5):
    """
    Rolling std-dev of test accuracy over `window` consecutive rounds.
    Lower = more stable convergence.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    for algo in ("fedavg", "fednova"):
        data = find_results(results_dir, algo, partition, num_clients)
        if data is None:
            continue
        accs   = [r["test_acc"] for r in data["history"]]
        rounds = [r["round"]    for r in data["history"]]
        rolling_std = [
            float(np.std(accs[max(0, i - window + 1): i + 1]))
            for i in range(len(accs))
        ]
        ax.plot(rounds, rolling_std, label=algo.upper(),
                color=COLORS[algo], linewidth=1.8)

    ax.set_xlabel("Round")
    ax.set_ylabel(f"Rolling Std Dev (window={window} rounds)")
    ax.set_title(f"Convergence Stability — {partition.upper()}, {num_clients} clients")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"variance_{partition}_{num_clients}c.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────
# Figure 8: Client fairness (std dev of per-client accuracy)
# ──────────────────────────────────────────────────────────────────

def plot_fairness(results_dir, out_dir, partition="noniid", num_clients=10):
    """
    Per-client accuracy std dev over rounds. Lower = fairer global model.
    Requires history entries to contain 'fairness_std'.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    found_any = False

    for algo in ("fedavg", "fednova"):
        data = find_results(results_dir, algo, partition, num_clients)
        if data is None or "fairness_std" not in data["history"][0]:
            continue
        rounds   = [r["round"]        for r in data["history"]]
        fairness = [r["fairness_std"] for r in data["history"]]
        ax.plot(rounds, fairness, label=algo.upper(),
                color=COLORS[algo], linewidth=1.8)
        found_any = True

    if not found_any:
        plt.close(fig)
        print(f"[fairness] No data with fairness_std — skipping {partition}")
        return

    ax.set_xlabel("Round")
    ax.set_ylabel("Std Dev of Per-Client Accuracy (↓ = fairer)")
    ax.set_title(f"Client Fairness Over Rounds — {partition.upper()}, {num_clients} clients")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"fairness_{partition}_{num_clients}c.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────
# Figure 9: Sensitivity to local epochs
# ──────────────────────────────────────────────────────────────────

def plot_local_epoch_sensitivity(results_dir, out_dir, partition="noniid",
                                  num_clients=10,
                                  local_epochs_list=(1, 3, 5, 10, 20)):
    """
    Final accuracy vs number of local epochs per round.
    Note: local_epochs=5 re-uses the base experiment (no _ep suffix).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, algo in zip(axes, ("fedavg", "fednova")):
        accs = []
        for ep in local_epochs_list:
            # local_epochs=5 is the default experiment (no _ep suffix)
            data = (find_results(results_dir, algo, partition, num_clients)
                    if ep == 5
                    else find_results(results_dir, algo, partition, num_clients,
                                      local_epochs=ep))
            accs.append(data["final_test_acc"] if data else None)

        valid_eps  = [e for e, a in zip(local_epochs_list, accs) if a is not None]
        valid_accs = [a for a in accs if a is not None]

        ax.plot(valid_eps, valid_accs, "o-", color=COLORS[algo],
                linewidth=1.8, markersize=7)
        ax.set_xlabel("Local Epochs per Round")
        ax.set_ylabel("Final Test Accuracy")
        ax.set_title(f"{algo.upper()} — Local Epoch Sensitivity\n"
                     f"({partition.upper()}, {num_clients} clients)")
        if valid_eps:
            ax.set_xticks(valid_eps)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir,
                            f"local_epoch_sensitivity_{partition}_{num_clients}c.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────
# Figure 10: Pruning vs Quantization comparison
# ──────────────────────────────────────────────────────────────────

def plot_pruning_vs_quant(results_dir, out_dir, num_clients=10, partition="noniid"):
    """
    Accuracy vs communication cost for quantization and pruning side by side.
    Lets the reader see which compression method gives better accuracy/cost tradeoff.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    bits_list    = [2, 4, 8, 16, 32]
    prune_ratios = [0.95, 0.90, 0.70, 0.50, 0.0]

    for quant_algo in ("quant_fedavg", "quant_fednova"):
        accs, comms = [], []
        for bits in bits_list:
            data = find_results(results_dir, quant_algo, partition,
                                num_clients, quant_bits=bits)
            if data:
                accs.append(data["final_test_acc"])
                comms.append(data["total_comm_GB"])
        if accs:
            label = quant_algo.replace("quant_", "").upper() + "+Quant"
            ax.plot(comms, accs, "o--", label=label,
                    color=COLORS[quant_algo], linewidth=1.5, markersize=7)
            for bits, c, a in zip(bits_list, comms, accs):
                ax.annotate(f"{bits}b", (c, a),
                            textcoords="offset points", xytext=(4, 4), fontsize=8)

    for prune_algo in ("prune_fedavg", "prune_fednova"):
        accs, comms = [], []
        for pr in prune_ratios:
            data = find_results(results_dir, prune_algo, partition,
                                num_clients, prune_ratio=pr)
            if data:
                accs.append(data["final_test_acc"])
                comms.append(data["total_comm_GB"])
        if accs:
            label = prune_algo.replace("prune_", "").upper() + "+Prune"
            ax.plot(comms, accs, "s-", label=label,
                    color=COLORS[prune_algo], linewidth=1.5, markersize=7)

    ax.set_xlabel("Total Communication Cost (GB)")
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title(f"Pruning vs Quantization — {partition.upper()}, {num_clients} clients")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(out_dir,
                            f"pruning_vs_quant_{partition}_{num_clients}c.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────
# Summary table printer
# ──────────────────────────────────────────────────────────────────

def print_summary_table(results_dir, num_clients=10):
    rows = []
    for partition in ("iid", "noniid"):
        for algo in ("fedavg", "fednova", "quant_fedavg", "quant_fednova"):
            for bits in ([None] if "quant" not in algo else [32, 16, 8, 4, 2]):
                data = find_results(results_dir, algo, partition, num_clients, bits)
                if data is None:
                    continue
                rows.append({
                    "algo": algo + (f"_{bits}b" if bits else ""),
                    "partition": partition,
                    "clients": num_clients,
                    "final_acc": data["final_test_acc"],
                    "comm_GB": round(data.get("total_comm_GB", 0), 3),
                })

    if not rows:
        print("No results found yet.")
        return

    print(f"\n{'Algorithm':<22} {'Partition':<10} {'Clients':<8} "
          f"{'Final Acc':<12} {'Comm (GB)':<10}")
    print("-" * 65)
    for r in rows:
        print(f"{r['algo']:<22} {r['partition']:<10} {r['clients']:<8} "
              f"{r['final_acc']:<12.4f} {r['comm_GB']:<10.3f}")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--out_dir",     default="./figures")
    p.add_argument("--num_clients", type=int, default=10)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Convergence curves
    plot_convergence(args.results_dir, args.out_dir, "iid",    client_counts=(2, 5, 10))
    plot_convergence(args.results_dir, args.out_dir, "noniid", client_counts=(2, 5, 10))

    # Final accuracy vs client count
    plot_client_count_bar(args.results_dir, args.out_dir)

    # Quantization
    plot_quant_tradeoff(args.results_dir, args.out_dir, args.num_clients, "iid")
    plot_quant_tradeoff(args.results_dir, args.out_dir, args.num_clients, "noniid")
    plot_quant_robustness(args.results_dir, args.out_dir, args.num_clients)

    # Communication cost
    plot_comm_cost(args.results_dir, args.out_dir, args.num_clients, "iid")
    plot_comm_cost(args.results_dir, args.out_dir, args.num_clients, "noniid")

    # New: stability, fairness, local epochs, pruning vs quant
    plot_variance(args.results_dir, args.out_dir, "iid",    args.num_clients)
    plot_variance(args.results_dir, args.out_dir, "noniid", args.num_clients)
    plot_fairness(args.results_dir, args.out_dir, "iid",    args.num_clients)
    plot_fairness(args.results_dir, args.out_dir, "noniid", args.num_clients)
    plot_local_epoch_sensitivity(args.results_dir, args.out_dir, "noniid", args.num_clients)
    plot_pruning_vs_quant(args.results_dir, args.out_dir, args.num_clients, "iid")
    plot_pruning_vs_quant(args.results_dir, args.out_dir, args.num_clients, "noniid")

    print_summary_table(args.results_dir, args.num_clients)


if __name__ == "__main__":
    main()
