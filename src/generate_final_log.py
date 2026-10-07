"""
generate_final_log.py
---------------------
Collects all experiment results and SLURM log excerpts into a single
FINAL_REPORT.txt for easy review.

Usage:
  python src/generate_final_log.py \
      --results_dir ./results \
      --logs_dir    ./logs \
      --out         ./FINAL_REPORT.txt
"""

from __future__ import annotations

import argparse
import json
import os
import glob
from datetime import datetime


SEP = "=" * 70


def section(title: str) -> str:
    return f"\n{SEP}\n  {title}\n{SEP}\n"


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def collect_results(results_dir: str) -> list[dict]:
    rows = []
    for metrics_path in sorted(glob.glob(os.path.join(results_dir, "*", "metrics.json"))):
        try:
            data = load_json(metrics_path)
            rows.append({"path": metrics_path, "data": data})
        except Exception as e:
            rows.append({"path": metrics_path, "error": str(e)})
    return rows


def format_results_table(rows: list[dict]) -> str:
    if not rows:
        return "  No metrics.json files found.\n"

    lines = []
    hdr = f"  {'Experiment':<45} {'FinalAcc':>9} {'CommGB':>8} {'Rounds':>7} {'TimeSec':>9}"
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    for row in rows:
        name = os.path.basename(os.path.dirname(row["path"]))
        if "error" in row:
            lines.append(f"  {name:<45} ERROR: {row['error']}")
            continue
        d = row["data"]
        lines.append(
            f"  {name:<45} "
            f"{d.get('final_test_acc', 0):>9.4f} "
            f"{d.get('total_comm_GB', 0):>8.3f} "
            f"{d.get('rounds', 0):>7} "
            f"{d.get('total_time_s', 0):>9.1f}"
        )
    return "\n".join(lines) + "\n"


def tail_file(path: str, n: int = 60) -> str:
    try:
        with open(path) as f:
            lines = f.readlines()
        kept = lines[-n:] if len(lines) > n else lines
        prefix = f"  [last {n} lines of {len(lines)} total]\n" if len(lines) > n else ""
        return prefix + "".join(kept)
    except FileNotFoundError:
        return f"  (file not found: {path})\n"
    except Exception as e:
        return f"  (error reading {path}: {e})\n"


def collect_slurm_logs(logs_dir: str) -> list[tuple[str, str]]:
    """Returns list of (label, content) for all .out and .err files."""
    entries = []
    for ext in ("*.out", "*.err"):
        for path in sorted(glob.glob(os.path.join(logs_dir, ext))):
            label = os.path.basename(path)
            entries.append((label, path))
    return entries


def format_per_experiment_history(rows: list[dict], last_n: int = 5) -> str:
    lines = []
    for row in rows:
        if "error" in row:
            continue
        name = os.path.basename(os.path.dirname(row["path"]))
        d = row["data"]
        history = d.get("history", [])
        if not history:
            continue
        lines.append(f"\n  --- {name} (last {last_n} rounds) ---")
        for r in history[-last_n:]:
            step = r.get('round', r.get('epoch', '?'))
            comm = f"  cumComm={r['cumulative_bytes']/1e9:.3f}GB" if 'cumulative_bytes' in r else ""
            lines.append(
                f"    Step {step:>3}  acc={r['test_acc']:.4f}  "
                f"loss={r['test_loss']:.4f}{comm}"
            )
    return "\n".join(lines) + "\n" if lines else "  No history data found.\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--logs_dir",    default="./logs")
    p.add_argument("--out",         default="./FINAL_REPORT.txt")
    p.add_argument("--log_tail",    type=int, default=60,
                   help="Lines to include from each SLURM log file")
    args = p.parse_args()

    lines = []

    lines.append(SEP)
    lines.append(f"  FINAL EXPERIMENT REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(SEP)

    # ── Results summary table ─────────────────────────────────────────
    lines.append(section("RESULTS SUMMARY"))
    rows = collect_results(args.results_dir)
    lines.append(f"  Found {len(rows)} result(s) in {args.results_dir}\n")
    lines.append(format_results_table(rows))

    # ── Per-experiment trailing history ──────────────────────────────
    lines.append(section("PER-EXPERIMENT FINAL ROUNDS"))
    lines.append(format_per_experiment_history(rows, last_n=5))

    # ── Missing experiments check ─────────────────────────────────────
    lines.append(section("MISSING EXPERIMENTS CHECK"))
    expected = []
    for algo in ("fedavg", "fednova"):
        for part in ("iid", "noniid"):
            for nc in (2, 5, 10):
                expected.append(f"{algo}_{part}_{nc}clients")
    for algo in ("quant_fedavg", "quant_fednova"):
        for part in ("iid", "noniid"):
            for bits in (2, 4, 8, 16, 32):
                expected.append(f"{algo}_{part}_10clients_{bits}bit")
    # EF + Top-k sparsification experiments (sujeongjo contribution)
    for algo in ("quant_fedavg", "quant_fednova"):
        for part in ("iid", "noniid"):
            for bits in (2, 4, 8, 16, 32):
                expected.append(f"{algo}_{part}_10clients_{bits}bit_top50k_ef")

    found_names = set()
    for row in rows:
        found_names.add(os.path.basename(os.path.dirname(row["path"])))

    missing = [e for e in expected if e not in found_names]
    if missing:
        lines.append(f"  {len(missing)} missing:\n")
        for m in missing:
            lines.append(f"    MISSING: {m}")
    else:
        lines.append("  All expected experiments found.")
    lines.append("")

    # ── SLURM logs ────────────────────────────────────────────────────
    lines.append(section("SLURM JOB LOGS"))
    slurm_logs = collect_slurm_logs(args.logs_dir)
    if not slurm_logs:
        lines.append(f"  No log files found in {args.logs_dir}")
    for label, path in slurm_logs:
        lines.append(f"\n  ---- {label} ----")
        lines.append(tail_file(path, n=args.log_tail))

    # ── Write output ──────────────────────────────────────────────────
    report = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(report)

    print(report)
    print(f"\n[Report written to {args.out}]")


if __name__ == "__main__":
    main()
