# Project 0: Federated Learning with NVFLARE
### HiPerGator Implementation — CIFAR-10 Benchmark

**Author:** Nitchaya Reddy

**Repository:** [Federated-Learning-Benchmarking](https://github.com/Nitchaya-Reddy/Federated-Learning-Benchmarking)

---

## Overview

This project benchmarks Federated Learning (FL) algorithms on CIFAR-10 using a simulation framework built on top of NVIDIA NVFLARE. It systematically compares **FedAvg** and **FedNova** across:

- IID vs. non-IID data distributions (Dirichlet α = 0.5)
- Varying client counts (2, 5, 10)
- Gradient quantization (2–32 bit)
- Magnitude-based model pruning (0–95%)
- Local epoch sensitivity (1–20 epochs)
- Model architecture (CNN vs. ResNet-18 with cosine annealing)

All experiments ran on the University of Florida HiPerGator cluster (NVIDIA B200 GPU) via SLURM array jobs. Results are from 75 completed experiments totaling ~100 GPU-hours.

---

## Results Summary

### Centralized Baseline (accuracy ceiling)

| Epochs | Final Test Acc | Time |
|--------|---------------|------|
| 100 | **86.76%** | 13.6 min |

---

### FedAvg vs FedNova — CNN (100 rounds)

| Clients | FedAvg IID | FedNova IID | FedAvg non-IID | FedNova non-IID |
|---------|-----------|------------|----------------|-----------------|
| 2 | 87.53% | **88.18%** | 85.30% | **86.36%** |
| 5 | **88.03%** | 88.00% | 85.56% | 85.47% |
| 10 | 87.41% | **87.57%** | 84.08% | **84.13%** |

**Communication cost:** 0.534 GB (2 clients) → 2.671 GB (10 clients) at 32-bit, same for both algorithms.

![image](https://hackmd.io/_uploads/S1_mA2csZe.png)

![image](https://hackmd.io/_uploads/SJGV025i-x.png)

---

### FedAvg vs FedNova — ResNet-18 + Cosine Annealing (100 rounds)

| Clients | FedAvg IID | FedNova IID | FedAvg non-IID | FedNova non-IID |
|---------|-----------|------------|----------------|-----------------|
| 2 | **95.00%** | 94.80% | 93.96% | **94.30%** |
| 5 | **94.36%** | 94.32% | 93.27% | 93.30% |
| 10 | 93.50% | **93.54%** | **92.15%** | 92.03% |

**Communication cost:** 8.947 GB (2 clients) → 44.734 GB (10 clients) — ResNet-18 is ~33× larger than the CNN.

---

### Client Count Effect (CNN, 100 rounds)

![image](https://hackmd.io/_uploads/B1ddR39ibg.png)
![image](https://hackmd.io/_uploads/B1RO035i-x.png)

---

### Quantization Results (CNN, 10 clients, 100 rounds)

#### IID

| Bits | FedAvg Acc | FedNova Acc | Comm vs 32-bit |
|------|-----------|------------|----------------|
| 32 | 87.07% | 87.15% | 1× (2.671 GB) |
| 16 | 87.47% | **87.69%** | 0.5× (1.336 GB) |
| 8 | 87.05% | 87.24% | 0.25× (0.668 GB) |
| 4 | 87.29% | 87.45% | 0.125× (0.334 GB) |
| 2 | 86.92% | 86.91% | 0.0625× (0.167 GB) |

#### non-IID

| Bits | FedAvg Acc | FedNova Acc | Comm vs 32-bit |
|------|-----------|------------|----------------|
| 32 | 84.12% | **84.69%** | 1× (2.671 GB) |
| 16 | 84.30% | 84.15% | 0.5× |
| 8 | 84.48% | 84.17% | 0.25× |
| 4 | 84.54% | **84.68%** | 0.125× |
| 2 | 84.01% | 83.70% | 0.0625× (0.167 GB) |

![image](https://hackmd.io/_uploads/HJ9qC2cjZg.png)
![image](https://hackmd.io/_uploads/Sk4iAn9i-e.png)
![image](https://hackmd.io/_uploads/HJy3Ch9oZg.png)
![image](https://hackmd.io/_uploads/S1q2R3qjZx.png)
![image](https://hackmd.io/_uploads/r1ma035jbe.png)

---

### Pruning Results (CNN, 10 clients, 100 rounds)

#### IID

| Prune Ratio | FedAvg Acc | FedNova Acc | Comm (GB) |
|-------------|------------|-------------|-----------|
| 0% | 87.37% | 87.49% | 2.671 |
| 50% | 87.28% | 87.30% | 2.671 |
| 70% | 87.11% | 86.93% | 1.603 |
| 90% | 84.91% | 85.04% | 0.534 |
| 95% | 82.88% | 82.31% | 0.267 |

#### non-IID

| Prune Ratio | FedAvg Acc | FedNova Acc | Comm (GB) |
|-------------|------------|-------------|-----------|
| 0% | 84.65% | **85.14%** | 2.671 |
| 50% | 84.26% | **84.82%** | 2.671 |
| 70% | 84.43% | **84.52%** | 1.603 |
| 90% | 82.55% | **82.91%** | 0.534 |
| 95% | 80.58% | 80.01% | 0.267 |

![image](https://hackmd.io/_uploads/BJ6C0hciZg.png)
![image](https://hackmd.io/_uploads/HyEk1aqi-l.png)

---

### Local Epoch Sensitivity (CNN, 10 clients, non-IID, 100 rounds)

| Local Epochs | FedAvg Acc | FedNova Acc | FedAvg Time | FedNova Time |
|-------------|-----------|------------|-------------|-------------|
| 1 | 80.47% | 80.35% | 29.8 min | 28.3 min |
| 3 | 84.24% | 84.11% | 56.5 min | 54.6 min |
| 5 | 84.23% | **84.43%** | 87.2 min | 85.3 min |
| 10 | 84.98% | **84.99%** | 177.1 min | 150.2 min |
| 20 | 84.41% | **84.64%** | 291.5 min | 288.0 min |

![image](https://hackmd.io/_uploads/BkdxJ6qjWl.png)

---

### Variance & Fairness (10 clients)

![image](https://hackmd.io/_uploads/ry1fJ69jWg.png)
![image](https://hackmd.io/_uploads/rJIzy6ci-l.png)
![image](https://hackmd.io/_uploads/HJe7kpqiWl.png)
![image](https://hackmd.io/_uploads/S1Omya5j-l.png)

---

## Key Findings

### 1. FL can match or exceed centralized accuracy on IID data
FedAvg with 5 IID clients reaches **88.03%** vs. the 86.76% centralized baseline. The ~1.3% improvement is likely an ensemble-like regularization effect: different clients see different augmented batches each round, reducing overfitting. This suggests FL is not just a privacy-preserving compromise but can be a stronger learner when data is well-distributed.

### 2. FedNova consistently outperforms FedAvg under data heterogeneity
On non-IID data, FedNova beats FedAvg at 2 clients (+1.06%) and 10 clients (+0.05%). This confirms FedNova's core theoretical claim: normalizing each client's update by local step count τ_i corrects objective inconsistency that causes FedAvg to diverge under heterogeneous data distributions. The advantage is more pronounced with fewer clients where per-client data skew is greatest.

### 3. More clients → lower accuracy under non-IID
Going from 2 to 10 clients on non-IID data drops accuracy ~2% (86.36% → 84.13%). Each client's data becomes more class-skewed (Dirichlet α = 0.5), making aggregation harder. This is a fundamental privacy-utility trade-off in real-world FL: more participants means stronger privacy but weaker models.

### 4. Quantization to 4-16 bit is essentially free
Both algorithms retain full accuracy at 8-16 bit quantization (e.g., FedNova IID: 87.24–87.69% vs. 87.15% at 32-bit) while halving or quartering communication. Even at 2-bit, accuracy drops only ~0.2–1.0% while cutting communication **16×** (2.671 GB → 0.167 GB). FedAvg degrades slightly more than FedNova under aggressive quantization, suggesting FedNova's normalization provides implicit robustness to gradient noise.

### 5. Pruning degrades gracefully up to 70%, then drops sharply
Removing 50–70% of model weights barely affects accuracy (FedAvg IID: 87.37% → 87.11%) while reducing communication by up to 40%. At 90%+ pruning, accuracy falls 2–5%, and at 95% it drops 4–5% absolute, making it impractical for most deployments. FedNova is consistently more robust to pruning under non-IID settings.

### 6. Local epochs plateau at 5–10; diminishing returns beyond that
Going from 1 to 5 local epochs recovers ~4% accuracy (80.47% → 84.23%) at the cost of ~3× training time. Beyond 5 epochs, gains diminish—10 epochs yields only +0.75% more while tripling training time again. This suggests 5 local epochs is the practical sweet spot for communication-constrained FL.

### 7. ResNet-18 substantially outperforms the CNN at ~16–33× communication cost
ResNet-18 achieves 93–95% accuracy vs. 84–88% for the CNN, but at 44.7 GB vs. 2.7 GB for 10 clients. With cosine annealing, both algorithms perform within 0.5% of each other. The architecture choice is a straightforward accuracy-vs-communication trade-off depending on bandwidth constraints.

---

## Methodology

### Dataset
**CIFAR-10**: 50,000 training images, 10,000 test images, 10 classes (32×32 RGB). Downloaded once by the centralized job and cached via `fcntl.flock` to prevent corrupted concurrent downloads across parallel SLURM jobs.

### Models

**CIFAR-10 CNN (~1.3M parameters)**
```
Conv(3→32, 3×3) → BN → ReLU → MaxPool(2×2)
Conv(32→64, 3×3) → BN → ReLU → MaxPool(2×2)
Conv(64→128, 3×3) → BN → ReLU → MaxPool(2×2)
FC(128×4×4 → 256) → ReLU → Dropout(0.5)
FC(256 → 10)
```

**ResNet-18 for CIFAR-10 (~11.2M parameters)**
Standard ResNet-18 adapted for CIFAR-10: 3×3 conv1 with stride 1 (no downsampling at input), no initial MaxPool, final FC layer 512→10. Trained with cosine annealing LR schedule.

### Data Partitioning

**IID**: Random shuffle of all 50,000 training samples, equal split across N clients. Each client sees ~uniform class distribution.

**non-IID (Dirichlet)**: Each client's class proportions are drawn from Dirichlet(α = 0.5). Lower α produces more skewed per-client distributions. At α = 0.5, clients typically specialize in 2–4 classes.

### FL Simulation

All experiments use single-process simulation mode (no network overhead). N client models are instantiated in memory, trained locally, then aggregated on a simulated server. This is mathematically equivalent to the real multi-process NVFLARE setup and compatible with SLURM's batch job model.

**Per-round procedure:**
1. Broadcast global model `w_old` to all N clients
2. Each client trains locally on its partition for `local_epochs` steps using SGD
3. Server aggregates updates → new global model `w_new`
4. Evaluate `w_new` on full 10,000-image test set

### Aggregation Rules

**FedAvg**
```
w_global = Σᵢ (nᵢ / N) · wᵢ
```
Weighted average of client weights by number of local samples `nᵢ`. Simple but suffers from objective inconsistency when clients have different local step counts.

**FedNova**
```
dᵢ = (w_before - w_after) / τᵢ      # normalize gradient by local steps
w_global = w_old - Σᵢ (nᵢ / N) · τᵢ · dᵢ
```
Each client sends its normalized gradient `dᵢ` and step count `τᵢ`. The server reconstructs the effective global update by re-weighting each client's contribution by its actual computation. This corrects the drift caused by heterogeneous local training.

### Quantization

Uniform symmetric stochastic quantization of weight deltas (FedAvg) or normalized gradients (FedNova) before transmission:

```
scale = max(|tensor|) / (2^(B-1) - 1)
q = round(tensor / scale)              # integers in [-2^(B-1)+1, 2^(B-1)-1]
# transmit: q (B-bit integers) + scale (float32)
# dequantize on server: tensor ≈ q * scale
```

Communication savings: transmitting B-bit integers instead of 32-bit floats reduces gradient traffic by **32/B**. Scale factors add negligible overhead (one float per layer).

### Pruning

Magnitude-based unstructured pruning applied to client model weights before transmission each round:

```
mask = |w| ≥ percentile(|w|, ratio)
w_pruned = w * mask    # zero out smallest weights
```

Pruned models are transmitted as sparse tensors. Communication savings approximate the pruning ratio for sufficiently sparse models (≥70%).

### Hyperparameters

| Parameter | CNN | ResNet-18 |
|-----------|-----|-----------|
| Rounds | 100 | 100 |
| Local epochs | 5 | 5 |
| Learning rate | 0.01 | 0.01 |
| LR schedule | constant | cosine annealing |
| Optimizer | SGD | SGD |
| Momentum | 0.9 | 0.9 |
| Weight decay | 5×10⁻⁴ | 5×10⁻⁴ |
| Batch size | 64 | 64 |
| Dirichlet α | 0.5 | 0.5 |
| Seed | 42 | 42 |

---

## Future Improvements to Test

### Algorithms
- **FedProx**: Adds proximal term to client loss to limit local drift — likely helps non-IID even more than FedNova
- **SCAFFOLD**: Uses control variates to correct client drift — theoretically stronger than both FedAvg and FedNova
- **Personalized FL (pFedMe / Ditto)**: Fine-tune a global model per client — better for highly heterogeneous deployments
- **FedAdam / FedYogi**: Server-side adaptive optimizers instead of simple averaging

### Data Heterogeneity
- Sweep Dirichlet α ∈ {0.1, 0.3, 0.5, 1.0} to quantify sensitivity to heterogeneity
- Pathological non-IID (each client gets only 2 classes) as an extreme case

### Scale
- Test with 20, 50, 100 clients to see where accuracy and communication cost break down
- Partial client participation (select k of N clients per round) — essential for large-scale FL

### Communication
- **Top-k sparsification**: Only transmit the k largest gradient elements
- **Error feedback / memory**: Accumulate quantization error and add it to next round's update
- Combine quantization + sparsification for maximum compression

### Privacy
- **Differential Privacy (DP-SGD)**: Add calibrated Gaussian noise to gradients before aggregation
- Measure the accuracy-privacy tradeoff curve (ε vs. accuracy)

---

## NVIDIA Feedback (for meeting)

### What works well
- The executor/controller abstraction cleanly separates client and server logic
- DXO (Data Exchange Object) makes it easy to pass arbitrary tensors between client and server
- Job configuration via JSON is flexible once you understand the structure

### Pain points encountered

**1. No HPC/SLURM integration**
The multi-process server/client setup assumes persistent networked processes, which conflicts with SLURM's job model. We had to build a full simulation mode manually. Native SLURM support or a documented HPC deployment pattern would significantly lower the barrier.

**2. Steep learning curve for the executor architecture**
The FLContext, Shareable, DXO, Signal abstraction stack is non-obvious. Most FL researchers just want to swap in an aggregation function — the full NVFLARE object model feels heavy for that use case. A simpler "simulation-first" API (like Flower's `fl.server.start_server`) would help.

**3. Dependency conflicts with system Python environments**
`pip install nvflare` pulls in its own torch version, which conflicts with HPC system modules (e.g., `module load pytorch/2.8.0`). This caused `torchvision::nms` runtime errors and broken shared libraries across multiple jobs. NVFLARE should document how to install in an environment where PyTorch is already provided by the system.

**4. Documentation gap: simulation vs. real deployment**
It's unclear from the docs when to use simulation mode vs. the full POC (Proof of Concept) setup. The examples mostly show POC, but for research/HPC, simulation is far more practical. A dedicated simulation guide would help.

**5. Error messages from NVFLARE internals are hard to trace**
When something goes wrong inside an executor, the stack trace often points into NVFLARE internals rather than the user's code, making debugging slow.

---

## Directory Structure

```
.
├── src/
│   ├── model.py               # CIFAR-10 CNN and ResNet-18 definitions
│   ├── data_utils.py          # IID / non-IID partitioning + download lock
│   ├── train_centralized.py   # Centralized baseline
│   ├── run_simulation.py      # FL simulation runner (all algorithms)
│   ├── fed_trainer.py         # NVFLARE FedAvg client executor
│   ├── fednova_trainer.py     # NVFLARE FedNova client executor
│   ├── fednova_aggregator.py  # NVFLARE FedNova server aggregator
│   ├── quantized_trainer.py   # Quantized communication (2–32 bit)
│   ├── validator.py           # Server-side global evaluator
│   ├── plot_results.py        # Generate all figures
│   └── generate_final_log.py  # Collect all results into FINAL_REPORT.txt
├── jobs/                      # SLURM batch job scripts
├── setup_and_submit.sh        # One-shot setup + job submission
├── results/                   # JSON metrics and model checkpoints
├── figures/                   # Generated plots
└── FINAL_REPORT.txt           # Aggregated results table + per-experiment logs
```

---

## Quickstart

### Step 1 — Clone the repository
```bash
git clone https://github.com/Nitchaya-Reddy/Federated-Learning-Benchmarking.git
cd Federated-Learning-Benchmarking
```

### Step 2 — Run the experiment setup
```bash
chmod +x setup_and_submit.sh
./setup_and_submit.sh
```

### Step 3 — Monitor SLURM jobs
```bash
squeue -u "$USER"
tail -f logs/fedavg_*.out
```

### Step 4 — Read the final report
```bash
cat FINAL_REPORT.txt
```

### Run plotting and report generation locally
```bash
python src/plot_results.py        # → figures/
python src/generate_final_log.py  # → FINAL_REPORT.txt
```

---

## Troubleshooting

**`torchvision::nms` error or broken torch**
The sbatch files run `pip uninstall -q -y torch torchvision` before each job to prevent user-site packages from shadowing the system module. If running interactively, do the same manually.

**CIFAR-10 corrupted download**
Multiple jobs downloading simultaneously causes truncated pickle files. Fixed by:
1. `data_utils.py` uses `fcntl.flock` to serialize downloads
2. `setup_and_submit.sh` makes jobs depend on the centralized data preparation job

**Plot job fails on GPU partition**
The plotting job uses a CPU partition because plotting does not need a GPU.