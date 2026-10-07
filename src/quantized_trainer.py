"""
quantized_trainer.py
--------------------
Optional component: quantized communication (FedPAQ-inspired).

Instead of sending full 32-bit float weights, each client quantizes
its weight update to B bits before transmission.  This dramatically
cuts communication cost at the expense of some quantization noise.

Quantization levels we test:
  B = 32  -> full precision (baseline)
  B = 16  -> float16  (2x compression)
  B = 8   -> int8     (4x compression)
  B = 4   -> int4     (8x compression)
  B = 2   -> int2     (16x compression)

Improvements added (v2):
  - Error Feedback: accumulates quantization error across rounds and
    adds it to the next round's delta before quantizing, reducing
    the drift caused by repeated rounding.
  - Top-k Sparsification: only the top-k% largest (by absolute value)
    gradient elements are transmitted; the rest are zeroed out.
    Combined with Error Feedback, the sparsified residual is also
    accumulated so nothing is permanently lost.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

from model import CIFAR10CNN
from data_utils import (
    get_datasets, iid_partition, dirichlet_partition, make_client_loader
)


# ------------------------------------------------------------------
# Quantization helpers
# ------------------------------------------------------------------

def uniform_quantize(arr: np.ndarray, bits: int) -> tuple[np.ndarray, float, float]:
    """
    Stochastic uniform quantization.
    Returns (quantized_int_array, scale, zero_point).
    If bits == 32 returns the original array unchanged.
    """
    if bits == 32:
        return arr, 1.0, 0.0

    levels = 2 ** bits - 1
    v_min, v_max = float(arr.min()), float(arr.max())

    if v_max == v_min:
        return np.zeros_like(arr, dtype=np.int32), 1.0, float(v_min)

    scale      = (v_max - v_min) / levels
    zero_point = v_min

    # Stochastic rounding: add uniform noise in [0, scale) before flooring
    noise    = np.random.uniform(0, scale, size=arr.shape).astype(np.float32)
    q_float  = (arr - zero_point + noise) / scale
    q_int    = np.clip(np.floor(q_float), 0, levels).astype(np.int32)

    return q_int, scale, zero_point


def dequantize(q_int: np.ndarray, scale: float, zero_point: float,
               bits: int) -> np.ndarray:
    if bits == 32:
        return q_int.astype(np.float32)
    return (q_int.astype(np.float32) * scale + zero_point)


def compute_bytes(q_int: np.ndarray, scale: float, bits: int) -> int:
    """
    Approximate bytes sent:
      - quantized tensor packed at `bits` bits per element
      - 2 floats (scale, zero_point) per layer (negligible)
    """
    if bits == 32:
        return q_int.size * 4
    return int(np.ceil(q_int.size * bits / 8)) + 8


# ------------------------------------------------------------------
# Top-k Sparsification helper
# ------------------------------------------------------------------

def topk_sparsify(arr: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only the top-k% elements by absolute value; zero out the rest.
    Returns (sparsified array, boolean mask of kept elements).
    ratio=0.1 means keep the largest 10% of values.
    """
    flat = arr.flatten()
    k = max(1, int(len(flat) * ratio))
    threshold_idx = np.argpartition(np.abs(flat), -k)[-k:]
    mask = np.zeros(flat.shape, dtype=bool)
    mask[threshold_idx] = True
    sparsified = np.where(mask, flat, 0.0).reshape(arr.shape)
    return sparsified.astype(np.float32), mask.reshape(arr.shape)


# ------------------------------------------------------------------
# Quantized FedAvg Trainer  (+ Error Feedback + Top-k)
# ------------------------------------------------------------------

class QuantizedFedAvgTrainer(Executor):
    """
    FedAvg client that quantizes weight deltas before sending.
    Now includes:
      - Error Feedback: leftover quantization error is added to next round's delta
      - Top-k Sparsification: only top-k% of delta elements are transmitted
    Works for both IID and non-IID partitions.
    """

    def __init__(
        self,
        data_dir: str       = "./data",
        num_clients: int    = 10,
        partition: str      = "iid",
        alpha: float        = 0.5,
        local_epochs: int   = 5,
        batch_size: int     = 64,
        lr: float           = 0.01,
        momentum: float     = 0.9,
        weight_decay: float = 5e-4,
        quant_bits: int     = 8,
        top_k_ratio: float  = 0.1,   # keep top 10% of gradient elements
        seed: int           = 42,
    ):
        super().__init__()
        self.data_dir      = data_dir
        self.num_clients   = num_clients
        self.partition     = partition
        self.alpha         = alpha
        self.local_epochs  = local_epochs
        self.batch_size    = batch_size
        self.lr            = lr
        self.momentum      = momentum
        self.weight_decay  = weight_decay
        self.quant_bits    = quant_bits
        self.top_k_ratio   = top_k_ratio
        self.seed          = seed

        self._train_loader   = None
        self._device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Error feedback buffers: one per layer, persists across rounds
        self._error_feedback: dict[str, np.ndarray] = {}

    def _lazy_init(self, fl_ctx: FLContext):
        if self._train_loader is not None:
            return
        client_name = fl_ctx.get_identity_name()
        client_id   = int(client_name.split("-")[1]) - 1

        train_ds, _ = get_datasets(self.data_dir)
        if self.partition == "iid":
            parts = iid_partition(train_ds, self.num_clients, self.seed)
        else:
            parts = dirichlet_partition(train_ds, self.num_clients, self.alpha, self.seed)

        self._train_loader = make_client_loader(train_ds, parts[client_id], self.batch_size)
        self._num_samples  = len(parts[client_id])
        self.log_info(fl_ctx,
            f"[QuantFedAvg-{self.quant_bits}bit+EF+TopK] Client {client_id}: "
            f"{self._num_samples} samples")

    def execute(self, task_name: str, shareable: Shareable,
                fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == "train":
            return self._train(shareable, fl_ctx, abort_signal)
        return make_reply(ReturnCode.TASK_UNKNOWN)

    def _train(self, shareable: Shareable, fl_ctx: FLContext,
               abort_signal: Signal) -> Shareable:

        self._lazy_init(fl_ctx)

        dxo = from_shareable(shareable)
        global_weights = dxo.data

        model = CIFAR10CNN().to(self._device)
        model.load_state_dict({k: torch.tensor(v) for k, v in global_weights.items()})

        w_before = {k: v.clone().cpu().numpy() for k, v in model.state_dict().items()}

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr,
                              momentum=self.momentum, weight_decay=self.weight_decay)

        model.train()
        for _ in range(self.local_epochs):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            for x, y in self._train_loader:
                x, y = x.to(self._device), y.to(self._device)
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                optimizer.step()

        w_after = {k: v.cpu().numpy() for k, v in model.state_dict().items()}

        quant_payload = {}
        total_bytes   = 0

        for k in w_before:
            # Raw delta
            delta = (w_before[k].astype(np.float32)
                     - w_after[k].astype(np.float32))

            # Step 1 — Error Feedback: add accumulated error from previous round
            if k in self._error_feedback:
                delta = delta + self._error_feedback[k]

            # Step 2 — Top-k Sparsification: keep only top-k% elements
            sparse_delta, mask = topk_sparsify(delta, self.top_k_ratio)

            # Accumulate the residual (what we didn't send) as error feedback
            self._error_feedback[k] = (delta - sparse_delta).astype(np.float32)

            # Step 3 — Quantize the sparse delta
            q_int, scale, zp = uniform_quantize(sparse_delta, self.quant_bits)
            total_bytes      += compute_bytes(q_int, scale, self.quant_bits)

            # Dequantize locally to simulate what the server receives
            deq_delta         = dequantize(q_int, scale, zp, self.quant_bits)
            quant_payload[k]  = (w_before[k] - deq_delta).astype(np.float32)

        out_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=quant_payload,
            meta={
                "num_samples": self._num_samples,
                "quant_bits":  self.quant_bits,
                "bytes_sent":  total_bytes,
                "top_k_ratio": self.top_k_ratio,
            },
        )
        return out_dxo.to_shareable()


# ------------------------------------------------------------------
# Quantized FedNova Trainer  (+ Error Feedback + Top-k)
# ------------------------------------------------------------------

class QuantizedFedNovaTrainer(Executor):
    """
    FedNova client that quantizes normalised gradients before sending.
    Now includes:
      - Error Feedback: leftover quantization error added to next round
      - Top-k Sparsification: only top-k% of gradient elements transmitted
    """

    def __init__(
        self,
        data_dir: str       = "./data",
        num_clients: int    = 10,
        partition: str      = "iid",
        alpha: float        = 0.5,
        local_epochs: int   = 5,
        batch_size: int     = 64,
        lr: float           = 0.01,
        momentum: float     = 0.9,
        weight_decay: float = 5e-4,
        quant_bits: int     = 8,
        top_k_ratio: float  = 0.1,
        seed: int           = 42,
    ):
        super().__init__()
        self.data_dir      = data_dir
        self.num_clients   = num_clients
        self.partition     = partition
        self.alpha         = alpha
        self.local_epochs  = local_epochs
        self.batch_size    = batch_size
        self.lr            = lr
        self.momentum      = momentum
        self.weight_decay  = weight_decay
        self.quant_bits    = quant_bits
        self.top_k_ratio   = top_k_ratio
        self.seed          = seed

        self._train_loader   = None
        self._device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._error_feedback: dict[str, np.ndarray] = {}

    def _lazy_init(self, fl_ctx: FLContext):
        if self._train_loader is not None:
            return
        client_name = fl_ctx.get_identity_name()
        client_id   = int(client_name.split("-")[1]) - 1

        train_ds, _ = get_datasets(self.data_dir)
        if self.partition == "iid":
            parts = iid_partition(train_ds, self.num_clients, self.seed)
        else:
            parts = dirichlet_partition(train_ds, self.num_clients, self.alpha, self.seed)

        self._train_loader = make_client_loader(train_ds, parts[client_id], self.batch_size)
        self._num_samples  = len(parts[client_id])

    def execute(self, task_name: str, shareable: Shareable,
                fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == "train":
            return self._train(shareable, fl_ctx, abort_signal)
        return make_reply(ReturnCode.TASK_UNKNOWN)

    def _train(self, shareable: Shareable, fl_ctx: FLContext,
               abort_signal: Signal) -> Shareable:

        self._lazy_init(fl_ctx)

        dxo = from_shareable(shareable)
        global_weights = dxo.data

        model = CIFAR10CNN().to(self._device)
        model.load_state_dict({k: torch.tensor(v) for k, v in global_weights.items()})
        w0 = {k: v.clone().cpu().numpy().astype(np.float32)
              for k, v in model.state_dict().items()}

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr,
                              momentum=self.momentum, weight_decay=self.weight_decay)

        tau = 0
        model.train()
        for _ in range(self.local_epochs):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            for x, y in self._train_loader:
                x, y = x.to(self._device), y.to(self._device)
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                optimizer.step()
                tau += 1

        w_i = {k: v.cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}

        # FedNova normalised gradient
        norm_grad = {k: (w0[k] - w_i[k]) / tau for k in w0}

        quant_grad  = {}
        total_bytes = 0

        for k, g in norm_grad.items():
            # Step 1 — Error Feedback
            if k in self._error_feedback:
                g = g + self._error_feedback[k]

            # Step 2 — Top-k Sparsification
            sparse_g, mask = topk_sparsify(g, self.top_k_ratio)

            # Accumulate residual as error feedback
            self._error_feedback[k] = (g - sparse_g).astype(np.float32)

            # Step 3 — Quantize
            q_int, scale, zp = uniform_quantize(sparse_g, self.quant_bits)
            total_bytes      += compute_bytes(q_int, scale, self.quant_bits)
            quant_grad[k]     = dequantize(q_int, scale, zp, self.quant_bits)

        out_dxo = DXO(
            data_kind=DataKind.WEIGHT_DIFF,
            data=quant_grad,
            meta={
                "num_samples": self._num_samples,
                "tau":         tau,
                "quant_bits":  self.quant_bits,
                "bytes_sent":  total_bytes,
                "top_k_ratio": self.top_k_ratio,
            },
        )
        return out_dxo.to_shareable()
