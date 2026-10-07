"""
fednova_trainer.py
------------------
NVFLARE client executor for FedNova (Wang et al., NeurIPS 2020).

Key idea vs FedAvg:
  FedAvg sends raw weight deltas.  When clients do different numbers of
  local steps (e.g. because their datasets differ in size), simply
  averaging those deltas introduces "objective inconsistency" -- the
  global update points in the wrong direction.

FedNova fix:
  Each client normalises its cumulative gradient by the number of local
  steps taken (tau_i), sending a *normalised* gradient instead of raw
  weights.  The server then scales each contribution by tau_i before
  aggregating, so the effective update aligns with the true federated
  objective.

  Mathematically:
      d_i  = (w_0 - w_i) / tau_i          # normalised gradient
      agg  = sum_i (n_i / N) * tau_i * d_i  # server reconstruction

  In practice we send (delta, tau) and let the server do the scaling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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


class FedNovaTrainer(Executor):
    """
    Client executor that implements the FedNova local update rule.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        num_clients: int = 10,
        partition: str = "iid",
        alpha: float = 0.5,
        local_epochs: int = 5,
        batch_size: int = 64,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        seed: int = 42,
        rho: float = 0.0,   # momentum correction coefficient (0 = no correction)
    ):
        super().__init__()
        self.data_dir     = data_dir
        self.num_clients  = num_clients
        self.partition    = partition
        self.alpha        = alpha
        self.local_epochs = local_epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay
        self.seed         = seed
        self.rho          = rho

        self._train_loader = None
        self._device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _lazy_init(self, fl_ctx: FLContext):
        if self._train_loader is not None:
            return
        client_name = fl_ctx.get_identity_name()
        client_id   = int(client_name.split("-")[1]) - 1

        train_ds, _ = get_datasets(self.data_dir)

        if self.partition == "iid":
            partitions = iid_partition(train_ds, self.num_clients, self.seed)
        else:
            partitions = dirichlet_partition(train_ds, self.num_clients,
                                             self.alpha, self.seed)

        self._train_loader = make_client_loader(
            train_ds, partitions[client_id], self.batch_size
        )
        self._num_samples = len(partitions[client_id])
        self.log_info(fl_ctx, f"[FedNova] Client {client_id}: "
                              f"{self._num_samples} samples ({self.partition})")

    def execute(self, task_name: str, shareable: Shareable,
                fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == "train":
            return self._train(shareable, fl_ctx, abort_signal)
        return make_reply(ReturnCode.TASK_UNKNOWN)

    def _train(self, shareable: Shareable, fl_ctx: FLContext,
               abort_signal: Signal) -> Shareable:

        self._lazy_init(fl_ctx)

        dxo    = from_shareable(shareable)
        global_weights = dxo.data

        model = CIFAR10CNN().to(self._device)
        model.load_state_dict({k: torch.tensor(v) for k, v in global_weights.items()})

        # Snapshot of global weights (w_0)
        w0 = {k: v.clone() for k, v in model.state_dict().items()}

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr,
                              momentum=self.momentum, weight_decay=self.weight_decay)

        # Count local steps (tau_i)
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

        # Compute normalised gradient: d_i = (w_0 - w_i) / tau_i
        w_i = model.state_dict()
        normalised_grad = {}
        for k in w0:
            delta = w0[k].float() - w_i[k].float()
            normalised_grad[k] = (delta / tau).cpu().numpy()

        out_dxo = DXO(
            data_kind=DataKind.WEIGHT_DIFF,
            data=normalised_grad,
            meta={
                "num_samples": self._num_samples,
                "tau": tau,          # server needs tau for re-scaling
                "rho": self.rho,
            },
        )
        return out_dxo.to_shareable()
