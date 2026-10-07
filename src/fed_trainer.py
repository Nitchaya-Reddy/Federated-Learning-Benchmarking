"""
fed_trainer.py
--------------
NVFLARE client-side executor for FedAvg.

How NVFLARE works (simplified):
  - The SERVER sends the global model weights to each client.
  - Each CLIENT runs this executor: loads weights, trains locally, sends
    updated weights + sample count back.
  - The SERVER aggregates (weighted average) and repeats.

This file is used by all FedAvg experiments (IID and non-IID).
The data partition is determined by the job config JSON.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.optim as optim

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

from model import CIFAR10CNN
from data_utils import (
    get_datasets, iid_partition, dirichlet_partition, make_client_loader
)


class FedAvgTrainer(Executor):
    """
    Client executor: receive global weights -> train locally -> return weights.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        num_clients: int = 10,
        partition: str = "iid",       # "iid" or "noniid"
        alpha: float = 0.5,           # Dirichlet alpha (used when noniid)
        local_epochs: int = 5,
        batch_size: int = 64,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        seed: int = 42,
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

        self._train_loader = None
        self._device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Called once when the FL job starts on this client
    # ------------------------------------------------------------------
    def _lazy_init(self, fl_ctx: FLContext):
        if self._train_loader is not None:
            return

        # Figure out which client index we are
        client_name = fl_ctx.get_identity_name()  # e.g. "site-1"
        client_id   = int(client_name.split("-")[1]) - 1  # 0-indexed

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
        self.log_info(fl_ctx, f"Client {client_id}: {self._num_samples} samples "
                              f"({self.partition} partition)")

    # ------------------------------------------------------------------
    # NVFLARE calls execute() for every task the server sends
    # ------------------------------------------------------------------
    def execute(self, task_name: str, shareable: Shareable,
                fl_ctx: FLContext, abort_signal: Signal) -> Shareable:

        if task_name == "train":
            return self._train(shareable, fl_ctx, abort_signal)

        self.log_error(fl_ctx, f"Unknown task: {task_name}")
        return make_reply(ReturnCode.TASK_UNKNOWN)

    # ------------------------------------------------------------------
    def _train(self, shareable: Shareable, fl_ctx: FLContext,
               abort_signal: Signal) -> Shareable:

        self._lazy_init(fl_ctx)

        # 1. Receive global model weights from server
        dxo    = from_shareable(shareable)
        global_weights = dxo.data

        # 2. Load into local model
        model = CIFAR10CNN().to(self._device)
        model.load_state_dict({k: torch.tensor(v) for k, v in global_weights.items()})

        # 3. Local training
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
                nn.CrossEntropyLoss()(model(x), y).backward()
                optimizer.step()

        # 4. Return updated weights + sample count for weighted averaging
        updated_weights = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
        out_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=updated_weights,
            meta={"num_samples": self._num_samples},
        )
        return out_dxo.to_shareable()
