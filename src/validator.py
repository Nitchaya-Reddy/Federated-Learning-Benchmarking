"""
validator.py
------------
NVFLARE server-side component that evaluates the global model on the
full CIFAR-10 test set after every aggregation round.

Results are logged to a JSON file so we can plot convergence curves.
"""

from __future__ import annotations

import json
import os
import time

import torch
import torch.nn as nn

from nvflare.apis.dxo import DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

from model import CIFAR10CNN
from data_utils import get_datasets, make_test_loader


class GlobalValidator(Executor):
    """
    Runs on the server node.  Receives global weights, evaluates on
    CIFAR-10 test set, and appends results to a JSON log file.
    """

    def __init__(
        self,
        data_dir: str  = "./data",
        log_file: str  = "./results/fl_metrics.json",
        batch_size: int = 256,
    ):
        super().__init__()
        self.data_dir   = data_dir
        self.log_file   = log_file
        self.batch_size = batch_size

        self._test_loader = None
        self._device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._history     = []
        self._round       = 0

    def _lazy_init(self):
        if self._test_loader is not None:
            return
        _, test_ds = get_datasets(self.data_dir)
        self._test_loader = make_test_loader(test_ds, self.batch_size)

    def execute(self, task_name: str, shareable: Shareable,
                fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == "validate":
            return self._validate(shareable, fl_ctx)
        return make_reply(ReturnCode.TASK_UNKNOWN)

    @torch.no_grad()
    def _validate(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        self._lazy_init()
        self._round += 1

        dxo    = from_shareable(shareable)
        weights = dxo.data

        model = CIFAR10CNN().to(self._device)
        model.load_state_dict({k: torch.tensor(v) for k, v in weights.items()})
        model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0

        t0 = time.time()
        for x, y in self._test_loader:
            x, y  = x.to(self._device), y.to(self._device)
            out   = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
            correct    += out.argmax(1).eq(y).sum().item()
            total      += x.size(0)

        acc  = correct / total
        loss = total_loss / total

        row = dict(
            round=self._round,
            test_acc=round(acc, 4),
            test_loss=round(loss, 4),
            eval_time_s=round(time.time() - t0, 2),
        )
        self._history.append(row)

        # Persist after every round so partial results survive job failure
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "w") as f:
            json.dump(self._history, f, indent=2)

        self.log_info(fl_ctx,
            f"[Round {self._round}] test_acc={acc:.4f}  test_loss={loss:.4f}")

        # Return the metrics so the controller can log them too
        out = Shareable()
        out["test_acc"]  = acc
        out["test_loss"] = loss
        out["round"]     = self._round
        return out
