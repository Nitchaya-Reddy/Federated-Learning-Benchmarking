"""
fednova_aggregator.py
---------------------
NVFLARE server-side aggregator for FedNova.

For FedAvg the server just does a weighted average of weight tensors.
For FedNova we receive normalised gradients + tau from each client and
reconstruct the corrected global update:

    delta_global = sum_i  (n_i / N) * tau_i * d_i

Then apply to the current global model:
    w_new = w_old - lr_server * delta_global

We set lr_server = 1.0 (the clients already absorbed the learning rate).
"""

from __future__ import annotations

import numpy as np

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator


class FedNovaAggregator(Aggregator):

    def __init__(self):
        super().__init__()
        self._contributions: list = []    # list of (normalised_grad, n_i, tau_i)
        self._global_weights = None

    # ------------------------------------------------------------------
    # Called by the server to store each client's submission
    # ------------------------------------------------------------------
    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        try:
            dxo = from_shareable(shareable)
            grad   = dxo.data
            n_i    = dxo.meta.get("num_samples", 1)
            tau_i  = dxo.meta.get("tau", 1)
            self._contributions.append((grad, n_i, tau_i))
            return True
        except Exception as e:
            self.log_error(fl_ctx, f"FedNovaAggregator.accept failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Called once all clients have reported; produce the new global model
    # ------------------------------------------------------------------
    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        if not self._contributions:
            self.log_error(fl_ctx, "No contributions to aggregate.")
            return Shareable()

        total_samples = sum(n for _, n, _ in self._contributions)
        keys = list(self._contributions[0][0].keys())

        # Weighted sum: (n_i / N) * tau_i * d_i
        agg_grad: dict = {k: np.zeros_like(self._contributions[0][0][k], dtype=np.float32)
                          for k in keys}

        for grad, n_i, tau_i in self._contributions:
            w = (n_i / total_samples) * tau_i
            for k in keys:
                agg_grad[k] += w * grad[k].astype(np.float32)

        # Apply update to global weights: w_new = w_old - delta
        if self._global_weights is not None:
            new_weights = {}
            for k in keys:
                new_weights[k] = self._global_weights[k] - agg_grad[k]
        else:
            # Fallback: just return the aggregated delta (controller handles it)
            new_weights = agg_grad

        self._contributions.clear()

        out_dxo = DXO(data_kind=DataKind.WEIGHTS, data=new_weights)
        return out_dxo.to_shareable()

    def set_global_weights(self, weights: dict):
        """Called by the controller before each round."""
        self._global_weights = {k: v.copy() for k, v in weights.items()}
