"""
train_centralized.py
--------------------
Baseline: train on all CIFAR-10 data in one place.
This gives the "best-case" accuracy ceiling we compare FL against.

Saves:
  results/centralized/metrics.json
  results/centralized/model.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import get_datasets, make_test_loader, make_client_loader
from model import CIFAR10CNN, count_parameters


# ------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += out.argmax(1).eq(y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x)
        total_loss += criterion(out, y).item() * x.size(0)
        correct    += out.argmax(1).eq(y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total


# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="./data")
    parser.add_argument("--out_dir",    default="./results/centralized")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.1)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, test_ds = get_datasets(args.data_dir)
    train_loader = make_client_loader(train_ds, list(range(len(train_ds))),
                                      batch_size=args.batch_size)
    test_loader  = make_test_loader(test_ds, batch_size=256)

    model     = CIFAR10CNN().to(device)
    print(f"Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        row = dict(epoch=epoch, train_loss=round(tr_loss, 4),
                   train_acc=round(tr_acc, 4), test_loss=round(te_loss, 4),
                   test_acc=round(te_acc, 4))
        history.append(row)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.5f}")

    elapsed = time.time() - t0
    final = dict(
        final_test_acc=history[-1]["test_acc"],
        final_test_loss=history[-1]["test_loss"],
        total_time_s=round(elapsed, 1),
        history=history,
    )

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(final, f, indent=2)

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
    print(f"\nDone. Final test accuracy: {final['final_test_acc']:.4f}")
    print(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
