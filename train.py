"""Train VGG6 on CIFAR-10 with a configurable setup and W&B logging (also used as the sweep entry point)."""
import argparse
import copy
import math
import os

import torch
import torch.nn as nn
import wandb

from data import get_loaders
from evaluate import evaluate
from model import ACTIVATIONS, build_model
from utils import OPTIMIZERS, build_optimizer, build_scheduler, get_device, set_seed


def str2bool(s):
    return str(s).lower() in ("1", "true", "yes")


def parse_args():
    p = argparse.ArgumentParser(description="Train VGG6 on CIFAR-10")
    p.add_argument("--activation", default="relu", choices=list(ACTIVATIONS))
    p.add_argument("--optimizer", default="nesterov", choices=OPTIMIZERS)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "none"])
    p.add_argument("--batch_norm", type=str2bool, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--project", default="vgg6-cifar10")
    p.add_argument("--save_path", default=None, help="checkpoint path (default: checkpoints/<run_name>.pt)")
    p.add_argument("--no_wandb", action="store_true", help="disable W&B logging")
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One pass over the training set; returns (mean loss, accuracy %) measured while training."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    args = parse_args()
    run = wandb.init(project=args.project, config=vars(args), mode="disabled" if args.no_wandb else "online")
    cfg = dict(run.config)  # sweep values override the CLI defaults
    run.name = (f"act_{cfg['activation']}_opt_{cfg['optimizer']}_bs_{cfg['batch_size']}"
                f"_lr_{cfg['lr']}_ep_{cfg['epochs']}_mom_{cfg['momentum']}_wd_{cfg['weight_decay']}")
    run.define_metric("val_acc", summary="max")
    run.define_metric("val_loss", summary="min")

    set_seed(cfg["seed"])
    device = get_device()
    train_loader, val_loader, test_loader = get_loaders(cfg["batch_size"], cfg["seed"], device)
    model = build_model(cfg["activation"], cfg["batch_norm"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["momentum"], cfg["weight_decay"])
    scheduler = build_scheduler(cfg["scheduler"], optimizer, cfg["epochs"])

    best_val, best_epoch, best_state = -1.0, -1, None
    for epoch in range(cfg["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if not math.isfinite(train_loss):  # diverged, stop wasting time
            print(f"Loss diverged at epoch {epoch}; stopping.")
            break
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()
        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss,
                   "val_acc": val_acc, "lr_current": optimizer.param_groups[0]["lr"]}, step=epoch)
        print(f"Epoch {epoch + 1}/{cfg['epochs']} train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}", flush=True)
        if val_acc > best_val:
            best_val, best_epoch, best_state = val_acc, epoch, copy.deepcopy(model.state_dict())

    if best_state is None:
        run.finish()
        return

    # Report test accuracy of the best-validation checkpoint.
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    run.summary.update({"best_val_acc": best_val, "best_epoch": best_epoch, "test_acc": test_acc, "test_loss": test_loss})
    print(f"Best val_acc={best_val:.2f} (epoch {best_epoch + 1}) | test_acc={test_acc:.2f}")

    save_path = cfg["save_path"] or os.path.join("checkpoints", f"{run.name}.pt")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({"state_dict": best_state, "config": cfg, "val_acc": best_val, "test_acc": test_acc}, save_path)
    run.finish()


if __name__ == "__main__":
    main()
