"""Evaluation utilities. Run as a script to evaluate a saved checkpoint."""
import argparse

import torch
import torch.nn as nn

from data import get_loaders
from model import build_model
from utils import get_device


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Return (mean loss, top-1 accuracy in %) over the loader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += criterion(out, y).item() * y.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved VGG6 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg["activation"], cfg["batch_norm"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    _, val_loader, test_loader = get_loaders(cfg["batch_size"], cfg["seed"], device)
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"val_loss={val_loss:.4f} val_acc={val_acc:.2f} | test_loss={test_loss:.4f} test_acc={test_acc:.2f}")


if __name__ == "__main__":
    main()
