"""Export all runs of a W&B project to results.csv and print summary tables for the report."""
import argparse

import pandas as pd
import wandb

KEYS = ["activation", "optimizer", "lr", "momentum", "weight_decay", "batch_size", "epochs"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="<entity>/<project>, e.g. myuser/vgg6-cifar10")
    args = p.parse_args()

    rows = []
    for r in wandb.Api().runs(args.path):
        if r.state != "finished":
            continue
        row = {k: r.config.get(k) for k in KEYS}
        row.update(name=r.name, best_val_acc=r.summary.get("best_val_acc"), test_acc=r.summary.get("test_acc"),
                   best_epoch=r.summary.get("best_epoch"))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("best_val_acc", ascending=False)
    df.to_csv("results.csv", index=False)
    print(df.to_string(index=False))
    print("\nBest run:\n", df.iloc[0].to_string())


if __name__ == "__main__":
    main()
