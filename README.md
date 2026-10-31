# VGG6 on CIFAR-10: activation / optimizer / hyperparameter study

CS6886W Assignment 1. A VGG6 (4 conv + 1 linear layer, BatchNorm, global average pooling) trained on CIFAR-10 with PyTorch, tracked with Weights & Biases (W&B).

**PDF report link / GitHub link:** `<paste repo URL here>`

## Repository layout

| File | Purpose |
|---|---|
| `data.py` | CIFAR-10 transforms, 45k/5k train/val split, loaders |
| `model.py` | VGG6 with configurable activation and BatchNorm |
| `train.py` | Training loop, W&B logging, checkpointing (also the sweep entry point) |
| `evaluate.py` | Evaluation function; CLI to evaluate a saved checkpoint |
| `utils.py` | Seeding, device selection, optimizer and scheduler factories |
| `export_results.py` | Dumps all W&B runs to `results.csv` for the report tables |
| `sweeps/*.yaml` | W&B sweep configs (activation, optimizer, batch/epochs/lr, random) |
| `models/best_model.pt` | Trained checkpoint of the final best configuration |

## Environment

- Python 3.9+, PyTorch >= 2.0 (tested with torch 2.8.0, torchvision 0.23.0, numpy 2.0.2, wandb 0.26.1, pandas 2.3.3)
- GPU strongly recommended (Colab / Kaggle / CUDA machine). Apple MPS and CPU also work but are slow.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wandb login
```

## Data and transforms (Q1a)

The 50k training images are split into 45k train / 5k validation with a fixed seed (the same split for every run). The 10k official test set is used only for the final test accuracy of the best-validation checkpoint.

- Train (applied on the GPU, per batch): `RandomCrop(32, padding=4)`, `RandomHorizontalFlip()` (p=0.5), `Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))`, `Cutout(1 hole, 16x16)`
- Validation / test: `Normalize(...)` with the same statistics

## Seeds and determinism

`--seed` (default `42`) seeds Python, NumPy, PyTorch, CUDA and the DataLoader shuffling generator. `cudnn.deterministic=True`, `cudnn.benchmark=False`, and `torch.use_deterministic_algorithms(True, warn_only=True)` are enabled. The train/val split uses a fixed seed of `0`. Results on the same hardware and library versions should reproduce; different GPUs or library versions can differ slightly.

## Run a single configuration

```bash
python train.py --activation relu --optimizer nesterov --lr 0.05 --momentum 0.9 \
  --weight_decay 0.0005 --batch_size 128 --epochs 30 --seed 42
# without W&B:
python train.py --no_wandb --epochs 1
```

Options: activation `relu|sigmoid|tanh|silu|gelu`; optimizer `sgd|nesterov|adam|adagrad|rmsprop|nadam`; scheduler `cosine|none` (default `cosine`). `sgd` uses `--momentum` (use `0` for plain SGD); `nesterov` forces momentum > 0; `rmsprop` uses `--momentum`; Adam-family optimizers ignore it. Weight decay is L2.

## Run the sweeps (Q2, Q3)

All sweeps log to the same W&B project (`vgg6-cifar10`). Run each, then start an agent with the printed sweep ID:

```bash
wandb sweep --project vgg6-cifar10 sweeps/sweep_activation.yaml   # 5 runs, 20 epochs   (Q2a)
wandb sweep --project vgg6-cifar10 sweeps/sweep_optimizer.yaml    # 18 runs, 15 epochs  (Q2b)
wandb sweep --project vgg6-cifar10 sweeps/sweep_hparams.yaml      # 18 runs  (Q2c)
wandb sweep --project vgg6-cifar10 sweeps/sweep_random.yaml       # random search for the parallel-coordinates plot
wandb agent <entity>/vgg6-cifar10/<sweep_id>
# limit random runs: wandb agent --count 40 <entity>/vgg6-cifar10/<sweep_id>
```

Export a table of all runs: `python export_results.py --path <entity>/vgg6-cifar10`

## Final model (Q4)

Best configuration by validation accuracy (from the W&B parallel-coordinates plot):

```
<fill in after sweeps>
```

Reproduce and save the checkpoint:

```bash
python train.py <flags of the best configuration> --save_path models/best_model.pt
python evaluate.py --checkpoint models/best_model.pt
```

Reported: best val acc = `<..>`, test top-1 acc = `<..>`.

## Results and analysis

`<Fill in after sweeps: tables and discussion for activation, optimizer, and batch size / epochs / learning rate.>`
