# VGG6 on CIFAR-10: activation / optimizer / hyperparameter study

CS6886W Assignment 1. A VGG6 (4 conv + 1 linear layer, BatchNorm, global average pooling) trained on CIFAR-10 with PyTorch, tracked with Weights & Biases (W&B).

**GitHub repository:** https://github.com/cs25m532/vgg6-cifar10
**W&B project (all runs and plots):** https://wandb.ai/cs25m532-iitm-india/vgg6-cifar10

## Repository layout

| File | Purpose |
|---|---|
| `data.py` | CIFAR-10 loading, 45k/5k train/val split, GPU-side augmentation and normalization |
| `model.py` | VGG6 with configurable activation and BatchNorm |
| `train.py` | Training loop, W&B logging, checkpointing (also the sweep entry point) |
| `evaluate.py` | Evaluation function; CLI to evaluate a saved checkpoint |
| `utils.py` | Seeding, device selection, optimizer and scheduler factories |
| `export_results.py` | Dumps all W&B runs to `results.csv` for the report tables |
| `sweeps/*.yaml` | W&B sweep configs (activation, optimizer, batch/epochs/lr, random) |
| `models/best_model.pt` | Trained checkpoint of the final best configuration |

## Environment

- Reported results were produced on Google Colab (Tesla T4 GPU, CUDA 13.0 driver, Python 3.13, wandb 0.28.1, the PyTorch build preinstalled on the Colab runtime).
- The code was also checked locally on Python 3.9.6 with torch 2.8.0, torchvision 0.23.0, numpy 2.0.2, wandb 0.26.1, pandas 2.3.3. `requirements.txt` lists the packages; any recent PyTorch >= 2.0 works.
- A GPU is strongly recommended (Colab / Kaggle / CUDA machine). Apple MPS and CPU also work but are slow. The whole dataset is kept on the device, so no data-loader workers are needed.

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

`--seed` (default `42`) seeds Python, NumPy, PyTorch and CUDA, plus a `torch.Generator` on the device that drives the training shuffle order and the random crop, flip and Cutout. `cudnn.deterministic=True`, `cudnn.benchmark=False`, and `torch.use_deterministic_algorithms(True, warn_only=True)` are enabled. The train/val split uses a fixed seed of `0`. Results on the same hardware and library versions reproduce exactly (the best configuration was re-run twice on a Colab T4 and gave identical accuracy and identical weights); different GPUs or library versions can differ slightly.

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
wandb agent cs25m532-iitm-india/vgg6-cifar10/<sweep_id>
# limit the random sweep: wandb agent --count 20 cs25m532-iitm-india/vgg6-cifar10/<sweep_id>
```

Export a table of all runs: `python export_results.py --path cs25m532-iitm-india/vgg6-cifar10`

Notes on the reported runs: the activation (5), optimizer (18) and batch size/epochs/lr (18) grids plus about 35 random-search runs were run on Colab. After Colab disconnections, 7 grid configurations were re-run by hand with `train.py` using the same arguments and seed, so they have no sweep ID in W&B. All W&B plots are built at the project level.

## Final model (Q4)

Best configuration by validation accuracy (from the W&B parallel-coordinates plot):

```
activation=relu  optimizer=nesterov  lr=0.05  momentum=0.9  weight_decay=0.0005
batch_size=64  epochs=30  scheduler=cosine  batch_norm=True  seed=42
```

Reproduce and save the checkpoint:

```bash
python train.py --activation relu --optimizer nesterov --lr 0.05 --momentum 0.9 \
  --weight_decay 0.0005 --batch_size 64 --epochs 30 --seed 42 --save_path models/best_model.pt
python evaluate.py --checkpoint models/best_model.pt
```

Sweep result: best val acc = 86.12 (epoch 30), test top-1 acc = 85.13.
Re-run result (`models/best_model.pt`): best val acc = 86.12 (epoch 30), test top-1 acc = 85.13, exactly matching the sweep run (re-run on a Colab T4).

## Results and analysis

All numbers are best validation accuracy (%) from a single seed (42), so gaps of about 1 point are within noise. Two identical runs (relu, nesterov, lr 0.05, bs 128, 20 epochs) gave exactly 83.54 val / 82.61 test, so runs are reproducible on the same setup.

### Activation (nesterov, lr 0.05, bs 128, 20 epochs)

| Activation | Val acc | Test acc |
|---|---|---|
| ReLU | 83.54 | 82.61 |
| GELU | 83.04 | 82.54 |
| SiLU | 81.60 | 80.76 |
| Tanh | 72.68 | 71.78 |
| Sigmoid | 61.74 | 61.16 |

Sigmoid and Tanh saturate, so gradients shrink through the layers; Sigmoid is also not zero-centered and has a maximum slope of 0.25, which is why it trains slowest. Tanh is zero-centered with slope up to 1, so it does better but still trails. BatchNorm keeps both trainable but they remain well behind within 20 epochs. ReLU, GELU and SiLU do not saturate for positive inputs and are within about 2 points of each other; ReLU was best in this short budget, and the smooth GELU and SiLU did not give a measurable gain.

### Optimizer (relu, bs 128, 15 epochs, momentum 0.9, wd 5e-4)

| Optimizer | lr 0.001 | lr 0.01 | lr 0.1 |
|---|---|---|---|
| SGD (momentum) | 55.40 | 73.84 | 82.78 |
| Nesterov SGD | 55.28 | 74.28 | 83.14 |
| Adam | 79.86 | 77.02 | 36.78 |
| Adagrad | 54.00 | 72.82 | 79.82 |
| RMSprop | 75.42 | 55.06 | 36.66 |
| Nadam | 79.80 | 79.80 | 33.78 |

- SGD and Nesterov SGD need a large learning rate (0.1) and are slow at 0.001. Nesterov is about 0.4 points ahead of plain momentum at lr 0.1 and 0.01, which is within noise.
- Adam and Nadam adapt the step per parameter, so they reach about 80% already at lr 0.001, but they break down at lr 0.1 (about 35%). Nadam (Adam with Nesterov momentum) behaves like Adam; it was more tolerant of lr 0.01.
- Adagrad accumulates squared gradients, so its effective step keeps shrinking; it needs a large initial lr (best at 0.1) and is slow at 0.001.
- RMSprop was best at lr 0.001 and degraded sharply at larger rates; here it also runs with momentum 0.9, which makes it less stable.
- The best result among all optimizers in this table is SGD-style (Nesterov, lr 0.1). Adam-style optimizers need much smaller learning rates. Weight decay is applied as L2 for every optimizer.

### Batch size, epochs and learning rate (relu, nesterov, momentum 0.9, wd 5e-4)

| Batch size | Epochs | lr 0.01 | lr 0.05 |
|---|---|---|---|
| 64 | 10 | 74.26 | 80.66 |
| 64 | 20 | 79.40 | 84.36 |
| 64 | 30 | 82.50 | 86.12 |
| 128 | 10 | 69.76 | 78.20 |
| 128 | 20 | 76.28 | 83.54 |
| 128 | 30 | 79.60 | 85.96 |
| 256 | 10 | 64.82 | 75.36 |
| 256 | 20 | 72.54 | 80.98 |
| 256 | 30 | 75.76 | 84.02 |

- More epochs help in every row; accuracy was still rising at 30 epochs, so the models are undertrained, not overfit.
- A higher learning rate (0.05 vs 0.01) gives 4 to 11 points more in this short budget; 0.01 is too slow to converge in 10 to 30 epochs.
- At a fixed number of epochs, smaller batches win (e.g. 30 epochs, lr 0.05: 86.12 / 85.96 / 84.02 for bs 64 / 128 / 256) because they make more weight updates per epoch and their gradients are noisier. Larger batches would need a larger learning rate to catch up.
