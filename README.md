# VGG6 on CIFAR-10

Code for CS6886W Assignment 1. I train a small VGG (VGG6) on CIFAR-10 and see how the activation function, optimizer, batch size, number of epochs and learning rate change the accuracy. Runs are tracked with Weights & Biases (W&B).

- GitHub: https://github.com/cs25m532/vgg6-cifar10
- W&B project with all runs and plots: https://wandb.ai/cs25m532-iitm-india/vgg6-cifar10

The model has 4 conv layers (64, 64, 128, 128 channels, 3x3, each followed by BatchNorm and the activation), a 2x2 max-pool after the 2nd and 4th conv, global average pooling and one linear layer.

## Files

| File | What it does |
|---|---|
| `data.py` | Loads CIFAR-10, splits 45k train / 5k validation, does augmentation and normalization on the GPU |
| `model.py` | VGG6 with a configurable activation |
| `train.py` | Training loop, W&B logging, saves the best checkpoint. Also the entry point for sweeps |
| `evaluate.py` | Evaluation function, and a command to evaluate a saved checkpoint |
| `utils.py` | Seeding, device selection, optimizers, LR scheduler |
| `export_results.py` | Pulls all runs from W&B into `results.csv` |
| `sweeps/*.yaml` | W&B sweep configs (activation, optimizer, batch size/epochs/lr, random search) |
| `models/best_model.pt` | Trained model for the best configuration |

## Environment

The results in this repo were run on Google Colab with a Tesla T4:

- Python 3.13, CUDA 13.0 driver
- torch 2.11.0 (+cu130), torchvision 0.26.0 (+cu130)
- numpy 2.1.3, pandas 2.2.3, wandb 0.28.1

`requirements.txt` has these versions. torch 2.11 needs Python 3.10 or newer. I also ran the code on my laptop (Python 3.9.6, torch 2.8.0, torchvision 0.23.0) to check it works, so any recent PyTorch (2.0+) should be fine, but the exact accuracy can change a little with a different GPU or library version.

You need a GPU to run the sweeps in a reasonable time. CPU and Apple MPS also work but are slow.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wandb login
```

## Data and augmentation (Q1a)

The 50k training images are split into 45k for training and 5k for validation. The split uses a fixed seed (0), so every run sees the same validation images. The 10k test images are only used at the end, to report the test accuracy of the checkpoint with the best validation accuracy.

Training transforms, applied on the GPU for each batch:
1. `RandomCrop(32, padding=4)`
2. `RandomHorizontalFlip(p=0.5)`
3. `Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))`
4. `Cutout`, one 16x16 hole

Validation and test images only get the same `Normalize`.

## Seeds

The default seed is 42 (`--seed`). It seeds Python, NumPy, PyTorch and CUDA, and a generator on the device that controls the shuffling and the random crop, flip and cutout. cuDNN is set to deterministic with benchmark off, and `torch.use_deterministic_algorithms(True, warn_only=True)` is on.

I re-ran the best configuration twice in separate Colab sessions. Both gave the same accuracy and the saved weights were identical.

## Training one configuration

```bash
python train.py --activation relu --optimizer nesterov --lr 0.05 --momentum 0.9 \
  --weight_decay 0.0005 --batch_size 128 --epochs 30 --seed 42

# quick test without W&B
python train.py --no_wandb --epochs 1
```

- Activations: `relu`, `sigmoid`, `tanh`, `silu`, `gelu`
- Optimizers: `sgd`, `nesterov`, `adam`, `adagrad`, `rmsprop`, `nadam`
- Scheduler: `cosine` (default) or `none`
- `sgd` uses `--momentum` (set it to 0 for plain SGD). `nesterov` always uses momentum, 0.9 if you pass 0. `rmsprop` also uses `--momentum`. Adam, Adagrad and Nadam ignore it.
- Weight decay is plain L2 for all optimizers.

## Running the sweeps (Q2, Q3)

```bash
wandb sweep --project vgg6-cifar10 sweeps/sweep_activation.yaml   # 5 runs, 20 epochs
wandb sweep --project vgg6-cifar10 sweeps/sweep_optimizer.yaml    # 18 runs, 15 epochs
wandb sweep --project vgg6-cifar10 sweeps/sweep_hparams.yaml      # 18 runs
wandb sweep --project vgg6-cifar10 sweeps/sweep_random.yaml       # random search

# each sweep prints an ID, then start an agent with it
wandb agent cs25m532-iitm-india/vgg6-cifar10/<sweep_id>

# stop the random sweep after 20 runs
wandb agent --count 20 cs25m532-iitm-india/vgg6-cifar10/<sweep_id>
```

To get all runs as a table: `python export_results.py --path cs25m532-iitm-india/vgg6-cifar10`

What I actually ran: the three grid sweeps and about 35 random-search runs, on Colab. Colab disconnected a few times and killed some runs, and the sweeps did not pick those configurations up again. I deleted the crashed runs and ran those 7 configurations by hand with `train.py`, using the same arguments and seed. That is why those runs have no sweep ID in W&B. The W&B plots are made from the whole project, so this does not matter for them.

## Best model (Q4)

Best validation accuracy across all runs:

```
activation=relu, optimizer=nesterov, lr=0.05, momentum=0.9, weight_decay=0.0005
batch_size=64, epochs=30, cosine scheduler, BatchNorm, seed=42
```

To reproduce it and save the model:

```bash
python train.py --activation relu --optimizer nesterov --lr 0.05 --momentum 0.9 \
  --weight_decay 0.0005 --batch_size 64 --epochs 30 --seed 42 --save_path models/best_model.pt
python evaluate.py --checkpoint models/best_model.pt
```

Result: best validation accuracy 86.12% (epoch 30), test accuracy 85.13%. The re-run on a Colab T4 gave the same numbers.

## Results

All numbers are the best validation accuracy (%) from one run with seed 42. I did not repeat runs with other seeds, so I would not read much into differences of about 1 point. Two identical runs (relu, nesterov, lr 0.05, batch size 128, 20 epochs) gave exactly the same result (83.54 val, 82.61 test).

### Activation

Nesterov SGD, lr 0.05, batch size 128, 20 epochs.

| Activation | Val acc | Test acc |
|---|---|---|
| ReLU | 83.54 | 82.61 |
| GELU | 83.04 | 82.54 |
| SiLU | 81.60 | 80.76 |
| Tanh | 72.68 | 71.78 |
| Sigmoid | 61.74 | 61.16 |

Sigmoid and Tanh are the worst. Both flatten out for large inputs, so the gradients get small as they go back through the layers. Sigmoid is also not centred around zero and its slope is at most 0.25, which fits with it being the slowest. Tanh is centred and has slope up to 1, so it does better, but it is still about 10 points below ReLU after 20 epochs. BatchNorm helps both, but not enough to catch up.

ReLU, GELU and SiLU are within about 2 points of each other. ReLU came out on top in this short training run. I did not see any gain from the smoother GELU and SiLU here, but with one seed I can't say the small gaps are real.

### Optimizer

ReLU, batch size 128, 15 epochs, momentum 0.9, weight decay 5e-4.

| Optimizer | lr 0.001 | lr 0.01 | lr 0.1 |
|---|---|---|---|
| SGD (momentum) | 55.40 | 73.84 | 82.78 |
| Nesterov SGD | 55.28 | 74.28 | 83.14 |
| Adam | 79.86 | 77.02 | 36.78 |
| Adagrad | 54.00 | 72.82 | 79.82 |
| RMSprop | 75.42 | 55.06 | 36.66 |
| Nadam | 79.80 | 79.80 | 33.78 |

- SGD and Nesterov SGD need a big learning rate. They reach about 83% at lr 0.1 and only about 55% at 0.001, since 15 epochs is too short for such small steps. Nesterov is about 0.4 points ahead of plain momentum at 0.1 and 0.01, which is small enough to be noise.
- Adam and Nadam do the opposite. They scale each parameter's step by running gradient statistics, so they get to about 80% with lr 0.001, but at lr 0.1 they fall apart (around 35%). Nadam is Adam with Nesterov momentum and behaves very similarly. It held up better at lr 0.01 (79.80 against 77.02).
- Adagrad keeps adding up squared gradients, so its step size only shrinks over time. It needs a large starting rate (best at 0.1) and is slow at 0.001.
- RMSprop was only good at lr 0.001 and got much worse for larger rates. In my setup it also ran with momentum 0.9, which probably makes it less stable.
- The best result in this table is Nesterov SGD at lr 0.1. The adaptive optimizers need a much smaller learning rate to work at all, so what matters most here is picking the learning rate that fits the optimizer.

### Batch size, epochs and learning rate

ReLU, Nesterov SGD, momentum 0.9, weight decay 5e-4.

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

- More epochs always helped, and accuracy was still going up at 30 epochs. The model is undertrained in these runs, not overfitting.
- lr 0.05 beat lr 0.01 by 4 to 11 points. With only 10 to 30 epochs, 0.01 is too slow.
- With the same number of epochs, smaller batches did better. At 30 epochs and lr 0.05 the results are 86.12, 85.96 and 84.02 for batch sizes 64, 128 and 256. A smaller batch means more weight updates per epoch and noisier gradients. A bigger batch would probably need a higher learning rate to catch up, but I did not test that.
