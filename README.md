# VGG6 on CIFAR-10

Code for CS6886W Assignment 1. A small VGG network (VGG6) is trained on CIFAR-10 with different activations, optimizers, batch sizes, epochs and learning rates. Runs are tracked with Weights & Biases (W&B).

- GitHub: https://github.com/cs25m532/vgg6-cifar10
- W&B project (all runs and plots): https://wandb.ai/cs25m532-iitm-india/vgg6-cifar10

Model: 4 conv layers (64, 64, 128, 128 channels, 3x3), each followed by BatchNorm and the activation. A 2x2 max-pool comes after the 2nd and 4th conv. Then global average pooling and one linear layer.

## Files

| File | Purpose |
|---|---|
| `data.py` | Loads CIFAR-10, splits it into 45k train and 5k validation, does augmentation and normalization on the GPU |
| `model.py` | VGG6 with a choice of activation |
| `train.py` | Training loop, W&B logging, saves the best checkpoint. Also used by the sweeps |
| `evaluate.py` | Evaluation code. Can also evaluate a saved checkpoint from the command line |
| `utils.py` | Seeding, device selection, optimizers, learning-rate scheduler |
| `export_results.py` | Downloads all runs from W&B into `results.csv` |
| `sweeps/*.yaml` | W&B sweep configs |
| `models/best_model.pt` | Trained model of the best configuration |

## Environment

The results were produced on Google Colab with a Tesla T4 GPU:

- Python 3.13, CUDA 13.0 driver
- torch 2.11.0 (+cu130), torchvision 0.26.0 (+cu130)
- numpy 2.1.3, pandas 2.2.3, wandb 0.28.1

`requirements.txt` lists these versions. torch 2.11 needs Python 3.10 or newer. The code was also tested on Python 3.9.6 with torch 2.8.0 and torchvision 0.23.0, so any recent PyTorch (2.0 or newer) should work. The exact accuracy can differ slightly with another GPU or library version.

A GPU is needed to run the sweeps in reasonable time. 

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wandb login
```

## Data and augmentation

The 50k training images are split into 45k for training and 5k for validation. The split uses a fixed seed (0), so all runs use the same validation images. The 10k test images are used only at the end, to report the test accuracy of the checkpoint with the best validation accuracy.

Training transforms, applied on the GPU for each batch:
1. `RandomCrop(32, padding=4)`
2. `RandomHorizontalFlip(p=0.5)`
3. `Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))`
4. `Cutout` with one 16x16 hole

Validation and test images are only normalized, with the same mean and std.

## Seeds

The default seed is 42 (`--seed`). It is used for Python, NumPy, PyTorch and CUDA, and for a generator on the device that controls shuffling and the random crop, flip and cutout. cuDNN runs in deterministic mode with benchmark off, and `torch.use_deterministic_algorithms(True, warn_only=True)` is enabled.

The best configuration was run twice in separate Colab sessions. Both runs gave the same accuracy and identical weights.

## Train one configuration

```bash
python train.py --activation relu --optimizer nesterov --lr 0.05 --momentum 0.9 \
  --weight_decay 0.0005 --batch_size 128 --epochs 30 --seed 42

# quick test without W&B
python train.py --no_wandb --epochs 1
```

- Activations: `relu`, `sigmoid`, `tanh`, `silu`, `gelu`
- Optimizers: `sgd`, `nesterov`, `adam`, `adagrad`, `rmsprop`, `nadam`
- Scheduler: `cosine` (default) or `none`
- `sgd` and `rmsprop` use `--momentum` (0 gives plain SGD). `nesterov` always uses momentum and falls back to 0.9 if 0 is given. Adam, Adagrad and Nadam ignore it.
- Weight decay is L2 for every optimizer.

## Run the sweeps

```bash
wandb sweep --project vgg6-cifar10 sweeps/sweep_activation.yaml   # 5 runs, 20 epochs
wandb sweep --project vgg6-cifar10 sweeps/sweep_optimizer.yaml    # 18 runs, 15 epochs
wandb sweep --project vgg6-cifar10 sweeps/sweep_hparams.yaml      # 18 runs
wandb sweep --project vgg6-cifar10 sweeps/sweep_random.yaml       # random search

# each command prints a sweep ID, use it to start an agent
wandb agent cs25m532-iitm-india/vgg6-cifar10/<sweep_id>

# stop the random sweep after 20 runs
wandb agent --count 20 cs25m532-iitm-india/vgg6-cifar10/<sweep_id>
```

To get all runs as a table: `python export_results.py --path cs25m532-iitm-india/vgg6-cifar10`

The reported results come from the three grid sweeps and about 35 random-search runs on Colab. Colab disconnected a few times and some runs crashed. The sweeps did not repeat those configurations, so the crashed runs were deleted and 7 configurations were run by hand with `train.py`, using the same arguments and seed. These runs have no sweep ID in W&B. The plots use all runs in the project, so this does not affect them.

## Best model

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

Result: validation accuracy 86.12% (epoch 30), test accuracy 85.13%. The re-run on a Colab T4 gave the same numbers.

## Results

All numbers are the best validation accuracy (%) of one run with seed 42. Runs were not repeated with other seeds, so differences of about 1 point may be noise. Two identical runs (relu, nesterov, lr 0.05, batch size 128, 20 epochs) gave exactly the same result (83.54 val, 82.61 test).

### Activation

Nesterov SGD, lr 0.05, batch size 128, 20 epochs.

| Activation | Val acc | Test acc |
|---|---|---|
| ReLU | 83.54 | 82.61 |
| GELU | 83.04 | 82.54 |
| SiLU | 81.60 | 80.76 |
| Tanh | 72.68 | 71.78 |
| Sigmoid | 61.74 | 61.16 |

Sigmoid and Tanh are the worst. Both flatten out for large inputs, so gradients become small as they pass back through the layers. Sigmoid is also not centred at zero and its slope is at most 0.25, so it is the slowest. Tanh is centred and its slope goes up to 1, so it does better, but it is still about 10 points below ReLU after 20 epochs. BatchNorm helps both, but not enough to close the gap.

ReLU, GELU and SiLU are within about 2 points of each other. ReLU was best in this short run. GELU and SiLU did not show a gain, but the gaps are small and come from a single seed.

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

- SGD and Nesterov SGD need a large learning rate. They reach about 83% at lr 0.1 but only about 55% at 0.001, because 15 epochs is too short for such small steps. Nesterov is about 0.4 points ahead of plain momentum at 0.1 and 0.01, which is within noise.
- Adam and Nadam work the other way. They scale each parameter's step using running gradient statistics, so they reach about 80% at lr 0.001. At lr 0.1 they break down (around 35%). Nadam is Adam with Nesterov momentum and behaves very similarly. It handled lr 0.01 better (79.80 against 77.02).
- Adagrad adds up squared gradients, so its step size keeps shrinking. It needs a large starting rate (best at 0.1) and is slow at 0.001.
- RMSprop was good only at lr 0.001 and got much worse at larger rates. It also ran with momentum 0.9 here, which probably makes it less stable.
- The best result in the table is Nesterov SGD at lr 0.1. Adaptive optimizers need a much smaller learning rate to work at all, so matching the learning rate to the optimizer matters a lot.

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

- More epochs helped in every case. Accuracy was still rising at 30 epochs, so the models are undertrained, not overfitting.
- lr 0.05 beat lr 0.01 by 4 to 11 points. With only 10 to 30 epochs, 0.01 is too slow.
- For the same number of epochs, smaller batches did better. At 30 epochs and lr 0.05 the results are 86.12, 85.96 and 84.02 for batch sizes 64, 128 and 256. A smaller batch gives more weight updates per epoch and noisier gradients. A larger batch probably needs a higher learning rate to catch up, but this was not tested.
