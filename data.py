"""CIFAR-10 data pipeline. The dataset lives on the device and augmentation is done per batch on the GPU."""
import math

import torch
import torch.nn.functional as F
from torchvision import datasets

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
SPLIT_SEED = 0  # fixed so every run sees the same train/val split
CUTOUT_LEN = 16
PAD = 4


class GPULoader:
    """Yields (images, labels) batches from uint8 tensors already on `device`.

    Train pipeline: RandomCrop(32, pad=4) -> RandomHorizontalFlip -> Normalize -> Cutout(16x16).
    Eval pipeline: Normalize only.
    """

    def __init__(self, images, labels, batch_size, device, augment=False, shuffle=False, generator=None):
        self.images, self.labels = images.to(device), labels.to(device)
        self.batch_size, self.device = batch_size, device
        self.augment, self.shuffle, self.gen = augment, shuffle, generator
        self.mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
        self.rng = torch.arange(32, device=device)

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self):
        n = len(self.labels)
        if self.shuffle:
            order = torch.randperm(n, device=self.device, generator=self.gen)
        else:
            order = torch.arange(n, device=self.device)
        for i in range(0, n, self.batch_size):
            idx = order[i:i + self.batch_size]
            x = self.images[idx].float().div_(255)
            if self.augment:
                x = self._crop_flip(x)
            x = (x - self.mean) / self.std
            if self.augment:
                x = self._cutout(x)
            yield x, self.labels[idx]

    def _randint(self, high, size):
        return torch.randint(0, high, size, device=self.device, generator=self.gen)

    def _crop_flip(self, x):
        b = x.size(0)
        xp = F.pad(x, (PAD, PAD, PAD, PAD))  # zero padding on raw pixels, as torchvision does
        dy = self._randint(2 * PAD + 1, (b, 1))
        dx = self._randint(2 * PAD + 1, (b, 1))
        flip = self._randint(2, (b, 1)).bool()
        cols = dx + torch.where(flip, 31 - self.rng, self.rng)
        rows = dy + self.rng
        batch = torch.arange(b, device=self.device)[:, None, None, None]
        chan = torch.arange(3, device=self.device)[None, :, None, None]
        return xp[batch, chan, rows[:, None, :, None], cols[:, None, None, :]]

    def _cutout(self, x):
        b = x.size(0)
        cy, cx = self._randint(32, (b, 1, 1)), self._randint(32, (b, 1, 1))
        half = CUTOUT_LEN // 2
        yy, xx = self.rng.view(1, 32, 1), self.rng.view(1, 1, 32)
        mask = (yy >= cy - half) & (yy < cy + half) & (xx >= cx - half) & (xx < cx + half)
        return x.masked_fill(mask[:, None], 0.0)


def _to_tensors(dataset):
    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).contiguous()  # N,3,32,32 uint8
    return images, torch.tensor(dataset.targets)


def get_loaders(batch_size, seed, device, val_size=5000, data_dir="./data"):
    """Build train (45k, augmented), val (5k) and test (10k) loaders."""
    train_x, train_y = _to_tensors(datasets.CIFAR10(data_dir, train=True, download=True))
    test_x, test_y = _to_tensors(datasets.CIFAR10(data_dir, train=False, download=True))

    split_gen = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(train_y), generator=split_gen)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    gen = torch.Generator(device=device).manual_seed(seed)  # drives shuffling and augmentation
    train_loader = GPULoader(train_x[train_idx], train_y[train_idx], batch_size, device,
                             augment=True, shuffle=True, generator=gen)
    val_loader = GPULoader(train_x[val_idx], train_y[val_idx], 512, device)
    test_loader = GPULoader(test_x, test_y, 512, device)
    return train_loader, val_loader, test_loader
