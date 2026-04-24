"""
Self-Pruning Neural Network on CIFAR-10  (Accuracy-Optimised)
==============================================================

Usage:
    pip install torch torchvision matplotlib
    python self_pruning_nn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# PrunableLinear Layer

class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 2.0))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        if not self.training:
            gates = (gates >= 0.5).float() * gates
        return F.linear(x, self.weight * gates, self.bias)

    def get_gates(self) -> torch.Tensor:
        g = torch.sigmoid(self.gate_scores).detach()
        return (g >= 0.5).float() * g

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# Network: CNN frontend + Prunable FC head

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )
    def forward(self, x):
        return self.net(x)


class SelfPruningNet(nn.Module):
    """
    Architecture:
        3x32x32
        ConvBlock(3->64)    -> 64x16x16
        ConvBlock(64->128)  -> 128x8x8
        Flatten             -> 8192
        Dropout + PrunableLinear(8192->512) + BN + ReLU
        Dropout + PrunableLinear(512->256)  + BN + ReLU
        Dropout + PrunableLinear(256->10)
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.conv1   = ConvBlock(3,  64)
        self.conv2   = ConvBlock(64, 128)
        self.flatten = nn.Flatten()

        self.drop1 = nn.Dropout(dropout)
        self.fc1   = PrunableLinear(8192, 512)
        self.bn1   = nn.BatchNorm1d(512)

        self.drop2 = nn.Dropout(dropout)
        self.fc2   = PrunableLinear(512, 256)
        self.bn2   = nn.BatchNorm1d(256)

        self.drop3 = nn.Dropout(dropout * 0.67)
        self.fc3   = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2(self.conv1(x))
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(self.drop1(x))))
        x = F.relu(self.bn2(self.fc2(self.drop2(x))))
        x = self.fc3(self.drop3(x))
        return x

    def prunable_layers(self):
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def gate_parameters(self):
        for layer in self.prunable_layers():
            yield layer.gate_scores

    def non_gate_parameters(self):
        gate_ids = {id(p) for p in self.gate_parameters()}
        for p in self.parameters():
            if id(p) not in gate_ids:
                yield p


# Sparsity loss + lambda schedule

def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    parts = []
    for layer in model.prunable_layers():
        parts.append(torch.sigmoid(layer.gate_scores).flatten())
    return torch.cat(parts).mean()


def get_lambda(epoch: int, target_lam: float,
               warmup_epochs: int, ramp_epochs: int) -> float:
    """
    Warmup-then-ramp schedule:
      epochs 1..warmup            : lam = 0
      epochs warmup+1..warmup+ramp: lam ramps 0 -> target_lam
      epochs after                : lam = target_lam
    """
    if epoch <= warmup_epochs:
        return 0.0
    prog = min(epoch - warmup_epochs, ramp_epochs) / ramp_epochs
    return target_lam * prog


# Data loaders

def get_cifar10_loaders(batch_size: int = 128):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True,
                                            download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(root="./data", train=False,
                                            download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256,
                              shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


# Training & evaluation helpers

def make_optimizers(model: SelfPruningNet, base_lr=1e-3, gate_lr=5e-3,
                    weight_decay=1e-4):
    return optim.Adam([
        {"params": list(model.non_gate_parameters()),
         "lr": base_lr, "weight_decay": weight_decay},
        {"params": list(model.gate_parameters()),
         "lr": gate_lr, "weight_decay": 0.0},
    ])


def train_one_epoch(model, loader, optimizer, device, lam: float,
                    label_smoothing: float = 0.1):
    model.train()
    total_clf, total_sp, n = 0.0, 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits   = model(images)
        clf_loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        sp_loss  = sparsity_loss(model)
        loss     = clf_loss + lam * sp_loss
        loss.backward()
        optimizer.step()
        total_clf += clf_loss.item()
        total_sp  += sp_loss.item()
        n += 1

    return total_clf / n, total_sp / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def compute_sparsity(model, threshold: float = 0.5) -> float:
    all_gates = []
    for layer in model.prunable_layers():
        all_gates.append(torch.sigmoid(layer.gate_scores).cpu().flatten())
    gates = torch.cat(all_gates)
    return 100.0 * (gates < threshold).float().mean().item()


@torch.no_grad()
def collect_all_gates(model) -> np.ndarray:
    parts = []
    for layer in model.prunable_layers():
        parts.append(torch.sigmoid(layer.gate_scores).cpu().numpy().flatten())
    return np.concatenate(parts)


# Main experiment

def run_experiment(lam: float, epochs: int, warmup: int, ramp: int,
                   train_loader, test_loader, device):
    print(f"\n{'='*60}")
    print(f"  lam_target={lam}  warmup={warmup}  ramp={ramp}  epochs={epochs}")
    print(f"{'='*60}")

    model     = SelfPruningNet().to(device)
    optimizer = make_optimizers(model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        cur_lam = get_lambda(epoch, lam, warmup, ramp)
        clf_l, sp_l = train_one_epoch(model, train_loader, optimizer, device, cur_lam)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate(model, test_loader, device)
            sp  = compute_sparsity(model)
            phase = ("warmup" if epoch <= warmup
                     else "ramp  " if epoch <= warmup + ramp
                     else "prune ")
            print(f"  [{phase}] Ep {epoch:3d}  lam={cur_lam:.2f}  "
                  f"clf={clf_l:.4f}  |  acc={acc:.2f}%  sparse={sp:.1f}%")

    final_acc   = evaluate(model, test_loader, device)
    final_sp    = compute_sparsity(model)
    final_gates = collect_all_gates(model)
    print(f"\n  Final  acc={final_acc:.2f}%   sparsity={final_sp:.2f}%")
    return final_acc, final_sp, final_gates, model


def plot_gate_distributions(lambdas, gates_list, sparsity_list,
                             accs_list, save_path="gate_distributions.png"):
    n = len(lambdas)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for ax, lam, gates, sp, acc, color in zip(axes, lambdas, gates_list,
                                               sparsity_list, accs_list, colors):
        ax.hist(gates, bins=80, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_title(f"lam = {lam}\nAcc={acc:.1f}%  Sparse={sp:.1f}%",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Gate value", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1.2, label="threshold=0.5")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Gate Distributions - CNN + Prunable FC",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {save_path}")
    plt.show()


def print_results_table(lambdas, accs, sparsities):
    print("\n\n" + "-" * 50)
    print("  Results Summary")
    print("-" * 50)
    print(f"  {'Lambda':<12} {'Test Acc (%)':>14} {'Sparsity (%)':>14}")
    print("-" * 50)
    for lam, acc, sp in zip(lambdas, accs, sparsities):
        print(f"  {lam:<12} {acc:>14.2f} {sp:>14.2f}")
    print("-" * 50)


# Entry point

if __name__ == "__main__":
    # Config
    EPOCHS        = 50          # increase to 60-70 for best results
    WARMUP_EPOCHS = 10          # pure classification warmup
    RAMP_EPOCHS   = 10          # lam ramps 0 -> target over these epochs
    BATCH_SIZE    = 128
    LAMBDAS       = [1.0, 5.0, 15.0]
    DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {DEVICE}")
    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)

    accs, sparsities, gates_list = [], [], []
    best_model, best_acc = None, -1.0

    for lam in LAMBDAS:
        acc, sp, gates, model = run_experiment(
            lam, EPOCHS, WARMUP_EPOCHS, RAMP_EPOCHS,
            train_loader, test_loader, DEVICE
        )
        accs.append(acc)
        sparsities.append(sp)
        gates_list.append(gates)
        if acc > best_acc:
            best_acc   = acc
            best_model = model

    print_results_table(LAMBDAS, accs, sparsities)

    best_idx = int(np.argmax(accs))
    print(f"\nBest model: lam={LAMBDAS[best_idx]}  "
          f"acc={accs[best_idx]:.2f}%  sparsity={sparsities[best_idx]:.2f}%")

    plot_gate_distributions(LAMBDAS, gates_list, sparsities, accs)

    print("\nPer-layer gate statistics (best model):")
    print(f"  {'Layer':<10} {'Total gates':>12} {'Pruned (<0.5)':>15} {'Sparsity %':>12}")
    print("  " + "-" * 52)
    for name, layer in best_model.named_modules():
        if isinstance(layer, PrunableLinear):
            g   = torch.sigmoid(layer.gate_scores).detach().cpu().flatten()
            pr  = (g < 0.5).sum().item()
            tot = g.numel()
            print(f"  {name:<10} {tot:>12} {pr:>15} {100*pr/tot:>11.1f}%")
