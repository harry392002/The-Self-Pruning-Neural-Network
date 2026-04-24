"""
Self-Pruning Neural Network on CIFAR-10
========================================
Implements a feed-forward network with learnable gate parameters that
encourage weight sparsity via an L1 regularization penalty during training.

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

# ─────────────────────────────────────────────
# Part 1: PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that associates each weight with a
    learnable scalar gate in [0, 1] via the Sigmoid function.

    Forward pass:
        gates        = sigmoid(gate_scores)          # shape: (out, in)
        pruned_w     = weight * gates                # element-wise
        output       = pruned_w @ x.T + bias
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight and bias (same init as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores – same shape as weight.
        # Initialised near 0 so initial gates ≈ sigmoid(0) = 0.5 (all half-open).
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming uniform init for weights (matches nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: map gate_scores → [0, 1]
        gates = torch.sigmoid(self.gate_scores)          # (out, in)

        # Step 2: mask the weights – gradients flow through BOTH weight and gate_scores
        pruned_weights = self.weight * gates             # (out, in)

        # Step 3: standard affine transform
        return F.linear(x, pruned_weights, self.bias)   # (batch, out)

    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (detached from the graph)."""
        return torch.sigmoid(self.gate_scores).detach()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ─────────────────────────────────────────────
# Network definition
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 (32×32×3 = 3072 inputs, 10 outputs).
    All linear layers are PrunableLinear so every weight can be gated.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512,  256)
        self.fc3 = PrunableLinear(256,  128)
        self.fc4 = PrunableLinear(128,  10)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

    def prunable_layers(self):
        """Yield every PrunableLinear in the network."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module


# ─────────────────────────────────────────────
# Part 2: Sparsity Loss
# ─────────────────────────────────────────────

def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """
    NORMALISED L1 norm of all gate values (mean, not sum).

    Dividing by the total gate count keeps the value in (0, 1) regardless
    of network size, making lambda scale-independent and preventing the
    sparsity term from overwhelming the classifier loss (~1-2).
    """
    gate_list = []
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)   # keep in graph!
        gate_list.append(gates.flatten())
    all_gates = torch.cat(gate_list)
    return all_gates.mean()   # always in (0, 1)


# ─────────────────────────────────────────────
# Part 3: Training & Evaluation helpers
# ─────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128):
    """Return (train_loader, test_loader) for CIFAR-10."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
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


def train_one_epoch(model, loader, optimizer, device, lam: float):
    """Run one epoch; return (avg_clf_loss, avg_sparsity_loss)."""
    model.train()
    total_clf, total_sp, n_batches = 0.0, 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        clf_loss  = F.cross_entropy(logits, labels)
        sp_loss   = sparsity_loss(model)
        loss      = clf_loss + lam * sp_loss

        loss.backward()
        optimizer.step()

        total_clf += clf_loss.item()
        total_sp  += sp_loss.item()
        n_batches += 1

    return total_clf / n_batches, total_sp / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    """Return test accuracy (0-100)."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def compute_sparsity(model, threshold: float = 1e-2) -> float:
    """
    Percentage of weights whose gate value < threshold.
    A gate < 0.01 means the weight contributes < 1 % of its value to output.
    """
    all_gates = []
    for layer in model.prunable_layers():
        all_gates.append(torch.sigmoid(layer.gate_scores).cpu().flatten())
    gates = torch.cat(all_gates)
    pruned = (gates < threshold).float().mean().item()
    return 100.0 * pruned


@torch.no_grad()
def collect_all_gates(model) -> np.ndarray:
    """Concatenate all gate values into a numpy array for plotting."""
    parts = []
    for layer in model.prunable_layers():
        parts.append(torch.sigmoid(layer.gate_scores).cpu().numpy().flatten())
    return np.concatenate(parts)


# ─────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────

def run_experiment(lam: float, epochs: int, train_loader, test_loader, device):
    """Train a fresh model with a given lambda; return (accuracy, sparsity, gates)."""
    print(f"\n{'='*55}")
    print(f"  λ = {lam}   |   epochs = {epochs}")
    print(f"{'='*55}")

    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        clf_l, sp_l = train_one_epoch(model, train_loader, optimizer, device, lam)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate(model, test_loader, device)
            sp  = compute_sparsity(model)
            print(f"  Epoch {epoch:3d}  |  clf={clf_l:.4f}  sp={sp_l:.1f}"
                  f"  |  acc={acc:.2f}%  sparse={sp:.1f}%")

    final_acc  = evaluate(model, test_loader, device)
    final_sp   = compute_sparsity(model)
    final_gates = collect_all_gates(model)
    print(f"\n  ✓ Final  acc={final_acc:.2f}%   sparsity={final_sp:.2f}%")
    return final_acc, final_sp, final_gates, model


def plot_gate_distributions(lambdas, gates_list, sparsity_list, save_path="gate_distributions.png"):
    """Plot gate-value histograms for each lambda in a grid."""
    n = len(lambdas)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for ax, lam, gates, sp, color in zip(axes, lambdas, gates_list, sparsity_list, colors):
        ax.hist(gates, bins=80, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_title(f"λ = {lam}\nSparsity = {sp:.1f}%", fontsize=12, fontweight="bold")
        ax.set_xlabel("Gate value", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.2, label="threshold=0.01")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Distribution of Gate Values for Different λ", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {save_path}")
    plt.show()


def print_results_table(lambdas, accs, sparsities):
    """Pretty-print a Markdown-style results table."""
    print("\n\n" + "─" * 50)
    print("  Results Summary")
    print("─" * 50)
    print(f"  {'Lambda':<12} {'Test Acc (%)':>14} {'Sparsity (%)':>14}")
    print("─" * 50)
    for lam, acc, sp in zip(lambdas, accs, sparsities):
        print(f"  {lam:<12} {acc:>14.2f} {sp:>14.2f}")
    print("─" * 50)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Config ──────────────────────────────────
    EPOCHS     = 30        # increase to 50-60 for better accuracy
    BATCH_SIZE = 128
    LAMBDAS    = [0.1, 1.0, 5.0]      # low / medium / high sparsity pressure (normalized loss)
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ────────────────────────────────────────────

    print(f"Device: {DEVICE}")
    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)

    accs, sparsities, gates_list = [], [], []
    best_model, best_acc = None, -1.0

    for lam in LAMBDAS:
        acc, sp, gates, model = run_experiment(
            lam, EPOCHS, train_loader, test_loader, DEVICE
        )
        accs.append(acc)
        sparsities.append(sp)
        gates_list.append(gates)
        if acc > best_acc:
            best_acc   = acc
            best_model = model

    print_results_table(LAMBDAS, accs, sparsities)

    # Identify index of best model (highest accuracy)
    best_idx = int(np.argmax(accs))
    print(f"\nBest model: λ={LAMBDAS[best_idx]}  "
          f"acc={accs[best_idx]:.2f}%  sparsity={sparsities[best_idx]:.2f}%")

    plot_gate_distributions(LAMBDAS, gates_list, sparsities)

    # ── Optional: print per-layer sparsity for best model ──
    print("\nPer-layer gate statistics (best model):")
    print(f"  {'Layer':<10} {'Total gates':>12} {'Pruned (<0.01)':>16} {'Sparsity %':>12}")
    print("  " + "-" * 52)
    for name, layer in best_model.named_modules():
        if isinstance(layer, PrunableLinear):
            g   = layer.get_gates().flatten()
            pr  = (g < 0.01).sum().item()
            tot = g.numel()
            print(f"  {name:<10} {tot:>12} {pr:>16} {100*pr/tot:>11.1f}%")
