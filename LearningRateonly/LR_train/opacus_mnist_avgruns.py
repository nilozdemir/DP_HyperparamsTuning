"""
opacus_mnist.py
---------------
Final DP-SGD training on the FULL MNIST dataset for every epsilon target,
using the best hyperparameters found by tuning_script.py.

Each (epsilon, sigma) point is trained N_RUNS=10 independent times and the
test accuracies are averaged (mean ± std), matching the evaluation protocol
of Figure 2 in Koskela & Kulkarni (2024).

Reads   : tuned_weights_eps{eps}.pth  and  tuned_config_eps{eps}.pth
Writes  : final_results.csv  — one row per epsilon with per-run accuracies,
          mean, and std (columns: epsilon, sigma, run_1 … run_10, mean_acc, std_acc)

Key fixes vs. original
  1. Weights loaded BEFORE make_private() — avoids GradSampleModule key mismatch.
  2. LR scaled from tuning-subset size to full dataset: η_final = η* × (N/m).
  3. Batch size from full dataset: int(GAMMA × N_FULL).
  4. noise_multiplier = sigma (plain, no /C when C=1).
  5. No momentum — plain DP-SGD (Equation 2.3).
  6. Tempered sigmoid activations (same arch as tuning_script.py).
  7. Each epsilon point averaged over 10 independent runs.
"""

import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# (epsilon → sigma) pairs — must match tuning_script.py
# ──────────────────────────────────────────────────────────────────────────────
EPS_SIGMA_PAIRS = [
    (0.450, 4.890169),
    (0.600, 2.593154),
    (0.800, 1.886150),
    (1.000, 1.601185),
    (1.200, 1.435335),
    (1.400, 1.312616),
    (1.600, 1.225438),
    (1.800, 1.155948),
    (2.000, 1.096290),
    (2.200, 1.047493),
    (2.400, 1.008691),
    (2.600, 0.979371),
    (2.800, 0.942858),
    (3.000, 0.911557),
]

# ──────────────────────────────────────────────────────────────────────────────
# Fixed hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
EPOCHS        = 40
GAMMA         = 0.0213
DELTA         = 1e-5
MAX_GRAD_NORM = 1.0
N_RUNS        = 10              # independent runs to average per epsilon point

OUTPUT_CSV = "final_results.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Tempered Sigmoid — identical to tuning_script.py
# ──────────────────────────────────────────────────────────────────────────────
class TemperedSigmoid(nn.Module):
    def __init__(self, t: float = 1.0):
        super().__init__()
        self.t = t

    def forward(self, x):
        return 2.0 * torch.sigmoid(x / self.t) - 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Model — identical to tuning_script.py
# ──────────────────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), TemperedSigmoid(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), TemperedSigmoid(), nn.MaxPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Linear(64 * 7 * 7, 128), TemperedSigmoid(),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.fc(self.conv(x).flatten(1))
        
def build_model() -> nn.Module:
    
    m = SimpleCNN().to(device)
    m = ModuleValidator.fix(m)
    ModuleValidator.validate(m, strict=True)
    return m


# ──────────────────────────────────────────────────────────────────────────────
# Data (full training set)
# ──────────────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

N_FULL     = len(train_dataset)               # 60 000
BATCH_SIZE = max(1, int(GAMMA * N_FULL))      # ~1278  (full-dataset batch)

test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

print(f"Full dataset N={N_FULL}  |  Final batch size={BATCH_SIZE}")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def eval_accuracy(model: nn.Module) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / total


def train_final(target_eps: float, sigma: float) -> float:
    """
    Load tuned weights + config, scale LR, run final DP-SGD on full dataset.
    Returns final test accuracy.
    """
    weights_path = f"tuned_weights_eps{target_eps}.pth"
    config_path  = f"tuned_config_eps{target_eps}.pth"

    if not os.path.exists(weights_path) or not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing {weights_path} or {config_path}. "
            "Run tuning_script.py first."
        )

    config  = torch.load(config_path, map_location=device)
    lr_tune = config["lr"]
    N_tune  = config["N_tune"]

    # Section 3.2: scale LR linearly with dataset size
    lr_final = lr_tune * (N_FULL / N_tune)
    print(f"  Tuned LR={lr_tune:.4e} on subset N={N_tune}")
    print(f"  Scaled LR={lr_final:.4e} for full N={N_FULL}")

    # ── Build model and load weights BEFORE make_private ────────────────────
    model = build_model()
    state_dict = torch.load(weights_path, map_location=device)
    # Strip the _module. prefix added by Opacus GradSampleModule
    state_dict = {
        k.replace("_module.", ""): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)

    # Plain SGD, no momentum
    optimizer = optim.SGD(model.parameters(), lr=lr_final)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,        # σ/C with C=MAX_GRAD_NORM=1
        max_grad_norm=MAX_GRAD_NORM,
    )

    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(EPOCHS):
        total_loss = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (out.argmax(1) == y).sum().item()
            total      += y.size(0)

        eps_spent = privacy_engine.get_epsilon(DELTA)
        print(
            f"  Epoch {epoch+1:>2}/{EPOCHS}  "
            f"loss={total_loss:.3f}  "
            f"train_acc={100*correct/total:.2f}%  "
            f"ε={eps_spent:.4f}"
        )

    return eval_accuracy(model)


# ──────────────────────────────────────────────────────────────────────────────
# Main — iterate over all epsilon targets, N_RUNS independent runs each
# ──────────────────────────────────────────────────────────────────────────────
def run_all():
    # CSV header: epsilon, sigma, run_1, run_2, ..., run_N, mean_acc, std_acc
    run_cols     = [f"run_{i+1}" for i in range(N_RUNS)]
    header       = ["epsilon", "sigma"] + run_cols + ["mean_acc", "std_acc"]
    write_header = not os.path.exists(OUTPUT_CSV)

    csv_file = open(OUTPUT_CSV, "a", newline="")
    writer   = csv.writer(csv_file)
    if write_header:
        writer.writerow(header)

    for target_eps, sigma in EPS_SIGMA_PAIRS:
        print(f"\n{'='*60}")
        print(f"ε={target_eps:.3f}   σ={sigma:.6f}   ({N_RUNS} independent runs)")
        print(f"{'='*60}")

        run_accs = []

        for run_idx in range(N_RUNS):
            print(f"\n  ── Run {run_idx+1}/{N_RUNS} ──────────────────────────────")
            acc = train_final(target_eps, sigma)
            run_accs.append(acc)
            print(f"  Run {run_idx+1} test accuracy: {100*acc:.2f}%")

        mean_acc = float(np.mean(run_accs))
        std_acc  = float(np.std(run_accs, ddof=1))   # sample std, matching the paper

        print(f"\n  ✓ ε={target_eps:.3f}  mean={100*mean_acc:.2f}%  "
              f"std={100*std_acc:.2f}%  "
              f"(runs: {[f'{100*a:.2f}' for a in run_accs]})")

        row = (
            [target_eps, sigma]
            + [round(a, 6) for a in run_accs]
            + [round(mean_acc, 6), round(std_acc, 6)]
        )
        writer.writerow(row)
        csv_file.flush()

    csv_file.close()
    print(f"\nAll done. Results → {OUTPUT_CSV}")


if __name__ == "__main__":
    run_all()
