"""
tuning_script.py
----------------
Implements DP hyperparameter tuning (Koskela & Kulkarni 2024) for every
(epsilon, sigma) pair from results_variant2.csv.

For each epsilon target:
  - Draws K ~ Poisson(MU)
  - Randomly samples K LR candidates from LR_GRID
  - Trains each on the Poisson-subsampled tuning set X1
  - Picks the best candidate by test accuracy
  - Saves best weights + config for opacus_mnist.py to consume

Output files
  tuning_results.csv               — one row per epsilon (appended)
  tuned_weights_eps{eps}.pth       — best model state dict
  tuned_config_eps{eps}.pth        — {"lr": best_lr, "N_tune": N_TUNE}
"""

import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# (epsilon → sigma) pairs from results_variant2.csv
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
# Fixed hyperparameters (Table 1 / Table 2 of the paper)
# ──────────────────────────────────────────────────────────────────────────────
EPOCHS        = 40
GAMMA         = 0.0213      # DP-SGD subsampling ratio
DELTA         = 1e-5
MAX_GRAD_NORM = 1.0         # clipping constant C
MU            = 15          # Poisson mean for number of candidate runs
Q_TUNE        = 0.1         # inclusion probability for tuning set X1

# LR grid (Table 2): 10^{-i} for i in {4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0}
LR_GRID = [10 ** (-i) for i in [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]]

OUTPUT_CSV = "tuning_results.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Tempered Sigmoid (Papernot et al. 2020)
# ──────────────────────────────────────────────────────────────────────────────
class TemperedSigmoid(nn.Module):
    def __init__(self, t: float = 1.0):
        super().__init__()
        self.t = t

    def forward(self, x):
        return 2.0 * torch.sigmoid(x / self.t) - 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Model
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
# Data
# ──────────────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

full_train = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

N_FULL = len(full_train)  # 60 000

# Draw the fixed Poisson tuning subset X1 once (same seed every run)
torch.manual_seed(42)
mask        = torch.bernoulli(torch.full((N_FULL,), Q_TUNE)).bool()
tune_idx    = mask.nonzero(as_tuple=False).squeeze().tolist()
tune_subset = Subset(full_train, tune_idx)
N_TUNE      = len(tune_subset)

# Batch size derived from SUBSET size so effective subsampling ratio stays = GAMMA
BATCH_SIZE_TUNE = max(1, int(GAMMA * N_TUNE))

print(f"Full N={N_FULL}  |  Tuning subset N={N_TUNE}  |  Batch={BATCH_SIZE_TUNE}")


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


def train_candidate(lr: float, sigma: float) -> tuple[float, dict]:
    """Train one DP candidate on the tuning subset; return (test_acc, state_dict)."""
    model     = build_model()
    # Plain SGD — no momentum (Equation 2.3 of the paper)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    privacy_engine = PrivacyEngine()
    model, optimizer, priv_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=DataLoader(
            tune_subset,
            batch_size=BATCH_SIZE_TUNE,
            shuffle=True,
            drop_last=False,
        ),
        noise_multiplier=sigma,     # Opacus convention: noise_multiplier = σ/C, C=1
        max_grad_norm=MAX_GRAD_NORM,
    )

    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(EPOCHS):
        for x, y in priv_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

    return eval_accuracy(model), {k: v.cpu() for k, v in model.state_dict().items()}


# ──────────────────────────────────────────────────────────────────────────────
# Main — iterate over all (eps, sigma) targets
# ──────────────────────────────────────────────────────────────────────────────
def run_all():
    write_header = not os.path.exists(OUTPUT_CSV)
    csv_file = open(OUTPUT_CSV, "a", newline="")
    writer   = csv.writer(csv_file)
    if write_header:
        writer.writerow(["epsilon", "sigma", "best_lr", "tuning_test_acc", "K"])

    for target_eps, sigma in EPS_SIGMA_PAIRS:
        print(f"\n{'='*60}")
        print(f"Target ε={target_eps:.3f}   σ={sigma:.6f}")
        print(f"{'='*60}")

        # ── Draw K ~ Poisson(MU) ────────────────────────────────────────────
        K = int(torch.poisson(torch.tensor(float(MU))).item())
        print(f"K = {K} candidate runs")

        if K == 0:
            print("K=0 — saving random init.")
            m = build_model()
            torch.save(m.state_dict(),
                       f"tuned_weights_eps{target_eps}.pth")
            torch.save({"lr": LR_GRID[0], "N_tune": N_TUNE},
                       f"tuned_config_eps{target_eps}.pth")
            writer.writerow([target_eps, sigma, LR_GRID[0], 0.0, K])
            csv_file.flush()
            continue

        # ── Sample K LRs uniformly from the grid ────────────────────────────
        lr_candidates = [
            LR_GRID[torch.randint(len(LR_GRID), (1,)).item()]
            for _ in range(K)
        ]
        print(f"LR candidates: {[f'{lr:.2e}' for lr in lr_candidates]}")

        best_acc = -1.0
        best_sd  = None
        best_lr  = None

        for i, lr in enumerate(lr_candidates):
            print(f"  [{i+1}/{K}] lr={lr:.2e} ...", end=" ", flush=True)
            acc, sd = train_candidate(lr, sigma)
            print(f"acc={100*acc:.2f}%")
            if acc > best_acc:
                best_acc, best_sd, best_lr = acc, sd, lr

        print(f"  ✓ Best: lr={best_lr:.2e}   acc={100*best_acc:.2f}%")

        # ── Persist for opacus_mnist.py ──────────────────────────────────────
        torch.save(best_sd, f"tuned_weights_eps{target_eps}.pth")
        torch.save({"lr": best_lr, "N_tune": N_TUNE},
                   f"tuned_config_eps{target_eps}.pth")

        writer.writerow([target_eps, sigma, best_lr, round(best_acc, 6), K])
        csv_file.flush()

    csv_file.close()
    print(f"\nAll done. Results → {OUTPUT_CSV}")


if __name__ == "__main__":
    run_all()
