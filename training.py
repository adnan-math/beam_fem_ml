import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

# ============================================================
# LOAD DATA
# ============================================================

data = np.load("beam_fem_dataset.npz")

X = data["X"]   # [L, x]
y = data["y"]   # uz

print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================================================
# TRAIN / TEST SPLIT (by beam length)
# ============================================================

L_values = np.unique(X[:, 0])

# =========================
# Define regions
# =========================
L_train = L_values[(L_values >= 0.7) & (L_values <= 1.3)]

L_extrap_left  = L_values[L_values < 0.7]
L_extrap_right = L_values[L_values > 1.3]

# =========================
# Masks
# =========================
train_mask = np.isin(X[:, 0], L_train)

test_interp_mask = np.isin(X[:, 0], L_train)   # interpolation region (holdout if needed)
test_extrap_mask = np.isin(X[:, 0], np.concatenate([L_extrap_left, L_extrap_right]))

# =========================
# Split data
# =========================
X_train, y_train = X[train_mask], y[train_mask]
X_test_interp, y_test_interp = X[test_interp_mask], y[test_interp_mask]
X_test_extrap, y_test_extrap = X[test_extrap_mask], y[test_extrap_mask]

# =========================
# Print summary
# =========================
print("Train samples:", len(X_train))
print("Interpolation test samples:", len(X_test_interp))
print("Extrapolation test samples:", len(X_test_extrap))

# ============================================================
# NORMALIZATION (fit on train only)
# ============================================================

X_mean = X_train.mean(axis=0)
X_std  = X_train.std(axis=0)

y_mean = y_train.mean()
y_std  = y_train.std()

# Training data
X_train = (X_train - X_mean) / X_std
y_train = (y_train - y_mean) / y_std

# Interpolation test set
X_test_interp = (X_test_interp - X_mean) / X_std
y_test_interp = (y_test_interp - y_mean) / y_std

# Extrapolation test set
X_test_extrap = (X_test_extrap - X_mean) / X_std
y_test_extrap = (y_test_extrap - y_mean) / y_std

# Save normalization stats
np.savez(
    "norm_stats.npz",
    X_mean=X_mean,
    X_std=X_std,
    y_mean=y_mean,
    y_std=y_std,
)

# ============================================================
# TORCH DATA
# ============================================================

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_interp_t = torch.tensor(X_test_interp, dtype=torch.float32)
y_test_interp_t = torch.tensor(y_test_interp, dtype=torch.float32).view(-1, 1)

X_test_extrap_t = torch.tensor(X_test_extrap, dtype=torch.float32)
y_test_extrap_t = torch.tensor(y_test_extrap, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

# ============================================================
# MODEL
# ============================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

model = MLP()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ============================================================
# TRAINING LOOP
# ============================================================

epochs = 1000

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for xb, yb in train_loader:

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss = {total_loss / len(train_loader):.6e}")

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(model.state_dict(), "beam_mlp_model.pt")

print("Model saved: beam_mlp_model.pt")