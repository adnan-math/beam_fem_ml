import numpy as np
import pandas as pd
from fem_dataset import BeamDataset   

# ============================================================
# CREATE DATASET
# ============================================================

dataset = BeamDataset()

# ============================================================
# PARAMETERS
# ============================================================

L_train = np.linspace(0.7, 1.3, 13)
n_points = 100

all_data = []

# ============================================================
# FEM DATA GENERATION LOOP
# ============================================================

for L in L_train:
    print(f"Running FEM for L = {L:.2f}")

    data_L = dataset.generate(L, n_points=n_points)
    all_data.append(data_L)

# Stack everything
all_data = np.vstack(all_data)

print("Final dataset shape:", all_data.shape)

# ============================================================
# SAVE CSV (optional)
# ============================================================

df = pd.DataFrame(all_data, columns=["L", "x", "uz"])
df.to_csv("beam_fem_dataset.csv", index=False)

print("Saved: beam_fem_dataset.csv")

# ============================================================
# SAVE NPZ (for ML)
# ============================================================

X = all_data[:, :2]
y = all_data[:, 2]

np.savez("beam_fem_dataset.npz", X=X, y=y)

print("Saved: beam_fem_dataset.npz")
