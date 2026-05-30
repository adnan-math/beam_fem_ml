# Parametric FEM + Neural Network Surrogate Model

## Overview

This project simulates a 3D clamped beam using the Finite Element Method (FEM) and trains a neural network to approximate the vertical displacement along the beam centerline as a function of beam length and spatial position.

The goal is to learn the mapping:

(L, x) → u_z(x, 0, 0)

where:
- L is the beam length
- x is the position along the beam
- u_z is the vertical displacement

---

## Physics Model

- 3D linear elasticity
- Beam dimensions:
  - Length: L ∈ [0.5, 1.5] m
  - Width: 0.2 m
  - Height: 0.2 m
- Material:
  - Young’s modulus: 30 GPa
  - Poisson ratio: 0.2
- Boundary conditions:
  - Left face (x = 0): fully clamped (u = 0)
- Loading:
  - Uniform traction on top surface: 1000 N/m² (downward)

---

## Methodology

### 1. FEM Simulation
- Implemented using FEniCSx (DOLFINx)
- Solves linear elasticity equations
- Extracts displacement along beam centerline (y = 0, z = 0)

### 2. Dataset Generation
Each sample contains:
- Input: (L, x)
- Output: u_z(x, 0, 0)

Saved as: beam_fem_dataset.npz


### 3. Neural Network
- Fully connected MLP
- Input: (L, x)
- Output: displacement u_z
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam

---


