# FEM Dataset and Neural Surrogate Model for Beam Deformation

## Overview

This project presents a complete workflow combining high-fidelity Finite Element Method (FEM) simulations with machine learning to construct a fast surrogate model for predicting 3D beam deformation under mechanical loading.

The pipeline includes:
- FEM-based dataset generation using FEniCSx
- Centerline extraction of displacement fields
- Neural network surrogate modeling
- Evaluation in interpolation and extrapolation regimes

---

## Repository Structure


├── fem_dataset.py # FEM simulation and dataset generation class

├── data_generator.py # Dataset generation pipeline

├── training.py # Neural network training script

├── beam_fem_dataset.npz # Generated dataset (input-output pairs)

├── beam_mlp_model.pt # Trained surrogate model

├── norm_stats.npz # Normalization parameters

├── fem_validation.ipynb # FEM verification notebook

├── analysis.ipynb # ML model analysis and results


---

## FEM Simulation (Physics Model)

The FEM solver is implemented using FEniCSx and models a 3D elastic beam under traction loading.

### Key features:
- Linear elasticity formulation
- Hexahedral mesh with geometry-aware refinement
- Clamped boundary condition at x = 0
- Traction applied on the beam surface
- Centerline displacement extraction

### Governing model:
- Stress-strain relation: linear elasticity
- Solved using PETSc LU direct solver (MUMPS)

---

## Dataset Generation

Run `data_generator.py` to generate the dataset:


python data_generator.py


## Notebooks

### FEM Validation
📌 Verifies correctness of FEM simulation results  
👉 https://colab.research.google.com/github/adnan-math/beam_fem_ml/blob/main/fem_validation.ipynb

### Model Analysis
📌 Evaluates ML surrogate performance and generalization  
👉 https://colab.research.google.com/github/adnan-math/beam_fem_ml/blob/main/analysis.ipynb

---

## Results

The surrogate model:

- Achieves high accuracy in interpolation regime  
- Preserves smooth physical deformation trends  
- Shows expected degradation under extrapolation beyond training range  

---

## Key Contributions

- End-to-end FEM-to-ML pipeline for structural mechanics  
- Geometry-aware dataset generation strategy  
- Compact centerline-based representation of 3D displacement fields  
- Fast surrogate model for engineering design exploration  

---

## Requirements

- Python 3.9+  
- FEniCSx (dolfinx)  
- PETSc / MPI  
- PyTorch  
- NumPy  
- Pandas  
