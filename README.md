# ⚙️ Latent Transportable Interventions (LTI) Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Preprint](https://img.shields.io/badge/Status-Preprint-blue.svg)]()

> **One-Shot Causal Direction Discovery:** A purely mathematical, training-data-free causal engine grounded in thermodynamic topology and Phase-Space Hysteresis.

## 📄 Research Paper
**[Read the official preprint: Latent Transportable Interventions (LTI)](./LTI_Research_Paper.pdf)**

Determining causal direction from a single, passive bivariate time-series is a notoriously hard problem in causal inference (constrained by the Markov Equivalence problem). This repository contains the official implementation of the **LTI Engine**, which overcomes this barrier without requiring large sample sets, interventions, or neural network training.

## 🔬 Mathematical Core

The LTI framework operates through a robust, two-layer deterministic pipeline:
1. **Thermodynamic Topology (Bond Graphs):** Classifies raw signals into Effort (E), Flow (F), or Quantity (Q) roles via derivative sparsity and zero-crossing morphology.
2. **Phase-Space Hysteresis (Shoelace Formula):** When linear derivative correlations tie, LTI invokes a second-order virtual intervention: calculating the directed area of the closed loop in phase-space to exploit physical causal inertia. 

## 🚀 Getting Started

### Prerequisites
The engine is computationally lightweight ($O(T)$ complexity) and designed for Edge AI deployment.
```bash
pip install numpy scipy pandas statsmodels
