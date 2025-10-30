# uq-for-meshsimulations-using-splitcp

Uncertainty quantification for mesh-based simulations using the Split Conformal Prediction framework, with spatially adaptive extensions and MeshGraphNet surrogates.

## MeshGraphNet Surrogate Model

The `meshgraph/` module contains a PyTorch re-implementation of [Learning Mesh-Based Simulation with Graph Networks](https://sites.google.com/view/meshgraphnets) for training GNN-based surrogate models on mesh simulations.

**Quick Start:**
```bash
cd meshgraph/

# Train on cylinder_flow dataset
python run_gnn.py +datasize=medium +noise=paper

# Train on flag_simple (3D) dataset
python run_gnn.py +dataset=flag_simple
```

**Reference:** Adapted from [gnn-physics](https://github.com/BurgerAndreas/gnn-physics). See [`meshgraph/README.md`](meshgraph/README.md) and [`meshgraph/results.md`](meshgraph/results.md) for details.
