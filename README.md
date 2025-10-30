# Uncertainty Quantification Using Conformal Prediction for Mesh-Based Simulations

Uncertainty quantification for mesh-based simulations using the Split Conformal Prediction framework, with spatially adaptive extensions and MeshGraphNet surrogates.

## MeshGraphNet Surrogate Training

The surrogate model training in `meshgraph/` is pretty standard MeshGraphNet implementation. For reference:
- Original DeepMind: [meshgraphnets](https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/README.md)
- NVIDIA's tutorial: [PhysicsNeMo MeshGraphNet](https://docs.nvidia.com/physicsnemo/latest/user-guide/model_architecture/meshgraphnet.html)

Our main research contribution is the **conformal prediction** framework built atop the meshgraphnet's predictions inside the `conformal/` module.

## Conformal Prediction

The `conformal/` module implements our uncertainty quantification framework. 

**Quick Start:**
```bash
# Compare all geometries (ℓ2, ℓ∞, Mahalanobis) at α=0.1
python cli.py cylinder_medium_noise -a 0.1 -c

# Alpha sweep across confidence levels
python cli.py cylinder_medium_noise -s 0.05,0.1,0.15,0.2 -c

# Spatially adaptive CP with basic features (p=5)
python cli.py cylinder_medium_noise -s 0.05,0.1,0.15,0.2 -ad -x 0.2

# Spatially adaptive CP with full features (p=17, Table 3)
python cli.py cylinder_medium_noise -s 0.05,0.1,0.15,0.2 -ad -x 0.2 -f

# Flag dataset (3D)
python cli.py flag_medium -a 0.1 -c
```

Results saved to `results/conformal_preds/`. See [`conformal/README.md`](conformal/README.md) for details.

