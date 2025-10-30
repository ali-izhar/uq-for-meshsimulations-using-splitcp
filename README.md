# Uncertainty Quantification Using Conformal Prediction for Mesh-Based Simulations

Uncertainty quantification for mesh-based simulations using the Split Conformal Prediction framework, with spatially adaptive extensions and MeshGraphNet surrogates.

## MeshGraphNet Surrogate Training

The surrogate model training in `meshgraph/` is pretty standard MeshGraphNet implementation. For reference:
- Original DeepMind: [meshgraphnets](https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/README.md)
- NVIDIA's tutorial: [PhysicsNeMo MeshGraphNet](https://docs.nvidia.com/physicsnemo/latest/user-guide/model_architecture/meshgraphnet.html)

Our main research contribution is the **conformal prediction** framework built atop the meshgraphnet's predictions inside the `conformal/` module.

## Conformal Prediction

