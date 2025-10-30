# Dataset Configuration Files

This directory contains dataset-specific configuration overrides for different MeshGraphNet datasets.

## Available Datasets

### `stanford.yaml` - Stanford Preprocessed Cylinder Flow

- **Dataset**: Stanford-prepared cylinder_flow subset
- **Features**: 11D node features (2D velocity + 7 node types + 2D position)
- **Targets**: 2D velocity changes (acceleration)
- **Trajectories**: 1-3 training + 1 test trajectory
- **Use case**: Quick prototyping with pre-processed data

### `flag_simple.yaml` - Flag Dynamics Simulation

- **Dataset**: 3D flag simulation
- **Features**: 12D node features (3D velocity + 9 node types)
- **Targets**: 3D velocity changes (acceleration)
- **Trajectories**: Single trajectory with 300 timesteps
- **Use case**: 3D fluid-structure interaction simulation

## Usage Examples

```bash
# Train on flag dataset
python run_gnn.py +dataset=flag_simple

# Train on Stanford preprocessed data
python run_gnn.py +dataset=stanford +datasize=medium
```

## Dataset Auto-Detection

The system automatically detects dataset type based on feature dimensions:

- **Cylinder Flow** (default, stanford): 11 node features → 2D targets
- **Flag Simple**: 12 node features → 3D targets

This allows seamless switching between datasets without manual configuration of model architecture.
