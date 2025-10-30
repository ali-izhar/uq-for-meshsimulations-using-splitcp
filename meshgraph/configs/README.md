## Configuration System

The project uses [Hydra](https://hydra.cc/docs/intro/) for hierarchical configuration management.

### Base Configuration

**`default.yaml`** - Base settings for all experiments
- Model: 10 layers, 10 hidden dimensions
- Training: 5000 epochs, batch size 16, Adam optimizer (lr=0.001, weight_decay=5e-4)
- Dataset: Single trajectory, cylinder_flow_pyg, no noise

### Config Groups

Config groups are composable modules that override specific settings. Use `+group=name` syntax.

#### **`datasize/`** - Training/Test Split Sizes (Cylinder Flow)

| Config  | Train Size | Test Size | Batch Size | Epochs | Use Case                 |
|---------|------------|-----------|------------|--------|--------------------------|
| `small` | 85         | 15        | 16         | 5000   | Quick prototyping        |
| `medium`| 500        | 90        | 16         | 5000   | Standard experiments     |
| `large` | 5990       | 2995      | 128        | 3000   | Multi-trajectory         |

**Note:** For flag_simple dataset, training parameters are defined in `dataset/flag_simple.yaml` (200/50 split, batch_size=1, epochs=2000). Override individual parameters as needed.

#### **`dataset/`** - Dataset Selection

| Config       | Description                      | Trajectories         | Features | Training Config                    |
|--------------|----------------------------------|----------------------|----------|-------------------------------------|
| `stanford`   | Preprocessed Stanford subset     | 1-3 train, 1 test    | 11D      | Use with datasize configs           |
| `flag_simple`| 3D flag dynamics simulation      | 1 trajectory (300 ts)| 12D      | Built-in: 200/50, batch=1, ep=2000 |

#### **`noise/`** - Training Regularization

| Config | Noise Scale | Target Mixing | Description                      |
|--------|-------------|---------------|----------------------------------|
| `paper`| 0.02        | 1.0           | Noise settings from original paper|

#### **`testset/`** - Train/Test Splitting Strategy

| Config      | Same Trajectory | Description                           |
|-------------|-----------------|---------------------------------------|
| `different` | False           | Train and test on separate trajectories|

### Usage Examples

```bash
# Default: small dataset, no noise
python run_gnn.py

# Medium dataset with paper noise (recommended for research)
python run_gnn.py +datasize=medium +noise=paper

# Large dataset
python run_gnn.py +datasize=large +noise=paper

# Flag dataset (3D simulation) - uses built-in training config
python run_gnn.py +dataset=flag_simple

# Flag dataset with more training data (override defaults)
python run_gnn.py +dataset=flag_simple training.train_size=240 training.test_size=60

# Override individual parameters
python run_gnn.py +datasize=medium training.epochs=1000 model.hidden_dim=128

# Test on different trajectories than training
python run_gnn.py +datasize=medium +testset=different +noise=paper
```

### Configuration Priority

Hydra applies configurations in this order (later overrides earlier):

1. `default.yaml` (base configuration)
2. Config groups (e.g., `+datasize=medium`, `+dataset=flag_simple`)
3. Command-line overrides (e.g., `training.epochs=1000`)

### Adding New Configurations

To add a new configuration variant:

1. Create a new YAML file in the appropriate directory (e.g., `configs/datasize/custom.yaml`)
2. Use `# @package _global_` at the top to merge into global config
3. Override only the parameters you need to change
4. Use it with `+groupname=filename` (without `.yaml` extension)

**Example** `configs/datasize/custom.yaml`:
```yaml
# @package _global_
training:
  train_size: 1000
  test_size: 200
  batch_size: 32
```

Usage: `python run_gnn.py +datasize=custom`
