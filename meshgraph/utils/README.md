## Utils Module

Core utilities for MeshGraphNet training, evaluation, and visualization.

`plot.py` contains:

- **Training visualization**:

  - `plot_training_loss(cfg)` - Training/test loss curves over epochs
  - `plot_rollout_error(cfg)` - Velocity RMSE during rollout prediction

- **Animation system**:

  - `create_velocity_animation(cfg, ...)` - Single animation with full parameter control
  - `create_standard_animations(cfg)` - Standard 4-animation suite (0/50 steps, train/test)
  - Animation types include:
    - **Single-step**: Ground truth inputs (no error accumulation)
    - **Rollout**: Previous predictions as inputs (shows error accumulation)
    - **Multi-panel**: Ground truth, prediction, and error side-by-side

- **Advanced Analysis**
  - `compare_noise_effects(base_config, ...)` - Noise impact on rollout stability
  - `_load_predictions_for_plotting(cfg, ...)` - Core data loading for visualization

Basic Usage:

```python
# Basic plots
from utils.plot import plot_training_loss, plot_rollout_error
plot_training_loss(cfg)
plot_rollout_error(cfg, start_step=50, num_steps=100)

# Animations
from utils.plot import create_velocity_animation
create_velocity_animation(cfg, start_step=0, num_steps=500, single_step=True)

# Noise comparison
from utils.plot import compare_noise_effects
compare_noise_effects(["+datasize=medium"], start_step=50, num_steps=100)
```

---

`process.py` contains:

- `normalize(tensor, mean, std)` / `unnormalize(tensor, mean, std)` - Z-score normalization
- `compute_dataset_statistics(data_list)` - Per-feature mean/std computation
- `load_dataset_split(dataset_path, ...)` - Train/test splitting with configurable strategies
- `get_dataset_info(data_list)` - Metadata extraction and validation

Basic Usage:

```python
from utils.process import compute_dataset_statistics, normalize

# Compute normalization statistics
stats = compute_dataset_statistics(dataset)
mean_x, std_x, mean_edge, std_edge, mean_y, std_y = stats

# Normalize features
normalized_features = normalize(raw_features, mean_x, std_x)
```

---

`evaluate.py` contains:

- `noise_comparison_analysis(cfg)` - Compare with/without training noise
- `rollout_analysis(cfg)` - Multi-window rollout error analysis
- `main()` - CLI dispatcher for different analysis types

**Analysis Types** are:

- **Noise Effects**: Side-by-side comparison of noise regularization impact
- **Rollout Windows**: Early (0-100), mid (50-150), late (100-200) timestep analysis
- **Config Flexibility**: Works with any Hydra configuration combination

Basic Usage:

```bash
# Noise comparison analysis
python scripts/evaluate.py --analysis noise_comparison +datasize=medium

# Full rollout analysis
python scripts/evaluate.py --analysis rollout_analysis +datasize=large

# Both analyses
python scripts/evaluate.py --analysis both +datasize=large +noise=paper
```
