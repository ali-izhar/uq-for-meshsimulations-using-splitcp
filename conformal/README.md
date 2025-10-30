## Conformal Prediction Module

**Uncertainty quantification for mesh-based simulations with finite-sample coverage guarantees.**

This module implements the conformal prediction framework from:
*"Uncertainty Quantification Using Conformal Prediction for Mesh-Based Simulations"*
by Mabtoul, Ali, and Ho (2025).

Works post-hoc on MeshGraphNet predictions without model retraining.

### Core Functionality

**`predictor.py`** - Split conformal prediction framework
- **`ConformalPredictor(nonconformity=...)`** - Main predictor class
  - `l2`: disk sets {‖y−ŷ‖₂ ≤ q}
  - `mahalanobis`: ellipse sets {(y−ŷ)ᵀΣ⁻¹(y−ŷ) ≤ q²} 
  - `box`: joint ℓ∞ boxes {max(|y−ŷ|) ≤ q}
- **Calibration**: quantile q at rank ⌈(m+1)(1−α)⌉ for coverage P(Y ∈ C_α(X)) ≥ 1-α
- **Metrics**: coverage, width (2q), area (geometry-dependent)

**`conformity.py`** - Nonconformity score functions
- Score computation for each geometry type
- Covariance estimation for Mahalanobis

**`boosted.py`** - Spatially adaptive conformal prediction  
- Learns heterogeneous scales ŝ(x) via gradient boosting
- Enables adaptive sets: r(x) = q * ŝ(x)

**`features.py`** - Domain-aware feature engineering
- Six categories: kinematic, spatial, topology, gradients, node types, ensemble
- Two modes: basic (p=5) vs full (p=17)

**`run_conformal.py`** - Command-line interface
- Single alpha or alpha sweep analysis
- Geometry comparison mode
- Optional JSON export and visualization

---

## Usage

### **Basic Single Alpha Analysis**

```bash
python run_conformal.py --predictions-file ../meshgraph/outputs/conformal_inputs/datasize_medium_0_500_conformal.pkl \
  --alpha 0.1 --nonconformity l2
```

### **Alpha Sweep with Visualization**

```bash
python run_conformal.py --predictions-file ../meshgraph/outputs/conformal_inputs/datasize_medium_0_500_conformal.pkl \
  --alpha-sweep 0.01,0.05,0.1,0.15,0.2 --nonconformity mahalanobis \
  --plot --output-dir ./results
```

### **Compare Modes (L2, Mahalanobis, Box)**

```bash
python run_conformal.py --predictions-file ../meshgraph/outputs/conformal_inputs/datasize_medium_0_500_conformal.pkl \
  --alpha-sweep 0.01,0.05,0.1,0.15,0.2,0.3 --compare --output-dir ./results
```

### **Custom Calibration Settings**

```bash
python run_conformal.py --predictions-file ../meshgraph/outputs/conformal_inputs/datasize_medium_0_500_conformal.pkl \
  --alpha 0.1 --calib-ratio 0.7 --nonconformity box --no-fluid-mask --save-results
```

### Input Data Format

Expects pickle files from MeshGraphNet with structure:
```python
{
    'predictions': np.ndarray,      # [T, N, 2|3] - predicted accelerations
    'ground_truth': np.ndarray,     # [T, N, 2|3] - true accelerations
    'metadata': {
        'fluid_mask': np.ndarray,   # [N] - boolean mask for fluid nodes
        'mesh_positions': np.ndarray, # [N, 2|3] - node coordinates
        'node_types': np.ndarray,   # [N] - node type encoding
        # ... additional mesh topology data
    }
}
```

---

### Prediction-Set Geometries

After calibration yields quantile q, prediction sets are constructed as:

| Geometry      | Set Definition | Width | Area (2D) |
|---------------|----------------|-------|-----------|
| ℓ2 disk       | {‖y−ŷ‖₂ ≤ q}  | 2q    | πq²       |
| Mahalanobis   | {(y−ŷ)ᵀΣ⁻¹(y−ŷ) ≤ q²} | 2q√λᵢ | πq²√\|Σ\| |
| Joint ℓ∞ box  | {max(\|y−ŷ\|) ≤ q} | 2q | (2q)² |
| Adaptive      | {‖y−ŷ‖₂ ≤ q·ŝ(x)} | 2q·ŝ(x) | varies |

**Reference:** See paper.tex §4 (Prediction-Set Geometries) for mathematical details.

## Workflow

```bash
# 1. Extract predictions from trained MeshGraphNet
cd meshgraph/
python run_gnn.py conformal +datasize=medium --start-step 0 --num-steps 500

# 2. Run conformal prediction analysis
cd ../conformal/
python run_conformal.py -f ../meshgraph/outputs/conformal_inputs/datasize_medium_0_500_conformal.pkl \
  -s 0.05,0.1,0.15,0.2 --compare -p

# 3. Results saved to conformal_results/
```

## Advanced Usage

### **Custom Analysis**

```python
from core import ConformalPredictor, load_predictions_from_file

# Load data
data = load_predictions_from_file('predictions.pkl')

# Create predictor (choose: 'l2', 'mahalanobis', or 'box')
predictor = ConformalPredictor(nonconformity='mahalanobis')

# Calibrate
q = predictor.calibrate(data['predictions'][:250], data['ground_truth'][:250], alpha=0.1)

# Evaluate coverage
coverage = predictor.evaluate_coverage(data['predictions'][250:], data['ground_truth'][250:])

# Size metrics
sizes = predictor.calculate_interval_widths()  # includes width (2q) and area

# Box-only intervals (componentwise)
if predictor.nonconformity == 'box':
    intervals = predictor.predict_intervals(data['predictions'][250:])
```

### **Batch Processing**

```python
from core import run_alpha_sweep

# Run analysis across multiple confidence levels
results = run_alpha_sweep(
    'predictions.pkl',
    alphas=[0.01, 0.05, 0.1, 0.15, 0.2],
    calib_ratio=0.6,
    use_fluid_mask=True
)
```
