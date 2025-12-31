# MeshGraphNet Training

Workflow for training MeshGraphNet using [DeepMind's original implementation](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) (TensorFlow 1.x) and performing split sensitivity analysis for downstream conformal prediction.

## Quick Links

- **[RESULTS.md](RESULTS.md)** - Diagnostics and sensitivity analysis
- **[../conformal/](../conformal/)** - Conformal prediction
- **[../assets/](../assets/)** - Figures

## Figures

<img src="../assets/temporal_grid.png" width="100%">

<img src="../assets/error_accumulation.png" width="100%">

---

### 1. Setup DeepMind Repository & Docker

```bash
# Clone DeepMind research repository (if not already cloned)
git clone https://github.com/google-deepmind/deepmind-research.git

# Run NVIDIA's TensorFlow 1.x container (patched for A100/H100 GPUs)
# Mount current directory to /workspace inside container.
# If your user isn't in the docker group, you may need to prefix with `sudo`.
docker run --gpus all -it --rm -v $(pwd):/workspace nvcr.io/nvidia/tensorflow:21.09-tf1-py3

 # Inside Docker container, install DeepMind dependencies.
 # IMPORTANT: TF1 requires NumPy < 1.20 (newer NumPy removes `np.object` and breaks TF1 import).
pip install "dm-sonnet<2" "tensorflow_probability<0.9" graph_nets matplotlib absl-py "numpy==1.19.5"
```

**Note:** The workspace directory is mounted at `/workspace` inside the container. The `deepmind-research` directory should be at `/workspace/deepmind-research`.

### 2. Download Datasets

```bash
# Inside Docker container
cd /workspace
bash download_and_prepare_data.sh
```

Downloads CylinderFlow (~16 GB) and Flag (~8 GB) datasets directly from DeepMind's Google Cloud Storage. The script `download_and_prepare_data.sh` downloads datasets to `data/cylinder_flow/cylinder_flow/` and `data/flag_simple/flag_simple/`

### 3. Create Data Splits

Create train/auxiliary/calibration/test splits for conformal prediction:

```bash
# Create CylinderFlow splits (28 trajectories)
python create_splits.py \
    --dataset cylinder \
    --num_trajectories 28 \
    --output data/splits/cylinder_splits.json \
    --seed 42

# Create Flag splits (18 trajectories)
python create_splits.py \
    --dataset flag \
    --num_trajectories 18 \
    --output data/splits/flag_splits.json \
    --seed 42
```

**Splits Created:**
- **CylinderFlow**: 17 train, 5 aux, 3 cal, 3 test trajectories
- **Flag**: 12 train, 3 aux, 1 cal, 2 test trajectories

**Script:** `create_splits.py` - Creates split JSON files with trajectory IDs and timestep ranges

**Optional:** the default Flag `test` window is only 10 timesteps, which can be too short/easy for meaningful held-out reporting. You can generate a longer held-out evaluation window (still disjoint from train/aux/cal) by allocating 30 timesteps to `test`:

```bash
python create_splits.py \
    --dataset flag \
    --num_trajectories 18 \
    --seed 42 \
    --aux_timesteps 40 \
    --cal_timesteps 30 \
    --test_timesteps 30 \
    --output data/splits/flag_splits_eval30.json
```

### 4. Filter Datasets

Filter TFRecord files to contain only the specified **trajectories** and the split's **timestep_range** (the script slices dynamic fields to the window and writes split-specific `meta.json` with updated `trajectory_length` and dynamic shapes for DeepMind compatibility):

```bash
# Filter CylinderFlow
python filter_trajectories_tf1.py \
    --splits_file data/splits/cylinder_splits.json \
    --input_dir data/cylinder_flow/cylinder_flow \
    --output_dir data/cylinder_flow_filtered \
    --dataset cylinder

# Filter Flag
python filter_trajectories_tf1.py \
    --splits_file data/splits/flag_splits.json \
    --input_dir data/flag_simple/flag_simple \
    --output_dir data/flag_simple_filtered \
    --dataset flag
```

**If using the 30-step Flag eval window**, filter with:

```bash
python filter_trajectories_tf1.py \
    --splits_file data/splits/flag_splits_eval30.json \
    --input_dir data/flag_simple/flag_simple \
    --output_dir data/flag_simple_filtered \
    --dataset flag
```

**Script:** `filter_trajectories_tf1.py` - Filters TFRecord datasets and creates correct file structure

- Each split directory contains `train.tfrecord` (filtered trajectories)
- Additional files (`valid.tfrecord`, `test.tfrecord`) are created as copies for DeepMind compatibility
- Root `meta.json` is copied for reference; each split directory gets its own `meta.json` updated for the split's `trajectory_length`

### 4.5 Validate Filtered Data

Before training, validate that:
- each split has the expected number of trajectories (TFRecord records),
- each split's `meta.json` has the correct `trajectory_length`,
- dynamic feature shapes have been updated to match the timestep window.

Use the included script: `validate_filtered_data.py`

```bash
# Primary (seed 42) filtered datasets
python validate_filtered_data.py \
  --filtered_root data/cylinder_flow_filtered \
  --splits_file data/splits/cylinder_splits.json

python validate_filtered_data.py \
  --filtered_root data/flag_simple_filtered \
  --splits_file data/splits/flag_splits.json

# Optional: validate one sensitivity split (seed 42)
python validate_filtered_data.py \
  --filtered_root data/cylinder_flow_filtered_seed42 \
  --splits_file data/splits_alternative/cylinder_splits_seed42.json

python validate_filtered_data.py \
  --filtered_root data/flag_flow_filtered_seed42 \
  --splits_file data/splits_alternative/flag_splits_seed42.json
```

### 5. Train Models

Train MeshGraphNet models to 200K steps:

```bash
cd /workspace/deepmind-research

# Train CylinderFlow (CFD model)
python -m meshgraphnets.run_model \
    --mode=train \
    --model=cfd \
    --checkpoint_dir=/workspace/checkpoints_cylinder_ts \
    --dataset_dir=/workspace/data/cylinder_flow_filtered/train \
    --num_training_steps=200000

# Train Flag (Cloth model)
python -m meshgraphnets.run_model \
    --mode=train \
    --model=cloth \
    --checkpoint_dir=/workspace/checkpoints_flag_ts \
    --dataset_dir=/workspace/data/flag_simple_filtered/train \
    --num_training_steps=200000
```

**Checkpoints saved to:**
- `/workspace/checkpoints_cylinder_ts/model.ckpt-200000`
- `/workspace/checkpoints_flag_ts/model.ckpt-200000`

### 6. Evaluate Models (Generate Rollouts)

After training, generate rollouts for all splits (auxiliary, calibration, test) using the trained checkpoints:

```bash
cd /workspace/deepmind-research

# CylinderFlow: Generate rollouts for all splits
python -m meshgraphnets.run_model \
    --mode=eval \
    --model=cfd \
    --checkpoint_dir=/workspace/checkpoints_cylinder_ts \
    --dataset_dir=/workspace/data/cylinder_flow_filtered/auxiliary \
    --rollout_path=/workspace/rollouts_200k/rollout_cylinder_auxiliary_200k.pkl \
    --num_rollouts=5

python -m meshgraphnets.run_model \
    --mode=eval \
    --model=cfd \
    --checkpoint_dir=/workspace/checkpoints_cylinder_ts \
    --dataset_dir=/workspace/data/cylinder_flow_filtered/calibration \
    --rollout_path=/workspace/rollouts_200k/rollout_cylinder_calibration_200k.pkl \
    --num_rollouts=3

python -m meshgraphnets.run_model \
    --mode=eval \
    --model=cfd \
    --checkpoint_dir=/workspace/checkpoints_cylinder_ts \
    --dataset_dir=/workspace/data/cylinder_flow_filtered/test \
    --rollout_path=/workspace/rollouts_200k/rollout_cylinder_test_200k.pkl \
    --num_rollouts=3

# Flag: Generate rollouts for all splits
python -m meshgraphnets.run_model \
    --mode=eval \
    --model=cloth \
    --checkpoint_dir=/workspace/checkpoints_flag_ts \
    --dataset_dir=/workspace/data/flag_simple_filtered/auxiliary \
    --rollout_path=/workspace/rollouts_200k/rollout_flag_auxiliary_200k.pkl \
    --num_rollouts=3

python -m meshgraphnets.run_model \
    --mode=eval \
    --model=cloth \
    --checkpoint_dir=/workspace/checkpoints_flag_ts \
    --dataset_dir=/workspace/data/flag_simple_filtered/calibration \
    --rollout_path=/workspace/rollouts_200k/rollout_flag_calibration_200k.pkl \
    --num_rollouts=1

python -m meshgraphnets.run_model \
    --mode=eval \
    --model=cloth \
    --checkpoint_dir=/workspace/checkpoints_flag_ts \
    --dataset_dir=/workspace/data/flag_simple_filtered/test \
    --rollout_path=/workspace/rollouts_200k/rollout_flag_test_200k.pkl \
    --num_rollouts=2
```

**Outputs:**
- Rollout `.pkl` files saved to `/workspace/rollouts_200k/` (default small splits)
- Each file contains a list of trajectory dictionaries with:
  - **CylinderFlow**: `pred_velocity`, `gt_velocity` (plus `faces`, `mesh_pos`)
  - **Flag**: `pred_pos`, `gt_pos` (plus `faces`, `mesh_pos`)
- Use `inspect_rollout_structure.py` to inspect file structure

**Note:** `num_rollouts` should match the number of trajectories in each split (see Section "Data Splits for Conformal Prediction" below).

### 7. Split Sensitivity Analysis

Run rollouts on alternative splits (5 different random seeds) **without retraining**:

```bash
# Run complete sensitivity analysis
python run_split_sensitivity_rollouts.py --all_datasets

# Or run for specific dataset
python run_split_sensitivity_rollouts.py --dataset cylinder --seeds 42 123 456 789 999
```

**What it does:**
1. Creates 5 alternative splits with different random seeds (`create_splits.py --seeds`)
2. Filters datasets for each alternative split (`filter_trajectories_tf1.py`)
3. Runs rollouts on auxiliary/calibration/test splits using **same 200K checkpoint**
4. Saves all results to `/workspace/rollouts_sensitivity/` (legacy) or your configured sensitivity output directory

**Outputs:**
- `rollout_{dataset}_{split}_seed{seed}.pkl` - Rollout predictions
- `rollout_{dataset}_{split}_seed{seed}.log` - Evaluation logs
- `summary.json` - Summary of all rollouts
- 30 total rollouts (5 seeds × 3 splits × 2 datasets)

### 8. Analyze Results

Compute metrics and assess split sensitivity:

```bash
# Run analysis after rollouts
python run_split_sensitivity_rollouts.py --all_datasets --analyze

# Run analysis only (if rollouts already done)
python run_split_sensitivity_rollouts.py --skip_splits --skip_filter --skip_rollouts --analyze
```

Computes RMSE and Coefficient of Variation (CV) across different seeds to assess robustness to split variations.

## Data Splits for Conformal Prediction

### Standard Conformal Prediction Workflow

- **Train**: Used to train the model (already done - 200K checkpoint)
- **Auxiliary**: Trains adaptive scaling function (uses `.pkl` rollouts)
- **Calibration**: Computes conformal scores/quantiles (uses `.pkl` rollouts)
- **Test**: Evaluates coverage (uses `.pkl` rollouts)

### Disjointness (what is actually separated)

This repo enforces **trajectory-disjoint splits** (each split uses different trajectory IDs).

- **Timesteps (two supported protocols):**
  - **Default split configs** use **disjoint timestep windows** per split (useful for studying horizon shift / autoregressive error growth).
  - **Optional shared-window configs** (see below) use a **shared timestep window** for auxiliary/calibration/evaluation (often matching the train window) to keep calibration and evaluation "in-regime comparable" for post-hoc conformal.

For the default split configs:

- **Trajectories are disjoint across splits**: each split uses different trajectory IDs.
- **Timesteps are disjoint across splits**: each split uses a non-overlapping `timestep_range` and `filter_trajectories_tf1.py` slices dynamic fields to that window.

- **CylinderFlow** timesteps: train $[0,400)$, auxiliary $[400,520)$, calibration $[520,570)$, test $[570,600)$
- **Flag** timesteps: train $[0,200)$, auxiliary $[200,260)$, calibration $[260,290)$, test $[290,300)$

### Optional: "big in-regime" splits (improves empirical CP validity + stability)

If you have access to the full raw TFRecords (1000 trajectories for each dataset), a strong reporting protocol for post-hoc conformal is:

- **Keep the training trajectory IDs fixed** (so you stay compatible with the already-trained 200K checkpoint).
- Sample **many more auxiliary/calibration/evaluation trajectories** from the remaining pool.
- Use a **shared timestep window** for auxiliary/calibration/evaluation (e.g. the train window), so calibration and evaluation are "in-regime comparable".

This tends to make empirical coverage much closer to nominal $1-\alpha$ (because cal and eval are not drawn from different time regimes), and improves adaptive efficiency (because $\sigma(x)$ is fit on far more auxiliary data).

Example (run inside TF1 container, from `/workspace/meshgraphnet`):

```bash
python create_splits.py \
  --dataset cylinder --num_trajectories 1000 --seed 42 \
  --keep_train_from data/splits/cylinder_splits.json \
  --timesteps_per_traj 600 --train_timesteps 400 \
  --nontrain_timestep_range 0 400 \
  --aux_n 100 --cal_n 100 --test_n 100 \
  --output data/splits/cylinder_splits_big_inregime.json

python create_splits.py \
  --dataset flag --num_trajectories 1000 --seed 42 \
  --keep_train_from data/splits/flag_splits.json \
  --timesteps_per_traj 401 --train_timesteps 200 \
  --nontrain_timestep_range 0 200 \
  --aux_n 100 --cal_n 100 --test_n 100 \
  --output data/splits/flag_splits_big_inregime.json
```

Then filter + rollouts into parallel folders (so you don't overwrite the default ones):

```bash
python filter_trajectories_tf1.py \
  --splits_file data/splits/cylinder_splits_big_inregime.json \
  --input_dir data/cylinder_flow/cylinder_flow \
  --output_dir data/cylinder_flow_filtered_big \
  --dataset cylinder

python filter_trajectories_tf1.py \
  --splits_file data/splits/flag_splits_big_inregime.json \
  --input_dir data/flag_simple/flag_simple \
  --output_dir data/flag_simple_filtered_big \
  --dataset flag

mkdir -p /workspace/meshgraphnet/rollouts_200k_big
cd /workspace/meshgraphnet/deepmind-research

# Cylinder (100 rollouts per split)
python -m meshgraphnets.run_model --mode=eval --model=cfd \
  --checkpoint_dir=/workspace/meshgraphnet/checkpoints_cylinder_ts \
  --dataset_dir=/workspace/meshgraphnet/data/cylinder_flow_filtered_big/auxiliary \
  --rollout_path=/workspace/meshgraphnet/rollouts_200k_big/rollout_cylinder_auxiliary_200k.pkl \
  --num_rollouts=100

python -m meshgraphnets.run_model --mode=eval --model=cfd \
  --checkpoint_dir=/workspace/meshgraphnet/checkpoints_cylinder_ts \
  --dataset_dir=/workspace/meshgraphnet/data/cylinder_flow_filtered_big/calibration \
  --rollout_path=/workspace/meshgraphnet/rollouts_200k_big/rollout_cylinder_calibration_200k.pkl \
  --num_rollouts=100

python -m meshgraphnets.run_model --mode=eval --model=cfd \
  --checkpoint_dir=/workspace/meshgraphnet/checkpoints_cylinder_ts \
  --dataset_dir=/workspace/meshgraphnet/data/cylinder_flow_filtered_big/test \
  --rollout_path=/workspace/meshgraphnet/rollouts_200k_big/rollout_cylinder_test_200k.pkl \
  --num_rollouts=100

# Flag (100 rollouts per split)
python -m meshgraphnets.run_model --mode=eval --model=cloth \
  --checkpoint_dir=/workspace/meshgraphnet/checkpoints_flag_ts \
  --dataset_dir=/workspace/meshgraphnet/data/flag_simple_filtered_big/auxiliary \
  --rollout_path=/workspace/meshgraphnet/rollouts_200k_big/rollout_flag_auxiliary_200k.pkl \
  --num_rollouts=100

python -m meshgraphnets.run_model --mode=eval --model=cloth \
  --checkpoint_dir=/workspace/meshgraphnet/checkpoints_flag_ts \
  --dataset_dir=/workspace/meshgraphnet/data/flag_simple_filtered_big/calibration \
  --rollout_path=/workspace/meshgraphnet/rollouts_200k_big/rollout_flag_calibration_200k.pkl \
  --num_rollouts=100

python -m meshgraphnets.run_model --mode=eval --model=cloth \
  --checkpoint_dir=/workspace/meshgraphnet/checkpoints_flag_ts \
  --dataset_dir=/workspace/meshgraphnet/data/flag_simple_filtered_big/test \
  --rollout_path=/workspace/meshgraphnet/rollouts_200k_big/rollout_flag_test_200k.pkl \
  --num_rollouts=100
```

**Note:** MeshGraphNet rollouts typically have $T = \text{trajectory length} - 2$ due to DeepMind preprocessing `val[1:-1]`, but the underlying timestep window you specify is still the one being evaluated (the rollout just omits the first/last step of that window).

**Sizes in this shared-window example:**
- **CylinderFlow**: train 17×400; auxiliary/calibration/evaluation 100×400 each
- **Flag**: train 12×200; auxiliary/calibration/evaluation 100×200 each

### Split Reproducibility

All split details are saved in JSON files for reproducibility:

**Primary splits (seed 42):**
- `data/splits/cylinder_splits.json` - Contains exact trajectory IDs and timestep ranges
- `data/splits/flag_splits.json` - Contains exact trajectory IDs and timestep ranges

**Alternative splits (sensitivity analysis):**
- `data/splits_alternative/cylinder_splits_seed{seed}.json` - Individual split files
- `data/splits_alternative/flag_splits_seed{seed}.json` - Individual split files
- `data/splits_alternative/cylinder_splits_summary.json` - Summary of all seeds with trajectory assignments
- `data/splits_alternative/flag_splits_summary.json` - Summary of all seeds with trajectory assignments

**Each split JSON file contains:**
- `trajectory_ids`: Exact trajectory indices (0-indexed) assigned to each split
- `timestep_range`: Start and end timesteps for each split
- `num_timesteps`: Number of timesteps per split
- `num_trajectories`: Number of trajectories per split
- `total_samples`: Total number of samples (trajectories × timesteps)
- `metadata`: Dataset name, total trajectories, seed used

**Example: Viewing split details:**
```bash
# View primary CylinderFlow split (seed 42)
cat data/splits/cylinder_splits.json | python -m json.tool

# View summary of all alternative splits
cat data/splits_alternative/cylinder_splits_summary.json | python -m json.tool
```

## References

- [DeepMind MeshGraphNet Paper](https://arxiv.org/abs/2010.03409)
- [DeepMind Research Repository](https://github.com/google-deepmind/deepmind-research)
- [MeshGraphNet GitHub](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
