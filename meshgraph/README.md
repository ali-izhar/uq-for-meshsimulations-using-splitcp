# Mesh-Based Fluid Simulation with GNNs

> This module is taken from [gnn-physics](https://github.com/BurgerAndreas/gnn-physics).

Re-implementation of [Learning Mesh-Based Simulation with Graph Networks](https://sites.google.com/view/meshgraphnets) for `cylinder_flow` and `flag_dataset` in PyTorch based on [this blog post](https://medium.com/stanford-cs224w/learning-mesh-based-flow-simulations-on-graph-networks-44983679cf2d).

[Look at results.md for a summary!](results.md)

## Installation

Get the code:

```bash
git clone git@github.com:BurgerAndreas/gnn-physics.git
# git clone https://github.com/BurgerAndreas/gnn-physics.git
```

Setup Conda environment:
(Tested on Ubuntu 22.04, RTX 3060, Cuda 12.3)

```bash
conda create -n meshgnn python=3.11 -y
conda activate meshgnn
pip3 install -r requirements.txt

# or try by hand
conda install -c conda-forge nbformat jupyter plotly matplotlib mediapy pip tqdm gdown -y
pip3 install torch torchvision torchaudio torch_scatter torch_sparse torch_cluster torch_spline_conv torch-geometric torchdata
pip3 install black hydra-core
pip3 install tensorflow tensorrt protobuf==3.20.3
```

## Run

Download the small dataset (1GB) from Google Drive:

```bash
cd data/datasets/cylinder_flow_pyg/
# https://drive.google.com/file/d/1AmQwNt2zsLnUSUWcH_f8rGIPY9VhPQZt/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AmQwNt2zsLnUSUWcH_f8rGIPY9VhPQZt' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AmQwNt2zsLnUSUWcH_f8rGIPY9VhPQZt" -O data_pt.tgz && rm -rf /tmp/cookies.txt

# Unzip the data
tar -zxvf data_pt.tgz
```

Run the code:

```bash
cd ../../..
# Run with default settings
python run_gnn.py
# Try additional configs like this:
python run_gnn.py +noise=paper +datasize=small
# Use the new large configuration for expanded dataset:
python run_gnn.py +datasize=large +dataset=expanded +noise=paper
# Plot training loss & predictions from loaded checkpoints like this:
python plot.py +datasize=medium
python animate.py +datasize=medium
```

## Dataset

The original `cylinder_flow` dataset contains 1,200 trajectories with 600 timesteps each.
The data is in the `.tfrecord` format. `.tfrecord` is highly optimized, but only works with TensorFlow and can be hard to handle.

I simplified the dataset to 4 trajectories (3 train, 1 test) saved as numpy arrays in a `.hdf5` file.
The 4 trajectories are provided via the Google Drive link above.

#### Optional: Get more data

If you want to download the full original `.tfrecord` dataset for `cylinder_flow` (16 GB):

```bash
chmod +x ./data/datasets/download_dataset.sh
bash ./data/datasets/download_dataset.sh cylinder_flow ./data/datasets
```

If you want to convert the `.tfrecord` dataset to numpy in `.hdf5`:

```bash
conda activate meshgnn
# -num_traj -1 means convert all trajectories
python ./data/datasets/tfrecord_to_hdf5.py -in 'data/datasets/cylinder_flow/train' -out 'data/datasets/cylinder_flow_hdf5/train' --num_traj 3
python ./data/datasets/tfrecord_to_hdf5.py -in 'data/datasets/cylinder_flow/test' -out 'data/datasets/cylinder_flow_hdf5/test' --num_traj 1
```

If you want to convert the `.hdf5` dataset to PyTorch graphs `.pt`:

```bash
conda activate meshgnn
python ./data/datasets/hdf5_to_pyg.py -in 'data/datasets/cylinder_flow_hdf5/train.hdf5' -out 'data/datasets/cylinder_flow_pyg/train.pt'
python ./data/datasets/hdf5_to_pyg.py -in 'data/datasets/cylinder_flow_hdf5/test.hdf5' -out 'data/datasets/cylinder_flow_pyg/test.pt'
```

#### Optional: Expanded dataset configuration

For users with access to more data, a new "large" configuration is available that supports:

- **10 training trajectories** (instead of 3)
- **5 test trajectories** (instead of 1)
- **Multi-trajectory training** for better generalization
- **Paper noise settings** (0.02 noise scale)

To use this configuration:

```bash
python run_gnn.py +datasize=large +dataset=expanded +noise=paper
```

This configuration automatically includes:

- **3000 training epochs** (reduced from 5000 for faster training) via `+datasize=large`
- **5990 training timesteps** (10 trajectories × ~599 each) via `+datasize=large`
- **2995 test timesteps** (5 trajectories × ~599 each) via `+datasize=large`
- **Multi-trajectory support** (separate train/test trajectories) via `+dataset=expanded`
- **Paper noise settings** (0.02 noise scale) via `+noise=paper`
- **Optimized batch size** (128 for large datasets) via `+datasize=large`

#### Optional: Get prior blog data

I based my code on this [blog post](https://medium.com/stanford-cs224w/learning-mesh-based-flow-simulations-on-graph-networks-44983679cf2d),
which ships with some data in `.pt` format.
Sadly, they did not include the code they used to transform the data.
My code still works on their data.
In practice, their data performs worse than my data conversion, for unknown reasons.

If you want to download their data:

```bash
python ./data/datasets/download_pyg_stanford_data.py
```

## Future Work

The [original codebase of the paper](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) implements the `cylinder_flow` and `flag_simple` domains.

- [ ] Change `./data/datasets/hdf5_to_pyg.py` to work with all datasets with different features

The original codebase also does not contain the prediction of the sizing field and the corresponding remesher.
Out of time constraints, we do not implement the sizing field prediction + remesher either.
To implement sizing field prediction:

- [ ] Build prediction head and combine with existing GNN
- [ ] Build sizing-based remesher (pseudo-code can be found in [this paper](http://graphics.berkeley.edu/papers/Narain-AAR-2012-11/Narain-AAR-2012-11.pdf) and [A3 of the original paper](https://arxiv.org/abs/2010.03409))
- [ ] Adapt training loop to learn sizing field prediction on `flag_dynamic_sizing`
      (Only the `flag_dynamic_sizing` (36 GB) and `sphere_dynamic_sizing` datasets include the necessary data to learn the sizing field)

## Resources

- Original [Paper](https://arxiv.org/abs/2010.03409)
  |
  [Website](https://sites.google.com/view/meshgraphnets)
  |
  [Code (TensorFlow)](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
- My code is based on this [blog](https://medium.com/stanford-cs224w/learning-mesh-based-flow-simulations-on-graph-networks-44983679cf2d)
  |
  [Code (PyTorch)](https://colab.research.google.com/drive/1mZAWP6k9R0DE5NxPzF8yL2HpIUG3aoDC?usp=sharing)

Follow-up papers:

- [MultiScale MeshGraphNets, 2022](https://arxiv.org/abs/2210.00612)
- [Predicting Physics in Mesh-reduced Space with Temporal Attention](https://arxiv.org/abs/2201.09113)
- [Graph network simulators can learn discontinuous, rigid contact dynamics](https://proceedings.mlr.press/v205/allen23a.html)
- [Learned Coarse Models for Efficient Turbulence Simulation, 2022](https://arxiv.org/abs/2112.15275)

---

```bash
### Medium Dataset (No Noise)
python run_gnn.py conformal +datasize=medium --start-step 0 --num-steps 500 --use-test-traj

### **Medium Dataset with Paper Noise
python run_gnn.py conformal +datasize=medium +noise=paper --start-step 0 --num-steps 500 --use-test-traj

### Custom Time Windows (if needed)
# Early timesteps (0-200)
python run_gnn.py conformal +datasize=medium +noise=paper --start-step 0 --num-steps 200 --use-test-traj

# Mid timesteps (200-400)
python run_gnn.py conformal +datasize=medium +noise=paper --start-step 200 --num-steps 200 --use-test-traj

# Late timesteps (400-600)
python run_gnn.py conformal +datasize=medium +noise=paper --start-step 400 --num-steps 200 --use-test-traj

### Training Loss Plots
# Medium dataset plots
python make_plots.py plots +datasize=medium

# Medium with noise
python make_plots.py plots +datasize=medium +noise=paper

# Large expanded dataset
python make_plots.py plots +datasize=large +dataset=expanded +noise=paper

### Velocity Field Animations
# Standard animations for each model
python make_plots.py animations +datasize=medium +noise=paper
python make_plots.py animations +datasize=large +dataset=expanded +noise=paper

# Custom animation with specific parameters
python run_gnn.py custom-anim +datasize=medium +noise=paper --start-step 50 --num-steps 200 --use-test-traj

### Static Frame Captures
# Create static frames at key timesteps
python make_plots.py frames +datasize=medium +noise=paper --start-step 50 --num-steps 100 --stride 10 --use-test-traj

# Compact research-style layout
python make_plots.py frames +datasize=medium +noise=paper --start-step 100 --num-steps 50 --stride 5 --compact --use-test-traj

### Mesh Topology and Flow Visualizations
# High-contrast mesh plots
python make_plots.py mesh +datasize=medium +noise=paper --start-step 140 --use-test-traj

### Create Everything at Once
# Both plots and animations for each model
python make_plots.py both +datasize=medium +noise=paper
python make_plots.py both +datasize=large +dataset=expanded +noise=paper
```
