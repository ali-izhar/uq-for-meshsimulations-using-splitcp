# Mesh-Based Fluid Simulation with GNNs

## Overview

This document summarizes experimental results from training the `MeshGraphNet` model on the `cylinder_flow` dataset with different configurations and dataset sizes.

### Configuration Comparison

| Parameter            | Small                | Medium                           | Large                            |
| -------------------- | -------------------- | -------------------------------- | -------------------------------- |
| **Training Data**    | 85 timesteps         | 500 timesteps (1-3 trajectories) | 5990 timesteps (multi-trajectory)|
| **Test Data**        | 15 timesteps         | 90 timesteps                     | 2995 timesteps                   |
| **Epochs**           | 5000                 | 5000                             | 3000                             |
| **Batch Size**       | 16                   | 16                               | 128                              |
| **Noise Scale**      | 0.0 (default)        | 0.0 (or 0.02 with +noise=paper)  | 0.02 (recommended with +noise=paper)|

## 1. Original Experiments (Medium Dataset + Noise)

### 1.1 - Qualitative results

First, we visualize a model trained on the medium dataset (500 timesteps).
5k epochs take about 4h on an RTX 3060.
We animate the next-step predictions for 500 timesteps of the same trajectory we trained on.

The model performs much worse during the first few timesteps (~2-10).
Since we train less than a single trajectory with randomly selected timesteps, the first few timesteps are rarely seen during training.

![Animation of next-step predictions](https://github.com/BurgerAndreas/gnn-physics/blob/main/data/animations/x_velocity_0_500_datasize_medium_anim.gif)

### 1.2 - Training on more data

We see a clear advantage when training on more data (going from 45 to 500 timesteps).
We are still far away from the original paper, which trained on $1,200 * 600 = 740,000$ timesteps!

Looking at the loss curve, it is probably much better to trade off fewer epochs in favor of more data.

![Test loss plot with more data (made via wandb)](https://github.com/BurgerAndreas/gnn-physics/blob/main/data/plots/more_data_test_loss.png)

### 1.3 - Including noise

Next, let us see the effect of adding noise during training.
So far, we have only looked at single-step predictions, in contrast to full rollouts.
During rollouts, we feed the model predictions back in as inputs for the next timestep, which leads to accumulating errors.
The model is only trained to do single-step prediction: the current true velocity is the input, the next velocity is the target.
The idea is that adding noise during training imitates making predictions with imperfect inputs.

Indeed, we see that adding noise during training can reduce the accumulation of error, at least for the timesteps 50-150.
![Error accumulation with and without noise during training](https://github.com/BurgerAndreas/gnn-physics/blob/main/data/plots/datasize_medium_50_150_rollout_noise.png)

### 1.4 - Generalize to new trajectories

We can also use the same model on an unseen trajectory, and it generalizes!

![Animation of next-step predictions on a trajectory unseen during training](https://github.com/BurgerAndreas/gnn-physics/blob/main/data/animations/x_velocity_testtraj_0_500_datasize_medium_anim.gif)

## 2. Comparison and Analysis

### 2.1 - Dataset Size Impact

Training on more data consistently improves model performance and generalization.

| Configuration | Train Size | Test Size | Use Case                    |
| ------------- | ---------- | --------- | --------------------------- |
| Small         | 85         | 15        | Quick prototyping           |
| Medium        | 500        | 90        | Standard experiments        |
| Large         | 5990       | 2995      | Large-scale training        |

### 2.2 - Training Efficiency

| Aspect          | Small/Medium | Large     | Notes                    |
| --------------- | ------------ | --------- | ------------------------ |
| Epochs          | 5000         | 3000      | Reduced due to more data |
| Batch Size      | 16           | 128       | Optimized for large data |
| Mixed Precision | Optional     | Recommended | AMP for efficiency     |

## 3. Reproduce Runs

### Training Configurations

```bash
# Default (small dataset)
python run_gnn.py

# Medium dataset
python run_gnn.py +datasize=medium

# Medium with noise (recommended)
python run_gnn.py +datasize=medium +noise=paper

# Large dataset
python run_gnn.py +datasize=large +noise=paper

# Flag dataset (3D simulation)
python run_gnn.py +dataset=flag_simple
```

## 4. Project Outline

There existed a [prior PyTorch project by three Stanford students](https://medium.com/stanford-cs224w/learning-mesh-based-flow-simulations-on-graph-networks-44983679cf2d)
which I build on. Thank you!
But the project:
(a) did not include code to generate their data
(b) did not add noise during training
(c) only tested on single-step predictions instead of full rollouts
(d) could not train and test on different trajectories
(e) was partially broken
(f) lacked infrastructure to run experiments

I:

- Added code to convert the original data (.tfrecord) into a general format (.hdf5)
- Rebuilt the Colab project code into a working codebase
- Added noise during training, as in the original paper
- Built evaluation to rollout trajectories
- Added experiment infrastructure (train/test on different trajectories, wandb logging, hydra configs, resume checkpoints)
- **Large-scale training support with optimized settings**
- **Modular configuration system for different dataset sizes**

I also looked into predicting the sizing field for a couple of hours.
One would need to implement the sizing field prediction and the sizing-based remesher.
