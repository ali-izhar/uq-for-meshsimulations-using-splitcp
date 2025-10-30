#!/usr/bin/env python3
"""
Modern, research-grade plotting utilities for cylinder/flag figures.

This module reuses data loading from meshgraph.utils.plot but applies a
unified, high-quality aesthetic (fonts, colors, vector saves) and produces
the specific panels used in the paper (speed + arrows, FEM streamlines,
topology highlights).
"""

from typing import Any, Tuple
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.tri import LinearTriInterpolator

# NOTE: This module is self-contained. Do not import from itself to avoid
# circular imports. The functions `_load_predictions_for_plotting`,
# `ensure_directory`, `name_from_config`, and `PLOTS_DIR` are defined below in
# this file.


def apply_modern_style(dpi: int = 400) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )


def _panel_colorbar(ax, mappable, label: str = "", size: str = "1.8%") -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=0.025)
    clb = ax.figure.colorbar(mappable, cax=cax)
    clb.ax.tick_params(labelsize=8)
    if label:
        clb.ax.set_title(label, fontsize=8)


def fig_speed_quiver(cfg: Any, step: int = 50, cmap: str = "cividis") -> str:
    data_pred, data_true, _ = _load_predictions_for_plotting(
        cfg, (step, step + 1), single_step=True, use_test_traj=False
    )
    data = data_true[0]
    pos = data.mesh_pos
    faces = data.cells
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
    vel = data.x[:, 0:2].cpu().numpy()
    speed = np.sqrt((vel**2).sum(axis=1))

    apply_modern_style()
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 1.6), facecolor="white")
    ax.set_aspect("equal")
    ax.set_axis_off()

    tpc = ax.tripcolor(
        triang, speed, shading="gouraud", cmap=cmap, edgecolors="0.8", linewidth=0.25
    )

    # Downsample and draw quiver
    stride = max(1, int(len(pos) / 500))
    idx = np.arange(0, len(pos), stride)
    ax.quiver(
        pos[idx, 0],
        pos[idx, 1],
        vel[idx, 0],
        vel[idx, 1],
        color="k",
        alpha=0.6,
        scale=50.0,
        width=0.0022,
        zorder=4,
    )

    _panel_colorbar(ax, tpc, label="|v| (m/s)")

    ensure_directory(PLOTS_DIR)
    out = os.path.join(PLOTS_DIR, f"modern_speed_quiver_step{step}.png")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return out


def fig_streamlines(cfg: Any, step: int = 50, cmap: str = "cividis") -> str:
    data_pred, data_true, _ = _load_predictions_for_plotting(
        cfg, (step, step + 1), single_step=True, use_test_traj=False
    )
    data = data_true[0]
    pos = data.mesh_pos
    faces = data.cells
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
    vel = data.x[:, 0:2].cpu().numpy()
    u, v = vel[:, 0], vel[:, 1]

    Iu = LinearTriInterpolator(triang, u)
    Iv = LinearTriInterpolator(triang, v)

    xmin, xmax = pos[:, 0].min().item(), pos[:, 0].max().item()
    ymin, ymax = pos[:, 1].min().item(), pos[:, 1].max().item()
    X = np.linspace(xmin, xmax, 500)
    Y = np.linspace(ymin, ymax, 140)
    XX, YY = np.meshgrid(X, Y)

    UU = np.ma.array(Iu(XX, YY))
    VV = np.ma.array(Iv(XX, YY))
    mask = np.ma.getmask(UU) | np.ma.getmask(VV)
    UU.mask = mask
    VV.mask = mask
    speed = np.sqrt(UU**2 + VV**2)

    smin, smax = float(speed.min()), float(speed.max())
    lw_min, lw_max = 0.3, 2.0
    if np.isfinite(smin) and np.isfinite(smax) and smax > 0:
        lw = lw_min + (lw_max - lw_min) * ((speed - smin) / (smax - smin))
    else:
        lw = lw_min

    apply_modern_style()
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 1.6), facecolor="white")
    ax.set_aspect("equal")
    ax.set_axis_off()

    strm = ax.streamplot(
        X, Y, UU, VV, color=speed, linewidth=lw, cmap=cmap, density=1.2, arrowsize=0.7
    )
    ax.triplot(triang, color="0.85", lw=0.3, alpha=0.6, zorder=1)
    _panel_colorbar(ax, strm.lines, label="|v| (m/s)")

    ensure_directory(PLOTS_DIR)
    out = os.path.join(PLOTS_DIR, f"modern_streamlines_step{step}.png")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return out


def fig_topology(cfg: Any, step: int = 0) -> str:
    data_pred, data_true, _ = _load_predictions_for_plotting(
        cfg, (step, step + 1), single_step=True, use_test_traj=False
    )
    data = data_true[0]
    pos = data.mesh_pos
    faces = data.cells
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
    type_ids = np.argmax(data.x[:, 2:].cpu().numpy(), axis=1)

    apply_modern_style()
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 1.6), facecolor="white")
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.triplot(triang, color="0.1", lw=0.4)

    # Minimal boundary markers
    unique = set(type_ids.tolist())
    wall_id = 3 if 3 in unique else None
    if wall_id is not None:
        w = type_ids == wall_id
        ax.scatter(pos[w, 0], pos[w, 1], s=22, c="#E24A33", edgecolors="white", lw=0.7)

    ensure_directory(PLOTS_DIR)
    out = os.path.join(PLOTS_DIR, f"modern_topology_step{step}.png")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return out


def build_cylinder_panels(cfg: Any, step: int = 50) -> Tuple[str, str]:
    """Convenience to produce both panels for Figure 2 at the same style."""
    return fig_speed_quiver(cfg, step=step), fig_streamlines(cfg, step=step)


"""
Plotting and visualization utilities for MeshGraphNet training and evaluation.

This module provides:
- Training/test loss curve plotting
- Velocity field animations (single-step and rollout)
- Rollout error analysis and comparison plots
- Noise effect visualization

Design principles:
- CPU-based matplotlib for compatibility and reproducibility
- Hydra configuration integration
- Comprehensive inline documentation
- Modular function design for easy composition
"""

import torch
import os
import copy
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import pickle
import math
import pathlib
from typing import Any, Tuple, List

# Matplotlib imports for triangular mesh plotting
from matplotlib import tri as mtri
from matplotlib.tri import LinearTriInterpolator
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm

# Project imports
import hydra
from omegaconf import DictConfig

from models.meshgraphnet import MeshGraphNet
from utils.process import get_stats, unnormalize

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================

# Compute the repository root relative to this file location
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]

# Data directories used by training/evaluation/visualization code
DATASET_DIR = os.path.join(ROOT_DIR, "datasets")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(ROOT_DIR, "outputs", "plots")
ANIM_DIR = os.path.join(ROOT_DIR, "outputs", "animations")

# Physical constants from the original dataset
DELTA_T = 0.01  # Simulation timestep size in seconds

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def name_from_config(cfg: Any) -> str:
    """
    Build a filesystem-friendly model name from the Hydra config.

    If `cfg.override_dirname` is set and resolvable, sanitize and use it;
    otherwise reconstruct the name from config parameters to match training naming.
    This creates consistent naming for checkpoints, plots, and animations.
    """
    try:
        override = getattr(cfg, "override_dirname", None)
        if override and not str(override).startswith("${"):
            # Replace characters that could be problematic in file paths
            override = str(override).replace(",", "-").replace("+", "")
            override = override.replace("=", "_").replace(".", "")
            # Remove training-time flag suffix if present
            override = override.replace("-resume_checkpoint_False", "")
            return override
    except (AttributeError, Exception):
        pass

    # Fallback to config_name or generate a descriptive name
    try:
        if hasattr(cfg, "config_name") and cfg.config_name != "default":
            return str(cfg.config_name)
    except (AttributeError, Exception):
        pass

        # Reconstruct the override directory name from config parameters
    # This matches how Hydra generates the override_dirname during training
    try:
        parts = []

        # Check for dataset overrides - only add if explicitly overridden
        # The default config has use_stanford_data=False, so we need to check if it was changed
        if hasattr(cfg, "data") and hasattr(cfg.data, "use_stanford_data"):
            # Dataset type is determined by the data configuration
            # The default config already handles this appropriately
            pass

        # Check for datasize overrides
        if hasattr(cfg, "training") and hasattr(cfg.training, "train_size"):
            if cfg.training.train_size == 500:
                parts.append("datasize_medium")
            elif cfg.training.train_size == 5990:
                parts.append("datasize_large")
            elif cfg.training.train_size == 45:
                parts.append("datasize_small")

        # Check for noise overrides
        if hasattr(cfg, "data") and hasattr(cfg.data, "noise_scale"):
            if cfg.data.noise_scale > 0:
                parts.append("noise_paper")

        # Check for testset overrides
        if hasattr(cfg, "data") and hasattr(cfg.data, "train_test_same_traj"):
            if not cfg.data.train_test_same_traj:
                parts.append("testset_different")

        if parts:
            return "-".join(parts)
    except (AttributeError, Exception):
        pass

    # Final fallback: generate a name from config parameters
    try:
        parts = []
        if (
            hasattr(cfg, "data")
            and hasattr(cfg.data, "noise_scale")
            and cfg.data.noise_scale > 0
        ):
            parts.append("noise")
        if hasattr(cfg, "training") and hasattr(cfg.training, "train_size"):
            parts.append(f"train{cfg.training.train_size}")
        if hasattr(cfg, "model") and hasattr(cfg.model, "hidden_dim"):
            parts.append(f"hd{cfg.model.hidden_dim}")

        if parts:
            return "_".join(parts)
    except (AttributeError, Exception):
        pass

    return "default"


def ensure_directory(path: str) -> None:
    """Ensure directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)


def _apply_research_style(dpi: int = 300) -> None:
    """Set high-contrast, research-grade matplotlib style consistently.

    This function is idempotent and safe to call before each figure.
    """
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
            "font.size": 10,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "lines.antialiased": True,
            "patch.antialiased": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# =============================================================================
# CORE PLOTTING FUNCTIONS
# =============================================================================


def plot_training_loss(cfg: Any) -> plt.Figure:
    """
    Plot training and test loss curves from saved DataFrame.

    Loads the losses DataFrame saved during training and creates a standard
    loss curve plot. Losses correspond to single-step predictions.

    Args:
        cfg: Hydra configuration containing model naming information

    Returns:
        matplotlib Figure object
    """
    model_name = name_from_config(cfg)
    path_df = os.path.join(CHECKPOINT_DIR, model_name + "_losses.pkl")

    # Load losses dataframe with columns [epoch, train_loss, test_loss, velocity_val_loss]
    df = pd.read_pickle(path_df)
    train_loss = df["train_loss"]
    test_loss = df["test_loss"]

    # Create and save plot
    ensure_directory(PLOTS_DIR)
    path_fig = os.path.join(PLOTS_DIR, model_name + ".png")

    _apply_research_style(dpi=300)
    fig = plt.figure(figsize=(10, 6))
    plt.title("Training Progress", fontsize=16)
    plt.plot(train_loss, label="Training Loss", linewidth=2)
    plt.plot(test_loss, label="Test Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (RMSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    fig.savefig(path_fig, bbox_inches="tight", dpi=300)
    print(f"Saved training loss plot to: {path_fig}")
    return fig


def create_velocity_animation(
    cfg: Any,
    start_step: int = 0,
    num_steps: int = 500,
    single_step: bool = True,
    use_test_traj: bool = False,
    fps: int = 10,
    dpi: int = 100,
) -> str:
    """
    Create animated GIF of velocity field predictions over time.

    Args:
        cfg: Hydra configuration
        start_step: Starting timestep for animation
        num_steps: Number of timesteps to animate
        single_step: If True, use ground truth inputs (no error accumulation).
                    If False, use rollout predictions (with error accumulation)
        use_test_traj: If True, use test trajectory; else use training trajectory
        fps: Frames per second for output GIF
        dpi: Resolution for output GIF

    Returns:
        Path to saved animation file
    """
    print(
        f"Generating velocity field animation ({start_step}-{start_step + num_steps})..."
    )

    # Generate prediction data
    data_pred, data_true, data_error = _load_predictions_for_plotting(
        cfg, (start_step, start_step + num_steps), single_step, use_test_traj
    )

    # Optimize frame rate for reasonable file size
    actual_steps = len(data_true)
    skip = _calculate_frame_skip(actual_steps)
    num_frames = actual_steps // skip

    print(f"Creating {num_frames} frames from {actual_steps} timesteps")

    # Calculate plot bounds from middle timestep for stability
    bounds = _calculate_plot_bounds(data_true, data_error, start_step, actual_steps)

    _apply_research_style(dpi=dpi)
    # Create figure with three panels: ground truth, prediction, error
    fig, axes = plt.subplots(3, 1, figsize=(20, 16))

    def animate_frame(frame_num: int) -> tuple:
        """Animation function called for each frame."""
        step = (frame_num * skip) % actual_steps

        for panel_idx, ax in enumerate(axes):
            ax.clear()
            ax.set_aspect("equal")
            ax.set_axis_off()

            # Get mesh geometry
            pos = data_true[step].mesh_pos
            faces = data_true[step].cells
            triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)

            # Select data and styling based on panel
            if panel_idx == 0:
                velocity = data_true[step].x[:, 0:2]
                title = "Ground Truth"
                vmin, vmax = bounds["velocity"]
            elif panel_idx == 1:
                velocity = data_pred[step].x[:, 0:2]
                title = "Prediction"
                vmin, vmax = bounds["velocity"]
            else:
                velocity = data_error[step].x[:, 0:2]
                title = "Error (Prediction - Ground Truth)"
                vmin, vmax = bounds["error"]

            # Create mesh plot
            mesh_plot = ax.tripcolor(
                triang, velocity[:, 0], vmin=vmin, vmax=vmax, shading="flat"
            )
            ax.triplot(triang, "ko-", ms=0.5, lw=0.3)
            ax.set_title(f"{title}: Step {step}", fontsize=20)

            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            clb = fig.colorbar(mesh_plot, cax=cax, orientation="vertical")
            clb.ax.tick_params(labelsize=20)
            if panel_idx == 0:
                clb.ax.set_title("x velocity (m/s)", fontdict={"fontsize": 20})

        # Optimize layout
        fig.subplots_adjust(
            left=0.001, right=0.95, bottom=0.01, top=0.95, wspace=0.2, hspace=0.2
        )
        return (fig,)

    # Create and save animation
    anim = animation.FuncAnimation(
        fig, animate_frame, frames=num_frames, interval=1000 // fps
    )

    # Generate output filename
    model_name = name_from_config(cfg)
    filename_parts = ["x_velocity"]
    if not single_step:
        filename_parts.append("rollout")
    if use_test_traj:
        filename_parts.append("testtraj")
    filename_parts.extend(
        [f"{start_step}_{start_step + num_steps}", model_name, "anim.gif"]
    )

    ensure_directory(ANIM_DIR)
    anim_path = os.path.join(ANIM_DIR, "_".join(filename_parts))

    # Save with appropriate writer
    writer = animation.PillowWriter(fps=fps)
    anim.save(anim_path, writer=writer, dpi=dpi)

    print(f"Saved velocity animation to: {anim_path}")
    return anim_path


def plot_rollout_error(
    cfg: Any, start_step: int = 50, num_steps: int = 100, use_test_traj: bool = False
) -> plt.Figure:
    """
    Plot velocity RMSE over time during rollout prediction.

    Computes autoregressive predictions (where model predictions become inputs
    for next timestep) and plots the accumulation of error over time.

    Args:
        cfg: Hydra configuration
        start_step: Starting timestep for rollout window
        num_steps: Number of timesteps to analyze
        use_test_traj: Whether to use test trajectory

    Returns:
        matplotlib Figure object
    """
    print(
        f"Computing rollout error analysis ({start_step}-{start_step + num_steps})..."
    )

    # Get rollout predictions
    data_pred, data_true, data_error = _load_predictions_for_plotting(
        cfg,
        (start_step, start_step + num_steps),
        single_step=False,
        use_test_traj=use_test_traj,
    )

    # Calculate RMSE per timestep
    velocity_rmse = _calculate_timestep_rmse(data_true, data_error)

    # Create plot
    model_name = name_from_config(cfg)
    ensure_directory(PLOTS_DIR)
    path_fig = os.path.join(
        PLOTS_DIR, f"{model_name}_{start_step}_{start_step + num_steps}_rollout.png"
    )

    _apply_research_style(dpi=300)
    fig = plt.figure(figsize=(10, 6))
    plt.title("Rollout Error Analysis", fontsize=16)
    plt.plot(velocity_rmse, label=model_name, linewidth=2)
    plt.xlabel("Timestep")
    plt.ylabel("Velocity RMSE")
    plt.legend()
    plt.grid(True, alpha=0.3)

    fig.savefig(path_fig, bbox_inches="tight", dpi=300)
    print(f"Saved rollout error plot to: {path_fig}")
    return fig


# =============================================================================
# ADVANCED ANALYSIS FUNCTIONS
# =============================================================================


def compare_noise_effects(
    base_config: List[str] = None,
    start_step: int = 50,
    num_steps: int = 100,
    use_test_traj: bool = False,
    output_name: str = None,
) -> plt.Figure:
    """
    Compare rollout RMSE with and without training noise.

    Creates side-by-side comparison of models trained with different noise settings
    to visualize the impact of noise regularization on rollout stability.

    Args:
        base_config: Base configuration overrides (e.g., ["+datasize=medium"])
        start_step: Starting timestep for analysis
        num_steps: Number of timesteps to analyze
        use_test_traj: Whether to use test trajectory
        output_name: Custom output filename (auto-generated if None)

    Returns:
        matplotlib Figure object
    """
    if base_config is None:
        base_config = ["+datasize=medium"]

    fig = plt.figure(figsize=(12, 8))
    plt.title("Training Noise Effect on Rollout Stability", fontsize=16)
    plt.xlabel("Timestep")
    plt.ylabel("Velocity RMSE")

    configs_to_test = [
        (base_config + ["+noise=paper"], "With Noise"),
        (base_config, "Without Noise"),
    ]

    plot_data = []

    for config_overrides, label in configs_to_test:
        try:
            # Load configuration
            hydra.core.global_hydra.GlobalHydra.instance().clear()
            with hydra.initialize(
                version_base=None, config_path="../configs", job_name="plot"
            ):
                cfg = hydra.compose(config_name="default", overrides=config_overrides)

            model_name = name_from_config(cfg)
            print(f"Analyzing model: {model_name} ({label})")

            # Get predictions and calculate RMSE
            data_pred, data_true, data_error = _load_predictions_for_plotting(
                cfg,
                (start_step, start_step + num_steps),
                single_step=False,
                use_test_traj=use_test_traj,
            )

            velocity_rmse = _calculate_timestep_rmse(data_true, data_error)

            plt.plot(velocity_rmse, label=f"{label} ({model_name})", linewidth=2)
            plot_data.append((label, model_name, velocity_rmse))

        except Exception as e:
            print(f"Error processing {label}: {e}")
            continue

    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    if output_name is None:
        output_name = f"noise_comparison_{start_step}_{start_step + num_steps}"

    ensure_directory(PLOTS_DIR)
    path_fig = os.path.join(PLOTS_DIR, f"{output_name}.png")
    fig.savefig(path_fig, bbox_inches="tight", dpi=300)

    print(f"Saved noise comparison plot to: {path_fig}")
    return fig


# =============================================================================
# HYDRA INTEGRATION FUNCTIONS
# =============================================================================


def create_standard_plots(cfg: DictConfig) -> None:
    """Create standard training loss and rollout error plots from config."""
    plot_training_loss(cfg)
    plot_rollout_error(cfg)


def create_standard_animations(cfg: DictConfig) -> None:
    """Create standard animation suite from config."""
    animation_configs = [
        {"start_step": 0, "num_steps": 500, "use_test_traj": True},
        {"start_step": 0, "num_steps": 500, "use_test_traj": False},
        {"start_step": 50, "num_steps": 500, "use_test_traj": True},
        {"start_step": 50, "num_steps": 500, "use_test_traj": False},
    ]

    for config in animation_configs:
        create_velocity_animation(cfg, single_step=True, **config)


# =============================================================================
# STATIC MESH FRAME PLOTS (FROM ANIMATION LOGIC)
# =============================================================================


def create_velocity_frames(
    cfg: Any,
    start_step: int = 0,
    num_steps: int = 100,
    stride: int = 10,
    single_step: bool = True,
    use_test_traj: bool = False,
    dpi: int = 200,
    # Visual style controls
    layout: str = "horizontal",  # "horizontal" (1x3) or "vertical" (3x1)
    compact: bool = True,
    cmap_vel: str = "cividis",
    cmap_err: str = "coolwarm",
    edge_alpha: float = 0.15,
    fmt: str = "png",
    velocity_component: int = 0,  # 0 -> x, 1 -> y
) -> list:
    """
    Create and save static mesh plots (frames) for selected timesteps, mirroring
    the three-panel animation layout (Ground Truth, Prediction, Error).

    Args:
        cfg: Hydra configuration
        start_step: starting timestep (inclusive)
        num_steps: number of timesteps to consider
        stride: save every k-th step for speed (>=1)
        single_step: if True, use ground truth inputs; else rollout predictions
        use_test_traj: whether to read from test trajectory file
        dpi: image resolution

    Returns:
        List of saved file paths.
    """
    end_step = start_step + num_steps
    print(
        f"Generating static velocity frames ({start_step}-{end_step}, stride={stride})..."
    )

    # Generate prediction data window
    data_pred, data_true, data_error = _load_predictions_for_plotting(
        cfg, (start_step, end_step), single_step, use_test_traj
    )

    actual_steps = len(data_true)
    if actual_steps == 0:
        print("No timesteps available for the requested window.")
        return []

    # Compute consistent bounds across frames
    bounds = _calculate_plot_bounds(
        data_true, data_error, start_step, actual_steps, velocity_component
    )

    # Output directory for frames
    ensure_directory(PLOTS_DIR)
    frames_dir = os.path.join(PLOTS_DIR, "frames")
    ensure_directory(frames_dir)

    model_name = name_from_config(cfg)
    tag_parts = []
    if not single_step:
        tag_parts.append("rollout")
    if use_test_traj:
        tag_parts.append("testtraj")
    tag = ("_".join(tag_parts) + "_") if tag_parts else ""

    saved = []
    for local_idx in range(0, actual_steps, max(1, stride)):
        step = start_step + local_idx

        # Figure layout
        if layout.lower().startswith("h"):
            nrows, ncols, figsize = 1, 3, (15.6, 4.8)
        else:
            nrows, ncols, figsize = 3, 1, (11.6, 13.5)
        _apply_research_style(dpi=dpi)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()

        fig.set_facecolor("white")
        for panel_idx, ax in enumerate(axes):
            ax.set_aspect("equal")
            ax.set_axis_off()

            pos = data_true[local_idx].mesh_pos
            faces = data_true[local_idx].cells
            triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)

            if panel_idx == 0:
                velocity = data_true[local_idx].x[:, 0:2]
                title = "Ground Truth"
                vmin, vmax = bounds["velocity"]
            elif panel_idx == 1:
                velocity = data_pred[local_idx].x[:, 0:2]
                title = "Prediction"
                vmin, vmax = bounds["velocity"]
            else:
                velocity = data_error[local_idx].x[:, 0:2]
                title = "Error (Prediction - Ground Truth)"
                vmin, vmax = bounds["error"]

            # Choose component and color normalization
            values = velocity[:, velocity_component]
            if panel_idx == 2:
                norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
            else:
                norm = None
            if norm is not None:
                mesh_plot = ax.tripcolor(
                    triang,
                    values,
                    shading="gouraud" if compact else "flat",
                    cmap=(cmap_vel if panel_idx in (0, 1) else cmap_err),
                    norm=norm,
                )
            else:
                mesh_plot = ax.tripcolor(
                    triang,
                    values,
                    vmin=vmin,
                    vmax=vmax,
                    shading="gouraud" if compact else "flat",
                    cmap=(cmap_vel if panel_idx in (0, 1) else cmap_err),
                )
            # Subtle mesh edges for context
            ax.triplot(triang, color="k", ms=0.0, lw=0.25, alpha=edge_alpha)
            # Compact, research-style titles
            ax.set_title(f"{title}: Step {step}", fontsize=14, pad=6)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.025)
            clb = fig.colorbar(mesh_plot, cax=cax, orientation="vertical")
            clb.ax.tick_params(labelsize=10)
            if panel_idx == 0:
                label = "x vel (m/s)" if velocity_component == 0 else "y vel (m/s)"
                clb.ax.set_title(label, fontdict={"fontsize": 10})

        if compact:
            fig.subplots_adjust(
                left=0.02, right=0.98, bottom=0.02, top=0.92, wspace=0.1, hspace=0.08
            )
        else:
            fig.subplots_adjust(
                left=0.04, right=0.96, bottom=0.04, top=0.95, wspace=0.2, hspace=0.2
            )

        filename = f"{model_name}_{tag}{start_step}_{end_step}_step{step}.{fmt}"
        out_path = os.path.join(frames_dir, filename)
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        saved.append(out_path)

    print(f"Saved {len(saved)} frames to: {frames_dir}")
    return saved


# =============================================================================
# MESH TOPOLOGY PLOT (HIGH-CONTRAST)
# =============================================================================


def create_mesh_topology_plot(
    cfg: Any,
    step: int = 0,
    use_test_traj: bool = False,
    dpi: int = 300,
    figsize: tuple = (12, 3.2),
    edge_color: str = "0.1",
    edge_lw: float = 0.4,
    show_fluid_points: bool = False,
) -> str:
    """
    Create a single, high-contrast figure showing only the mesh topology and
    boundary/obstacle nodes. Designed for paper-quality inclusion.

    - Triangular faces are drawn with dark edges on white background
    - Boundary-type nodes (non-fluid) are highlighted with contrasting colors
    - Optionally show fluid nodes as light, semi-transparent dots

    Returns: path to saved PNG.
    """
    # Load a single timestep window [step, step+1)
    data_pred, data_true, _ = _load_predictions_for_plotting(
        cfg, (step, step + 1), single_step=True, use_test_traj=use_test_traj
    )
    data = data_true[0]

    pos = data.mesh_pos
    faces = data.cells
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)

    # Node-type IDs
    type_ids = torch.argmax(data.x[:, 2:], dim=1).cpu().numpy()
    unique = set(type_ids.tolist())
    # Heuristic defaults based on prior usage (normal=0, inflow=4, outflow=5, wall=3 if present)
    normal_id = int(np.bincount(type_ids).argmax())
    inflow_id = 4 if 4 in unique else None
    outflow_id = 5 if 5 in unique else None
    wall_id = 3 if 3 in unique else None

    # Fluid mask: normal and outflow if available
    fluid_mask = type_ids == normal_id
    if outflow_id is not None:
        fluid_mask = np.logical_or(fluid_mask, type_ids == outflow_id)

    # Boundary/others
    boundary_mask = ~fluid_mask
    wall_mask = (type_ids == wall_id) if wall_id is not None else boundary_mask
    inflow_mask = (
        (type_ids == inflow_id)
        if inflow_id is not None
        else np.zeros_like(type_ids, dtype=bool)
    )
    outflow_mask = (
        (type_ids == outflow_id)
        if outflow_id is not None
        else np.zeros_like(type_ids, dtype=bool)
    )

    # Figure
    _apply_research_style(dpi=dpi)
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Draw mesh edges only (no face fill)
    ax.triplot(triang, color=edge_color, lw=edge_lw)

    # Highlight node categories (order: wall -> inflow -> outflow -> fluid [optional])
    if wall_mask.any():
        ax.scatter(
            pos[wall_mask, 0],
            pos[wall_mask, 1],
            s=25,  # Increased from 8 to 25 for better paper visibility
            c="#E24A33",
            edgecolors="white",
            linewidths=0.8,  # Increased edge width for better contrast
            zorder=3,
            label="wall/obstacle",
        )
    if inflow_mask.any():
        ax.scatter(
            pos[inflow_mask, 0],
            pos[inflow_mask, 1],
            s=6,
            c="#348ABD",
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
            label="inflow",
        )
    if outflow_mask.any():
        ax.scatter(
            pos[outflow_mask, 0],
            pos[outflow_mask, 1],
            s=6,
            c="#988ED5",
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
            label="outflow",
        )
    if show_fluid_points:
        fluid_idx = np.where(fluid_mask)[0]
        ax.scatter(
            pos[fluid_idx, 0],
            pos[fluid_idx, 1],
            s=2,
            c="0.6",
            alpha=0.25,
            edgecolors="none",
            zorder=2,
            label="fluid",
        )

    # Compose output path
    ensure_directory(PLOTS_DIR)
    out_dir = os.path.join(PLOTS_DIR, "frames")
    ensure_directory(out_dir)
    model_name = name_from_config(cfg)
    tag = "testtraj_" if use_test_traj else ""
    out_path = os.path.join(out_dir, f"{model_name}_{tag}mesh_topology_step{step}.png")

    # Tight layout and save
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0, dpi=dpi)
    plt.close(fig)

    print(f"Saved mesh topology plot to: {out_path}")
    return out_path


# =============================================================================
# CONFORMAL PREDICTION INTERFACE
# =============================================================================


def create_mesh_flow_plot(
    cfg: Any,
    step: int = 0,
    use_test_traj: bool = False,
    dpi: int = 300,
    figsize: tuple = (12, 3.2),
    edge_color: str = "0.15",
    edge_lw: float = 0.3,
    cmap: str = "cividis",
    show_quiver: bool = True,
    quiver_stride: int = 30,
    quiver_scale: float = 50.0,
) -> str:
    """
    High-contrast mesh visualization with fluid colored by instantaneous speed
    (or component) and optional quiver arrows to suggest local flow direction.
    Boundary categories are highlighted as in `create_mesh_topology_plot`.
    """
    # Load single step
    data_pred, data_true, _ = _load_predictions_for_plotting(
        cfg, (step, step + 1), single_step=True, use_test_traj=use_test_traj
    )
    data = data_true[0]

    pos = data.mesh_pos
    faces = data.cells
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)

    vel = data.x[:, 0:2]
    speed = torch.sqrt((vel**2).sum(dim=1)).cpu().numpy()

    # Node-type masks
    type_ids = torch.argmax(data.x[:, 2:], dim=1).cpu().numpy()
    unique = set(type_ids.tolist())
    normal_id = int(np.bincount(type_ids).argmax())
    inflow_id = 4 if 4 in unique else None
    outflow_id = 5 if 5 in unique else None
    wall_id = 3 if 3 in unique else None

    fluid_mask = type_ids == normal_id
    if outflow_id is not None:
        fluid_mask = np.logical_or(fluid_mask, type_ids == outflow_id)
    wall_mask = (type_ids == wall_id) if wall_id is not None else ~fluid_mask
    inflow_mask = (
        (type_ids == inflow_id)
        if inflow_id is not None
        else np.zeros_like(type_ids, dtype=bool)
    )
    outflow_mask = (
        (type_ids == outflow_id)
        if outflow_id is not None
        else np.zeros_like(type_ids, dtype=bool)
    )

    # Figure
    _apply_research_style(dpi=dpi)
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Face coloring by speed with subtle edges
    tpc = ax.tripcolor(
        triang,
        speed,
        shading="gouraud",
        cmap=cmap,
        edgecolors=edge_color,
        linewidth=edge_lw,
    )

    # Overlay boundary nodes
    if wall_mask.any():
        ax.scatter(
            pos[wall_mask, 0],
            pos[wall_mask, 1],
            s=25,  # Increased from 8 to 25 for better paper visibility
            c="#E24A33",
            edgecolors="white",
            linewidths=0.8,  # Increased edge width for better contrast
            zorder=3,
            label="wall/obstacle",
        )
    if inflow_mask.any():
        ax.scatter(
            pos[inflow_mask, 0],
            pos[inflow_mask, 1],
            s=6,
            c="#348ABD",
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
            label="inflow",
        )
    if outflow_mask.any():
        ax.scatter(
            pos[outflow_mask, 0],
            pos[outflow_mask, 1],
            s=6,
            c="#988ED5",
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
            label="outflow",
        )

    # Optional quiver arrows (downsample fluid nodes)
    if show_quiver:
        idx = np.where(fluid_mask)[0]
        if len(idx) > 0:
            idx = idx[:: max(1, quiver_stride)]
            ax.quiver(
                pos[idx, 0],
                pos[idx, 1],
                vel[idx, 0].cpu().numpy(),
                vel[idx, 1].cpu().numpy(),
                color="k",
                alpha=0.6,
                scale=quiver_scale,
                width=0.0025,
                zorder=4,
            )

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1.6%", pad=0.025)
    clb = fig.colorbar(tpc, cax=cax)
    clb.ax.tick_params(labelsize=8)
    clb.ax.set_title("|v| (m/s)", fontsize=8)

    ensure_directory(PLOTS_DIR)
    out_dir = os.path.join(PLOTS_DIR, "frames")
    ensure_directory(out_dir)
    model_name = name_from_config(cfg)
    tag = "testtraj_" if use_test_traj else ""
    out_path = os.path.join(out_dir, f"{model_name}_{tag}mesh_flow_step{step}.png")

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0, dpi=dpi)
    plt.close(fig)

    print(f"Saved mesh flow plot to: {out_path}")
    return out_path


def create_mesh_streamplot(
    cfg: Any,
    step: int = 0,
    use_test_traj: bool = False,
    field: str = "truth",  # "truth" or "pred"
    dpi: int = 300,
    figsize: tuple = (12, 3.2),
    grid_nx: int = 500,
    grid_ny: int = 120,
    density: float = 1.2,
    cmap: str = "cividis",
    lw_min: float = 0.3,
    lw_max: float = 2.0,
) -> str:
    """
    Mathematically precise streamlines from the unstructured mesh using
    linear finite-element interpolation over the triangulation.

    - Interpolation: matplotlib.tri.LinearTriInterpolator on the original
      unstructured mesh (no smoothing). This preserves vector directions.
    - Streamlines: matplotlib.streamplot on a regular grid inside the domain.
    - Coloring: by speed |v| with a perceptually uniform colormap.
    """
    # Load single time step
    data_pred, data_true, _ = _load_predictions_for_plotting(
        cfg, (step, step + 1), single_step=True, use_test_traj=use_test_traj
    )
    data = data_true[0] if field == "truth" else data_pred[0]

    pos = data.mesh_pos
    faces = data.cells
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)

    # Velocity at nodes (exact for that step)
    vel = data.x[:, 0:2].cpu().numpy()
    u, v = vel[:, 0], vel[:, 1]

    # Interpolators over the triangulated domain
    Iu = LinearTriInterpolator(triang, u)
    Iv = LinearTriInterpolator(triang, v)

    # Regular grid spanning the mesh bounds
    xmin, xmax = pos[:, 0].min().item(), pos[:, 0].max().item()
    ymin, ymax = pos[:, 1].min().item(), pos[:, 1].max().item()
    X = np.linspace(xmin, xmax, grid_nx)
    Y = np.linspace(ymin, ymax, grid_ny)
    XX, YY = np.meshgrid(X, Y)

    UU = Iu(XX, YY)
    VV = Iv(XX, YY)

    # Mask points outside the triangulation
    mask = np.ma.getmask(UU) | np.ma.getmask(VV)
    UU = np.ma.array(UU, mask=mask)
    VV = np.ma.array(VV, mask=mask)

    speed = np.sqrt(UU**2 + VV**2)
    # Normalize for linewidths
    smin, smax = float(speed.min()), float(speed.max())
    if np.isfinite(smin) and np.isfinite(smax) and smax > 0:
        lw = lw_min + (lw_max - lw_min) * ((speed - smin) / (smax - smin))
    else:
        lw = lw_min

    # Figure
    _apply_research_style(dpi=dpi)
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Streamlines colored by speed
    strm = ax.streamplot(
        X,
        Y,
        UU,
        VV,
        color=speed,
        linewidth=lw,
        cmap=cmap,
        density=density,
        minlength=0.1,
        arrowsize=0.7,
    )

    # Overlay a light mesh edge for context
    ax.triplot(triang, color="0.85", lw=0.3, alpha=0.6, zorder=1)

    # Boundary categories for orientation (small markers)
    type_ids = torch.argmax(data.x[:, 2:], dim=1).cpu().numpy()
    unique = set(type_ids.tolist())
    wall_id = 3 if 3 in unique else None
    if wall_id is not None:
        wmask = type_ids == wall_id
        if wmask.any():
            ax.scatter(
                pos[wmask, 0],
                pos[wmask, 1],
                s=20,  # Increased from 6 to 20 for better paper visibility
                c="#E24A33",
                edgecolors="white",
                linewidths=0.6,  # Increased edge width for better contrast
                zorder=3,
            )

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1.6%", pad=0.025)
    clb = fig.colorbar(strm.lines, cax=cax)
    clb.ax.tick_params(labelsize=8)
    clb.ax.set_title("|v| (m/s)", fontsize=8)

    # Save
    ensure_directory(PLOTS_DIR)
    out_dir = os.path.join(PLOTS_DIR, "frames")
    ensure_directory(out_dir)
    model_name = name_from_config(cfg)
    tag = "testtraj_" if use_test_traj else ""
    out_path = os.path.join(out_dir, f"{model_name}_{tag}stream_{field}_step{step}.png")

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0, dpi=dpi)
    plt.close(fig)

    print(f"Saved mesh streamplot to: {out_path}")
    return out_path


def extract_predictions_for_conformal(
    cfg: Any,
    steps: Tuple[int, int] = (0, 500),
    use_test_traj: bool = False,
    return_metadata: bool = True,
) -> dict:
    """
    Extract raw model predictions for conformal prediction analysis.

    This function provides a clean interface for conformal prediction modules
    to consume MeshGraphNet predictions without visualization dependencies.

    Args:
        cfg: Hydra configuration
        steps: (start_step, end_step) tuple for prediction window
        use_test_traj: Whether to use test trajectory
        return_metadata: Whether to include mesh geometry and node types

    Returns:
        Dictionary containing:
        - 'predictions': Raw acceleration predictions [num_steps, num_nodes, 2]
        - 'ground_truth': Ground truth accelerations [num_steps, num_nodes, 2]
        - 'velocities': Input velocities [num_steps, num_nodes, 2]
        - 'metadata': Optional mesh info (positions, node types, etc.)
    """
    print(f"Extracting predictions for conformal analysis ({steps[0]}-{steps[1]})...")

    # Load model and generate predictions
    data_pred, data_true, data_error = _load_predictions_for_plotting(
        cfg, steps, single_step=True, use_test_traj=use_test_traj
    )

    num_steps = len(data_pred)
    num_nodes = data_pred[0].x.shape[0]

    # Determine velocity dimensions based on dataset type (detect from data)
    if data_pred[0].x.shape[1] == 12 and data_pred[0].y.shape[1] == 3:
        vel_dim = 3  # 3D velocities for flag
        dataset_type = "flag_simple"
    else:
        vel_dim = 2  # 2D velocities for cylinder
        dataset_type = "cylinder_flow"

    # Extract raw acceleration predictions (before velocity integration)
    # These are the direct model outputs that conformal prediction needs
    predictions = torch.zeros(num_steps, num_nodes, vel_dim)
    ground_truth = torch.zeros(num_steps, num_nodes, vel_dim)
    velocities = torch.zeros(num_steps, num_nodes, vel_dim)

    for t in range(num_steps):
        # Store input velocities (model inputs)
        velocities[t] = data_pred[t].x[:, 0:vel_dim]

        # Extract raw acceleration predictions (model outputs before integration)
        # We need to reverse the integration: pred_accel = (pred_vel - input_vel) / dt
        pred_vel = data_pred[t].x[:, 0:vel_dim]
        input_vel = data_true[t].x[:, 0:vel_dim] - data_true[t].y * DELTA_T
        predictions[t] = (pred_vel - input_vel) / DELTA_T

        # Ground truth accelerations
        ground_truth[t] = data_true[t].y

    result = {
        "predictions": predictions.numpy(),  # Convert to numpy for conformal
        "ground_truth": ground_truth.numpy(),
        "velocities": velocities.numpy(),
        "num_steps": num_steps,
        "num_nodes": num_nodes,
        "delta_t": DELTA_T,
        "model_name": name_from_config(cfg),
    }

    if return_metadata:
        # Include mesh geometry and node type information
        metadata = {
            "mesh_positions": data_pred[0].mesh_pos.numpy(),
            "node_types": torch.argmax(data_pred[0].x[:, 2:], dim=1).numpy(),
            "fluid_mask": _create_fluid_mask(data_pred[0]).numpy(),
            "cells": (
                data_pred[0].cells.numpy() if hasattr(data_pred[0], "cells") else None
            ),
            "edge_index": (
                data_pred[0].edge_index.numpy()
                if hasattr(data_pred[0], "edge_index")
                else None
            ),
        }
        result["metadata"] = metadata

    print(f"Extracted {num_steps} timesteps × {num_nodes} nodes for conformal analysis")
    return result


def batch_extract_predictions(
    cfg: Any,
    batch_steps: List[Tuple[int, int]],
    use_test_traj: bool = False,
    return_metadata: bool = True,
) -> List[dict]:
    """
    Extract predictions for multiple time windows (batch processing).

    Useful for conformal prediction when you need predictions across
    different time ranges or multiple trajectories.

    Args:
        cfg: Hydra configuration
        batch_steps: List of (start_step, end_step) tuples
        use_test_traj: Whether to use test trajectory
        return_metadata: Whether to include metadata

    Returns:
        List of prediction dictionaries for each time window
    """
    results = []
    for start_step, end_step in batch_steps:
        try:
            result = extract_predictions_for_conformal(
                cfg, (start_step, end_step), use_test_traj, return_metadata
            )
            results.append(result)
        except Exception as e:
            print(f"Error extracting predictions for {start_step}-{end_step}: {e}")
            continue

    return results


def _create_fluid_mask(data: Any) -> torch.Tensor:
    """Create boolean mask for fluid nodes (dataset-specific)."""
    # Detect dataset type based on feature dimensions
    if data.x.shape[1] == 12 and data.y.shape[1] == 3:
        # Flag dataset: node types start at column 3, use type 0 (normal)
        normal = torch.tensor(0)
        return torch.argmax(data.x[:, 3:], dim=1) == normal
    else:
        # Cylinder dataset: node types start at column 2, use types 0 (normal) and 5 (outflow)
        normal, outflow = torch.tensor(0), torch.tensor(5)
        return torch.logical_or(
            torch.argmax(data.x[:, 2:], dim=1) == normal,
            torch.argmax(data.x[:, 2:], dim=1) == outflow,
        )


# =============================================================================
# INTERNAL HELPER FUNCTIONS
# =============================================================================


def _load_predictions_for_plotting(
    cfg: Any,
    steps: Tuple[int, int] = (50, 550),
    single_step: bool = True,
    use_test_traj: bool = False,
) -> Tuple[List, List, List]:
    """
    Load model and data to generate predictions for visualization.

    Returns three lists of PyG Data objects: predicted, ground-truth, and error,
    where `.x[:, 0:2]` holds the x/y velocity fields to visualize.
    """
    # Set seeds for reproducibility
    torch.manual_seed(cfg.rseed)
    random.seed(cfg.rseed)
    np.random.seed(cfg.rseed)

    # Use CPU for plotting stability
    device = torch.device("cpu")

    # Load dataset
    datapath = cfg.data.datapath
    if use_test_traj:
        datapath = datapath.replace("train", "test")

    dataset_path = os.path.join(DATASET_DIR, datapath)
    dataset = torch.load(dataset_path)[steps[0] : steps[1]]  # noqa: E203

    # Load trained model
    model_name = name_from_config(cfg)
    model_path = os.path.join(CHECKPOINT_DIR, model_name + "_model.pt")
    infos_path = os.path.join(CHECKPOINT_DIR, model_name + "_infos.pkl")

    with open(infos_path, "rb") as f:
        num_node_features, num_edge_features, num_classes, _ = pickle.load(f)

    # Recompute stats on plotting subset for consistency
    stats_list = get_stats(dataset)

    # Detect dataset type for proper model initialization
    temp_sample = dataset[0] if dataset else None
    if temp_sample is not None:
        if temp_sample.x.shape[1] == 12 and temp_sample.y.shape[1] == 3:
            dataset_type = "flag_simple"
        else:
            dataset_type = "cylinder_flow"
    else:
        dataset_type = "cylinder_flow"  # Default fallback

    # Initialize model with dataset type
    model = MeshGraphNet(
        num_node_features,
        num_edge_features,
        cfg.model.hidden_dim,
        num_classes,
        cfg,
        dataset_type,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Unpack normalization statistics
    mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y = [
        tensor.to(device) for tensor in stats_list
    ]

    # Generate predictions
    if single_step:
        return _generate_single_step_predictions(
            model,
            dataset,
            [
                mean_vec_x,
                std_vec_x,
                mean_vec_edge,
                std_vec_edge,
                mean_vec_y,
                std_vec_y,
            ],
            device,
        )
    else:
        return _generate_rollout_predictions(
            model,
            dataset,
            [
                mean_vec_x,
                std_vec_x,
                mean_vec_edge,
                std_vec_edge,
                mean_vec_y,
                std_vec_y,
            ],
            device,
        )


def _generate_single_step_predictions(
    model: MeshGraphNet,
    dataset: List,
    stats: List[torch.Tensor],
    device: str = "cpu",
) -> Tuple[List, List, List]:
    """Generate single-step predictions using ground truth inputs."""
    model.eval()
    data_pred = copy.deepcopy(dataset)
    data_true = copy.deepcopy(dataset)
    data_error = copy.deepcopy(dataset)

    # Unpack required statistics including output (y) stats for unnormalization
    mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y = stats

    for t in range(len(dataset)):
        dataset[t] = dataset[t].to(device)
        data_pred[t] = data_pred[t].to(device)

        with torch.no_grad():
            # Predict acceleration
            pred = model(dataset[t], mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)

            # Convert predicted acceleration from normalized units to physical units
            pred_phys = unnormalize(pred, mean_vec_y, std_vec_y)

            # Integrate to get velocity: new_vel = old_vel + acceleration * dt
            # Handle variable velocity dimensions (2D for cylinder, 3D for flag)
            vel_dim = pred_phys.shape[
                1
            ]  # Get actual velocity dimension from prediction
            data_pred[t].x[:, 0:vel_dim] = (
                dataset[t].x[:, 0:vel_dim] + pred_phys * DELTA_T
            )
            data_true[t].x[:, 0:vel_dim] = (
                dataset[t].x[:, 0:vel_dim] + dataset[t].y * DELTA_T
            )
            data_error[t].x[:, 0:vel_dim] = (
                data_pred[t].x[:, 0:vel_dim] - data_true[t].x[:, 0:vel_dim]
            )

    return data_pred, data_true, data_error


def _generate_rollout_predictions(
    model: MeshGraphNet,
    dataset: List,
    stats: List[torch.Tensor],
    device: str = "cpu",
) -> Tuple[List, List, List]:
    """Generate rollout predictions using previous predictions as inputs."""
    if len(dataset) > 599:
        print(
            "Warning: Dataset longer than single trajectory. Results may be incorrect past 599 steps."
        )

    model.eval()
    data_pred = copy.deepcopy(dataset)
    data_true = copy.deepcopy(dataset)
    data_error = copy.deepcopy(dataset)

    # Initialize with first timestep velocity
    last_velocity = copy.deepcopy(data_pred[0].x[:, 0:2]).to(device)

    # Unpack required statistics including output (y) stats for unnormalization
    mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y = stats

    for t in range(len(dataset)):
        with torch.no_grad():
            # Use previous prediction as input
            data_pred[t].x[:, 0:2] = last_velocity
            pred = model(
                data_pred[t], mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge
            )

            # Convert predicted acceleration from normalized units to physical units
            pred_phys = unnormalize(pred, mean_vec_y, std_vec_y)

            # Integrate prediction
            data_pred[t].x[:, 0:2] = data_pred[t].x[:, 0:2] + pred_phys * DELTA_T
            last_velocity = copy.deepcopy(data_pred[t].x[:, 0:2])

            # Compute ground truth
            data_true[t].x[:, 0:2] = dataset[t].x[:, 0:2] + dataset[t].y * DELTA_T
            data_error[t].x[:, 0:2] = data_pred[t].x[:, 0:2] - data_true[t].x[:, 0:2]

    return data_pred, data_true, data_error


def _calculate_timestep_rmse(data_true: List, data_error: List) -> List[float]:
    """Calculate velocity RMSE per timestep over fluid nodes only."""
    velocity_rmse = []
    normal, outflow = torch.tensor(0), torch.tensor(5)

    for t in range(len(data_true)):
        # Create mask for fluid nodes (normal and outflow types)
        loss_mask = torch.logical_or(
            torch.argmax(data_true[t].x[:, 2:], dim=1) == normal,
            torch.argmax(data_true[t].x[:, 2:], dim=1) == outflow,
        )

        # Calculate MSE over x and y velocity components
        error = torch.sum(data_error[t].x[:, :2] ** 2, axis=1)
        rmse = torch.sqrt(torch.mean(error[loss_mask]))
        velocity_rmse.append(rmse.item())

    return velocity_rmse


def _calculate_frame_skip(num_steps: int) -> int:
    """Calculate frame skip to keep animations under 50 frames for reasonable file size."""
    if num_steps <= 50:
        return 1
    elif num_steps == 500:
        return 10
    else:
        return int(math.ceil(num_steps / 50))


def _calculate_plot_bounds(
    data_true: List,
    data_error: List,
    start_step: int,
    num_steps: int,
    velocity_component: int = 0,
) -> dict:
    """Calculate plot bounds for consistent scaling across animation frames."""
    # Use a local index into the provided window; lists are 0-indexed within window
    # Clamp to valid range to avoid IndexError for short windows
    step_middle = max(0, min(len(data_true) - 1, int(num_steps / 2)))

    # Velocity bounds from ground truth
    velocity_min = data_true[step_middle].x[:, velocity_component].min().item()
    velocity_max = data_true[step_middle].x[:, velocity_component].max().item()

    # Error bounds from fluid nodes only
    normal, outflow = torch.tensor(0), torch.tensor(5)
    loss_mask = torch.logical_or(
        torch.argmax(data_true[0].x[:, 2:], dim=1) == normal,
        torch.argmax(data_true[0].x[:, 2:], dim=1) == outflow,
    )

    masked_error = data_error[step_middle].x[loss_mask]
    # Symmetric error bounds around 0 with mild shrink to avoid outlier domination
    err_vals = masked_error[:, velocity_component]
    emax = float(err_vals.abs().quantile(0.98))  # robust max
    error_min, error_max = -emax, emax

    return {
        "velocity": (velocity_min, velocity_max),
        "error": (error_min, error_max),
    }
