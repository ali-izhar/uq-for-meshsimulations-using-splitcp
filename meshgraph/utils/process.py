"""
This file is heavily based on:
https://colab.research.google.com/drive/1mZAWP6k9R0DE5NxPzF8yL2HpIUG3aoDC?usp=sharing


Data processing utilities for MeshGraphNet training and evaluation.

This module provides:
- Dataset normalization and statistics computation
- Directory path configuration
- Data loading and preprocessing utilities

Design principles:
- Device-aware tensor operations (CPU/GPU compatibility)
- Memory-efficient statistics computation with early stopping
- Robust numerical stability with epsilon clamping
- Single-pass statistics computation for large datasets
"""

import torch
import os
import pathlib
from typing import List

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================

# Compute the repository root relative to this file location
# This ensures the project works regardless of current working directory
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]

# Data directories used throughout the codebase
# Using strings for compatibility with existing os.path.* calls
DATASET_DIR = os.path.join(ROOT_DIR, "datasets")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(ROOT_DIR, "outputs", "plots")
ANIM_DIR = os.path.join(ROOT_DIR, "outputs", "animations")

# Physical constants from the original CFD dataset
DELTA_T = 0.01  # Simulation timestep size in seconds

# =============================================================================
# NORMALIZATION UTILITIES
# =============================================================================


def normalize(
    to_normalize: torch.Tensor, mean_vec: torch.Tensor, std_vec: torch.Tensor
) -> torch.Tensor:
    """
    Normalize tensor using element-wise mean and standard deviation.

    Applies z-score normalization: (x - μ) / σ

    Args:
        to_normalize: Input tensor of shape [..., F]
        mean_vec: Per-feature means of shape [F]
        std_vec: Per-feature standard deviations of shape [F]

    Returns:
        Normalized tensor of same shape as input

    Example:
        >>> features = torch.randn(100, 5)  # 100 samples, 5 features
        >>> mean = features.mean(dim=0)     # [5]
        >>> std = features.std(dim=0)       # [5]
        >>> normalized = normalize(features, mean, std)
    """
    return (to_normalize - mean_vec) / std_vec


def unnormalize(
    to_unnormalize: torch.Tensor, mean_vec: torch.Tensor, std_vec: torch.Tensor
) -> torch.Tensor:
    """
    Reverse normalization to original scale.

    Applies inverse z-score: σ * x + μ

    Args:
        to_unnormalize: Normalized tensor of shape [..., F]
        mean_vec: Per-feature means of shape [F]
        std_vec: Per-feature standard deviations of shape [F]

    Returns:
        Unnormalized tensor of same shape as input
    """
    return to_unnormalize * std_vec + mean_vec


# =============================================================================
# DATASET STATISTICS COMPUTATION
# =============================================================================


def get_stats(data_list: List) -> List[torch.Tensor]:
    """
    Compute comprehensive normalization statistics across a dataset.

    Efficiently computes per-feature mean and standard deviation for node features,
    edge features, and output targets using a single pass through the data with
    early stopping for memory protection.

    Args:
        data_list: List of PyG Data objects with attributes .x, .edge_attr, .y

    Returns:
        List of six tensors: [mean_x, std_x, mean_edge, std_edge, mean_y, std_y]
        All tensors are on the same device/dtype as the input data.

    Example:
        >>> dataset = [data1, data2, ...]  # List of PyG Data objects
        >>> stats = compute_dataset_statistics(dataset)
        >>> mean_x, std_x, mean_edge, std_edge, mean_y, std_y = stats
    """
    if not data_list:
        raise ValueError("Dataset list cannot be empty")

    # Initialize statistics tensors on same device/dtype as first data point
    first_data = data_list[0]
    device, dtype = first_data.x.device, first_data.x.dtype

    # Node feature statistics
    mean_x = torch.zeros(first_data.x.shape[1:], device=device, dtype=dtype)
    std_x = torch.zeros(first_data.x.shape[1:], device=device, dtype=dtype)

    # Edge feature statistics
    mean_edge = torch.zeros(first_data.edge_attr.shape[1:], device=device, dtype=dtype)
    std_edge = torch.zeros(first_data.edge_attr.shape[1:], device=device, dtype=dtype)

    # Output target statistics
    mean_y = torch.zeros(first_data.y.shape[1:], device=device, dtype=dtype)
    std_y = torch.zeros(first_data.y.shape[1:], device=device, dtype=dtype)

    # Numerical stability epsilon
    eps = torch.as_tensor(1e-8, device=device, dtype=dtype)

    # Memory protection: limit total accumulations to prevent OOM
    max_accumulations = 10**6

    # Accumulation counters
    num_nodes_total = 0
    num_edges_total = 0
    num_outputs_total = 0

    # First pass: compute sums and squared sums
    for data_point in data_list:
        # Node accumulation
        mean_x += torch.sum(data_point.x, dim=0)
        std_x += torch.sum(data_point.x**2, dim=0)
        num_nodes_total += data_point.x.shape[0]

        # Edge accumulation
        mean_edge += torch.sum(data_point.edge_attr, dim=0)
        std_edge += torch.sum(data_point.edge_attr**2, dim=0)
        num_edges_total += data_point.edge_attr.shape[0]

        # Output accumulation
        mean_y += torch.sum(data_point.y, dim=0)
        std_y += torch.sum(data_point.y**2, dim=0)
        num_outputs_total += data_point.y.shape[0]

        # Early exit if memory limit reached
        if any(
            [
                num_nodes_total > max_accumulations,
                num_edges_total > max_accumulations,
                num_outputs_total > max_accumulations,
            ]
        ):
            print(
                f"Warning: Early stopping statistics computation at {len(data_list)} samples"
            )
            break

    # Finalize node statistics: E[x], std = sqrt(E[x²] - E[x]²)
    mean_x = mean_x / num_nodes_total
    variance_x = std_x / num_nodes_total - mean_x**2
    std_x = torch.maximum(torch.sqrt(variance_x), eps)

    # Finalize edge statistics
    mean_edge = mean_edge / num_edges_total
    variance_edge = std_edge / num_edges_total - mean_edge**2
    std_edge = torch.maximum(torch.sqrt(variance_edge), eps)

    # Finalize output statistics
    mean_y = mean_y / num_outputs_total
    variance_y = std_y / num_outputs_total - mean_y**2
    std_y = torch.maximum(torch.sqrt(variance_y), eps)

    print(
        f"Computed statistics over {num_nodes_total} nodes, {num_edges_total} edges, {num_outputs_total} outputs"
    )

    return [mean_x, std_x, mean_edge, std_edge, mean_y, std_y]


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================


def load_dataset_split(
    dataset_path: str,
    train_size: int = None,
    test_size: int = None,
    train_test_same_traj: bool = True,
    single_traj: bool = True,
    seed: int = 0,
    dataset_type: str = "cylinder_flow",
) -> tuple:
    """
    Load and split dataset according to configuration.

    Args:
        dataset_path: Path to dataset file (.pt format)
        train_size: Number of training timesteps
        test_size: Number of test timesteps
        train_test_same_traj: If True, split from same trajectory
        single_traj: If True, sample from single trajectory only
        seed: Random seed for reproducible splits
        dataset_type: Type of dataset ('cylinder_flow' or 'flag_simple')

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    import random

    # Set seed for reproducible splits
    random.seed(seed)
    torch.manual_seed(seed)

    # Load full dataset
    full_dataset = torch.load(dataset_path)

    if train_test_same_traj and single_traj:
        # Sequential split from same trajectory
        train_dataset = full_dataset[:train_size] if train_size else full_dataset
        test_start = train_size if train_size else len(full_dataset)
        test_end = test_start + test_size if test_size else len(full_dataset)
        test_dataset = full_dataset[test_start:test_end]

    elif train_test_same_traj:
        # Random split from same trajectory pool
        random.shuffle(full_dataset)
        train_dataset = full_dataset[:train_size] if train_size else full_dataset
        test_start = train_size if train_size else len(full_dataset)
        test_end = test_start + test_size if test_size else len(full_dataset)
        test_dataset = full_dataset[test_start:test_end]

    else:
        # Different trajectories for train/test
        train_dataset = full_dataset[:train_size] if train_size else full_dataset

        # Load separate test trajectory
        test_path = dataset_path.replace("train", "test")
        if os.path.exists(test_path):
            test_full = torch.load(test_path)
            test_dataset = test_full[:test_size] if test_size else test_full
        else:
            raise FileNotFoundError(f"Test dataset not found at {test_path}")

    # Shuffle datasets for training
    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    print(f"Loaded {len(train_dataset)} training and {len(test_dataset)} test samples")
    return train_dataset, test_dataset


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def ensure_directory_exists(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def get_dataset_info(data_list: List) -> dict:
    """
    Extract metadata about a dataset.

    Args:
        data_list: List of PyG Data objects

    Returns:
        Dictionary with dataset information
    """
    if not data_list:
        return {"num_samples": 0}

    first_sample = data_list[0]

    return {
        "num_samples": len(data_list),
        "num_node_features": first_sample.x.shape[1],
        "num_edge_features": first_sample.edge_attr.shape[1],
        "num_output_features": first_sample.y.shape[1],
        "avg_nodes_per_sample": sum(data.x.shape[0] for data in data_list)
        / len(data_list),  # noqa: W503
        "avg_edges_per_sample": sum(data.edge_attr.shape[0] for data in data_list)
        / len(data_list),  # noqa: W503
        "device": str(first_sample.x.device),
        "dtype": str(first_sample.x.dtype),
    }


def detect_dataset_type(data_sample) -> str:
    """
    Automatically detect dataset type based on feature dimensions.

    Args:
        data_sample: A single PyG Data object from the dataset

    Returns:
        str: Dataset type ('cylinder_flow' or 'flag_simple')
    """
    node_feature_dim = data_sample.x.shape[1]
    target_dim = data_sample.y.shape[1]

    # Cylinder: 11 node features (2D vel + 7 node types + 2D pos), 2D targets
    if node_feature_dim == 11 and target_dim == 2:
        return "cylinder_flow"

    # Flag: 12 node features (3D vel + 9 node types), 3D targets
    elif node_feature_dim == 12 and target_dim == 3:
        return "flag_simple"

    else:
        # Default fallback
        print(
            f"Warning: Unknown dataset type with {node_feature_dim} node features and {target_dim} target dimensions"
        )
        return "unknown"


def get_dataset_specific_info(dataset_type: str) -> dict:
    """
    Get dataset-specific information for processing.

    Args:
        dataset_type: Type of dataset ('cylinder_flow' or 'flag_simple')

    Returns:
        Dictionary with dataset-specific parameters
    """
    if dataset_type == "cylinder_flow":
        return {
            "velocity_dim": 2,
            "node_type_start_idx": 2,
            "num_node_types": 7,
            "valid_node_types": [0, 5],  # normal and outflow
            "has_position_features": True,
            "position_start_idx": 9,
            "description": "2D cylinder flow simulation",
        }
    elif dataset_type == "flag_simple":
        return {
            "velocity_dim": 3,
            "node_type_start_idx": 3,
            "num_node_types": 9,
            "valid_node_types": [0],  # normal fluid nodes
            "has_position_features": False,
            "position_start_idx": None,
            "description": "3D flag simulation",
        }
    else:
        return {
            "velocity_dim": 2,
            "node_type_start_idx": 2,
            "num_node_types": 7,
            "valid_node_types": [0],
            "has_position_features": False,
            "position_start_idx": None,
            "description": "Unknown dataset type",
        }


def validate_dataset_consistency(data_list: List) -> bool:
    """
    Validate that all samples in dataset have consistent feature dimensions.

    Args:
        data_list: List of PyG Data objects to validate

    Returns:
        True if dataset is consistent, raises ValueError otherwise
    """
    if not data_list:
        return True

    first_sample = data_list[0]
    expected_node_features = first_sample.x.shape[1]
    expected_edge_features = first_sample.edge_attr.shape[1]
    expected_output_features = first_sample.y.shape[1]

    for i, data in enumerate(data_list):
        if data.x.shape[1] != expected_node_features:
            raise ValueError(
                f"Sample {i}: Expected {expected_node_features} node features, got {data.x.shape[1]}"
            )
        if data.edge_attr.shape[1] != expected_edge_features:
            raise ValueError(
                f"Sample {i}: Expected {expected_edge_features} edge features, got {data.edge_attr.shape[1]}"
            )
        if data.y.shape[1] != expected_output_features:
            raise ValueError(
                f"Sample {i}: Expected {expected_output_features} output features, got {data.y.shape[1]}"
            )

    return True
