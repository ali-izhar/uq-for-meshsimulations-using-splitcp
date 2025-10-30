# This file is heavily based on
# https://colab.research.google.com/drive/1mZAWP6k9R0DE5NxPzF8yL2HpIUG3aoDC?usp=sharing

"""
Train MeshGraphNet using Hydra configuration, evaluate periodically, and log.

This script composes the dataset, model, and training settings via `conf/` and
uses W&B for logging. It saves the best model checkpoint and loss curves.
"""

import torch
import os
import random
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import argparse
import sys

from torch_geometric.loader import DataLoader

import numpy as np
import torch.optim as optim
import tqdm
import copy
import pickle

from models.meshgraphnet import MeshGraphNet
from utils.process import (
    get_stats,
    unnormalize,
    CHECKPOINT_DIR,
    DATASET_DIR,
    DELTA_T,
    detect_dataset_type,
    get_dataset_specific_info,
)
from utils.plot import (  # noqa: F401
    name_from_config,
    plot_training_loss,
    create_velocity_animation,
    create_standard_plots,
    create_standard_animations,
)


# -----------------------------------------------------------------------------
# Performance and training utilities
# -----------------------------------------------------------------------------
def _build_dataloader_kwargs(cfg):
    """
    Construct GPU-optimized DataLoader keyword arguments for maximum throughput.

    - num_workers: optimized for GPU memory bandwidth
    - pin_memory: essential for fast GPU memory transfers
    - persistent_workers: keep workers alive across epochs
    - prefetch_factor: tuned for GPU processing speed
    """
    # GPU-optimized worker settings with better defaults for high-end hardware
    num_workers = getattr(cfg.training, "num_workers", min(os.cpu_count() or 8, 16))
    # Avoid None; ensure non-negative int
    if num_workers is None:
        num_workers = 0

    # High-end GPUs benefit greatly from pinned memory
    pin_memory = getattr(cfg.training, "pin_memory", cfg.device == "cuda")
    persistent_workers = getattr(cfg.training, "persistent_workers", True)

    # High-end GPUs can handle more prefetching due to faster processing
    prefetch_factor = getattr(cfg.training, "prefetch_factor", 4)

    # Advanced optimization settings
    use_amp = getattr(
        cfg.training, "use_amp", True
    )  # Enable by default for modern GPUs
    compile_model = getattr(
        cfg.training, "compile_model", False
    )  # PyTorch 2.0 compilation

    # Only valid when workers > 0; otherwise DataLoader will error
    if num_workers <= 0:
        persistent_workers = False
        prefetch_factor = None

    kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers),
    }
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def _seed_worker(worker_id):
    """
    Seed numpy and python random in each DataLoader worker for reproducibility.
    """
    # Derive a different seed for each worker
    worker_seed = torch.initial_seed() % 2**32
    # Initialize numpy RNG for potential numpy ops in workers
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # Optionally also seed torch here if you rely on torch ops in workers
    torch.manual_seed(worker_seed)


def add_noise(dataset, cfg):
    """Add noise to each timestep.

    noise_field: 'velocity' for cylinder_flow in the original codebase
    noise_scale: 0.02 for cylinder_flow in the original codebase
    noise_gamma: 1.0 for cylinder_flow in the original codebase

    Similar to split_and_preprocess() from
    https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets
    """
    # dataset_before = copy.deepcopy(dataset)
    for datapoint in dataset:
        # each datapoint is class torch_geometric.data.Data
        # add noise to velocity
        momentum = datapoint.x[:, :2]  # (tsteps, 2)
        node_type = datapoint.x[:, 2:]  # (tsteps, 7)
        # noise
        # Allocate on same device/dtype as momentum for efficiency
        noise = torch.empty_like(momentum).normal_(mean=0.0, std=cfg.data.noise_scale)
        # but don't apply noise to boundary nodes
        # Identify boundary nodes (mask where the first type-channel is 1)
        condition = node_type[:, 0] == torch.ones_like(node_type[:, 0])  # (tsteps)
        condition = condition.unsqueeze(1)  # (tsteps, 1)
        condition = condition.repeat(1, 2)  # (tsteps, 2)
        # noise (tsteps, 2)
        noise = torch.where(
            condition=condition, input=noise, other=torch.zeros_like(momentum)
        )
        momentum += noise  # In-place update OK, maintained in concatenation below
        datapoint.x = torch.cat((momentum, node_type), dim=-1).type(torch.float)
        datapoint.y += (1.0 - cfg.data.noise_gamma) * noise  # (tsteps, 2)
        # print('Still the same?', dataset_before[0].x == datapoint.x)
    return dataset


def train(data_train, data_test, stats_list, cfg, dataset_type="cylinder_flow"):
    """
    Performs a training loop on the dataset for MeshGraphNets. Also calls
    test and validation functions.

     Args:
         data_train (List[Data]): Training graphs.
         data_test (List[Data]): Test graphs.
         stats_list (List[Tensor]): Normalization stats in order
             [mean_x, std_x, mean_edge, std_edge, mean_y, std_y].
         cfg (DictConfig): Hydra configuration.
         dataset_type (str): Type of dataset ('cylinder_flow' or 'flag_simple').
    """

    # add noise to the training data
    if cfg.data.noise_scale > 0.0:
        data_train = add_noise(data_train, cfg)

    assert (
        len(data_train) > 0 and len(data_test) > 0
    ), f"Start training on {len(data_train)} train and {len(data_test)} test datapoints"

    # Configure DataLoader for maximum throughput while preserving correctness
    dl_kwargs = _build_dataloader_kwargs(cfg)

    # torch_geometric DataLoaders handle lists of graphs.
    # Data is already shuffled upstream if desired, so we keep shuffle=False here.
    loader = DataLoader(
        data_train,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        worker_init_fn=_seed_worker,
        **dl_kwargs,
    )
    test_loader = DataLoader(
        data_test,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        worker_init_fn=_seed_worker,
        **dl_kwargs,
    )

    # The statistics of the data are decomposed
    [
        mean_vec_x,
        std_vec_x,
        mean_vec_edge,
        std_vec_edge,
        mean_vec_y,
        std_vec_y,
    ] = stats_list
    (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y) = (
        mean_vec_x.to(cfg.device),
        std_vec_x.to(cfg.device),
        mean_vec_edge.to(cfg.device),
        std_vec_edge.to(cfg.device),
        mean_vec_y.to(cfg.device),
        std_vec_y.to(cfg.device),
    )

    # Define the model name for saving checkpoint
    model_name = name_from_config(cfg)
    path_model_checkpoint = os.path.join(CHECKPOINT_DIR, model_name + "_model.pt")
    path_infos = os.path.join(CHECKPOINT_DIR, model_name + "_infos.pkl")
    path_df = os.path.join(CHECKPOINT_DIR, model_name + "_losses.pkl")
    # saving model
    if not os.path.isdir(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    # look for checkpoint
    # if it exists, continue from previous checkpoint
    if os.path.exists(path_model_checkpoint) and cfg.training.resume_checkpoint:
        # get infos
        with open(path_infos, "rb") as f:
            infos = pickle.load(f)
        (num_node_features, num_edge_features, num_classes, stats_list) = infos

        # instantiate model
        model = MeshGraphNet(
            num_node_features,
            num_edge_features,
            cfg.model.hidden_dim,
            num_classes,
            cfg,
            dataset_type,
        ).to(cfg.device)
        model.load_state_dict(
            torch.load(path_model_checkpoint, map_location=cfg.device)
        )

        # load existing losses DataFrame
        if os.path.exists(path_df):
            df = pd.read_pickle(path_df)
            print("Continuing from previous checkpoint.")
            print(f"Loaded {len(df)} previous loss records.")
        else:
            # create new DataFrame if losses file doesn't exist
            df = pd.DataFrame(
                columns=["epoch", "train_loss", "test_loss", "velocity_val_loss"]
            )
            print("Continuing from previous checkpoint (no previous losses found).")

    else:
        # build model
        num_node_features = data_train[0].x.shape[1]
        num_edge_features = data_test[0].edge_attr.shape[1]
        # Dynamically determine output dimensions based on dataset type
        num_classes = data_train[0].y.shape[1]  # 2 for cylinder, 3 for flag

        # save data infos
        infos = (num_node_features, num_edge_features, num_classes, stats_list)

        model = MeshGraphNet(
            num_node_features,
            num_edge_features,
            cfg.model.hidden_dim,
            num_classes,
            cfg,
            dataset_type,
        ).to(cfg.device)

        # dataframe with losses
        df = pd.DataFrame(
            columns=["epoch", "train_loss", "test_loss", "velocity_val_loss"]
        )

        print("No previous checkpoint found. Starting training from scratch.")

    # Advanced optimizer configuration
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # Optional learning rate scheduler for longer training
    scheduler = None
    if hasattr(cfg.training, "use_scheduler") and cfg.training.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.lr * 0.01
        )

    # Enhanced mixed precision configuration for GPU optimization
    use_amp = (cfg.device == "cuda") and getattr(cfg.training, "use_amp", True)
    scaler_init_scale = getattr(cfg.training, "grad_scaler_init", 65536.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=scaler_init_scale)

    # PyTorch 2.0 model compilation for additional speedup
    compile_model_flag = getattr(cfg.training, "compile_model", False)
    if compile_model_flag and hasattr(torch, "compile"):
        print("Compiling model with PyTorch 2.0...")
        model = torch.compile(model, mode="max-autotune")

    # Gradient clipping for training stability
    gradient_clip_val = getattr(cfg.training, "gradient_clip_val", None)

    # GPU mixed precision - ensure we're using it optimally
    if use_amp and cfg.device == "cuda":
        print("Mixed Precision Training: Enabled (AMP)")
        print(f"Current batch size: {cfg.training.batch_size}")

    # Enable GPU-optimized settings for maximum performance
    if cfg.device == "cuda":
        # cuDNN autotuner for optimal convolution performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Allow optimizations

        # Modern GPU optimizations
        try:
            # Use high precision matmul (beneficial on modern GPUs)
            torch.set_float32_matmul_precision("high")
            # Enable tensor cores for mixed precision
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        # Memory optimization for large models
        torch.cuda.empty_cache()
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.opt_restart)

    # train
    losses = []
    test_losses = []
    velocity_val_losses = []
    best_test_loss = np.inf
    best_model = None
    for epoch in tqdm.trange(cfg.training.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        num_loops = 0
        for batch in loader:
            # Note that normalization must be done before it's called. The unnormalized
            # data needs to be preserved in order to correctly calculate the loss
            batch = batch.to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # Autocast forward/backward to bf16/fp16 where beneficial
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
                loss = model.loss(pred, batch, mean_vec_y, std_vec_y)
            if use_amp:
                scaler.scale(loss).backward()
                # Gradient clipping for training stability
                if gradient_clip_val is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_val
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Gradient clipping for training stability
                if gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_val
                    )
                optimizer.step()

            # Step scheduler if available
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            num_loops += 1
        total_loss /= num_loops
        losses.append(total_loss)

        # Every tenth epoch, calculate acceleration test loss (prediction)
        # and velocity validation loss
        if epoch % 10 == 0:
            test_loss, velocity_val_rmse = test(
                test_loader,
                cfg.device,
                model,
                mean_vec_x,
                std_vec_x,
                mean_vec_edge,
                std_vec_edge,
                mean_vec_y,
                std_vec_y,
                dataset_type=dataset_type,
                use_amp=use_amp,
            )
            velocity_val_losses.append(velocity_val_rmse.item())
            test_losses.append(test_loss.item())

            # save the model if the current one is better than the previous best
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = copy.deepcopy(model)

        else:
            # If not the tenth epoch, append the previously calculated loss to the
            # list in order to be able to plot it on the same plot as the training losses
            test_losses.append(test_losses[-1])
            velocity_val_losses.append(velocity_val_losses[-1])

        # log to dataframe
        df.loc[len(df.index)] = [
            epoch,
            losses[-1],
            test_losses[-1],
            velocity_val_losses[-1],
        ]

        wandb.log(
            {
                "train_loss": losses[-1],
                "test_loss": test_losses[-1],
                "velocity_loss": velocity_val_losses[-1],
            }
        )

        if epoch % 100 == 0:
            tqdm.tqdm.write(
                f"train loss {round(total_loss, 2)} | "
                f"test loss {round(test_loss.item(), 2)} | "
                f"velocity loss {round(velocity_val_rmse.item(), 5)}"
            )

            if cfg.training.save_best_model:
                # model
                torch.save(best_model.state_dict(), path_model_checkpoint)
                # data infos
                with open(path_infos, "wb") as f:
                    pickle.dump(infos, f)
                # losses
                df.to_pickle(path_df)

    print("Finished training!")
    print("Min test set loss:                {0}".format(min(test_losses)))
    print("Minimum loss:                     {0}".format(min(losses)))
    print("Minimum velocity validation loss: {0}".format(min(velocity_val_losses)))

    if (best_model is not None) and cfg.training.save_best_model:
        # model
        torch.save(best_model.state_dict(), path_model_checkpoint)
        # data infos
        with open(path_infos, "wb") as f:
            pickle.dump(infos, f)
        # losses
        df.to_pickle(path_df)
        print("Saving best model to", str(path_model_checkpoint))

    return


def test(
    loader,
    device,
    test_model,
    mean_vec_x,
    std_vec_x,
    mean_vec_edge,
    std_vec_edge,
    mean_vec_y,
    std_vec_y,
    dataset_type="cylinder_flow",
    is_validation=True,
    use_amp=False,
):
    """
    Calculates test set losses and validation set errors.
    """

    loss = 0
    velocity_rmse = 0
    num_loops = 0

    for data in loader:
        data = data.to(device, non_blocking=True)
        with torch.no_grad():
            # calculate the loss for the model given the test set
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = test_model(
                    data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge
                )
                test_loss = test_model.loss(pred, data, mean_vec_y, std_vec_y)
            loss += test_loss

            # calculate validation error
            # Like for the MeshGraphNets model,
            # build the mask over which we calculate the flow loss
            # and add this calculated RMSE value to our val error
            if dataset_type == "cylinder_flow":
                # Cylinder: node types start at column 2, use types 0 (normal) and 5 (outflow)
                normal = torch.tensor(0)
                outflow = torch.tensor(5)
                loss_mask = torch.logical_or(
                    (torch.argmax(data.x[:, 2:], dim=1) == normal),
                    (torch.argmax(data.x[:, 2:], dim=1) == outflow),
                )
            elif dataset_type == "flag_simple":
                # Flag: node types start at column 3, use type 0 (normal)
                normal = torch.tensor(0)
                loss_mask = torch.argmax(data.x[:, 3:], dim=1) == normal
            else:
                # Fallback: use all nodes
                loss_mask = torch.ones(data.x.shape[0], dtype=torch.bool, device=device)

            # Handle velocity dimensions based on dataset type
            if dataset_type == "flag_simple":
                # 3D velocities for flag dataset
                eval_velocity = (
                    data.x[:, 0:3]
                    + unnormalize(pred[:], mean_vec_y, std_vec_y) * DELTA_T
                )
                true_velocity = data.x[:, 0:3] + data.y[:] * DELTA_T
            else:
                # 2D velocities for cylinder dataset (default)
                eval_velocity = (
                    data.x[:, 0:2]
                    + unnormalize(pred[:], mean_vec_y, std_vec_y) * DELTA_T
                )
                true_velocity = data.x[:, 0:2] + data.y[:] * DELTA_T

            error = torch.sum((eval_velocity - true_velocity) ** 2, axis=1)
            velocity_rmse += torch.sqrt(torch.mean(error[loss_mask]))

        num_loops += 1

    return (loss / num_loops), (velocity_rmse / num_loops)


def load_train_plot(cfg: DictConfig) -> None:
    """Load data, train model, log plots and save checkpoints.

    Composes configuration from `conf/` and follows split policy in cfg.data
    to build train/test sets. Uses W&B for logging.
    """
    print(OmegaConf.to_yaml(cfg))

    # Handle Hydra interpolation variables for manual config composition
    # Set these values directly to avoid interpolation issues
    cfg.job_name = "manual_train"
    cfg.config_name = "default"
    cfg.override_dirname = "manual_override"

    wandb.login()
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project="MeshGraphNets", name=name_from_config(cfg))

    print("Cuda is available to torch:", torch.cuda.is_available())
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # torch.set_default_device(cfg.device)

    # Set the random seeds for all random number generators
    torch.manual_seed(cfg.rseed)  # Torch
    random.seed(cfg.rseed)  # Python
    np.random.seed(cfg.rseed)  # NumPy

    # load the data for training and testing
    # the configuration handles the correct splitting based on train_size and test_size
    # for multi-trajectory datasets, use appropriate datasize configs (e.g., +datasize=large)

    # Safety check for data configuration
    if cfg.data is None:
        raise ValueError(
            "Data configuration is missing! Make sure to use +dataset=flag_simple or check your config files."
        )

    if not hasattr(cfg.data, "datapath") or cfg.data.datapath is None:
        raise ValueError(
            "Dataset path is missing! Make sure cfg.data.datapath is set in your configuration."
        )

    file_path = os.path.join(DATASET_DIR, cfg.data.datapath)

    # Load initial dataset to detect type
    full_dataset = torch.load(file_path)
    dataset_type = detect_dataset_type(full_dataset[0])
    dataset_info = get_dataset_specific_info(dataset_type)

    print(f"Detected dataset type: {dataset_type}")
    print(f"Dataset info: {dataset_info['description']}")
    print(
        f"Node features: velocity({dataset_info['velocity_dim']}D) + node_types({dataset_info['num_node_types']}) + position({dataset_info.get('has_position_features', False)})"
    )

    if cfg.data.train_test_same_traj and cfg.data.single_traj:
        # test on later timesteps of the same trajectory
        # if you specify more than 599 steps, it will still select data from multiple trajectories
        dataset_train = full_dataset[: cfg.training.train_size]
        dataset_test = full_dataset[
            cfg.training.train_size : (  # noqa: E203
                cfg.training.train_size + cfg.training.test_size
            )
        ]
    elif cfg.data.train_test_same_traj:
        # take random timesteps from the same soup of trajectories
        random.shuffle(full_dataset)
        dataset_train = full_dataset[: cfg.training.train_size]
        # test
        dataset_test = full_dataset[
            cfg.training.train_size : (  # noqa: E203
                cfg.training.train_size + cfg.training.test_size
            )
        ]
    else:
        # test on a different trajectory
        test_file_path = file_path.replace("train", "test")
        dataset_train = full_dataset[: cfg.training.train_size]
        dataset_test = torch.load(test_file_path)[: cfg.training.test_size]

    # timesteps in random order
    random.shuffle(dataset_train)
    random.shuffle(dataset_test)

    # maybe it would be better to load the full data to compute the statistics
    # this would ensure that we can use the same model checkpoint on different sets of data.
    # currently we have to recompute the statistic for each loaded dataset
    # stats has to happen on the CPU, because the dataset is a list
    stats_list = get_stats(dataset_train + dataset_test)

    # Training
    train(
        data_train=dataset_train,
        data_test=dataset_test,
        stats_list=stats_list,
        cfg=cfg,
        dataset_type=dataset_type,
    )

    # f = plot_training_loss(cfg)
    # wandb.log({"figure": f})

    # anim_path = create_velocity_animation(cfg)
    # wandb.log({"animation": anim_path})

    return


def run_visualization_mode(cfg: DictConfig, mode: str) -> None:
    """Run visualization tasks based on the specified mode."""
    print(f"Running visualization mode: {mode}")
    print(OmegaConf.to_yaml(cfg))

    if mode == "plots":
        print("Creating standard training plots...")
        create_standard_plots(cfg)
    elif mode == "animations":
        print("Creating standard animations...")
        create_standard_animations(cfg)
    elif mode == "both":
        print("Creating both plots and animations...")
        create_standard_plots(cfg)
        create_standard_animations(cfg)
    else:
        print(f"Unknown visualization mode: {mode}")
        return

    print("Visualization complete!")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="MeshGraphNet: Train, plot, and animate fluid simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model with default config
  python run_gnn.py train

  # Train with custom config
  python run_gnn.py train +datasize=large +noise=paper

  # Create plots from existing checkpoint
  python run_gnn.py plots +datasize=medium

  # Create animations from existing checkpoint
  python run_gnn.py animations +datasize=medium

  # Create both plots and animations
  python run_gnn.py both +datasize=large

  # Custom animation with specific parameters
  python run_gnn.py custom-anim +datasize=medium --start-step 50 --num-steps 200

  # Extract predictions for conformal analysis
  python run_gnn.py conformal +datasize=medium --start-step 0 --num-steps 500

Note: All config overrides (e.g., +datasize=medium) work with all modes.
        """,
    )

    parser.add_argument(
        "mode",
        choices=["train", "plots", "animations", "both", "custom-anim", "conformal"],
        help="Operation mode: train model, create plots/animations, or conformal extraction",
    )

    # Custom animation parameters
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Starting timestep for custom animation (default: 0)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=500,
        help="Number of timesteps for custom animation (default: 500)",
    )
    parser.add_argument(
        "--single-step",
        action="store_true",
        help="Use ground truth inputs (no error accumulation) for animation",
    )
    parser.add_argument(
        "--use-test-traj",
        action="store_true",
        help="Use test trajectory for animation (default: training trajectory)",
    )

    # Parse known args to separate Hydra overrides from our custom args
    args, hydra_overrides = parser.parse_known_args()

    # Check if no arguments provided and show help
    if len(sys.argv) == 1:
        parser.print_help()
        return

    if args.mode == "train":
        # Training mode - compose config manually with overrides
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with hydra.initialize(
            version_base=None, config_path="configs", job_name="train"
        ):
            cfg = hydra.compose(config_name="default", overrides=hydra_overrides)
        load_train_plot(cfg)
    elif args.mode == "custom-anim":
        # Custom animation mode - need to compose config manually
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with hydra.initialize(
            version_base=None, config_path="configs", job_name="custom_anim"
        ):
            cfg = hydra.compose(config_name="default", overrides=hydra_overrides)

        print(
            f"Creating custom animation: {args.start_step}-{args.start_step + args.num_steps}"
        )
        print(f"Single-step: {args.single_step}, Test trajectory: {args.use_test_traj}")

        create_velocity_animation(
            cfg,
            start_step=args.start_step,
            num_steps=args.num_steps,
            single_step=args.single_step,
            use_test_traj=args.use_test_traj,
        )
    elif args.mode == "conformal":
        # Conformal prediction mode - extract raw predictions
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with hydra.initialize(
            version_base=None, config_path="configs", job_name="conformal_extraction"
        ):
            cfg = hydra.compose(config_name="default", overrides=hydra_overrides)

        print(
            f"Extracting predictions for conformal analysis: {args.start_step}-{args.start_step + args.num_steps}"
        )
        print(f"Test trajectory: {args.use_test_traj}")

        # Import the conformal extraction function
        from utils.plot import extract_predictions_for_conformal

        # Extract predictions and save to file for conformal module
        predictions = extract_predictions_for_conformal(
            cfg,
            steps=(args.start_step, args.start_step + args.num_steps),
            use_test_traj=args.use_test_traj,
            return_metadata=True,
        )

        # Save predictions to a standardized location for conformal module
        import pickle
        import os

        output_dir = os.path.join("outputs", "conformal_inputs")
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(
            output_dir,
            f"{predictions['model_name']}_{args.start_step}_{args.start_step + args.num_steps}_conformal.pkl",
        )

        with open(output_file, "wb") as f:
            pickle.dump(predictions, f)

        print(f"Saved conformal predictions to: {output_file}")
        print(f"Shape: {predictions['predictions'].shape}")
        print("Ready for conformal prediction analysis!")

    else:
        # Visualization modes - compose config and run
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with hydra.initialize(
            version_base=None, config_path="configs", job_name="visualization"
        ):
            cfg = hydra.compose(config_name="default", overrides=hydra_overrides)

        run_visualization_mode(cfg, args.mode)


if __name__ == "__main__":
    main()
