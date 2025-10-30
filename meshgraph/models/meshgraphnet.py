# This file is heavily based on
# https://colab.research.google.com/drive/1mZAWP6k9R0DE5NxPzF8yL2HpIUG3aoDC?usp=sharing

"""
High-level overview of MeshGraphNet:
- Implements a MeshGraphNet model with explicit node/edge encoders, a stack of
  message-passing processor layers, and a node-only decoder.
- Message passing updates edge embeddings first (based on sender/receiver node
  embeddings plus current edge embeddings), then updates node embeddings with
  aggregated edge messages and a residual connection.
"""

import torch
import torch_scatter
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data

import utils.process as stats


class MeshGraphNet(torch.nn.Module):
    """MeshGraphNets-style GNN for CFD rollout.

    Encodes node and edge features into latent embeddings, applies a stack of
    processor layers (message passing), and decodes to per-node outputs.

    Supports both cylinder_flow (2D, 11 node features, 2 outputs) and
    flag_simple (3D, 12 node features, 3 outputs) datasets dynamically.

    Args:
        input_dim_node (int): Number of node feature channels in `data.x`.
        input_dim_edge (int): Number of edge feature channels in `data.edge_attr`.
        hidden_dim (int): Hidden dimensionality for encoders/processors/decoder.
        output_dim (int): Number of per-node output channels (e.g., 2 for cylinder, 3 for flag).
        cfg (omegaconf.DictConfig): Hydra config with at least `model.num_layers`.
        dataset_type (str): Dataset type ('cylinder_flow' or 'flag_simple') for dataset-specific handling.
    """

    def __init__(
        self,
        input_dim_node,
        input_dim_edge,
        hidden_dim,
        output_dim,
        cfg,
        dataset_type="cylinder_flow",
    ):
        super(MeshGraphNet, self).__init__()
        """
        This model is built upon Deepmind's 2021 paper, and consists of three parts:
        (1) Preprocessing: encoder (2) Processor (3) postproccessing: decoder.
        - Encoder has an edge and node decoders respectively.
        - Processor has two processors for edge and node respectively. Note that edge
        attributes have to be updated first.
        - Decoder is only for nodes.

        Input_dim: dynamic variables + node_type + node_position (dataset-dependent)
        Hidden_dim: 128 in deepmind's paper
        Output_dim: dynamic variables: velocity changes (2D for cylinder, 3D for flag)

        Args:
            input_dim_node (int): Number of node feature channels in `data.x`.
            input_dim_edge (int): Number of edge feature channels in `data.edge_attr`.
            hidden_dim (int): Hidden dimensionality for encoders/processors/decoder.
            output_dim (int): Number of per-node output channels (e.g., 2 for cylinder, 3 for flag).
            cfg (omegaconf.DictConfig): Hydra config with at least `model.num_layers`.
            dataset_type (str): Dataset type for dataset-specific processing.
        """
        # Store dataset type for dataset-specific processing
        self.dataset_type = dataset_type
        self.output_dim = output_dim

        # Number of stacked message passing layers (depth of the processor)
        self.num_layers = cfg.model.num_layers

        # Node encoder: maps raw node features -> latent node embeddings of size hidden_dim
        # Shapes: input [num_nodes, input_dim_node] -> output [num_nodes, hidden_dim]
        self.node_encoder = Sequential(
            Linear(input_dim_node, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
        )

        # Edge encoder: maps raw edge features -> latent edge embeddings of size hidden_dim
        # Shapes: input [num_edges, input_dim_edge] -> output [num_edges, hidden_dim]
        self.edge_encoder = Sequential(
            Linear(input_dim_edge, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
        )

        # Processor: list of message-passing layers. Each layer updates edges then nodes.
        # We build num_layers independent ProcessorLayer instances.
        self.processor = nn.ModuleList()
        assert self.num_layers >= 1, "Number of message passing layers is not >=1"

        processor_layer = self.build_processor_model()
        # Build N identical processor layers; each layer updates edges then nodes
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim, hidden_dim))

        # Decoder: maps final node embeddings -> predicted physical quantities per node
        # Shapes: input [num_nodes, hidden_dim] -> output [num_nodes, output_dim]
        self.decoder = Sequential(
            Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, output_dim)
        )

    def build_processor_model(self):
        # Returns the ProcessorLayer class (factory method for clarity/overrides)
        return ProcessorLayer

    def forward(
        self,
        data: Data,
        mean_vec_x: torch.Tensor,
        std_vec_x: torch.Tensor,
        mean_vec_edge: torch.Tensor,
        std_vec_edge: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        # Unpack required graph attributes
        # x: node features [num_nodes, input_dim_node]
        # edge_index: COO indices [2, num_edges] (source=row 0, target=row 1 by PyG convention)
        # edge_attr: edge features [num_edges, input_dim_edge]
        x, edge_index, edge_attr = (
            data.x,
            data.edge_index,
            data.edge_attr,
        )
        # `pressure = data.p` not used in the model

        # Normalize inputs using dataset statistics for stable training/inference
        x = stats.normalize(x, mean_vec_x, std_vec_x)
        edge_attr = stats.normalize(edge_attr, mean_vec_edge, std_vec_edge)

        # Step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x)  # node embeddings: [num_nodes, hidden_dim]

        edge_attr = self.edge_encoder(
            edge_attr
        )  # edge embeddings: [num_edges, hidden_dim]

        # step 2: perform message passing with latent node/edge embeddings
        for layer in self.processor:
            # Each ProcessorLayer returns updated (node_embeddings, edge_embeddings)
            x, edge_attr = layer(x, edge_index, edge_attr)

        # step 3: decode latent node embeddings into physical quantities of interest
        return self.decoder(x)

    def loss(
        self,
        pred: torch.Tensor,
        inputs: Data,
        mean_vec_y: torch.Tensor,
        std_vec_y: torch.Tensor,
    ) -> torch.Tensor:
        # Dataset-specific loss computation based on node types
        if self.dataset_type == "cylinder_flow":
            # Cylinder: node types start at column 2, use types 0 (normal) and 5 (outflow)
            node_types = torch.argmax(inputs.x[:, 2:], dim=1)
            loss_mask = (node_types == 0) | (node_types == 5)
        elif self.dataset_type == "flag_simple":
            # Flag: node types start at column 3 (after 3D velocity), use type 0 (normal)
            # Flag dataset typically uses type 0 for normal fluid nodes
            node_types = torch.argmax(inputs.x[:, 3:], dim=1)
            loss_mask = node_types == 0
        else:
            # Fallback: compute loss on all nodes
            loss_mask = torch.ones(
                inputs.x.shape[0], dtype=torch.bool, device=inputs.x.device
            )

        # Normalize labels with dataset statistics
        labels = stats.normalize(inputs.y, mean_vec_y, std_vec_y)

        # Sum of squared errors per node
        # Shapes: labels/pred [num_nodes, output_dim] -> error [num_nodes]
        error = (labels - pred).pow(2).sum(dim=1)

        # Handle case where no valid nodes are found
        if loss_mask.sum() == 0:
            # Return a small loss connected to the computation graph
            return error.mean() * 0.01

        # Root of mean error over the selected node types
        loss = torch.sqrt(torch.mean(error[loss_mask]))

        return loss


class ProcessorLayer(MessagePassing):
    """Message passing layer that updates edges then nodes.

    Edge update: concatenates source, target node embeddings and edge features,
    processes via MLP with residual add.
    Node update: aggregates updated incoming edges, concatenates with node
    embedding, applies MLP with residual add.

    Args:
        in_channels (int): Node embedding dimension.
        out_channels (int): Output embedding dimension for both nodes and edges.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ProcessorLayer, self).__init__(**kwargs)
        """
        in_channels: dim of node embeddings [128]
        out_channels: dim of edge embeddings [128]
        """
        # Edge MLP consumes concatenated [sender, receiver, edge] embeddings
        # Therefore its input width is 3 * in_channels
        self.edge_mlp = Sequential(
            Linear(3 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels),
        )

        # Node MLP consumes concatenated [self_node, aggregated_edge_messages]
        # Therefore its input width is 2 * in_channels
        self.node_mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels),
        )

        # Initialize linear layers for more stable training
        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        # Reset Linear layers (0 and 2) in each Sequential; LayerNorm has robust defaults
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        size=None,
    ):
        """
        Update edges then nodes via message passing.

        Note that message passing and aggregation are handled by the propagate
        function, and the update

        x has shape [node_num , in_channels] (node embeddings)
        edge_index: [2, edge_num]
        edge_attr: [E, in_channels]

        """
        # propagate() calls: message() -> aggregate() -> update()
        # Here, message returns updated edge embeddings; aggregate returns
        # a tuple (node_aggregates, updated_edges) so we can keep edges too.
        out, updated_edges = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=size
        )  # out: aggregated messages per node [num_nodes, in_channels]

        # Concatenate self node embeddings with aggregated messages [N, 2*in_channels]
        updated_nodes = torch.cat([x, out], dim=1)

        # Node update with residual connection for better gradient flow/stability
        updated_nodes = x + self.node_mlp(updated_nodes)

        # Return updated node embeddings and updated edge embeddings
        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        """
        source_node: x_i has the shape of [E, in_channels]
        target_node: x_j has the shape of [E, in_channels]
        target_edge: edge_attr has the shape of [E, out_channels]

        The messages that are passed are the raw embeddings. These are not processed.
        """
        # Build edge update input by concatenating sender (x_i), receiver (x_j), and current edge features
        updated_edges = torch.cat([x_i, x_j, edge_attr], dim=1)  # [E, 3*in_channels]

        # Edge update with residual connection to preserve identity mapping if needed
        updated_edges = self.edge_mlp(updated_edges) + edge_attr
        return updated_edges

    def aggregate(self, updated_edges, edge_index, dim_size=None):
        """
        Aggregate incoming updated edges per node (sum reduction).

        Args:
            updated_edges (torch.Tensor): Edge embeddings [E, out_channels].
            edge_index (torch.Tensor): Edge indices [2, E].
            dim_size (int|None): Optional number of nodes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Node-aggregated edge sums [N, out_channels],
            and passthrough updated_edges [E, out_channels].
        """
        # Aggregate edge messages into nodes using segment-wise sum over edges.
        # We index by edge_index[0, :] (sender nodes) consistent with the
        # chosen convention for message flow in this implementation.
        node_dim = 0  # Axis representing nodes
        out = torch_scatter.scatter(
            updated_edges, edge_index[0, :], dim=node_dim, reduce="sum"
        )  # [num_nodes, in_channels]

        # Return both the per-node aggregated features and the updated per-edge features
        return out, updated_edges
