# MeshGraphNet Model

**MeshGraphNet** learns to simulate fluid dynamics on irregular meshes by treating the mesh as a graph and using message passing to capture local fluid interactions, following [Pfaff et al. (2021)](https://arxiv.org/abs/2010.03409).

## Overview

Think of fluid simulation as **information flowing between neighboring mesh points**. Each mesh node needs to "know" what its neighbors are doing to predict how the fluid will move next. This is exactly what GNNs excel at!

### 1. **Encoder: Understanding the Scene**

**What it does**: Converts raw mesh data into a common "language" the network can understand.

- **Node Encoder**: Takes each mesh point's current state (velocity, whether it's a wall/fluid/boundary, position) and creates a rich internal representation
- **Edge Encoder**: Looks at connections between mesh points (distance, relative position) and encodes their relationship

**Why this works**: Just like you need context to understand a conversation, the network needs to process raw measurements into meaningful features before making predictions.

**Mathematics**: Given mesh graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:

- **Node Encoder**: $\mathbf{v}_i^{(0)} = \phi^v(\mathbf{x}_i)$ where $\mathbf{x}_i \in \mathbb{R}^9$ (velocity + node type + position)
- **Edge Encoder**: $\mathbf{e}_{ij}^{(0)} = \phi^e(\mathbf{a}_{ij})$ (edge features)
- Both: $\mathbb{R}^{d_{in}} \xrightarrow{\text{MLP}} \mathbb{R}^{d_h}$ via Linear → ReLU → Linear → LayerNorm

### 2. **Processor: Simulating Fluid Flow**

**What it does**: Multiple rounds of "communication" between neighboring mesh points to simulate how momentum spreads through the fluid.

**The Process** (repeated L=10 times):

1. **Edge Update**: Each connection between mesh points asks "How should we interact?" by looking at both endpoints
2. **Node Update**: Each mesh point collects all incoming "messages" from neighbors and updates its state

**Why this works**: Real fluid dynamics is fundamentally about **local interactions** - what happens at each point depends mainly on its immediate neighbors. After 10 rounds of message passing, information has propagated far enough to capture the relevant physics.

**Physical Intuition**: Each message-passing step is like one "time increment" of momentum diffusion - pressure and velocity information spreads outward from each mesh point to influence its neighbors.

**Mathematics**: For layer $\ell \in \{1, \ldots, L\}$:

- **Edge Update**: $\mathbf{e}_{ij}^{(\ell)} = \mathbf{e}_{ij}^{(\ell-1)} + \psi^e([\mathbf{v}_i^{(\ell-1)} \| \mathbf{v}_j^{(\ell-1)} \| \mathbf{e}_{ij}^{(\ell-1)}])$
- **Node Update**: $\mathbf{v}_i^{(\ell)} = \mathbf{v}_i^{(\ell-1)} + \psi^v([\mathbf{v}_i^{(\ell-1)} \| \sum_{j \in \mathcal{N}(i)} \mathbf{e}_{ij}^{(\ell)}])$

where $\mathcal{N}(i)$ are neighbors, $\|$ is concatenation, $\psi^e, \psi^v$ are MLPs.

### 3. **Decoder: Making Predictions**

**What it does**: Converts the final rich representations back into physical quantities we care about - specifically, how much the velocity will change at each fluid point.

**Why acceleration, not velocity?** We predict the _change_ in velocity because we integrate these predictions over time: `new_velocity = old_velocity + predicted_acceleration * dt`. This is more stable than directly predicting velocities.

**Mathematics**: $\mathbf{a}_i = \phi^{out}(\mathbf{v}_i^{(L)}) \in \mathbb{R}^2$

### Loss Function

**Smart Training**: We only compute error on fluid nodes (not walls), because walls don't move - no point in predicting their "acceleration"!

**Mathematics**: $\mathcal{L} = \sqrt{\frac{1}{|\mathcal{V}_{fluid}|} \sum_{i \in \mathcal{V}_{fluid}} \|\tilde{\mathbf{a}}_i - \mathbf{a}_i\|_2^2}$

## Key Insights

- **Mesh-Adaptive**: Handles irregular triangular meshes (unlike CNNs on grids)
- **Permutation Invariant**: Graph structure preserved under node reordering
- **Local Dynamics**: Message passing captures neighborhood fluid interactions
- **Residual Connections**: Enable deep networks ($L=10$ layers) with stable gradients

**Physical Interpretation**: Each message-passing step simulates one "diffusion" of momentum information across mesh edges, approximating the continuous Navier-Stokes dynamics.
