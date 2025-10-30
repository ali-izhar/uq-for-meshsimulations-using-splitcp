#!/usr/bin/env python3
"""
Domain-Aware Feature Engineering for Spatially Adaptive Conformal Prediction

Implements the six feature categories for learning heterogeneous uncertainty
patterns in mesh-based fluid simulations:

    1. Kinematic & Temporal (3): speed magnitudes, time normalization
    2. Spatial (2): mesh coordinates
    3. Topology (4): node degree, wall proximity, edge statistics
    4. Gradient & Vorticity (4): velocity gradients, curl, divergence
    5. Node Classification (1): one-hot node types (wall/fluid/boundary)
    6. Ensemble Statistics (3): prediction variance across ensemble (optional)

Two modes:
    - basic (p=5): kinematic (3) + spatial (2) - minimal baseline
    - full (p=17): all 6 categories - complete feature set

These features enable the meta-model to predict spatially varying scales ŝ(x),
adapting uncertainty sets to local flow complexity.

Reference:
    See paper.tex §4.4 and Table 3 for feature definitions and ablation studies.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy.spatial import cKDTree


class PhysicsFeatures:
    """
    Domain-aware feature extraction for spatially adaptive conformal prediction.

    Extracts physics-informed features from mesh simulations to predict
    heterogeneous uncertainty patterns. Implements the six feature categories
    from Table 3 of the paper.

    Modes:
        - "basic" (p=5): kinematic (3) + spatial (2) - baseline
        - "full" (p=17): all 6 categories - complete feature set

    Reference:
        Paper §4.4 (Spatially Adaptive Scaling) and Table 3
    """

    def __init__(self, mode: str = "full"):
        """
        Initialize feature extractor.

        Args:
            mode: "basic" for minimal features (p=5) or "full" for complete set (p=17)

        Raises:
            ValueError: If mode not in {"basic", "full"}
        """
        if mode not in {"basic", "full"}:
            raise ValueError(f"mode must be 'basic' or 'full', got {mode}")
        self.mode = mode
        self._wall_kdtree: Optional[cKDTree] = None

    def compute_enhanced_features(
        self,
        predictions: np.ndarray,
        velocities: np.ndarray,
        metadata: Dict,
    ) -> np.ndarray:
        """
        Extract domain-aware features for uncertainty scale prediction.

        Implements the feature categories from Table 3:
            1. Kinematic & Temporal (3)
            2. Spatial (2) [basic mode stops here]
            3. Topology (4) [full mode]
            4. Gradient & Vorticity (4) [full mode]
            5. Node Classification (1) [full mode]
            6. Flow Constraints (2) [full mode]

        Args:
            predictions: Model predictions ŷ [T, N, d] where d ∈ {2,3}
            velocities: Initial velocities v₀ [T, N, d]
            metadata: Dict with 'mesh_positions', 'node_types', optional 'cells'

        Returns:
            Feature matrix [T*N, p] where p=5 (basic) or p=17 (full)

        Reference:
            Paper Table 3 (Feature Categories and Ablation Study)
        """
        T, N, d = predictions.shape

        # Extract mesh topology
        positions = metadata["mesh_positions"]  # [N, 2|3]
        node_types = metadata["node_types"]  # [N]
        faces = metadata.get("cells", None)  # [F, 3] triangulation

        features = []

        # Category 1: Kinematic & Temporal (3 features)
        # φ_kin = [‖ŷ(x)‖₂, ‖v₀(x)‖₂, t/T_max]
        pred_speed = np.linalg.norm(predictions, axis=-1)  # [T, N]
        vel_speed = np.linalg.norm(velocities, axis=-1)  # [T, N]
        time_normalized = (np.arange(T, dtype=np.float32) / max(1, T - 1))[
            :, None
        ]  # [T, 1]
        time_feat = np.broadcast_to(time_normalized, (T, N))  # [T, N]

        features.extend(
            [
                pred_speed[..., None],
                vel_speed[..., None],
                time_feat[..., None],
            ]
        )

        if self.mode == "basic":
            # Category 2: Spatial (2 features) - basic mode stops here
            # φ_spatial = [x, y]
            features.append(np.broadcast_to(positions[None, :, :], (T, N, 2)))

        else:  # Full mode: add categories 3-6

            # Category 3: Topology (4 features) - mesh connectivity & quality
            # φ_topo = [degree(x)/degree_max, aspect_ratio(x), min_angle(x), d_wall(x)]
            wall_distances = self._compute_wall_distances(positions, node_types)

            # Wall distance features (2)
            features.append(np.broadcast_to(wall_distances[None, :, None], (T, N, 1)))
            features.append(
                np.broadcast_to(
                    np.log(np.maximum(wall_distances, 1e-6))[None, :, None], (T, N, 1)
                )
            )

            # Category 4: Gradient & Vorticity (2 features)
            # φ_grad = [log(‖∇v₀‖₂ + ε), log(‖∇ŷ‖₂ + ε)]
            grad_v0 = self._compute_velocity_gradients(velocities, positions, faces)
            grad_pred = self._compute_velocity_gradients(predictions, positions, faces)
            features.extend([grad_v0[..., None], grad_pred[..., None]])

            # Category 5: Mesh Topology (3 features)
            # φ_mesh = [degree/degree_max, aspect_ratio, min_angle]
            if faces is not None:
                degrees = self._compute_mesh_degrees(faces, N)
                aspect_ratios, min_angles = self._compute_mesh_quality(
                    positions, faces, N
                )
                features.extend(
                    [
                        np.broadcast_to(degrees[None, :, None], (T, N, 1)),
                        np.broadcast_to(aspect_ratios[None, :, None], (T, N, 1)),
                        np.broadcast_to(min_angles[None, :, None], (T, N, 1)),
                    ]
                )
            else:
                # Fallback if no mesh connectivity available
                features.extend([np.zeros((T, N, 1))] * 3)

            # Category 6: Flow Constraints (2 features)
            # φ_phys = [|∇·v₀|, log(Re_local + 1)]
            divergence = self._compute_velocity_divergence(velocities, positions, faces)
            re_local = self._compute_local_reynolds(velocities, wall_distances)
            features.extend([divergence[..., None], re_local[..., None]])

            # Spatial coordinates (2 features)
            # φ_spatial = [x, y]
            features.append(np.broadcast_to(positions[None, :, :], (T, N, 2)))

            # Domain-specific features (3 features)
            # [d_center, 𝟙_boundary, p_boundary]
            center = positions.mean(axis=0)
            center_dist = np.linalg.norm(positions - center, axis=1)

            majority_type = np.bincount(node_types).argmax()
            is_boundary = (node_types != majority_type).astype(np.float32)

            d_95 = np.percentile(wall_distances, 95)
            p_boundary = 1.0 - np.clip(wall_distances / d_95, 0, 1)

            features.extend(
                [
                    np.broadcast_to(center_dist[None, :, None], (T, N, 1)),
                    np.broadcast_to(is_boundary[None, :, None], (T, N, 1)),
                    np.broadcast_to(p_boundary[None, :, None], (T, N, 1)),
                ]
            )

        # Concatenate all features and flatten
        X = np.concatenate(features, axis=-1)  # [T, N, p]
        X_flat = X.reshape(-1, X.shape[-1])  # [T*N, p]

        # Validate feature count
        expected_dims = 5 if self.mode == "basic" else 17
        if X_flat.shape[1] != expected_dims:
            raise ValueError(
                f"Expected {expected_dims} features in '{self.mode}' mode, got {X_flat.shape[1]}"
            )

        return X_flat

    def _compute_wall_distances(
        self, positions: np.ndarray, node_types: np.ndarray
    ) -> np.ndarray:
        """
        Compute distance from each node to nearest wall node: d_wall(x)

        Uses k-d tree for efficient nearest-neighbor query. Identifies wall nodes
        from node type encoding and caches the tree for repeated calls.
        """
        # Identify wall nodes (type > 0 indicates boundary/wall)
        unique_types = np.unique(node_types)
        wall_candidates = [t for t in unique_types if t > 0]
        wall_type = min(wall_candidates) if wall_candidates else unique_types[0]

        wall_mask = node_types == wall_type
        wall_positions = positions[wall_mask]

        # Fallback: use minority nodes if no explicit wall type
        if wall_positions.shape[0] == 0:
            majority_type = np.bincount(node_types).argmax()
            wall_mask = node_types != majority_type
            wall_positions = positions[wall_mask]

        if wall_positions.shape[0] == 0:
            return np.full(positions.shape[0], 0.1)  # Constant fallback

        # Build/update k-d tree for wall positions
        if (
            self._wall_kdtree is None
            or self._wall_kdtree.data.shape[0] != wall_positions.shape[0]
        ):
            self._wall_kdtree = cKDTree(wall_positions)

        # Query nearest wall distance for all nodes
        distances, _ = self._wall_kdtree.query(positions, k=1)

        # Clip minimum distance to avoid numerical issues
        min_dist = (
            np.percentile(distances[distances > 0], 5)
            if np.any(distances > 0)
            else 1e-3
        )
        return np.maximum(distances, min_dist)

    def _compute_velocity_gradients(
        self,
        velocity_field: np.ndarray,
        positions: np.ndarray,
        faces: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compute log(‖∇v‖₂ + ε) using linear finite element gradients.

        For 2D meshes, computes ∇v for each component (u, v) using FEM,
        then returns log of the averaged gradient magnitude.

        Args:
            velocity_field: Velocity field [T, N, d]
            positions: Node coordinates [N, 2]
            faces: Triangle connectivity [F, 3]

        Returns:
            log(gradient magnitude) [T, N]
        """
        if faces is None:
            return self._compute_gradient_knn_fallback(velocity_field, positions)

        T, N, d = velocity_field.shape
        grad_magnitudes = np.zeros((T, N))

        for t in range(T):
            for component in range(min(d, 2)):  # u, v components
                grad_x, grad_y = self._compute_fem_gradient(
                    velocity_field[t, :, component], positions, faces
                )
                grad_magnitudes[t] += np.sqrt(grad_x**2 + grad_y**2)

        grad_magnitudes /= 2.0  # Average over components
        return np.log(np.maximum(grad_magnitudes, 1e-8))

    def _compute_fem_gradient(
        self,
        field: np.ndarray,
        positions: np.ndarray,
        faces: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ∇f using piecewise linear finite elements on triangular mesh.

        For each triangle, computes constant gradient via:
            ∇f = (1/2A) × [(f_j - f_i)(p_k - p_i) - (f_k - f_i)(p_j - p_i)]
        where A is triangle area. Averages gradients at each node.

        Args:
            field: Scalar field values [N]
            positions: Node coordinates [N, 2]
            faces: Triangle connectivity [F, 3]

        Returns:
            (∂f/∂x, ∂f/∂y) gradient components [N]
        """
        N = positions.shape[0]
        grad_x, grad_y = np.zeros(N), np.zeros(N)
        node_counts = np.zeros(N)

        for i, j, k in faces:
            p1, p2, p3 = positions[i], positions[j], positions[k]

            # Compute 2×area via cross product
            area_x2 = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (
                p2[1] - p1[1]
            )
            if abs(area_x2) < 1e-12:  # Degenerate triangle
                continue

            # Constant gradient on triangle via finite element formula
            f1, f2, f3 = field[i], field[j], field[k]
            grad_x_tri = (
                (f2 - f1) * (p3[1] - p1[1]) - (f3 - f1) * (p2[1] - p1[1])
            ) / area_x2
            grad_y_tri = (
                (f3 - f1) * (p2[0] - p1[0]) - (f2 - f1) * (p3[0] - p1[0])
            ) / area_x2

            # Accumulate gradients at vertices
            for idx in (i, j, k):
                grad_x[idx] += grad_x_tri
                grad_y[idx] += grad_y_tri
                node_counts[idx] += 1

        # Average gradients from all incident triangles
        valid = node_counts > 0
        grad_x[valid] /= node_counts[valid]
        grad_y[valid] /= node_counts[valid]

        return grad_x, grad_y

    def _compute_gradient_knn_fallback(
        self, velocity_field: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """
        Fallback gradient estimation using weighted k-NN least squares.

        Used when mesh connectivity (faces) is unavailable. Estimates gradient
        at each node by fitting local linear model to k nearest neighbors.
        """
        T, N, _ = velocity_field.shape
        k_neighbors = min(8, N // 4)

        tree = cKDTree(positions)
        distances, indices = tree.query(positions, k=k_neighbors + 1)
        grad_magnitudes = np.zeros((T, N))

        for t in range(T):
            for i in range(N):
                nbr_idx = indices[i, 1:]  # Exclude self
                nbr_dist = distances[i, 1:]

                if nbr_dist.max() > 1e-8:
                    weights = 1.0 / (nbr_dist + 1e-8)
                    vel_diff = velocity_field[t, nbr_idx] - velocity_field[t, i]
                    pos_diff = positions[nbr_idx] - positions[i]

                    try:
                        # Weighted least squares: ∇v ≈ argmin Σ w_j ‖∇v·Δp_j - Δv_j‖²
                        A = pos_diff * weights[:, None]
                        b = vel_diff * weights[:, None]
                        grad_est = np.linalg.lstsq(A, b, rcond=None)[0]
                        grad_magnitudes[t, i] = np.linalg.norm(grad_est, "fro")
                    except:
                        grad_magnitudes[t, i] = 0.0

        return np.log(np.maximum(grad_magnitudes, 1e-8))

    def _compute_mesh_degrees(self, faces: np.ndarray, num_nodes: int) -> np.ndarray:
        """
        Compute normalized node degree: degree(x) / degree_max

        Counts number of triangles incident to each node, normalized by maximum.
        Higher degree indicates denser local mesh refinement.
        """
        degrees = np.bincount(faces.ravel(), minlength=num_nodes)
        max_degree = degrees.max() if degrees.max() > 0 else 1.0
        return degrees / max_degree

    def _compute_mesh_quality(
        self, positions: np.ndarray, faces: np.ndarray, num_nodes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mesh quality metrics: aspect ratio and minimum angle.

        - Aspect ratio: perimeter²/(12√3·area), normalized by equilateral (=1)
        - Min angle: smallest angle in incident triangles, normalized by 60°

        Poor quality (high aspect ratio, small angles) correlates with higher
        numerical error and uncertainty in CFD simulations.
        """
        aspect_ratios = np.zeros(num_nodes)
        min_angles = np.zeros(num_nodes)
        node_counts = np.zeros(num_nodes)

        for i, j, k in faces:
            p1, p2, p3 = positions[i], positions[j], positions[k]

            # Compute edge lengths
            edges = [p2 - p1, p3 - p2, p1 - p3]
            edge_lens = [np.linalg.norm(e) for e in edges]
            e1, e2, e3 = edge_lens

            if min(edge_lens) < 1e-12:  # Degenerate triangle
                continue

            # Triangle area via cross product
            area = 0.5 * abs(
                (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            )
            if area < 1e-12:
                continue

            # Aspect ratio (1 = equilateral, >1 = stretched)
            perimeter = sum(edge_lens)
            aspect_ratio = perimeter**2 / (12 * np.sqrt(3) * area)

            # Compute triangle angles via cosine law
            angles = []
            for e_opp, e_adj1, e_adj2 in [(e1, e2, e3), (e2, e3, e1), (e3, e1, e2)]:
                cos_angle = np.clip(
                    (e_adj1**2 + e_adj2**2 - e_opp**2) / (2 * e_adj1 * e_adj2), -1, 1
                )
                angles.append(np.arccos(cos_angle))

            # Accumulate at vertices
            for idx in (i, j, k):
                aspect_ratios[idx] += aspect_ratio
                min_angles[idx] += min(angles)
                node_counts[idx] += 1

        # Average over incident triangles
        valid = node_counts > 0
        aspect_ratios[valid] /= node_counts[valid]
        min_angles[valid] /= node_counts[valid]

        # Transform for numerical stability
        aspect_ratios = np.log(np.maximum(aspect_ratios, 1.1))
        min_angles = min_angles / (np.pi / 3)  # Normalize by 60° (equilateral)

        return aspect_ratios, min_angles

    def _compute_velocity_divergence(
        self,
        velocity_field: np.ndarray,
        positions: np.ndarray,
        faces: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compute |∇·v| measuring incompressibility violation.

        For incompressible flow, ∇·v = ∂u/∂x + ∂v/∂y should be zero.
        Non-zero values indicate numerical error or compressibility effects,
        which correlate with higher prediction uncertainty.

        Returns:
            |∇·v| [T, N]
        """
        if faces is None:
            return np.zeros(velocity_field.shape[:2])

        T, N, _ = velocity_field.shape
        divergence = np.zeros((T, N))

        for t in range(T):
            # Compute ∂u/∂x + ∂v/∂y
            grad_u_x, _ = self._compute_fem_gradient(
                velocity_field[t, :, 0], positions, faces
            )
            _, grad_v_y = self._compute_fem_gradient(
                velocity_field[t, :, 1], positions, faces
            )
            divergence[t] = grad_u_x + grad_v_y

        return np.abs(divergence)

    def _compute_local_reynolds(
        self,
        velocity_field: np.ndarray,
        wall_distances: np.ndarray,
    ) -> np.ndarray:
        """
        Compute local Reynolds number: Re_local = |v|·h/ν

        where h = d_wall is the characteristic length scale and ν is kinematic
        viscosity. Higher Re indicates more turbulent/complex flow with higher
        prediction uncertainty.

        Returns:
            log(Re_local + 1) [T, N] for numerical stability
        """
        vel_magnitudes = np.linalg.norm(velocity_field, axis=-1)  # [T, N]
        length_scale = np.maximum(wall_distances, 1e-6)  # [N]
        nu = 1e-6  # Kinematic viscosity (water-like, order of magnitude)

        re_local = (vel_magnitudes * length_scale[None, :]) / nu
        return np.log1p(re_local)  # log(1 + x) for stability
