"""
Extension Step 3 — Hungarian Matcher
======================================
Matches visual step embeddings (from ActionFormer mean-pooling, Step 1) to
task graph node embeddings (from EgoVLP text encoder) via the Hungarian algorithm.

The assignment is one-to-one: each visual step is matched to at most one graph
node, and vice versa.  min(N_vis, N_nodes) pairs are always returned unless
filtered by min_score.

References:
  Kuhn, H.W. (1955). "The Hungarian Method for the assignment problem."
  scipy.optimize.linear_sum_assignment
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Union


# Type alias: accepts both torch Tensors and numpy arrays
ArrayLike = Union[torch.Tensor, np.ndarray]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert Tensor or ndarray to float32 numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    # avoid division by zero for zero vectors
    norms = np.where(norms < 1e-8, 1.0, norms)
    return x / norms


def match_visual_to_graph(
    V: ArrayLike,
    T: ArrayLike,
    min_score: float = 0.0,
) -> List[Tuple[int, int]]:
    """
    Find the optimal one-to-one assignment between visual steps and graph nodes.

    Both V and T are L2-normalized internally, so the similarity metric is
    cosine similarity regardless of the input scale.

    Args:
        V:          (N_vis, 256)   visual step embeddings (ActionFormer mean-pool)
        T:          (N_nodes, 256) text node embeddings (EgoVLP text encoder)
        min_score:  discard matched pairs whose cosine similarity is below this
                    threshold (default 0.0 keeps all pairs; useful for ablations)

    Returns:
        List of (vis_idx, node_idx) tuples, sorted by vis_idx.
        Length is min(N_vis, N_nodes) before min_score filtering.
    """
    V_np = _l2_normalize(_to_numpy(V))   # (N_vis, 256)
    T_np = _l2_normalize(_to_numpy(T))   # (N_nodes, 256)

    # Cosine similarity matrix: entry [i, j] = similarity(visual_i, node_j)
    # Shape: (N_vis, N_nodes)
    sim_matrix = V_np @ T_np.T   # dot product of L2-normalized vectors = cosine sim

    # Hungarian algorithm minimizes cost → negate similarity to turn it into a cost
    cost_matrix = -sim_matrix    # (N_vis, N_nodes)
    row_ids, col_ids = linear_sum_assignment(cost_matrix)

    # Build result, optionally filtering low-confidence pairs
    matches: List[Tuple[int, int]] = []
    for vis_idx, node_idx in zip(row_ids.tolist(), col_ids.tolist()):
        score = sim_matrix[vis_idx, node_idx]
        if score >= min_score:
            matches.append((vis_idx, node_idx))

    return matches


def compute_similarity_matrix(V: ArrayLike, T: ArrayLike) -> np.ndarray:
    """
    Return the full (N_vis, N_nodes) cosine similarity matrix.

    Useful for debugging and visualizing the quality of the matching.
    Not needed by the main pipeline — only called in sanity checks / ablations.
    """
    V_np = _l2_normalize(_to_numpy(V))
    T_np = _l2_normalize(_to_numpy(T))
    return V_np @ T_np.T
