"""
Extension Step 3 — Realization Builder
========================================
Assembles the "realization" of a task graph for a given video: a DAG where
matched nodes have their features updated by fusing textual and visual
information, and unmatched nodes retain their text-only features.

NOTE: this file is NOT part of the EXTENSION substep 4 training pipeline.
The EXTENSION substep 4 training loop (extension/step4/train_dag_classifier.py) uses
TaskGraphDataset (extension/step4/dataset.py), which builds a pre-fusion
cache storing raw text_feats and vis_feats; the DifferenceFusionProjector
is applied inside DAGClassifier.forward() at training time.
This file exists only for manual sanity checks in hungarian_matching.ipynb.

Public symbols:
  - TaskRealization  : dataclass with node features, edge_index, matched_mask.
  - build_realization(): assembles one TaskRealization given pre-computed embeddings
                         and matches. Accepts any nn.Module as projector.

Note: fusion projectors (NodeFusionProjector v1, DifferenceFusionProjector v2)
are defined in extension/step4/dag_classifier.py, not here.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from extension.step3.task_graph_loader import TaskGraph


@dataclass
class TaskRealization:
    """
    A task graph realization for a single video, ready for GNN input.

    node_features : (N_nodes, 256) float32
        Text features for unmatched nodes.
        Fused features (via projector) for matched nodes.
    edge_index    : (2, N_edges) int64
        DAG edges in PyTorch Geometric convention: edge_index[0]=src, edge_index[1]=dst.
    matched_mask  : (N_nodes,) bool
        True for nodes that were matched to a detected visual step.
    recording_id  : str   e.g. "1_7"
    activity_id   : int   recipe identifier
    label         : int   0=correct execution, 1=incorrect execution
    """
    node_features : torch.Tensor
    edge_index    : torch.Tensor
    matched_mask  : torch.Tensor
    recording_id  : str
    activity_id   : int
    label         : int


def build_realization(
    recording_id     : str,
    activity_id      : int,
    label            : int,
    visual_embeddings: torch.Tensor,          # (N_vis, 256)
    task_graph       : TaskGraph,
    text_embeddings  : torch.Tensor,          # (N_nodes, 256)
    matches          : List[Tuple[int, int]], # [(vis_idx, node_idx), ...]
    projector        : nn.Module,             # any fusion projector, e.g. NodeFusionProjector
) -> TaskRealization:
    """
    Assemble a TaskRealization for one video.

    Unmatched nodes keep their text_embeddings unchanged.
    Matched nodes are updated via the projector: Linear([text ; visual]).

    Args:
        recording_id:      video identifier, e.g. "1_7"
        activity_id:       recipe id (for grouping in LOO)
        label:             0=correct, 1=incorrect
        visual_embeddings: (N_vis, 256) step-level EgoVLP features from Step 1
        task_graph:        TaskGraph loaded by task_graph_loader.py
        text_embeddings:   (N_nodes, 256) from EgoVLPTextEncoder.encode()
        matches:           output of match_visual_to_graph() — (vis_idx, node_idx) pairs
        projector:         fusion projector nn.Module (shared, learnable)

    Returns:
        TaskRealization with updated node features, edge_index, and matched_mask.
    """
    N_nodes = len(task_graph.nodes)
    device  = text_embeddings.device

    # Start from text-only features; matched nodes will be overwritten below
    node_features = text_embeddings.clone()                      # (N_nodes, 256)
    matched_mask  = torch.zeros(N_nodes, dtype=torch.bool, device=device)

    # Fuse text + visual features for all matched nodes in one batched forward pass
    if matches:
        vis_indices  = [vis_idx  for vis_idx,  _         in matches]
        node_indices = [node_idx for _,         node_idx in matches]

        matched_text   = text_embeddings[node_indices]                        # (M, 256)
        matched_visual = visual_embeddings[vis_indices].to(device)            # (M, 256)
        matched_visual = F.normalize(matched_visual, p=2, dim=-1)            # (M, 256) — align scale with text

        fused = projector(matched_text, matched_visual)                      # (M, 256)
        node_features[node_indices] = fused
        matched_mask[node_indices]  = True

    # Build edge_index in PyTorch Geometric format: shape (2, N_edges), dtype long
    if task_graph.edges:
        src = torch.tensor([e[0] for e in task_graph.edges], dtype=torch.long, device=device)
        dst = torch.tensor([e[1] for e in task_graph.edges], dtype=torch.long, device=device)
        edge_index = torch.stack([src, dst], dim=0)              # (2, N_edges)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

    return TaskRealization(
        node_features=node_features,
        edge_index=edge_index,
        matched_mask=matched_mask,
        recording_id=recording_id,
        activity_id=activity_id,
        label=label,
    )
