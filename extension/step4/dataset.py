"""
Extension Step 4 — Task Graph Dataset
======================================
Builds a PyTorch Geometric dataset where each item is a Data(x, edge_index, y)
object representing the task graph realization of one video recording.

Design decisions:
  - build_realization() is expensive (text encoder + Hungarian matching),
    so realizations are cached to disk as .pt files after the first build.
  - The cache stores text_feats and vis_feats separately (pre-fusion).
    The DifferenceFusionProjector is applied in DAGClassifier.forward(),
    so its parameters are optimised end-to-end with the GNN.
  - Label rule: label=1 if ANY step in the recording has an error (same as B2).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data

from extension.step3.task_graph_loader import load_all_task_graphs, TaskGraph
from extension.step3.egovlp_text_encoder import EgoVLPTextEncoder
from extension.step3.hungarian_matcher import match_visual_to_graph


# ---------------------------------------------------------------------------
# Label extraction — identical logic to B2 TaskVerificationDataset
# ---------------------------------------------------------------------------

def load_samples(annotations_path: str) -> List[Dict]:
    """
    Parse complete_step_annotations.json and return one entry per recording.

    Each entry:
        {
            "recording_id": str,         e.g. "1_7"
            "activity_id":  int,         recipe identifier (0-23)
            "label":        int,         1=incorrect, 0=correct
        }

    Label rule (same as B2): label=1 if ANY step in the recording has an error.
    """
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    samples = []
    for recording_id, info in annotations.items():
        activity_id = info["activity_id"]
        label = int(any(step["has_errors"] for step in info["steps"]))
        samples.append({
            "recording_id": recording_id,
            "activity_id":  activity_id,
            "label":        label,
        })
    return samples


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class TaskGraphDataset(Dataset):
    """
    PyTorch Geometric dataset for task graph classification (B4).

    Each __getitem__ returns a torch_geometric.data.Data object with PRE-FUSION fields:
        data.text_feats  : (N_nodes, 256) float32 — text-only node features (EgoVLP text encoder)
        data.vis_feats   : (N_nodes, 256) float32 — visual features (zeros for unmatched nodes)
        data.matched_mask: (N_nodes,)     bool    — True where visual info is available
        data.edge_index  : (2, N_edges)   int64   — DAG edges
        data.y           : (1,)           float32 — binary label (BCEWithLogitsLoss)
        data.recording_id (stored as metadata for LOO bookkeeping)
        data.activity_id  (stored as metadata for LOO bookkeeping)

    The DifferenceFusionProjector is NOT applied here. It runs in DAGClassifier.forward()
    so its parameters are jointly optimised with the GNN. The cache is therefore invariant
    across all training configurations — no rebuild needed when changing the projector.

    Args:
        annotations_path:   path to complete_step_annotations.json
        step_embeddings_dir: directory with {recording_id}_step_embeddings.npz (B1 output)
        graphs_dir:         directory with task graph JSON files (annotations/task_graphs/)
        egovlp_repo:        path to showlab/EgoVLP clone (for text encoder)
        egovlp_ckpt:        path to egovlp.pth checkpoint
        cache_dir:          if provided, realizations are cached here as .pt files
        video_ids:          if provided, filter to exactly these recording IDs
        activity_ids:       if provided, filter to only these recipe IDs (LOO)
        sim_threshold:      minimum cosine similarity for a Hungarian match to be accepted

    Note: video_ids takes precedence over activity_ids if both are provided.
    """

    def __init__(
        self,
        annotations_path:    str,
        step_embeddings_dir: str,
        graphs_dir:          str,
        egovlp_repo:         str,
        egovlp_ckpt:         str,
        cache_dir:           Optional[str] = None,
        video_ids:           Optional[List[str]] = None,
        activity_ids:        Optional[List[int]] = None,
        sim_threshold:       float = 0.0,
    ):
        super().__init__()

        self.step_embeddings_dir = Path(step_embeddings_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.sim_threshold = sim_threshold

        all_samples = load_samples(annotations_path)
        if video_ids is not None:
            keep = set(video_ids)
            all_samples = [s for s in all_samples if s["recording_id"] in keep]
        elif activity_ids is not None:
            all_samples = [s for s in all_samples if s["activity_id"] in activity_ids]
        self.samples = all_samples

        self.task_graphs: Dict[int, TaskGraph] = load_all_task_graphs(graphs_dir, annotations_path)

        self._encoder: Optional[EgoVLPTextEncoder] = None
        self._egovlp_repo = egovlp_repo
        self._egovlp_ckpt = egovlp_ckpt
        self._text_embeddings_cache: Dict[int, torch.Tensor] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_encoder(self) -> EgoVLPTextEncoder:
        if self._encoder is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._encoder = EgoVLPTextEncoder(
                egovlp_repo=self._egovlp_repo,
                ckpt_path=self._egovlp_ckpt,
                device=device,
            )
        return self._encoder

    def _get_text_embeddings(self, activity_id: int) -> torch.Tensor:
        if activity_id not in self._text_embeddings_cache:
            graph = self.task_graphs[activity_id]
            encoder = self._get_encoder()
            embeddings = encoder.encode(graph.nodes)  # (N_nodes, 256)
            self._text_embeddings_cache[activity_id] = embeddings
        return self._text_embeddings_cache[activity_id]

    def _load_visual_embeddings(self, recording_id: str) -> Optional[torch.Tensor]:
        path = self.step_embeddings_dir / f"{recording_id}_step_embeddings.npz"
        if not path.exists():
            return None
        data = np.load(path)
        return torch.from_numpy(data["step_embeddings"].astype(np.float32))  # (N_steps, 256)

    def _build_pyg_data(self, sample: Dict) -> Data:
        """
        Build a pre-fusion PyG Data object for one video.

        Steps:
          1. Load visual step embeddings from B1 .npz
          2. Get text embeddings for the recipe task graph (cached per recipe)
          3. Run Hungarian matching (visual ↔ textual)
          4. Store text_feats and vis_feats separately — NO projector applied here
          5. Pack into Data(text_feats, vis_feats, matched_mask, edge_index, y)
        """
        recording_id      = sample["recording_id"]
        activity_id       = sample["activity_id"]
        label             = sample["label"]
        graph             = self.task_graphs[activity_id]
        N_nodes           = len(graph.nodes)
        text_embeddings   = self._get_text_embeddings(activity_id).clone()  # (N_nodes, 256)
        visual_embeddings = self._load_visual_embeddings(recording_id)

        if visual_embeddings is not None and visual_embeddings.shape[0] > 0:
            matches = match_visual_to_graph(visual_embeddings, text_embeddings, min_score=self.sim_threshold)
        else:
            matches = []

        vis_feats    = torch.zeros(N_nodes, 256)
        matched_mask = torch.zeros(N_nodes, dtype=torch.bool)
        if matches:
            vis_indices  = [v for v, _ in matches]
            node_indices = [n for _, n in matches]
            matched_vis  = visual_embeddings[vis_indices]
            matched_vis  = F.normalize(matched_vis.float(), p=2, dim=-1)
            vis_feats[node_indices]    = matched_vis
            matched_mask[node_indices] = True

        if graph.edges:
            src = torch.tensor([e[0] for e in graph.edges], dtype=torch.long)
            dst = torch.tensor([e[1] for e in graph.edges], dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        data = Data(
            text_feats   = text_embeddings,
            vis_feats    = vis_feats,
            matched_mask = matched_mask,
            edge_index   = edge_index,
            y            = torch.tensor([label], dtype=torch.float32),
            num_nodes    = N_nodes,
        )
        data.recording_id = recording_id
        data.activity_id  = activity_id
        return data

    def _cache_path(self, recording_id: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{recording_id}.pt"

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        sample = self.samples[idx]
        recording_id = sample["recording_id"]

        cache_path = self._cache_path(recording_id)
        if cache_path is not None and cache_path.exists():
            return torch.load(cache_path, weights_only=False)

        data = self._build_pyg_data(sample)

        if cache_path is not None:
            torch.save(data, cache_path)

        return data

    def prebuild_cache(self) -> None:
        """Pre-build and cache all realizations before training starts."""
        missing = [
            s for s in self.samples
            if self._cache_path(s["recording_id"]) is None
            or not self._cache_path(s["recording_id"]).exists()
        ]
        if not missing:
            print(f"[TaskGraphDataset] Cache complete ({len(self.samples)} items).")
            return

        print(f"[TaskGraphDataset] Building cache for {len(missing)} items...")
        for i, sample in enumerate(missing):
            data = self._build_pyg_data(sample)
            cache_path = self._cache_path(sample["recording_id"])
            if cache_path is not None:
                torch.save(data, cache_path)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(missing)} done")
        print("[TaskGraphDataset] Cache build complete.")

    def get_activity_ids(self) -> List[int]:
        return sorted({s["activity_id"] for s in self.samples})
