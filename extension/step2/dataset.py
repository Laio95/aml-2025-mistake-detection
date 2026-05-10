import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def load_samples(annotations_path: str, step_embeddings_dir: str) -> list:
    """
    Parse complete_step_annotations.json and return one dict per recording
    that has a matching step_embeddings .npz produced by Extension Step 1.

    Each dict contains:
        recording_id  (str) e.g. "1_7"
        activity_id   (int) e.g. 1       -- used to group recordings by recipe for LOO
        activity_name (str) e.g. "Microwave Egg Sandwich"  -- for logging only
        label         (int) 0=correct, 1=incorrect
    """
    with open(annotations_path) as f:
        data = json.load(f)

    samples, n_missing = [], 0

    for recording_id, info in data.items():
        npz_path = os.path.join(
            step_embeddings_dir, f"{recording_id}_step_embeddings.npz"
        )
        if not os.path.exists(npz_path):
            n_missing += 1
            continue

        # A video is incorrect if ANY of its steps has an error (including omissions)
        label = int(any(step["has_errors"] for step in info["steps"]))

        samples.append({
            "recording_id": recording_id,
            "activity_id":  info["activity_id"],
            "activity_name": info["activity_name"],
            "label":        label,
        })

    n_correct   = sum(s["label"] == 0 for s in samples)
    n_incorrect = sum(s["label"] == 1 for s in samples)
    n_recipes   = len({s["activity_id"] for s in samples})
    print(f"Loaded {len(samples)} samples | "
          f"correct={n_correct}, incorrect={n_incorrect} | "
          f"recipes={n_recipes} | skipped={n_missing} (no .npz)")
    return samples


class TaskVerificationDataset(Dataset):
    """
    Video-level binary classification dataset for Task Verification.

    Each video is represented as a variable-length sequence of step-level
    EgoVLP embeddings produced by ActionFormer mean-pooling (Extension Step 1).

    Designed for batch_size=1: no padding or masking needed since
    torch.stack([single_tensor]) works for any sequence length.

    Returns per sample:
        embeddings  : (N_steps, 256) float32  -- detected steps from ActionFormer
        label       : scalar         float32  -- 0.0=correct, 1.0=incorrect
        recording_id: str                     -- for logging / debugging
    """

    def __init__(self, samples: list, step_embeddings_dir: str):
        """
        Args:
            samples:             output of load_samples(), or a subset for one LOO fold
            step_embeddings_dir: folder containing {recording_id}_step_embeddings.npz
        """
        self.samples = samples
        self.step_embeddings_dir = step_embeddings_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        npz_path = os.path.join(
            self.step_embeddings_dir, f"{s['recording_id']}_step_embeddings.npz"
        )
        # step_embeddings key is guaranteed by build_step_embeddings.py (Extension Step 1)
        embeddings = np.load(npz_path)["step_embeddings"]  # (N_steps, 256) float32

        return (
            torch.from_numpy(embeddings.astype(np.float32)),  # (N_steps, 256)
            torch.tensor(s["label"], dtype=torch.float32),    # scalar
        )
