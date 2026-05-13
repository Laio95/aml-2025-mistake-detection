"""
Extension Step 4 — LOO Training for DAGClassifier (DAGNN, faithful to Thost & Chen ICLR 2021)
==============================================================================================
Trains a DAGClassifier on task graph realizations using Leave-One-Out
cross-validation (one fold per recipe), mirroring the protocol of B2.

Key differences from B2 (extension/step2/loo_train.py):
  - Dataset: TaskGraphDataset (PyG Data objects) instead of TaskVerificationDataset
  - DataLoader: torch_geometric.loader.DataLoader (handles variable-size graphs)
  - Model: DAGClassifier (DAGNN + readout on target nodes + MLP head)
  - NodeFusionProjector is INSIDE the model — no separate projector variable needed
  - forward() call passes raw text_feats / vis_feats / matched_mask / ptr
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from extension.step4.dataset_dag import TaskGraphDataset
from extension.step4.dag_classifier import DAGClassifier

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_epoch(
    model:     DAGClassifier,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.BCEWithLogitsLoss,
    device:    torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)

        logits = model(
            batch.text_feats,
            batch.vis_feats,
            batch.matched_mask,
            batch.edge_index,
            batch.batch,
            batch.ptr,
        )  # (B, 1)

        loss = criterion(logits.squeeze(-1), batch.y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_fold(
    model:     DAGClassifier,
    loader:    DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    threshold: float,
    device:    torch.device,
) -> Dict:
    model.eval()

    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)

        logits = model(
            batch.text_feats,
            batch.vis_feats,
            batch.matched_mask,
            batch.edge_index,
            batch.batch,
            batch.ptr,
        )  # (B, 1)

        loss = criterion(logits.squeeze(-1), batch.y.view(-1))
        total_loss += loss.item()

        all_logits.append(logits.squeeze(-1).cpu())
        all_labels.append(batch.y.view(-1).cpu())

    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    preds_np  = (torch.sigmoid(torch.tensor(logits_np)) >= threshold).float().numpy()

    try:
        auc = roc_auc_score(labels_np, logits_np)
    except ValueError:
        auc = float("nan")  # only one class in this fold

    return {
        "loss":     total_loss / max(len(loader), 1),
        "auc":      auc,
        "f1":       f1_score(labels_np, preds_np, zero_division=0),
        "accuracy": accuracy_score(labels_np, preds_np),
    }


# ---------------------------------------------------------------------------
# LOO split utility
# ---------------------------------------------------------------------------

def make_loo_splits(activity_ids: List[int]):
    """Yield (fold_idx, test_id, train_ids) for each fold."""
    for fold_idx, test_id in enumerate(activity_ids, start=1):
        train_ids = [aid for aid in activity_ids if aid != test_id]
        yield fold_idx, test_id, train_ids


# ---------------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------------

def train_fold(
    fold_idx:           int,
    test_activity_id:   int,
    train_activity_ids: List[int],
    args:               argparse.Namespace,
    dataset_kwargs:     Dict,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Datasets (share the same pre-built cache) ---
    train_ds = TaskGraphDataset(**dataset_kwargs, activity_ids=train_activity_ids)
    test_ds  = TaskGraphDataset(**dataset_kwargs, activity_ids=[test_activity_id])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # --- Model (fresh per fold) ---
    torch.manual_seed(args.seed)
    model = DAGClassifier(
        in_channels=256,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # --- Loss: pos_weight from training split ---
    n_pos = sum(s["label"] for s in train_ds.samples)
    n_neg = len(train_ds.samples) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Optimizer: all model parameters (includes projector, DAGNN, head) ---
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-5,
    )

    # --- WandB ---
    use_wandb = args.enable_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project="gnn_task_verification_loo",
            name=f"fold_{fold_idx}_recipe_{test_activity_id}",
            reinit=True,
        )

    # --- Checkpoint path ---
    ckpt_dir  = Path(args.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fold_{fold_idx}_recipe_{test_activity_id}_best.pt"

    best_auc     = -1.0
    best_metrics: Dict = {}

    # --- Training loop ---
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        metrics    = eval_fold(model, test_loader, criterion, args.threshold, device)

        scheduler.step(metrics["auc"] if not np.isnan(metrics["auc"]) else 0.0)

        if use_wandb:
            wandb.log({
                "epoch":         epoch,
                "train_loss":    train_loss,
                "test_loss":     metrics["loss"],
                "test_auc":      metrics["auc"],
                "test_f1":       metrics["f1"],
                "test_accuracy": metrics["accuracy"],
                "lr":            optimizer.param_groups[0]["lr"],
            })

        if not np.isnan(metrics["auc"]) and metrics["auc"] > best_auc:
            best_auc     = metrics["auc"]
            best_metrics = metrics.copy()
            torch.save({
                "model":   model.state_dict(),
                "epoch":   epoch,
                "metrics": metrics,
            }, ckpt_path)

    if use_wandb:
        wandb.finish()

    return {
        "fold":        fold_idx,
        "activity_id": test_activity_id,
        **best_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LOO training for DAGNN classifier (B4)")

    # --- Paths ---
    parser.add_argument("--annotations_path",    required=True)
    parser.add_argument("--step_embeddings_dir", required=True)
    parser.add_argument("--graphs_dir",          required=True)
    parser.add_argument("--egovlp_repo",         required=True)
    parser.add_argument("--egovlp_ckpt",         required=True)
    parser.add_argument("--cache_dir",           required=True)
    parser.add_argument("--output_dir",          required=True)

    # --- Model ---
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.5)

    # --- Training ---
    parser.add_argument("--num_epochs",   type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--threshold",    type=float, default=0.5)
    parser.add_argument("--num_workers",  type=int,   default=2)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--enable_wandb", action="store_true")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dataset_kwargs = dict(
        annotations_path    = args.annotations_path,
        step_embeddings_dir = args.step_embeddings_dir,
        graphs_dir          = args.graphs_dir,
        egovlp_repo         = args.egovlp_repo,
        egovlp_ckpt         = args.egovlp_ckpt,
        cache_dir           = args.cache_dir,
    )

    # --- Pre-build cache once (reused by all folds) ---
    print("Building pre-fusion cache (runs once, reused by all folds)...")
    full_ds = TaskGraphDataset(**dataset_kwargs)
    full_ds.prebuild_cache()
    all_activity_ids = full_ds.get_activity_ids()
    print(f"LOO: {len(all_activity_ids)} folds | {len(full_ds)} total videos\n")

    # --- LOO loop (with resume: skips folds whose checkpoint already exists) ---
    all_results = []
    ckpt_dir = Path(args.output_dir) / "checkpoints"
    for fold_idx, test_id, train_ids in make_loo_splits(all_activity_ids):
        ckpt_path = ckpt_dir / f"fold_{fold_idx}_recipe_{test_id}_best.pt"
        if ckpt_path.exists():
            saved  = torch.load(ckpt_path, weights_only=False)
            result = {"fold": fold_idx, "activity_id": test_id, **saved["metrics"]}
            all_results.append(result)
            print(f"Fold {fold_idx:02d}/{len(all_activity_ids)} — SKIP (checkpoint exists)")
            continue

        print(f"Fold {fold_idx:02d}/{len(all_activity_ids)} — test recipe {test_id}")
        result = train_fold(fold_idx, test_id, train_ids, args, dataset_kwargs)
        all_results.append(result)
        print(
            f"  AUC={result.get('auc', float('nan')):.4f}  "
            f"F1={result.get('f1', float('nan')):.4f}  "
            f"Acc={result.get('accuracy', float('nan')):.4f}"
        )

    # --- Aggregate ---
    valid    = [r for r in all_results if not np.isnan(r.get("auc", float("nan")))]
    mean_auc = np.mean([r["auc"]      for r in valid])
    std_auc  = np.std( [r["auc"]      for r in valid])
    mean_f1  = np.mean([r["f1"]       for r in valid])
    std_f1   = np.std( [r["f1"]       for r in valid])
    mean_acc = np.mean([r["accuracy"] for r in valid])
    std_acc  = np.std( [r["accuracy"] for r in valid])

    print(f"\n=== LOO Summary ({len(valid)} folds) ===")
    print(f"AUC:      {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"F1:       {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    # --- Save CSV ---
    csv_path   = Path(args.output_dir) / "results.csv"
    fieldnames = ["fold", "activity_id", "auc", "f1", "accuracy"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
        writer.writerow({
            "fold": "mean", "activity_id": "",
            "auc": f"{mean_auc:.4f}", "f1": f"{mean_f1:.4f}", "accuracy": f"{mean_acc:.4f}",
        })
        writer.writerow({
            "fold": "std", "activity_id": "",
            "auc": f"{std_auc:.4f}", "f1": f"{std_f1:.4f}", "accuracy": f"{std_acc:.4f}",
        })

    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
