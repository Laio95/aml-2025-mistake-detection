"""
Extension Step 4 — Train/Val/Test Training for DAGClassifier (DAGNN)
======================================================================
Single-model training with a fixed train/val/test split at the recipe level.

The 24 recipes are split into train/val/test using a deterministic shuffle
(seed=42 by default): 16 train / 4 val / 4 test.

Split rationale:
  - Split is at the RECIPE level — all recordings from the same recipe go to
    the same partition, preventing any leakage between related videos.
  - Val set drives LR scheduling and early stopping.
  - Test set is evaluated ONCE at the end on the best-val-AUC checkpoint.

The LOO version is archived in extension/step4/loo/train_dag_classifier_loo.py.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from extension.step3.dataset import TaskGraphDataset
from extension.step4.dag_classifier import DAGClassifier

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Recipe-level split
# ---------------------------------------------------------------------------

def split_recipes(
    activity_ids: List[int],
    n_val:  int = 4,
    n_test: int = 4,
    seed:   int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Deterministic random split of recipe IDs into train / val / test.
    Same seed always produces the same partition.

    Returns:
        (train_ids, val_ids, test_ids) — each a sorted list of activity_ids.
    """
    rng      = np.random.default_rng(seed)
    shuffled = rng.permutation(sorted(activity_ids)).tolist()
    test_ids  = sorted(shuffled[:n_test])
    val_ids   = sorted(shuffled[n_test:n_test + n_val])
    train_ids = sorted(shuffled[n_test + n_val:])
    return train_ids, val_ids, test_ids


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
        batch  = batch.to(device)
        logits = model(
            batch.text_feats, batch.vis_feats, batch.matched_mask,
            batch.edge_index, batch.batch, batch.ptr,
        )
        loss = criterion(logits.squeeze(-1), batch.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_split(
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
        batch  = batch.to(device)
        logits = model(
            batch.text_feats, batch.vis_feats, batch.matched_mask,
            batch.edge_index, batch.batch, batch.ptr,
        )
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
        auc = float("nan")

    return {
        "loss":     total_loss / max(len(loader), 1),
        "auc":      auc,
        "f1":       f1_score(labels_np, preds_np, zero_division=0),
        "accuracy": accuracy_score(labels_np, preds_np),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train/Val/Test training for DAGNN classifier (B4)"
    )

    # Paths
    parser.add_argument("--annotations_path",    required=True)
    parser.add_argument("--step_embeddings_dir", required=True)
    parser.add_argument("--graphs_dir",          required=True)
    parser.add_argument("--egovlp_repo",         required=True)
    parser.add_argument("--egovlp_ckpt",         required=True)
    parser.add_argument("--cache_dir",           required=True)
    parser.add_argument("--output_dir",          required=True)

    # Split
    parser.add_argument("--n_val",  type=int, default=4,
                        help="number of recipes held out for validation")
    parser.add_argument("--n_test", type=int, default=4,
                        help="number of recipes held out for test")
    parser.add_argument("--seed",   type=int, default=42)

    # Model
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.5)

    # Training
    parser.add_argument("--num_epochs",   type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--threshold",    type=float, default=0.5)
    parser.add_argument("--patience",     type=int,   default=15,
                        help="early stopping: max epochs without val AUC improvement")
    parser.add_argument("--num_workers",  type=int,   default=2)
    parser.add_argument("--enable_wandb", action="store_true")

    args   = parser.parse_args()
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_kwargs = dict(
        annotations_path    = args.annotations_path,
        step_embeddings_dir = args.step_embeddings_dir,
        graphs_dir          = args.graphs_dir,
        egovlp_repo         = args.egovlp_repo,
        egovlp_ckpt         = args.egovlp_ckpt,
        cache_dir           = args.cache_dir,
    )

    # --- Verify cache, derive split ---
    print("Verifying pre-fusion cache...")
    full_ds = TaskGraphDataset(**dataset_kwargs)
    full_ds.prebuild_cache()
    all_ids = full_ds.get_activity_ids()

    train_ids, val_ids, test_ids = split_recipes(
        all_ids, n_val=args.n_val, n_test=args.n_test, seed=args.seed,
    )
    print(f"\nSplit (seed={args.seed}): "
          f"train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)} recipes")
    print(f"  train : {train_ids}")
    print(f"  val   : {val_ids}")
    print(f"  test  : {test_ids}")

    # Save split for reproducibility
    with open(out / "split_info.json", "w") as f:
        json.dump({"seed": args.seed, "n_val": args.n_val, "n_test": args.n_test,
                   "train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2)

    train_ds = TaskGraphDataset(**dataset_kwargs, activity_ids=train_ids)
    val_ds   = TaskGraphDataset(**dataset_kwargs, activity_ids=val_ids)
    test_ds  = TaskGraphDataset(**dataset_kwargs, activity_ids=test_ids)
    print(f"Recordings — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}\n")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    # --- Model ---
    torch.manual_seed(args.seed)
    model = DAGClassifier(
        in_channels=256,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # pos_weight derived from train split only (no leakage from val/test)
    n_pos = sum(s["label"] for s in train_ds.samples)
    n_neg = len(train_ds.samples) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-5)

    # --- WandB ---
    use_wandb = args.enable_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project="gnn_task_verification", name="train_val_test", reinit=True)

    # --- Checkpoint ---
    ckpt_dir  = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"

    best_val_auc   = -1.0
    no_improve_cnt = 0

    # --- Training loop ---
    for epoch in range(1, args.num_epochs + 1):
        train_loss  = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = eval_split(model, val_loader, criterion, args.threshold, device)

        val_auc = val_metrics["auc"] if not np.isnan(val_metrics["auc"]) else 0.0
        scheduler.step(val_auc)

        if use_wandb:
            wandb.log({
                "epoch":        epoch,
                "train_loss":   train_loss,
                "val_loss":     val_metrics["loss"],
                "val_auc":      val_metrics["auc"],
                "val_f1":       val_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
                "lr":           optimizer.param_groups[0]["lr"],
            })

        improved = val_auc > best_val_auc
        if improved:
            best_val_auc   = val_auc
            no_improve_cnt = 0
            torch.save({
                "model":       model.state_dict(),
                "epoch":       epoch,
                "val_metrics": val_metrics,
            }, ckpt_path)
        else:
            no_improve_cnt += 1

        print(
            f"Epoch {epoch:03d}/{args.num_epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_auc={val_metrics['auc']:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}"
            + ("  ← best" if improved else "")
        )

        if no_improve_cnt >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no val AUC improvement for {args.patience} epochs).")
            break

    # --- Final test evaluation on best checkpoint ---
    print("\nLoading best checkpoint for test evaluation...")
    saved = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(saved["model"])
    test_metrics = eval_split(model, test_loader, criterion, args.threshold, device)

    if use_wandb:
        wandb.log({
            "test_auc":      test_metrics["auc"],
            "test_f1":       test_metrics["f1"],
            "test_accuracy": test_metrics["accuracy"],
        })
        wandb.finish()

    print(f"\n=== Final Test Results ===")
    print(f"AUC      : {test_metrics['auc']:.4f}")
    print(f"F1       : {test_metrics['f1']:.4f}")
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"(Best val AUC: {best_val_auc:.4f} at epoch {saved['epoch']})")

    # --- Save results.csv ---
    csv_path = out / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "auc", "f1", "accuracy"])
        writer.writeheader()
        writer.writerow({
            "split":    "val",
            "auc":      f"{saved['val_metrics']['auc']:.4f}",
            "f1":       f"{saved['val_metrics']['f1']:.4f}",
            "accuracy": f"{saved['val_metrics']['accuracy']:.4f}",
        })
        writer.writerow({
            "split":    "test",
            "auc":      f"{test_metrics['auc']:.4f}",
            "f1":       f"{test_metrics['f1']:.4f}",
            "accuracy": f"{test_metrics['accuracy']:.4f}",
        })
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
