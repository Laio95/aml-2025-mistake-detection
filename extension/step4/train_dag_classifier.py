"""
Extension Step 4 — DAGClassifier (DAGNN) Training
==================================================
Three split modes via --split_mode:

  video     CaptainCook4D recording-level split (combined_recordings.json).
            All 24 recipes appear in every split.

  recipe    16 train / 4 val / 4 test recipes, seed-controlled (--recipe_split_seed).
            All recordings of each recipe go to the same split.

  loo       Leave-One-Out: one fold per recipe (24 folds), same protocol as B2.
            Enables protocol-consistent comparison with TaskVerifier (B2).

TVT modes (video / recipe):
  Val set drives LR scheduling and checkpoint selection.
  Test set is evaluated once at the end on the best-val-AUC checkpoint.

LOO mode:
  No separate val set — the test fold drives the LR scheduler.
  One checkpoint per fold; results aggregated as mean ± std over 24 folds.
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

from extension.step4.dataset import TaskGraphDataset, load_samples
from extension.step4.dag_classifier import DAGClassifier

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Split builders
# ---------------------------------------------------------------------------

def load_split_from_json(splits_json: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Load train / val / test recording IDs from combined_recordings.json.
    Returns three sorted lists of recording_id strings.
    """
    with open(splits_json, "r") as f:
        data = json.load(f)
    db = data.get("database", data)
    split: dict = {"train": [], "val": [], "test": []}
    for vid, item in db.items():
        raw = str(item.get("subset", "")).strip().lower()
        if raw in {"training", "train"}:
            split["train"].append(vid)
        elif raw in {"validation", "val", "valid"}:
            split["val"].append(vid)
        elif raw in {"test", "testing"}:
            split["test"].append(vid)
    return sorted(split["train"]), sorted(split["val"]), sorted(split["test"])


def make_recipe_split(
    annotations_path: str,
    seed: int = 42,
    n_train: int = 16,
    n_val: int = 4,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split the 24 recipes into 16 train / 4 val / 4 test (seed-controlled).
    All recordings of a recipe land in the same split, preventing leakage
    across recipes.
    Returns three sorted lists of recording_id strings.
    """
    samples = load_samples(annotations_path)
    activity_ids = sorted({s["activity_id"] for s in samples})
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(activity_ids).tolist()
    train_recipes = set(shuffled[:n_train])
    val_recipes   = set(shuffled[n_train:n_train + n_val])
    test_recipes  = set(shuffled[n_train + n_val:])
    train_ids = sorted(s["recording_id"] for s in samples if s["activity_id"] in train_recipes)
    val_ids   = sorted(s["recording_id"] for s in samples if s["activity_id"] in val_recipes)
    test_ids  = sorted(s["recording_id"] for s in samples if s["activity_id"] in test_recipes)
    return train_ids, val_ids, test_ids


def make_loo_splits(activity_ids: List[int]):
    """
    Generate LOO folds: for each recipe (activity_id), yield
    (fold_idx, test_activity_id, train_activity_ids).
    """
    for fold_idx, test_id in enumerate(activity_ids, start=1):
        yield fold_idx, test_id, [aid for aid in activity_ids if aid != test_id]


# ---------------------------------------------------------------------------
# Shared training primitives  (used by both TVT and LOO modes)
# ---------------------------------------------------------------------------

def train_epoch(
    model:     DAGClassifier,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.BCEWithLogitsLoss,
    device:    torch.device,
) -> float:
    """Run one full training epoch. Returns average loss over batches."""
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
def eval_loader(
    model:     DAGClassifier,
    loader:    DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    threshold: float,
    device:    torch.device,
) -> Dict:
    """
    Evaluate the model on all batches in loader.
    Returns a dict with keys: loss, auc, f1, accuracy.
    AUC is NaN when only one class is present in the split (e.g. small test folds).
    """
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
# TVT (Train / Val / Test) mode  —  official split or recipe split
# ---------------------------------------------------------------------------

def train_tvt(args: argparse.Namespace, dataset_kwargs: Dict) -> None:
    """
    Train once with a fixed train/val/test split (official or recipe mode).

    Val AUC drives the LR scheduler and selects the best checkpoint.
    Test set is evaluated exactly once at the end on the best-val checkpoint.
    Results written to results.csv (val row + test row).
    """
    out_dir = Path(args.output_dir)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build split ---
    if args.split_mode == "video":
        if args.splits_json is None:
            raise ValueError("--splits_json required for --split_mode video")
        print("Loading video train/val/test split...")
        train_ids, val_ids, test_ids = load_split_from_json(args.splits_json)
        split_meta = {"split_mode": "video", "splits_json": args.splits_json}
    else:
        print(f"Building recipe split (seed={args.recipe_split_seed})...")
        train_ids, val_ids, test_ids = make_recipe_split(
            args.annotations_path, seed=args.recipe_split_seed,
        )
        split_meta = {"split_mode": "recipe", "recipe_split_seed": args.recipe_split_seed}

    print(f"Split: train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)} recordings")
    with open(out_dir / "split_info.json", "w") as f:
        json.dump({**split_meta, "train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2)

    # --- Datasets and loaders ---
    train_ds = TaskGraphDataset(**dataset_kwargs, video_ids=train_ids)
    val_ds   = TaskGraphDataset(**dataset_kwargs, video_ids=val_ids)
    test_ds  = TaskGraphDataset(**dataset_kwargs, video_ids=test_ids)
    print(f"Recordings — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    print("Verifying pre-fusion cache...")
    TaskGraphDataset(**dataset_kwargs).prebuild_cache()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Model, loss, optimizer ---
    torch.manual_seed(args.seed)
    model = DAGClassifier(in_channels=256, hidden_dim=args.hidden_dim,
                          num_layers=args.num_layers, dropout=args.dropout).to(device)

    n_pos     = sum(s["label"] for s in train_ds.samples)
    n_neg     = len(train_ds.samples) - n_pos
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=device)
    )
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-5)

    # --- WandB ---
    use_wandb = args.enable_wandb and WANDB_AVAILABLE
    if use_wandb:
        run_name = f"tvt_{args.split_mode}"
        if args.split_mode == "recipe":
            run_name += f"_s{args.recipe_split_seed}"
        wandb.init(project=args.wandb_project or "gnn_task_verification", name=run_name, reinit=True)

    # --- Training loop (val-guided checkpoint selection) ---
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path    = ckpt_dir / "best.pt"
    best_val_auc = -1.0

    for epoch in range(1, args.num_epochs + 1):
        train_loss  = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = eval_loader(model, val_loader, criterion, args.threshold, device)
        val_auc     = val_metrics["auc"] if not np.isnan(val_metrics["auc"]) else 0.0
        scheduler.step(val_auc)

        if use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss,
                       "val_loss": val_metrics["loss"], "val_auc": val_metrics["auc"],
                       "val_f1": val_metrics["f1"], "val_accuracy": val_metrics["accuracy"],
                       "lr": optimizer.param_groups[0]["lr"]})

        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_metrics": val_metrics}, ckpt_path)

        print(f"Epoch {epoch:03d}/{args.num_epochs}  train_loss={train_loss:.4f}"
              f"  val_auc={val_metrics['auc']:.4f}  val_f1={val_metrics['f1']:.4f}"
              + ("  ← best" if is_best else ""))

    # --- Final test evaluation on best checkpoint ---
    print("\nLoading best checkpoint for test evaluation...")
    saved        = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(saved["model"])
    test_metrics = eval_loader(model, test_loader, criterion, args.threshold, device)

    if use_wandb:
        wandb.log({"test_auc": test_metrics["auc"], "test_f1": test_metrics["f1"],
                   "test_accuracy": test_metrics["accuracy"]})
        wandb.finish()

    print(f"\n=== Final Test Results ===")
    print(f"AUC      : {test_metrics['auc']:.4f}")
    print(f"F1       : {test_metrics['f1']:.4f}")
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"(Best val AUC: {best_val_auc:.4f} at epoch {saved['epoch']})")

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "auc", "f1", "accuracy"])
        writer.writeheader()
        writer.writerow({"split": "val",  "auc": f"{saved['val_metrics']['auc']:.4f}",
                         "f1": f"{saved['val_metrics']['f1']:.4f}",
                         "accuracy": f"{saved['val_metrics']['accuracy']:.4f}"})
        writer.writerow({"split": "test", "auc": f"{test_metrics['auc']:.4f}",
                         "f1": f"{test_metrics['f1']:.4f}",
                         "accuracy": f"{test_metrics['accuracy']:.4f}"})
    print(f"Results saved to {csv_path}")


# ---------------------------------------------------------------------------
# LOO (Leave-One-Out) mode  —  24 folds, one per recipe
# ---------------------------------------------------------------------------

def _train_loo_fold(
    fold_idx:           int,
    test_activity_id:   int,
    train_activity_ids: List[int],
    args:               argparse.Namespace,
    dataset_kwargs:     Dict,
) -> Dict:
    """
    Train and evaluate a single LOO fold.

    There is no separate val set: the test fold drives the LR scheduler and
    checkpoint selection (same as B2 protocol).
    Returns a dict with fold metadata and best test metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Datasets and loaders ---
    train_ds = TaskGraphDataset(**dataset_kwargs, activity_ids=train_activity_ids)
    test_ds  = TaskGraphDataset(**dataset_kwargs, activity_ids=[test_activity_id])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Model, loss, optimizer ---
    torch.manual_seed(args.seed)
    model = DAGClassifier(in_channels=256, hidden_dim=args.hidden_dim,
                          num_layers=args.num_layers, dropout=args.dropout).to(device)

    n_pos     = sum(s["label"] for s in train_ds.samples)
    n_neg     = len(train_ds.samples) - n_pos
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=device)
    )
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-5)

    # --- WandB ---
    use_wandb = args.enable_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=args.wandb_project or "gnn_task_verification_loo",
                   name=f"fold_{fold_idx}_recipe_{test_activity_id}", reinit=True)

    # --- Per-fold checkpoint ---
    ckpt_dir  = Path(args.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fold_{fold_idx}_recipe_{test_activity_id}_best.pt"

    best_test_auc = -1.0
    best_metrics: Dict = {}

    # --- Training loop (test-fold-guided checkpoint selection, no val set) ---
    for epoch in range(1, args.num_epochs + 1):
        train_loss   = train_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = eval_loader(model, test_loader, criterion, args.threshold, device)
        scheduler.step(test_metrics["auc"] if not np.isnan(test_metrics["auc"]) else 0.0)

        if use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss,
                       "test_loss": test_metrics["loss"], "test_auc": test_metrics["auc"],
                       "test_f1": test_metrics["f1"], "test_accuracy": test_metrics["accuracy"],
                       "lr": optimizer.param_groups[0]["lr"]})

        if not np.isnan(test_metrics["auc"]) and test_metrics["auc"] > best_test_auc:
            best_test_auc = test_metrics["auc"]
            best_metrics  = test_metrics.copy()
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "metrics": test_metrics}, ckpt_path)

    if use_wandb:
        wandb.finish()

    return {"fold": fold_idx, "activity_id": test_activity_id, **best_metrics}


def train_loo(args: argparse.Namespace, dataset_kwargs: Dict) -> None:
    """
    Run the full 24-fold LOO training.

    The pre-fusion cache is built once and shared across all folds.
    Folds whose checkpoint already exists are skipped (safe to resume after interruption).
    Final results (mean ± std over folds) are printed and saved to results.csv.
    """
    out_dir = Path(args.output_dir)

    # --- Build cache once, shared across all folds ---
    print("Verifying pre-fusion cache (shared across all folds)...")
    full_ds          = TaskGraphDataset(**dataset_kwargs)
    full_ds.prebuild_cache()
    all_activity_ids = full_ds.get_activity_ids()
    print(f"LOO: {len(all_activity_ids)} folds | {len(full_ds)} total videos\n")

    # --- LOO loop with automatic resume ---
    all_results = []
    ckpt_dir    = out_dir / "checkpoints"

    for fold_idx, test_id, train_ids in make_loo_splits(all_activity_ids):
        ckpt_path = ckpt_dir / f"fold_{fold_idx}_recipe_{test_id}_best.pt"
        if ckpt_path.exists():
            saved  = torch.load(ckpt_path, weights_only=False)
            result = {"fold": fold_idx, "activity_id": test_id, **saved["metrics"]}
            all_results.append(result)
            print(f"Fold {fold_idx:02d}/{len(all_activity_ids)} — SKIP (checkpoint exists)")
            continue

        print(f"Fold {fold_idx:02d}/{len(all_activity_ids)} — test recipe {test_id}")
        result = _train_loo_fold(fold_idx, test_id, train_ids, args, dataset_kwargs)
        all_results.append(result)
        print(f"  AUC={result.get('auc', float('nan')):.4f}  "
              f"F1={result.get('f1', float('nan')):.4f}  "
              f"Acc={result.get('accuracy', float('nan')):.4f}")

    # --- Aggregate over valid folds (skip folds where AUC is NaN) ---
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

    csv_path   = out_dir / "results.csv"
    fieldnames = ["fold", "activity_id", "auc", "f1", "accuracy"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
        writer.writerow({"fold": "mean", "activity_id": "",
                         "auc": f"{mean_auc:.4f}", "f1": f"{mean_f1:.4f}", "accuracy": f"{mean_acc:.4f}"})
        writer.writerow({"fold": "std",  "activity_id": "",
                         "auc": f"{std_auc:.4f}",  "f1": f"{std_f1:.4f}",  "accuracy": f"{std_acc:.4f}"})
    print(f"Results saved to {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DAGClassifier (DAGNN) training — B4")

    # Paths
    parser.add_argument("--annotations_path",    required=True)
    parser.add_argument("--step_embeddings_dir", required=True)
    parser.add_argument("--graphs_dir",          required=True)
    parser.add_argument("--egovlp_repo",         required=True)
    parser.add_argument("--egovlp_ckpt",         required=True)
    parser.add_argument("--cache_dir",           required=True)
    parser.add_argument("--output_dir",          required=True)
    parser.add_argument("--splits_json",         default=None,
                        help="path to combined_recordings.json (required for --split_mode video)")

    # Split mode
    parser.add_argument("--split_mode", choices=["video", "recipe", "loo"], default="video")
    parser.add_argument("--recipe_split_seed", type=int, default=42,
                        help="RNG seed for recipe split (ignored for other modes)")

    # Model + training
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--hidden_dim",    type=int,   default=32)
    parser.add_argument("--num_layers",    type=int,   default=1)
    parser.add_argument("--dropout",       type=float, default=0.5)
    parser.add_argument("--num_epochs",    type=int,   default=50)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--threshold",     type=float, default=0.5)
    parser.add_argument("--num_workers",   type=int,   default=2)
    parser.add_argument("--enable_wandb",  action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB project name. Defaults to 'gnn_task_verification' "
                             "(tvt) or 'gnn_task_verification_loo' (loo).")

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

    if args.split_mode == "loo":
        train_loo(args, dataset_kwargs)
    else:
        train_tvt(args, dataset_kwargs)


if __name__ == "__main__":
    main()
