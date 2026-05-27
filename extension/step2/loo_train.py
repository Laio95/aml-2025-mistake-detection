"""
loo_train.py
============
Extension Step 2 — Task Verification Baseline.

Leave-One-Out (LOO) training and evaluation of TaskVerifier.
For each recipe k (identified by activity_id):
    - train on all videos of recipes {1..K} \ {k}
    - test  on videos of recipe k

Final output: mean ± std of AUC, F1, Accuracy over all folds,
saved to output_dir/results.csv.
"""

import argparse
import csv
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import wandb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from extension.step2.dataset import TaskVerificationDataset, load_samples
from extension.step2.model import TaskVerifier


# ─────────────────────────────────────────────────────────────────────────────
# LOO split construction
# ─────────────────────────────────────────────────────────────────────────────

def make_loo_splits(samples: list) -> list:
    """
    Group samples by activity_id and build Leave-One-Out folds.

    Returns:
        List of (activity_id, train_samples, test_samples), sorted by activity_id.
    """
    by_recipe = defaultdict(list)
    for s in samples:
        by_recipe[s["activity_id"]].append(s)

    all_ids = sorted(by_recipe.keys())
    folds = []
    for held_out in all_ids:
        test_samples  = by_recipe[held_out]
        train_samples = [s for aid in all_ids if aid != held_out
                           for s in by_recipe[aid]]
        folds.append((held_out, train_samples, test_samples))
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — one pass over a DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def eval_fold(model, loader, criterion, device, threshold: float = 0.5):
    """
    Evaluate model on one DataLoader.

    Returns:
        losses  : list of per-batch scalar loss values
        metrics : dict {auc, f1, accuracy}  (auc=nan if test fold has one class)
    """
    model.eval()
    all_probs, all_labels, losses = [], [], []

    with torch.no_grad():
        for embeddings, label in tqdm(loader, desc="  eval", leave=False):
            embeddings = embeddings.to(device)   # (1, N_steps, 256)
            label      = label.to(device)         # (1,)

            logit = model(embeddings)             # (1,)
            loss  = criterion(logit, label)
            losses.append(loss.item())

            all_probs.append(torch.sigmoid(logit).cpu().item())
            all_labels.append(label.cpu().item())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs >= threshold).astype(int)

    # AUC is undefined when the test fold contains only one class
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        warnings.warn("AUC undefined for this fold (single class in test set). Logging nan.")
        auc = float("nan")

    return losses, {
        "auc":      auc,
        "f1":       f1_score(all_labels, preds, zero_division=0),
        "accuracy": accuracy_score(all_labels, preds),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training — one LOO fold
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(fold_idx, activity_id, train_loader, test_loader, args, device, output_dir, pos_weight: float):
    """
    Train TaskVerifier for one LOO fold and return the best metrics achieved.

    Follows the same training conventions as base.py:
        - Adam optimizer
        - BCEWithLogitsLoss with pos_weight
        - ReduceLROnPlateau scheduler (monitors AUC)
        - gradient clipping (max_norm=1.0)
        - best model saved by AUC on test fold
        - wandb logging (one run per fold)
    """
    model = TaskVerifier(
        d_model=256,
        nhead=4,
        num_layers=args.num_layers,
        dim_feedforward=512,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10,
        threshold=1e-4, threshold_mode="abs", min_lr=1e-5,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )

    ckpt_dir  = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fold_{fold_idx}_recipe_{activity_id}_best.pt"

    best = {"auc": 0.0, "f1": 0.0, "accuracy": 0.0}

    if args.enable_wandb:
        wandb.init(
            project="task_verification_loo",
            name=f"fold_{fold_idx}_recipe_{activity_id}",
            config=vars(args),
            reinit=True,   # required when calling wandb.init multiple times in one process
        )

    for epoch in range(1, args.num_epochs + 1):

        # ── train ─────────────────────────────────────────────────────────
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"  epoch {epoch}/{args.num_epochs}", leave=False)
        for embeddings, label in pbar:
            embeddings = embeddings.to(device)   # (1, N_steps, 256)
            label      = label.to(device)         # (1,)

            optimizer.zero_grad()
            logit = model(embeddings)             # (1,)
            loss  = criterion(logit, label)

            if torch.isnan(loss):
                print(f"    [WARN] NaN loss at epoch {epoch}, skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # ── eval ──────────────────────────────────────────────────────────
        test_losses, metrics = eval_fold(
            model, test_loader, criterion, device, args.threshold
        )
        avg_test_loss = float(np.mean(test_losses))

        # scheduler monitors AUC; fall back to 0 when AUC is nan (single-class fold)
        scheduler.step(metrics["auc"] if not np.isnan(metrics["auc"]) else 0.0)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Fold {fold_idx} | Recipe {activity_id:2d} | Epoch {epoch:3d} | "
            f"train_loss={avg_train_loss:.4f}  test_loss={avg_test_loss:.4f}  "
            f"AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  "
            f"Acc={metrics['accuracy']:.4f}  lr={current_lr:.2e}"
        )

        if args.enable_wandb:
            wandb.log({
                "epoch":        epoch,
                "train_loss":   avg_train_loss,
                "test_loss":    avg_test_loss,
                "test_auc":     metrics["auc"],
                "test_f1":      metrics["f1"],
                "test_accuracy": metrics["accuracy"],
                "lr":           current_lr,
            })

        # ── checkpoint ────────────────────────────────────────────────────
        if not np.isnan(metrics["auc"]) and metrics["auc"] > best["auc"]:
            best = metrics.copy()
            torch.save(model.state_dict(), ckpt_path)

    if args.enable_wandb:
        wandb.finish()

    print(
        f"  → Best: AUC={best['auc']:.4f}  "
        f"F1={best['f1']:.4f}  Acc={best['accuracy']:.4f}"
    )
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LOO training for Task Verification baseline (Extension Step 2)"
    )
    # ── paths ──────────────────────────────────────────────────────────────
    parser.add_argument("--annotations_path",    required=True,
                        help="path to annotations/annotation_json/complete_step_annotations.json")
    parser.add_argument("--step_embeddings_dir", required=True,
                        help="directory containing *_step_embeddings.npz files (Step 1 output)")
    parser.add_argument("--output_dir",          required=True,
                        help="where to save checkpoints/ and results.csv")

    # ── training ───────────────────────────────────────────────────────────
    parser.add_argument("--num_epochs",   type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--threshold",    type=float, default=0.5,
                        help="sigmoid threshold for F1 and Accuracy (default: 0.5)")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--num_workers",  type=int,   default=2)

    # ── model ──────────────────────────────────────────────────────────────
    parser.add_argument("--num_layers",   type=int,   default=2)
    parser.add_argument("--dropout",      type=float, default=0.5)

    # ── logging ────────────────────────────────────────────────────────────
    parser.add_argument("--enable_wandb", action="store_true",
                        help="enable Weights & Biases logging (one run per fold)")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load dataset ───────────────────────────────────────────────────────
    samples = load_samples(args.annotations_path, args.step_embeddings_dir)

    # ── pos_weight info ────────────────────────────────────────────────────
    n_correct   = sum(s["label"] == 0 for s in samples)
    n_incorrect = sum(s["label"] == 1 for s in samples)
    print(f"Class distribution — correct: {n_correct}, incorrect: {n_incorrect}")
    print(f"Global pos_weight = {n_correct / n_incorrect:.4f} (fold-adaptive per fold)\n")

    # ── build LOO folds ────────────────────────────────────────────────────
    folds = make_loo_splits(samples)
    print(f"LOO: {len(folds)} folds (one per recipe)\n")

    loader_kwargs = dict(batch_size=1, num_workers=args.num_workers, pin_memory=False)

    # ── LOO loop ───────────────────────────────────────────────────────────
    all_results = []
    for fold_idx, (activity_id, train_samples, test_samples) in enumerate(folds, start=1):
        print(f"\n{'='*70}")
        print(f"Fold {fold_idx}/{len(folds)} — Recipe {activity_id} "
              f"| train={len(train_samples)}  test={len(test_samples)}")
        print(f"{'='*70}")

        train_ds = TaskVerificationDataset(train_samples, args.step_embeddings_dir)
        test_ds  = TaskVerificationDataset(test_samples,  args.step_embeddings_dir)

        train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
        test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

        n_pos   = sum(s["label"] == 1 for s in train_samples)
        n_neg   = sum(s["label"] == 0 for s in train_samples)
        fold_pw = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"  fold pos_weight = {fold_pw:.4f}  "
              f"(n_correct={n_neg}, n_incorrect={n_pos})")

        best = train_fold(
            fold_idx=fold_idx,
            activity_id=activity_id,
            train_loader=train_loader,
            test_loader=test_loader,
            args=args,
            device=device,
            output_dir=args.output_dir,
            pos_weight=fold_pw,
        )
        all_results.append({"fold": fold_idx, "activity_id": activity_id, **best})

    # ── aggregate across folds ─────────────────────────────────────────────
    valid_aucs = [r["auc"] for r in all_results if not np.isnan(r["auc"])]
    f1s        = [r["f1"]       for r in all_results]
    accs       = [r["accuracy"] for r in all_results]

    summary = {
        "auc_mean":      np.mean(valid_aucs), "auc_std":      np.std(valid_aucs),
        "f1_mean":       np.mean(f1s),        "f1_std":       np.std(f1s),
        "accuracy_mean": np.mean(accs),       "accuracy_std": np.std(accs),
    }

    print(f"\n{'='*70}")
    print("LOO RESULTS (mean ± std over all folds):")
    print(f"  AUC:      {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"  F1:       {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"  Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    if len(valid_aucs) < len(folds):
        print(f"  [NOTE] AUC computed on {len(valid_aucs)}/{len(folds)} folds "
              f"({len(folds)-len(valid_aucs)} skipped: single-class test fold)")
    print(f"{'='*70}\n")

    # ── save results CSV ───────────────────────────────────────────────────
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"

    with open(csv_path, "w", newline="") as f:
        fieldnames = ["fold", "activity_id", "auc", "f1", "accuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
        writer.writerow({
            "fold":       "mean±std",
            "activity_id": "-",
            "auc":        f"{summary['auc_mean']:.4f}±{summary['auc_std']:.4f}",
            "f1":         f"{summary['f1_mean']:.4f}±{summary['f1_std']:.4f}",
            "accuracy":   f"{summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f}",
        })

    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
