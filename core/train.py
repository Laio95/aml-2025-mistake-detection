import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# --- FIX FOR MODULE IMPORTS ---
# This ensures that 'core' is recognized as a package by adding the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# ------------------------------

from core.dataset import CaptainCook4DDataset
from core.models.er_former import ErFormer
from constants import Constants as const

def train():
    parser = argparse.ArgumentParser(description='Train Mistake Detection Baselines')
    parser.add_argument('--variant', type=str, default='Transformer', choices=['Transformer', 'MLP'])
    parser.add_argument('--backbone', type=str, default='omnivore')
    parser.add_argument('--split', type=str, default='step', choices=['step', 'recordings'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"--- Training {args.variant} on {args.split} split using {args.backbone} ---")

    # 1. Load Datasets
    train_ds = CaptainCook4DDataset(args, split='train')
    test_ds = CaptainCook4DDataset(args, split='test')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Initialize Model (Transformer / Variant V2)
    model = ErFormer(args).to(args.device)
    
    # 3. Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0.0
    checkpoint_dir = os.path.join(parent_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(args.device), labels.to(args.device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(args.device)
                outputs = model(data)
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        # Metrics
        binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
        acc = accuracy_score(all_labels, binary_preds)
        f1 = f1_score(all_labels, binary_preds)
        auc = roc_auc_score(all_labels, all_preds)

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(checkpoint_dir, f"best_{args.variant}_{args.split}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")

if __name__ == '__main__':
    train()