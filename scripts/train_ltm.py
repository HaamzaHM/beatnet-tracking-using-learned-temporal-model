#!/usr/bin/env python3
"""
Train LearnedTemporalModel on beat/downbeat annotations with class imbalance handling.

Usage:
    python3 scripts/train_ltm.py --epochs 10 --batch_size 8 --lr 1e-3
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from BeatNet.ltm_model import LearnedTemporalModel

# Import dataset loader
sys.path.insert(0, str(Path(__file__).parent))
from dataset_loader_ltm import BeatLTMFrameDataset, collate_fn_ltm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def compute_pos_weights(train_dataset, device):
    """
    Compute positive weights for beat and downbeat from training set.
    pos_weight = num_negatives / num_positives
    
    Args:
        train_dataset: PyTorch Subset with training samples
        device: torch device
        
    Returns:
        Tuple of (beat_pos_weight, downbeat_pos_weight) as scalars
    """
    beat_counts = [0, 0]      # [negative, positive]
    downbeat_counts = [0, 0]  # [negative, positive]
    
    logger.info("\nComputing class weights from training set...")
    for i in range(len(train_dataset)):
        try:
            _, target, _ = train_dataset.dataset[train_dataset.indices[i]]
            
            # Count beat labels
            beat_positive = (target[:, 0] == 1).sum().item()
            beat_negative = (target[:, 0] == 0).sum().item()
            beat_counts[0] += beat_negative
            beat_counts[1] += beat_positive
            
            # Count downbeat labels
            downbeat_positive = (target[:, 1] == 1).sum().item()
            downbeat_negative = (target[:, 1] == 0).sum().item()
            downbeat_counts[0] += downbeat_negative
            downbeat_counts[1] += downbeat_positive
        except Exception:
            continue
    
    # Compute positive weights (ratio of negative to positive)
    beat_pos_weight = beat_counts[0] / max(beat_counts[1], 1)
    downbeat_pos_weight = downbeat_counts[0] / max(downbeat_counts[1], 1)
    
    logger.info(f"Beat:      {beat_counts[1]:,} positive, {beat_counts[0]:,} negative → pos_weight: {beat_pos_weight:.4f}")
    logger.info(f"Downbeat:  {downbeat_counts[1]:,} positive, {downbeat_counts[0]:,} negative → pos_weight: {downbeat_pos_weight:.4f}")
    
    return beat_pos_weight, downbeat_pos_weight


def compute_metrics(outputs, targets, lengths, threshold=0.5):
    """
    Compute beat and downbeat accuracy and F1.
    
    Args:
        outputs: (batch, max_frames, 2) logits
        targets: (batch, max_frames, 2) binary labels
        lengths: (batch,) actual lengths before padding
        threshold: threshold for binarization
        
    Returns:
        Dict with beat_acc, downbeat_acc, beat_f1, downbeat_f1
    """
    max_frames = outputs.shape[1]
    
    # Create mask for valid frames
    mask = torch.arange(max_frames, device=outputs.device).unsqueeze(0) < lengths.unsqueeze(1)
    
    # Apply sigmoid and threshold
    probs = torch.sigmoid(outputs)
    preds = (probs >= threshold).float()
    
    # Get valid predictions and targets
    valid_preds = preds[mask]      # (num_valid_frames, 2)
    valid_targets = targets[mask]  # (num_valid_frames, 2)
    
    # Compute metrics per output dimension
    metrics = {}
    for dim, label in enumerate(["beat", "downbeat"]):
        dim_preds = valid_preds[:, dim]
        dim_targets = valid_targets[:, dim]
        
        # Accuracy
        acc = (dim_preds == dim_targets).float().mean().item()
        metrics[f"{label}_acc"] = acc
        
        # F1 (simple: TP / (TP + 0.5*(FP + FN)))
        tp = ((dim_preds == 1) & (dim_targets == 1)).float().sum().item()
        fp = ((dim_preds == 1) & (dim_targets == 0)).float().sum().item()
        fn = ((dim_preds == 0) & (dim_targets == 1)).float().sum().item()
        
        if tp + fp + fn == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f"{label}_f1"] = f1
    
    return metrics


def train_epoch(model, train_loader, beat_criterion, downbeat_criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    num_skipped = 0
    
    all_metrics = {
        "beat_acc": [],
        "downbeat_acc": [],
        "beat_f1": [],
        "downbeat_f1": [],
    }
    
    for ltm_input, target, audio_ids, lengths in train_loader:
        try:
            ltm_input = ltm_input.to(device)
            target = target.to(device)
            lengths = lengths.to(device)
            
            # Forward
            output = model(ltm_input)  # (batch, max_frames, 2)
            
            # Create mask for valid frames
            max_frames = ltm_input.shape[1]
            mask = torch.arange(max_frames, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            
            # Loss: compute separately for beat and downbeat with their pos_weights
            beat_loss = beat_criterion(output[mask, 0], target[mask, 0])
            downbeat_loss = downbeat_criterion(output[mask, 1], target[mask, 1])
            loss = beat_loss + downbeat_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Compute metrics
            with torch.no_grad():
                batch_metrics = compute_metrics(output, target, lengths)
                for key, val in batch_metrics.items():
                    all_metrics[key].append(val)
        except Exception as e:
            num_skipped += 1
            logger.warning(f"Skipped batch: {str(e)[:100]}")
            continue
    
    # Average metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_metrics = {key: np.mean(vals) for key, vals in all_metrics.items()}
    
    if num_skipped > 0:
        logger.warning(f"Skipped {num_skipped} batches during training")
    
    return avg_loss, avg_metrics


def validate(model, val_loader, beat_criterion, downbeat_criterion, device):
    """Validate on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    num_skipped = 0
    
    all_metrics = {
        "beat_acc": [],
        "downbeat_acc": [],
        "beat_f1": [],
        "downbeat_f1": [],
    }
    
    with torch.no_grad():
        for ltm_input, target, audio_ids, lengths in val_loader:
            try:
                ltm_input = ltm_input.to(device)
                target = target.to(device)
                lengths = lengths.to(device)
                
                # Forward
                output = model(ltm_input)
                
                # Create mask for valid frames
                max_frames = ltm_input.shape[1]
                mask = torch.arange(max_frames, device=device).unsqueeze(0) < lengths.unsqueeze(1)
                
                # Loss: compute separately for beat and downbeat with their pos_weights
                beat_loss = beat_criterion(output[mask, 0], target[mask, 0])
                downbeat_loss = downbeat_criterion(output[mask, 1], target[mask, 1])
                loss = beat_loss + downbeat_loss
                
                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1
                
                # Compute metrics
                batch_metrics = compute_metrics(output, target, lengths)
                for key, val in batch_metrics.items():
                    all_metrics[key].append(val)
            except Exception as e:
                num_skipped += 1
                logger.warning(f"Skipped batch: {str(e)[:100]}")
                continue
    
    # Average metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_metrics = {key: np.mean(vals) if vals else 0.0 for key, vals in all_metrics.items()}
    
    if num_skipped > 0:
        logger.warning(f"Skipped {num_skipped} batches during validation")
    
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train LearnedTemporalModel with class imbalance handling")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--cache_dir", type=str, default="data/crnn_cache", help="Cache directory")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames per sample")
    parser.add_argument("--model_type", type=str, default="tcn", choices=["tcn", "transformer"], help="Model architecture")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    
    args = parser.parse_args()
    
    # Setup
    project_root = Path(__file__).parent.parent
    index_csv = project_root / "data" / "dataset_index.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create dataset
    logger.info(f"\nLoading dataset from {index_csv}...")
    dataset = BeatLTMFrameDataset(
        index_csv=index_csv,
        project_root=project_root,
        fps=50,
        max_frames=args.max_frames,
        cache_dir=args.cache_dir,
        verbose=False,
    )
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    logger.info(f"Train: {train_size}, Val: {val_size}")
    
    # Compute positive weights from training set
    beat_pos_weight, downbeat_pos_weight = compute_pos_weights(train_dataset, device)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_ltm,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_ltm,
        num_workers=0,
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    logger.info(f"\nCreating model ({args.model_type})...")
    model = LearnedTemporalModel(
        input_dim=2,
        hidden_dim=128,
        num_layers=4,
        output_dim=2,
        architecture=args.model_type,
        device=device,
    )
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Loss functions with pos_weight for class imbalance
    beat_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(beat_pos_weight, device=device))
    downbeat_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(downbeat_pos_weight, device=device))
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # Training loop
    logger.info(f"\nTraining for {args.epochs} epochs...\n")
    
    best_val_loss = float("inf")
    best_epoch = 0
    
    for epoch in range(args.epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, beat_criterion, downbeat_criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, beat_criterion, downbeat_criterion, device)
        scheduler.step()
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Beat F1: {val_metrics['beat_f1']:.4f} | Val Downbeat F1: {val_metrics['downbeat_f1']:.4f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_weights_path = models_dir / "ltm_weights.pt"
            torch.save(model.state_dict(), best_weights_path)
            logger.info(f"  ✓ Saved best model to {best_weights_path}")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("Training Complete!")
    logger.info("="*80)
    logger.info(f"Best epoch: {best_epoch}/{args.epochs}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best weights saved to: {models_dir / 'ltm_weights.pt'}")
    logger.info(f"Model architecture: {args.model_type}")
    logger.info(f"Batch size: {args.batch_size}, Learning rate: {args.lr}")
    logger.info(f"Beat pos_weight: {beat_pos_weight:.4f}, Downbeat pos_weight: {downbeat_pos_weight:.4f}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
