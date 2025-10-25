"""Stage-1 encoder pretraining script."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import yaml

from infrastructure.config_loader import load_config as load_env_config
from models.encoder import StageOneEncoder
from data.price_action_dataset import PriceActionDataset, train_val_split

try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the AGV2 encoder")
    parser.add_argument("--data", required=True, help="Path to parquet file with training windows")
    parser.add_argument("--config", default="config/encoder.yaml", help="Encoder YAML config path")
    parser.add_argument("--env", default=".env", help="Optional .env file for DB/Broker settings")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    diff = target.unsqueeze(-1) - pred
    loss = torch.maximum(quantiles * diff, (quantiles - 1.0) * diff)
    return loss.mean()


def _compute_losses(
    config: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    device: torch.device,
    mask: torch.Tensor | None,
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    features = batch["features"].to(device)

    if "masked_reconstruction" in predictions:
        target_dim = predictions["masked_reconstruction"].shape[-1]
        mask_tensor = mask
        if mask_tensor is None:
            ratio = float(config.get("heads", {}).get("masked_reconstruction", {}).get("mask_ratio", 0.15))
            mask_tensor = torch.rand(features.size(0), features.size(1), device=device) < ratio
        preds = predictions["masked_reconstruction"]
        target = features[..., :target_dim]
        masked_pred = preds[mask_tensor]
        masked_target = target[mask_tensor]
        losses["masked_reconstruction"] = F.mse_loss(masked_pred, masked_target)

    if "regime_classifier" in predictions and "regime_label" in batch:
        logits = predictions["regime_classifier"]
        target = batch["regime_label"].to(device)
        smoothing = float(config.get("training", {}).get("label_smoothing", 0.0))
        if smoothing > 0:
            num_classes = logits.size(-1)
            smooth = smoothing / (num_classes - 1)
            one_hot = torch.full_like(logits, smooth)
            one_hot.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
            log_probs = F.log_softmax(logits, dim=-1)
            losses["regime_classifier"] = -(one_hot * log_probs).sum(dim=-1).mean()
        else:
            losses["regime_classifier"] = F.cross_entropy(logits, target)

    if "sr_heatmap" in predictions and "sr_heatmap" in batch:
        raw_preds = predictions["sr_heatmap"]
        target = torch.clamp(batch["sr_heatmap"].to(device), 0.0, 1.0)
        losses["sr_heatmap"] = F.binary_cross_entropy_with_logits(raw_preds, target)

    if "quantile_forecasting" in predictions and "future_returns" in batch:
        preds = predictions["quantile_forecasting"]
        target = batch["future_returns"].to(device)
        head_cfg = config.get("heads", {}).get("quantile_forecasting", {})
        if head_cfg.get("quantiles"):
            quantiles = torch.tensor(head_cfg["quantiles"], device=device, dtype=torch.float32)
        else:
            num_quantiles = preds.size(-1)
            quantiles = torch.linspace(1.0 / (num_quantiles + 1), num_quantiles / (num_quantiles + 1), num_quantiles, device=device)
        losses["quantile_forecasting"] = _pinball_loss(preds, target, quantiles)

    if "pattern_classifier" in predictions and "pattern_label" in batch:
        logits = predictions["pattern_classifier"]
        target = batch["pattern_label"].to(device)
        losses["pattern_classifier"] = F.cross_entropy(logits, target)

    return losses


def _weighted_total(losses: Dict[str, torch.Tensor], config: Dict[str, Any]) -> torch.Tensor:
    if not losses:
        return torch.tensor(0.0)
    heads_cfg = config.get("heads", {})
    total = torch.zeros((), device=next(iter(losses.values())).device)
    for name, loss in losses.items():
        weight = float(heads_cfg.get(name, {}).get("loss_weight", 1.0))
        total = total + loss * weight
    return total


def run_epoch(
    model: StageOneEncoder,
    loader: DataLoader,
    optimizer,
    config: Dict[str, Any],
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    train: bool,
) -> Dict[str, float]:
    model.train(mode=train)
    mask_cfg = config.get("heads", {}).get("masked_reconstruction", {})
    mask_ratio = float(mask_cfg.get("mask_ratio", 0.15)) if mask_cfg else 0.0
    accumulation = int(config.get("training", {}).get("accumulation_steps", 1))
    clip_norm = float(config.get("training", {}).get("gradient_clip_norm", 0.0))
    stats: Dict[str, float] = {}
    steps = 0
    if train:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        features = batch["features"].to(device)
        mask = None
        masked_features = features
        if mask_cfg:
            mask = (torch.rand(features.size(0), features.size(1), device=device) < mask_ratio)
            masked_features = features.clone()
            masked_features[mask] = 0.0
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            preds = model(masked_features)
            losses = _compute_losses(config, batch, preds, device, mask)
            if not losses:
                continue
            total_loss = _weighted_total(losses, config)
            if train:
                total_loss = total_loss / accumulation
        if train:
            scaler.scale(total_loss).backward()
            if (step + 1) % accumulation == 0:
                if clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        for name, loss in losses.items():
            stats[name] = stats.get(name, 0.0) + float(loss.detach())
        stats["total"] = stats.get("total", 0.0) + float(total_loss.detach() * (accumulation if train else 1.0))
        steps += 1
    if steps == 0:
        return stats
    return {k: v / steps for k, v in stats.items()}


def main() -> None:
    args = _build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    if args.env and os.path.exists(args.env):
        load_env_config(args.env)

    config = _load_yaml(args.config)
    arch_cfg = config.get("architecture", {})
    input_cfg = arch_cfg.get("input", {})
    window_len = int(input_cfg.get("window_length", 512))
    feature_dim = int(input_cfg.get("feature_dim", 12))

    dataset = PriceActionDataset(args.data, window_len, feature_dim, required_columns=["features"])
    training_cfg = config.get("training", {})
    val_split = float(training_cfg.get("val_split", 0.2))
    train_ds, val_ds = train_val_split(dataset, val_split, seed=args.seed)

    batch_size = int(training_cfg.get("batch_size", 64))
    num_workers = int(config.get("compute", {}).get("num_workers", 0))
    pin_memory = bool(config.get("compute", {}).get("pin_memory", False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    device = torch.device(args.device or config.get("compute", {}).get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model = StageOneEncoder(config).to(device)

    optimizer_name = training_cfg.get("optimizer", "adamw").lower()
    lr = float(training_cfg.get("learning_rate", 3e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    betas = tuple(training_cfg.get("betas", (0.9, 0.999)))
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer {optimizer_name}")

    scheduler_cfg = training_cfg.get("lr_scheduler", {})
    scheduler = None
    if scheduler_cfg.get("type") == "cosine_annealing":
        t_max = int(scheduler_cfg.get("T_max", training_cfg.get("num_epochs", 100)))
        min_lr = float(scheduler_cfg.get("min_lr", 1e-5))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)

    compute_cfg = config.get("compute", {})
    amp_enabled = bool(compute_cfg.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val = math.inf
    early_cfg = training_cfg.get("early_stopping", {})
    patience = int(early_cfg.get("patience", 20)) if early_cfg.get("enabled", False) else None
    epochs = int(training_cfg.get("num_epochs", 100))
    patience_ctr = 0

    use_mlflow = mlflow is not None and config.get("logging", {}).get("use_mlflow", False)
    if use_mlflow:
        mlflow.set_experiment(config.get("logging", {}).get("experiment_name", "agv2_encoder_pretraining"))
        mlflow.start_run()
        mlflow.log_params({"optimizer": optimizer_name, "learning_rate": lr, "weight_decay": weight_decay})

    for epoch in range(1, epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, config, device, scaler, train=True)
        val_stats = run_epoch(model, val_loader, optimizer, config, device, scaler, train=False)
        if scheduler is not None:
            scheduler.step()
        val_loss = val_stats.get("total", float("inf"))
        if use_mlflow:
            mlflow.log_metrics({f"train/{k}": v for k, v in train_stats.items()}, step=epoch)
            mlflow.log_metrics({f"val/{k}": v for k, v in val_stats.items()}, step=epoch)
        print(f"Epoch {epoch:03d} | train_total={train_stats.get('total', 0.0):.4f} | val_total={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            save_cfg = config.get("checkpointing", {})
            if save_cfg.get("save_best", True):
                save_dir = Path(save_cfg.get("save_dir", "models/encoders"))
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "config": config}, save_dir / "encoder_best.pt")
        else:
            patience_ctr += 1
        if patience is not None and patience_ctr >= patience:
            print("Early stopping triggered")
            break
    if use_mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    main()
