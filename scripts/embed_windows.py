"""Add encoder embeddings to window datasets."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from data.price_action_dataset import PriceActionDataset
from models.encoder import StageOneEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run StageOneEncoder to add embeddings to a parquet dataset")
    parser.add_argument("--input", required=True, help="Input parquet with encoder windows")
    parser.add_argument("--checkpoint", required=True, help="Path to encoder checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Destination parquet with embeddings")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--device", default=None, help="Torch device override (cpu/cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    config = state.get("config")
    if config is None:
        raise ValueError("Checkpoint must contain 'config'")

    arch_cfg = config.get("architecture", {})
    input_cfg = arch_cfg.get("input", {})
    window_len = int(input_cfg.get("window_length", 512))
    feature_dim = int(input_cfg.get("feature_dim", 12))

    dataset = PriceActionDataset(args.input, window_length=window_len, feature_dim=feature_dim, required_columns=["features"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device_str = args.device or config.get("compute", {}).get("device", "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    model = StageOneEncoder(config)
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()

    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            outputs = model(feats)
            emb = outputs["embedding"].detach().cpu().numpy()
            embeddings.append(emb)
    if not embeddings:
        raise ValueError("No embeddings produced; dataset may be empty")
    full = np.concatenate(embeddings, axis=0)

    frame = pl.read_parquet(args.input)
    if frame.height != full.shape[0]:
        raise ValueError("Embedding count does not match dataset rows")
    embedding_series = pl.Series("embedding", [row.tolist() for row in full])
    frame = frame.drop("embedding") if "embedding" in frame.columns else frame
    frame = frame.with_columns(embedding_series)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    print(f"Wrote embeddings to {output_path}")


if __name__ == "__main__":
    main()
