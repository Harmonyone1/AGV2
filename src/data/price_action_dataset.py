"""Dataset helpers for encoder pretraining."""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


def _reshape_feature_array(arr: Sequence[float], window_len: int, feature_dim: int) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float32)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    expected = window_len * feature_dim
    if flat.size != expected:
        raise ValueError(f"Feature vector has length {flat.size}, expected {expected}")
    return flat.reshape(window_len, feature_dim)


class PriceActionDataset(Dataset):
    """Loads sliding windows of price-action features and optional targets."""

    def __init__(
        self,
        parquet_path: str,
        window_length: int,
        feature_dim: int,
        required_columns: Optional[Sequence[str]] = None,
        chunk_size: int = 256,
    ) -> None:
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {parquet_path}")
        self.parquet_path = str(path)
        self.window_length = int(window_length)
        self.feature_dim = int(feature_dim)
        self.chunk_size = int(max(1, chunk_size))
        self._features_file: Optional[str] = None
        self._features: Optional[np.memmap] = None
        self._regime: Optional[np.ndarray] = None
        self._pattern: Optional[np.ndarray] = None
        self._sr: Optional[List[Optional[List[float]]]] = None
        self._quantiles: Optional[List[Optional[List[float]]]] = None

        self.length = self._count_rows()
        if self.length == 0:
            raise ValueError("Dataset parquet contains no rows")
        self._load_in_chunks(required_columns)

    def _count_rows(self) -> int:
        lazy = pl.scan_parquet(self.parquet_path)
        return int(lazy.select(pl.len()).collect().item())

    def _load_in_chunks(self, required_columns: Optional[Sequence[str]]) -> None:
        lazy = pl.scan_parquet(self.parquet_path)
        first_chunk = lazy.slice(0, min(self.chunk_size, self.length)).collect(streaming=True)
        missing = []
        for col in required_columns or []:
            if col not in first_chunk.columns:
                missing.append(col)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self._features_file = tempfile.NamedTemporaryFile(prefix="agv2_features_", suffix=".mmap", delete=False).name
        self._features = np.memmap(
            self._features_file,
            dtype=np.float32,
            mode="w+",
            shape=(self.length, self.window_length, self.feature_dim),
        )

        if "regime_label" in first_chunk.columns:
            self._regime = np.empty(self.length, dtype=np.int64)
        if "pattern_label" in first_chunk.columns:
            self._pattern = np.empty(self.length, dtype=np.int64)
        if "sr_heatmap" in first_chunk.columns:
            self._sr = [None] * self.length
        if "future_returns" in first_chunk.columns:
            self._quantiles = [None] * self.length

        offset = 0
        for chunk in self._chunk_iterator(first_chunk):
            self._process_chunk(chunk, offset)
            offset += chunk.height

    def _chunk_iterator(self, first_chunk: pl.DataFrame):
        yield first_chunk
        processed = first_chunk.height
        while processed < self.length:
            remaining = self.length - processed
            take = min(self.chunk_size, remaining)
            chunk = pl.scan_parquet(self.parquet_path).slice(processed, take).collect(streaming=True)
            yield chunk
            processed += chunk.height

    def _process_chunk(self, chunk: pl.DataFrame, offset: int) -> None:
        if self._features is None:
            raise RuntimeError("Feature storage not initialized")
        feature_lists = chunk["features"].to_list()
        block = np.stack([
            _reshape_feature_array(row, self.window_length, self.feature_dim) for row in feature_lists
        ])
        end = offset + block.shape[0]
        self._features[offset:end] = block

        if self._regime is not None and "regime_label" in chunk.columns:
            self._regime[offset:end] = chunk["regime_label"].to_numpy().astype(np.int64)
        if self._pattern is not None and "pattern_label" in chunk.columns:
            self._pattern[offset:end] = chunk["pattern_label"].to_numpy().astype(np.int64)
        if self._sr is not None and "sr_heatmap" in chunk.columns:
            sr_values = chunk["sr_heatmap"].to_list()
            for idx, value in enumerate(sr_values):
                self._sr[offset + idx] = value
        if self._quantiles is not None and "future_returns" in chunk.columns:
            fut_values = chunk["future_returns"].to_list()
            for idx, value in enumerate(fut_values):
                self._quantiles[offset + idx] = value

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._features is None:
            raise RuntimeError("Dataset not initialized correctly")
        sample: Dict[str, Any] = {
            "features": torch.from_numpy(np.array(self._features[idx], copy=False)),
        }
        if self._regime is not None:
            sample["regime_label"] = torch.tensor(int(self._regime[idx]), dtype=torch.long)
        if self._pattern is not None:
            sample["pattern_label"] = torch.tensor(int(self._pattern[idx]), dtype=torch.long)
        if self._sr is not None and self._sr[idx] is not None:
            sample["sr_heatmap"] = torch.tensor(self._sr[idx], dtype=torch.float32)
        if self._quantiles is not None and self._quantiles[idx] is not None:
            sample["future_returns"] = torch.tensor(self._quantiles[idx], dtype=torch.float32)
        return sample

    def __del__(self) -> None:
        try:
            if self._features is not None:
                self._features._mmap.close()  # type: ignore[attr-defined]
            if self._features_file and os.path.exists(self._features_file):
                os.remove(self._features_file)
        except Exception:
            pass


def train_val_split(dataset: PriceActionDataset, val_fraction: float, seed: int = 42):
    length = len(dataset)
    indices = np.arange(length)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = int(math.floor(length * val_fraction))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, val_idx)
