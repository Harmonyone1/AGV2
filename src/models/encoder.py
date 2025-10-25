"""Stage-1 encoder backbone and head factory for AGV2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderOutputs:
    """Container for encoder representations."""

    sequence: torch.Tensor  # shape: (batch, seq_len, embed_dim)
    pooled: torch.Tensor    # shape: (batch, embed_dim)


class CausalConv1d(nn.Module):
    """1D convolution with causal left padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Two-layer temporal block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + residual


class TemporalConvNet(nn.Module):
    """Stack of temporal blocks with exponentially increasing dilation."""

    def __init__(self, input_channels: int, channels: List[int], kernel_size: int, dropout: float, dilation_base: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = input_channels
        dilation = 1
        for out_ch in channels:
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
            dilation *= dilation_base
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EncoderBackbone(nn.Module):
    """Configurable encoder backbone (TCN implemented; others placeholder)."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        arch_cfg = config.get("architecture", {})
        model_type = arch_cfg.get("model_type", "tcn").lower()
        input_cfg = arch_cfg.get("input", {})
        self.input_channels = int(input_cfg.get("feature_dim", 1))
        self.embedding_dim = int(arch_cfg.get(model_type, {}).get("embedding_dim", 256))
        self.model_type = model_type

        if model_type == "tcn":
            tcn_cfg = arch_cfg.get("tcn", {})
            channels = list(tcn_cfg.get("num_channels", [64, 128, 256]))
            kernel_size = int(tcn_cfg.get("kernel_size", 3))
            dropout = float(tcn_cfg.get("dropout", 0.1))
            dilation_base = int(tcn_cfg.get("dilation_base", 2))
            if channels[-1] != self.embedding_dim:
                channels[-1] = self.embedding_dim
            self.backbone = TemporalConvNet(
                input_channels=self.input_channels,
                channels=channels,
                kernel_size=kernel_size,
                dropout=dropout,
                dilation_base=dilation_base,
            )
        else:
            raise NotImplementedError(f"model_type '{model_type}' not implemented yet")

        self.project = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, x: torch.Tensor) -> EncoderOutputs:
        # x shape: (batch, seq_len, feature_dim)
        x = x.transpose(1, 2)  # -> (batch, feature_dim, seq_len)
        features = self.backbone(x)  # (batch, embed_dim, seq_len)
        sequence = features.transpose(1, 2).contiguous()
        pooled = self.project(sequence[:, -1, :])
        return EncoderOutputs(sequence=sequence, pooled=pooled)


class MaskedReconstructionHead(nn.Module):
    def __init__(self, embed_dim: int, cfg: Dict[str, Any]) -> None:
        super().__init__()
        output_dim = int(cfg.get("output_dim", 4))
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, output_dim),
        )
        self.loss_weight = float(cfg.get("loss_weight", 1.0))
        self.mask_ratio = float(cfg.get("mask_ratio", 0.15))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: (batch, seq_len, embed_dim)
        return self.proj(sequence)


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim: int, cfg: Dict[str, Any], num_classes_key: str) -> None:
        super().__init__()
        num_classes = int(cfg.get(num_classes_key))
        hidden_dim = int(cfg.get("hidden_dim", embed_dim))
        dropout = float(cfg.get("dropout", 0.0))
        layers: List[nn.Module] = [nn.Linear(embed_dim, hidden_dim), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)
        self.loss_weight = float(cfg.get("loss_weight", 1.0))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class SRHeatmapHead(nn.Module):
    def __init__(self, embed_dim: int, cfg: Dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(cfg.get("hidden_dim", embed_dim))
        grid_size = int(cfg.get("grid_size", 64))
        layers = [nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, grid_size)]
        self.net = nn.Sequential(*layers)
        activation = cfg.get("output_activation", "sigmoid").lower()
        self.activation_fn = torch.sigmoid if activation == "sigmoid" else torch.tanh
        self.loss_weight = float(cfg.get("loss_weight", 1.0))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(self.net(pooled))


class QuantileHead(nn.Module):
    def __init__(self, embed_dim: int, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.num_quantiles = int(cfg.get("num_quantiles", 9))
        horizons = cfg.get("horizons", [1])
        self.horizons = list(horizons)
        hidden_dim = int(cfg.get("hidden_dim", embed_dim))
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_quantiles * len(self.horizons)),
        )
        self.loss_weight = float(cfg.get("loss_weight", 1.0))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        out = self.encoder(pooled)
        batch = out.shape[0]
        return out.view(batch, len(self.horizons), self.num_quantiles)


class MultiTaskHeads(nn.Module):
    """Bundle of enabled heads defined in configuration."""

    def __init__(self, embed_dim: int, cfg: Dict[str, Any]) -> None:
        super().__init__()
        heads_cfg = cfg.get("heads", {})
        self.masked: Optional[MaskedReconstructionHead] = None
        self.regime: Optional[ClassificationHead] = None
        self.sr: Optional[SRHeatmapHead] = None
        self.quantile: Optional[QuantileHead] = None
        self.pattern: Optional[ClassificationHead] = None

        mask_cfg = heads_cfg.get("masked_reconstruction")
        if mask_cfg and mask_cfg.get("enabled", True):
            self.masked = MaskedReconstructionHead(embed_dim, mask_cfg)

        regime_cfg = heads_cfg.get("regime_classifier")
        if regime_cfg and regime_cfg.get("enabled", True):
            self.regime = ClassificationHead(embed_dim, regime_cfg, "num_classes")

        sr_cfg = heads_cfg.get("sr_heatmap")
        if sr_cfg and sr_cfg.get("enabled", True):
            self.sr = SRHeatmapHead(embed_dim, sr_cfg)

        quant_cfg = heads_cfg.get("quantile_forecasting")
        if quant_cfg and quant_cfg.get("enabled", True):
            self.quantile = QuantileHead(embed_dim, quant_cfg)

        pattern_cfg = heads_cfg.get("pattern_classifier")
        if pattern_cfg and pattern_cfg.get("enabled", True):
            self.pattern = ClassificationHead(embed_dim, pattern_cfg, "num_classes")

    def forward(self, outputs: EncoderOutputs) -> Dict[str, torch.Tensor]:
        preds: Dict[str, torch.Tensor] = {}
        seq = outputs.sequence
        pooled = outputs.pooled
        if self.masked is not None:
            preds["masked_reconstruction"] = self.masked(seq)
        if self.regime is not None:
            preds["regime_classifier"] = self.regime(pooled)
        if self.sr is not None:
            preds["sr_heatmap"] = self.sr(pooled)
        if self.quantile is not None:
            preds["quantile_forecasting"] = self.quantile(pooled)
        if self.pattern is not None:
            preds["pattern_classifier"] = self.pattern(pooled)
        return preds


class StageOneEncoder(nn.Module):
    """Full encoder + multi-task prediction heads."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.backbone = EncoderBackbone(config)
        self.heads = MultiTaskHeads(self.backbone.embedding_dim, config)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(x)
        predictions = self.heads(outputs)
        predictions["embedding"] = outputs.pooled
        predictions["sequence_embedding"] = outputs.sequence
        return predictions


__all__ = [
    "StageOneEncoder",
    "EncoderBackbone",
    "MultiTaskHeads",
    "EncoderOutputs",
]
