"""Feature engineering surface for AGV2."""

from .price_features import engineer_price_features, DEFAULT_HORIZONS
from .volatility import realized_volatility, parkinson_volatility
from .support_resistance import SupportResistanceConfig, sr_heatmap_series
from .regime_labels import RegimeConfig, REGIME_TO_ID, assign_regimes
from .windowing import WindowConfig, build_encoder_windows

__all__ = [
    "engineer_price_features",
    "DEFAULT_HORIZONS",
    "realized_volatility",
    "parkinson_volatility",
    "SupportResistanceConfig",
    "sr_heatmap_series",
    "RegimeConfig",
    "REGIME_TO_ID",
    "assign_regimes",
    "WindowConfig",
    "build_encoder_windows",
]
