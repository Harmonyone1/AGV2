"""Pydantic models for AGV2 API requests and responses."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================================
# Dashboard Models
# ============================================================================


class MetricsSummary(BaseModel):
    """Summary metrics for dashboard overview."""

    total_pnl: float = Field(..., description="Total cumulative PnL")
    total_pnl_pct: float = Field(..., description="Total PnL as percentage of initial equity")
    today_pnl: float = Field(..., description="Today's PnL")
    today_pnl_pct: float = Field(..., description="Today's PnL percentage")
    sharpe_ratio: float = Field(..., description="Annualized Sharpe ratio")
    win_rate: float = Field(..., description="Win rate (0-1)")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    max_drawdown_duration_days: float = Field(..., description="Max drawdown duration in days")
    active_positions: int = Field(..., description="Number of active positions")
    total_trades: int = Field(..., description="Total number of trades")
    avg_trade_duration_hours: float = Field(..., description="Average trade duration in hours")
    current_equity: float = Field(..., description="Current equity value")
    initial_equity: float = Field(..., description="Initial equity value")


class EquityCurvePoint(BaseModel):
    """Single point on equity curve."""

    timestamp: datetime
    equity: float
    pnl: float
    drawdown_pct: float


class Position(BaseModel):
    """Active trading position."""

    symbol: str
    entry_price: float
    current_price: float
    position_size: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_time: datetime
    duration_hours: float
    stop_loss: Optional[float] = None
    trailing_stop: Optional[float] = None


class Trade(BaseModel):
    """Completed trade record."""

    symbol: str
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    duration_hours: float
    exit_reason: str


# ============================================================================
# Inference Models
# ============================================================================


class InferenceRequest(BaseModel):
    """Request for ML inference."""

    symbol: str = Field(..., description="Trading symbol (e.g., 'ETH', 'BTC')")
    features: Optional[list[float]] = Field(
        None, description="Pre-computed features (if not using latest market data)"
    )
    current_position: float = Field(0.0, description="Current position size")
    current_equity: float = Field(10000.0, description="Current equity")
    steps_in_position: int = Field(0, description="Steps holding current position")


class InferenceResponse(BaseModel):
    """Response from ML inference."""

    symbol: str
    action: float = Field(..., description="Raw policy action (-1 to 1)")
    adjusted_action: float = Field(
        ..., description="Risk-adjusted action after position sizing"
    )
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime
    confidence: Optional[float] = Field(None, description="Model confidence score")


# ============================================================================
# Training Models
# ============================================================================


class TrainingJobCreate(BaseModel):
    """Request to create a new training job."""

    symbol: str
    job_type: str = Field(..., description="'encoder' or 'policy'")
    config_overrides: Optional[dict] = Field(None, description="Config overrides")


class TrainingJobStatus(BaseModel):
    """Training job status."""

    job_id: str
    status: str = Field(..., description="'pending', 'running', 'completed', 'failed'")
    progress: float = Field(..., description="Progress 0-1")
    current_step: int
    total_steps: int
    metrics: dict = Field(default_factory=dict, description="Training metrics")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


# ============================================================================
# Backtest Models
# ============================================================================


class BacktestRequest(BaseModel):
    """Request to run a backtest."""

    symbol: str
    start_date: datetime
    end_date: datetime
    policy_path: str = Field(..., description="Path to policy model")
    encoder_path: Optional[str] = Field(None, description="Path to encoder model")
    initial_equity: float = Field(10000.0, description="Initial equity")
    config_overrides: Optional[dict] = None


class BacktestResult(BaseModel):
    """Backtest results."""

    backtest_id: str
    symbol: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_duration_hours: float
    equity_curve: list[EquityCurvePoint]
    trades: list[Trade]
    metrics: dict = Field(default_factory=dict, description="Additional metrics")
    completed_at: datetime


# ============================================================================
# Risk Control Models
# ============================================================================


class RiskLimits(BaseModel):
    """Current risk limits and status."""

    daily_loss_limit_pct: float
    daily_profit_target_pct: float
    max_concurrent_positions: int
    max_trades_per_day: int
    current_daily_pnl: float
    current_daily_trades: int
    kill_switch_active: bool
    last_reset_time: datetime


# ============================================================================
# Configuration Models
# ============================================================================


class ConfigUpdate(BaseModel):
    """Request to update configuration."""

    section: str = Field(..., description="Config section (e.g., 'risk', 'position_sizing')")
    parameters: dict = Field(..., description="Parameters to update")


class ConfigResponse(BaseModel):
    """Configuration response."""

    section: str
    parameters: dict
    updated_at: datetime


# ============================================================================
# Health Check Models
# ============================================================================


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    api_version: str
    model_loaded: bool
    encoder_loaded: bool
    timestamp: datetime


__all__ = [
    "MetricsSummary",
    "EquityCurvePoint",
    "Position",
    "Trade",
    "InferenceRequest",
    "InferenceResponse",
    "TrainingJobCreate",
    "TrainingJobStatus",
    "BacktestRequest",
    "BacktestResult",
    "RiskLimits",
    "ConfigUpdate",
    "ConfigResponse",
    "HealthResponse",
]
