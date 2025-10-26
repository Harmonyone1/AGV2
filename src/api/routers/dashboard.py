"""Dashboard API endpoints."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from api.inference import get_inference_engine
from api.models import (
    EquityCurvePoint,
    InferenceRequest,
    InferenceResponse,
    MetricsSummary,
    Position,
    Trade,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Mock Data Store (Replace with real database/state management)
# ============================================================================


class DashboardState:
    """In-memory dashboard state (for demo purposes)."""

    def __init__(self):
        # Mock metrics
        self.total_pnl = 1247.32
        self.total_pnl_pct = 12.47
        self.today_pnl = 87.45
        self.today_pnl_pct = 0.78
        self.sharpe_ratio = 1.85
        self.win_rate = 0.58
        self.max_drawdown = -8.32
        self.max_drawdown_duration_days = 4.2
        self.total_trades = 127
        self.avg_trade_duration_hours = 6.8
        self.current_equity = 11247.32
        self.initial_equity = 10000.0

        # Mock active positions
        self.active_positions = [
            Position(
                symbol="ETH",
                entry_price=2458.32,
                current_price=2485.67,
                position_size=0.5,
                unrealized_pnl=13.67,
                unrealized_pnl_pct=1.11,
                entry_time=datetime.now() - timedelta(hours=3, minutes=24),
                duration_hours=3.4,
                stop_loss=2420.15,
                trailing_stop=None,
            )
        ]

        # Mock recent trades
        now = datetime.now()
        self.recent_trades = [
            Trade(
                symbol="ETH",
                entry_price=2445.21,
                exit_price=2467.89,
                position_size=0.5,
                pnl=11.34,
                pnl_pct=0.93,
                entry_time=now - timedelta(hours=8, minutes=15),
                exit_time=now - timedelta(hours=4, minutes=45),
                duration_hours=3.5,
                exit_reason="trailing_stop",
            ),
            Trade(
                symbol="ETH",
                entry_price=2478.55,
                exit_price=2461.23,
                position_size=-0.3,
                pnl=-5.20,
                pnl_pct=-0.70,
                entry_time=now - timedelta(hours=12, minutes=30),
                exit_time=now - timedelta(hours=9, minutes=10),
                duration_hours=3.3,
                exit_reason="stop_loss",
            ),
            Trade(
                symbol="ETH",
                entry_price=2433.12,
                exit_price=2456.78,
                position_size=0.6,
                pnl=14.20,
                pnl_pct=0.97,
                entry_time=now - timedelta(hours=18, minutes=5),
                exit_time=now - timedelta(hours=13, minutes=20),
                duration_hours=4.75,
                exit_reason="policy",
            ),
        ]

        # Mock equity curve
        self.equity_curve = self._generate_mock_equity_curve()

    def _generate_mock_equity_curve(self) -> list[EquityCurvePoint]:
        """Generate mock equity curve for visualization."""
        points = []
        now = datetime.now()
        equity = self.initial_equity
        peak_equity = equity

        # Generate last 30 days
        for i in range(30):
            timestamp = now - timedelta(days=30 - i)

            # Random walk with slight upward drift
            pnl_change = np.random.randn() * 50 + 10
            equity += pnl_change
            pnl = equity - self.initial_equity

            # Calculate drawdown
            peak_equity = max(peak_equity, equity)
            drawdown_pct = ((equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0.0

            points.append(
                EquityCurvePoint(
                    timestamp=timestamp,
                    equity=equity,
                    pnl=pnl,
                    drawdown_pct=drawdown_pct,
                )
            )

        # Set final equity to match current
        points[-1].equity = self.current_equity
        points[-1].pnl = self.total_pnl

        return points


# Global state (will be replaced with proper database)
dashboard_state = DashboardState()


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/metrics/summary", response_model=MetricsSummary)
async def get_metrics_summary() -> MetricsSummary:
    """Get summary metrics for dashboard overview.

    Returns:
        MetricsSummary with key performance metrics
    """
    return MetricsSummary(
        total_pnl=dashboard_state.total_pnl,
        total_pnl_pct=dashboard_state.total_pnl_pct,
        today_pnl=dashboard_state.today_pnl,
        today_pnl_pct=dashboard_state.today_pnl_pct,
        sharpe_ratio=dashboard_state.sharpe_ratio,
        win_rate=dashboard_state.win_rate,
        max_drawdown=dashboard_state.max_drawdown,
        max_drawdown_duration_days=dashboard_state.max_drawdown_duration_days,
        active_positions=len(dashboard_state.active_positions),
        total_trades=dashboard_state.total_trades,
        avg_trade_duration_hours=dashboard_state.avg_trade_duration_hours,
        current_equity=dashboard_state.current_equity,
        initial_equity=dashboard_state.initial_equity,
    )


@router.get("/positions/active", response_model=list[Position])
async def get_active_positions() -> list[Position]:
    """Get all active trading positions.

    Returns:
        List of active positions with current P&L
    """
    return dashboard_state.active_positions


@router.get("/trades/recent", response_model=list[Trade])
async def get_recent_trades(limit: int = Query(10, ge=1, le=100)) -> list[Trade]:
    """Get recent completed trades.

    Args:
        limit: Maximum number of trades to return (1-100)

    Returns:
        List of recent trades ordered by exit time (newest first)
    """
    return dashboard_state.recent_trades[:limit]


@router.get("/equity-curve", response_model=list[EquityCurvePoint])
async def get_equity_curve(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> list[EquityCurvePoint]:
    """Get equity curve data for charting.

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        List of equity curve points
    """
    curve = dashboard_state.equity_curve

    # Apply date filters if provided
    if start_date:
        curve = [p for p in curve if p.timestamp >= start_date]
    if end_date:
        curve = [p for p in curve if p.timestamp <= end_date]

    return curve


@router.post("/inference/predict", response_model=InferenceResponse)
async def predict_action(request: InferenceRequest) -> InferenceResponse:
    """Get ML model prediction for current market state.

    Args:
        request: Inference request with features and position state

    Returns:
        Model prediction with action and confidence
    """
    engine = get_inference_engine()

    if not engine.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Build observation from request
        # For minimal example, we'll use position state only
        # In production, this would include encoded features
        observation = np.array(
            [
                request.current_position,
                request.current_equity / 10000.0,  # Normalize
                min(request.steps_in_position / 100.0, 1.0),  # Normalize
            ],
            dtype=np.float32,
        )

        # If features provided, use them (assuming they're already encoded)
        if request.features:
            features = np.array(request.features, dtype=np.float32)
            observation = np.concatenate([features, observation])

        # Get prediction
        action, extra_info = engine.predict(observation, deterministic=True)

        # For this minimal example, adjusted_action = action
        # In production, this would apply position sizing and risk controls
        adjusted_action = action

        return InferenceResponse(
            symbol=request.symbol,
            action=action,
            adjusted_action=adjusted_action,
            model_version=engine.policy_version,
            timestamp=datetime.now(),
            confidence=None,  # Could add value estimate from policy
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


__all__ = ["router"]
