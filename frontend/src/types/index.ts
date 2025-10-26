// TypeScript types matching the backend Pydantic models

export interface MetricsSummary {
  total_pnl: number;
  total_pnl_pct: number;
  today_pnl: number;
  today_pnl_pct: number;
  sharpe_ratio: number;
  win_rate: number;
  max_drawdown: number;
  max_drawdown_duration_days: number;
  active_positions: number;
  total_trades: number;
  avg_trade_duration_hours: number;
  current_equity: number;
  initial_equity: number;
}

export interface EquityCurvePoint {
  timestamp: string;
  equity: number;
  pnl: number;
  drawdown_pct: number;
}

export interface Position {
  symbol: string;
  entry_price: number;
  current_price: number;
  position_size: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  entry_time: string;
  duration_hours: number;
  stop_loss: number | null;
  trailing_stop: number | null;
}

export interface Trade {
  symbol: string;
  entry_price: number;
  exit_price: number;
  position_size: number;
  pnl: number;
  pnl_pct: number;
  entry_time: string;
  exit_time: string;
  duration_hours: number;
  exit_reason: string;
}

export interface HealthResponse {
  status: string;
  api_version: string;
  model_loaded: boolean;
  encoder_loaded: boolean;
  timestamp: string;
}
