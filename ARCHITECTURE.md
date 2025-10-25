# AGV2 - Advanced Trading Agent Architecture

## Overview

AGV2 is an end-to-end reinforcement learning-based trading system that learns price-action patterns across multiple asset classes (crypto, metals, indices) using a 2-stage approach:

1. **Stage 1**: Self-supervised encoder that learns market structure, support/resistance, regime changes
2. **Stage 2**: RL policy (PPO/SAC) that makes trading decisions with explicit risk and cost modeling

## Target Markets

- **Crypto**: ETH (24/7, high frequency)
- **Metals**: XAUUSD, XAGUSD (23+ hours, session effects)
- **Indices**: US30, NAS100, Russell (session gaps, volatility clusters)

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│ TradeLocker API → Postgres → DuckDB/Parquet                     │
│ - Historical OHLCV data                                          │
│ - Real-time streaming (future: Kafka/Redis)                     │
│ - Multi-timeframe aggregation (1m, 5m, 15m, 1h)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│ Price-Only Features (NO lagging indicators):                    │
│ - Log returns (multiple horizons: 1, 3, 12, 60 bars)           │
│ - Ranges & wicks (H-L)/C, upper/lower wicks                    │
│ - Gaps (indices): (O_t - C_{t-1})/C_{t-1}                      │
│ - Volatility proxy: realized vol, Parkinson estimator          │
│ - Session flags: Asia/London/NY, day-of-week                   │
│ - Support/Resistance levels: fractal pivots, KDE clustering    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              ENCODER (STAGE 1 - Pretraining)                     │
├─────────────────────────────────────────────────────────────────┤
│ Architecture: Causal Transformer / TCN / LSTM                   │
│ Input: Sliding windows (L=512 bars) × multi-timeframe          │
│                                                                  │
│ Multi-Task Heads:                                               │
│ 1. Masked-candle reconstruction (self-supervised)              │
│ 2. Regime classification (trend/consolidation/volatile)         │
│ 3. S/R heatmap (probability of price reaction at levels)       │
│ 4. Next-return distribution (quantile regression)              │
│ 5. Breakout/reversal probability                               │
│                                                                  │
│ Output: Dense state embedding (dim=256) + task predictions     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                RL POLICY (STAGE 2 - Training)                    │
├─────────────────────────────────────────────────────────────────┤
│ Algorithm: PPO (Stable-Baselines3)                              │
│                                                                  │
│ Observation Space:                                              │
│ - Encoder state embedding (256-dim)                            │
│ - S/R heatmap (centered around current price)                  │
│ - Volatility estimate                                          │
│ - Position state: side, size, unrealized PnL, time-in-trade   │
│                                                                  │
│ Action Space (Discrete Hybrid):                                 │
│ - Position: {FLAT, LONG, SHORT}                                │
│ - Size: {0.25, 0.5, 0.75, 1.0} × max_size                     │
│ - Stop distance: {1.0, 1.5, 2.0, 2.5} × ATR                   │
│ - Take profit: {1.5, 2.0, 3.0, 4.0} × stop_distance           │
│                                                                  │
│ Reward Function:                                                │
│   r_t = log(1 + pnl_t / equity_t)                              │
│         - λ_dd × Δ(max_drawdown)                               │
│         - λ_trades × 1{trade_executed}                         │
│         - λ_hold × exposure_fraction                           │
│         - costs(slippage + spread + commission)                │
│                                                                  │
│ Environment (Gymnasium):                                        │
│ - Vectorized backtesting with domain randomization             │
│ - Random start indices, cost perturbations                     │
│ - Simulated slippage, partial fills, latency                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   RISK & EXECUTION LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│ Position Sizing:                                                │
│ - Volatility targeting: notional = k × (target_σ / σ_actual)  │
│ - Fractional Kelly cap (conservative: 0.1-0.3)                 │
│                                                                  │
│ Risk Controls:                                                  │
│ - Structure-aware stops (beyond nearest S/R cluster)           │
│ - Dynamic trailing stops (volatility bands)                    │
│ - Trade throttles: min time in position, cooldown              │
│ - Daily loss limit → kill-switch                               │
│ - Max concurrent positions per market                          │
│                                                                  │
│ Execution:                                                      │
│ - Simulated Broker (backtesting)                               │
│ - TradeLocker API (live trading)                               │
│ - Server-side stops and take-profits                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              VALIDATION & BACKTESTING LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│ Cross-Validation:                                               │
│ - Purged K-fold (avoid data leakage)                           │
│ - Embargoed periods between folds                              │
│ - Walk-forward analysis (expanding/rolling windows)            │
│                                                                  │
│ Metrics:                                                        │
│ - Sharpe, Sortino, Deflated Sharpe, Calmar                     │
│ - Max drawdown, underwater periods                             │
│ - Hit rate, profit factor, expectancy                          │
│ - Turnover, trade frequency                                    │
│ - Tail risk (CVaR 95%, 99%)                                    │
│                                                                  │
│ Robustness Tests:                                               │
│ - Monte Carlo bootstrap (block resampling)                     │
│ - Stress scenarios (2008, 2020 COVID, flash crashes)          │
│ - Parameter sensitivity analysis                               │
│ - Ablation studies (remove features/heads)                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              DEPLOYMENT & MONITORING LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│ Serving:                                                        │
│ - FastAPI decision service                                      │
│ - TorchScript compiled models                                  │
│ - Redis for state management                                   │
│                                                                  │
│ Monitoring:                                                     │
│ - MLflow experiment tracking                                   │
│ - Evidently (data drift detection - PSI/KL)                    │
│ - Live PnL vs backtest expectation bands                       │
│ - Predictive entropy (uncertainty detection)                   │
│ - Prometheus metrics + Grafana dashboards                      │
│                                                                  │
│ Safeguards:                                                     │
│ - Heartbeat monitor                                            │
│ - Drift tripwires (halt if PSI > threshold)                   │
│ - Drawdown breakers                                            │
│ - Human-in-loop confirmation (optional)                        │
│ - Audit logs (every decision + attribution)                    │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Core Infrastructure (Migrated from AG1)
- **Broker API**: TradeLocker SDK (existing)
- **Database**: PostgreSQL (existing) + DuckDB (analytics)
- **Configuration**: `.env` for credentials (existing)
- **Event Bus**: Custom async event queue (can migrate)

### New Components
- **Deep Learning**: PyTorch 2.x, PyTorch Lightning
- **RL Framework**: Stable-Baselines3 (PPO, SAC)
- **Data Storage**: DuckDB + Parquet files
- **Feature Store**: Polars DataFrames
- **Experiment Tracking**: MLflow
- **Hyperparameter Optimization**: Optuna
- **Backtesting**: Custom Gymnasium environment
- **Deployment**: FastAPI, Docker
- **Monitoring**: Evidently, OpenTelemetry, Prometheus/Grafana

## Key Design Principles

### 1. Price-Action First
- **NO traditional indicators** (RSI, MACD, Bollinger Bands, etc.)
- Pure price patterns: ranges, wicks, gaps, volatility
- Support/Resistance derived from fractal pivots
- Regime detection via statistical methods

### 2. Leakage-Free Validation
- Purged cross-validation (no train/test overlap)
- Embargo periods between folds
- Walk-forward analysis with locked hyperparameters
- Triple-barrier labeling (no look-ahead bias)

### 3. Cost-Aware Learning
- Slippage, spreads, commissions in reward function
- Trade frequency penalty (prevent overtrading)
- Drawdown penalty (risk-adjusted returns)
- Domain randomization (cost perturbations in training)

### 4. Explainability & Trust
- Attention rollout (which candles influenced decision)
- Integrated Gradients (feature attribution)
- Per-trade saliency maps stored in MLflow
- Regime/S/R predictions as interpretable signals

### 5. Production-Ready Design
- Modular, testable, typed Python code
- Configuration-driven (YAML files)
- Comprehensive logging with correlation IDs
- Graceful degradation and error handling
- Kill-switch and circuit breakers

## Implementation Phases

### Phase 1: Data Foundation (Week 1-2)
- Migrate TradeLocker → Postgres → DuckDB pipeline
- Multi-timeframe bar aggregation
- Price-only feature engineering
- S/R level detection (fractal pivots + KDE)
- Data quality checks

### Phase 2: Encoder Pretraining (Week 3-5)
- Implement Causal Transformer architecture
- Multi-task heads: regime, S/R, quantile forecasting
- Self-supervised pretraining (masked reconstruction)
- Validation on held-out periods
- Save embeddings for RL stage

### Phase 3: RL Environment (Week 6-7)
- Gymnasium-compatible trading environment
- Vectorized backtesting (fast simulation)
- Domain randomization (costs, slippage)
- Observation/action space implementation
- Reward function engineering

### Phase 4: Policy Training (Week 8-10)
- PPO training loop with Stable-Baselines3
- Hyperparameter optimization (Optuna)
- Walk-forward validation
- Ablation studies
- Explainability analysis

### Phase 5: Production Deployment (Week 11-12)
- FastAPI decision service
- TorchScript model compilation
- Integration with TradeLocker API
- Monitoring dashboards
- Paper trading validation

### Phase 6: Live Trading (Week 13+)
- Start with smallest position sizes
- Progressive ramp-up with monitoring
- Continuous drift detection
- Performance attribution
- Online fine-tuning (optional)

## Migration from AG1

### Components to Reuse
✅ `tradelocker_api.py` - Broker connectivity
✅ `database.py` - Postgres connection utilities
✅ `events/event_queue.py` - Event bus (can adapt)
✅ `.env` configuration pattern
✅ Logging infrastructure (with correlation IDs)

### Components to Discard
❌ All existing strategies (ema_crossover, stat_arb, etc.)
❌ `strategy_loader.py` - replaced by RL policy
❌ `risk/risk_engine.py` - replaced by RL-native risk
❌ Old backtesting framework - replaced by Gymnasium env

### Components to Rebuild
🔄 Feature engineering (price-only, no indicators)
🔄 Position sizing (volatility targeting)
🔄 Execution layer (adapt for RL actions)
🔄 Backtesting (Gymnasium-based)

## Directory Structure

```
D:/AGV2/
├── .env                          # Credentials (migrate from AG1)
├── .gitignore
├── README.md
├── ARCHITECTURE.md               # This file
├── requirements.txt
├── pyproject.toml
├── config/
│   ├── markets.yaml              # Market specs per symbol
│   ├── costs.yaml                # Slippage, spreads, commissions
│   ├── encoder.yaml              # Encoder architecture config
│   ├── rl_policy.yaml            # RL hyperparameters
│   ├── risk.yaml                 # Risk limits, position sizing
│   └── backtest.yaml             # Validation settings
├── data/
│   ├── raw/                      # Downloaded from TradeLocker
│   ├── processed/                # DuckDB + Parquet
│   └── features/                 # Cached feature tensors
├── models/
│   ├── encoders/                 # Pretrained encoders
│   ├── policies/                 # Trained RL policies
│   └── explainability/           # Attribution artifacts
├── mlruns/                       # MLflow experiment tracking
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tradelocker_client.py    # Migrated from AG1
│   │   ├── postgres_manager.py      # Migrated from AG1
│   │   ├── duckdb_store.py          # New: analytics layer
│   │   ├── bar_aggregator.py        # Multi-timeframe bars
│   │   └── quality_checks.py        # Data validation
│   ├── features/
│   │   ├── __init__.py
│   │   ├── price_features.py        # Log returns, wicks, gaps
│   │   ├── volatility.py            # Realized vol, Parkinson
│   │   ├── support_resistance.py    # Fractal pivots, KDE
│   │   ├── regime_labels.py         # Change-point detection
│   │   └── windowing.py             # Sliding window tensors
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py               # Causal Transformer/TCN
│   │   ├── heads.py                 # Regime, S/R, quantile heads
│   │   ├── pretraining.py           # Self-supervised tasks
│   │   └── policy.py                # RL policy network (if custom)
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── environment.py           # Gymnasium trading env
│   │   ├── rewards.py               # Reward engineering
│   │   ├── wrappers.py              # Env wrappers (normalization)
│   │   └── callbacks.py             # Training callbacks
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── simulator.py             # Vectorized backtester
│   │   ├── walk_forward.py          # WF validation
│   │   ├── purged_cv.py             # Leakage-free CV
│   │   ├── metrics.py               # Performance metrics
│   │   └── attribution.py           # Explainability analysis
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── position_sizer.py        # Vol targeting, Kelly
│   │   ├── stop_logic.py            # Structure-aware stops
│   │   └── limits.py                # Daily loss, max positions
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── order_executor.py        # Migrated from AG1 (adapted)
│   │   ├── simulated_broker.py      # For backtesting
│   │   └── live_executor.py         # TradeLocker integration
│   ├── deploy/
│   │   ├── __init__.py
│   │   ├── server.py                # FastAPI service
│   │   ├── model_loader.py          # TorchScript loading
│   │   └── monitoring.py            # Drift detection, alerts
│   └── utils/
│       ├── __init__.py
│       ├── config.py                # YAML config loader
│       ├── logging_config.py        # Migrated from AG1
│       └── constants.py             # Migrated from AG1
├── scripts/
│   ├── download_data.py             # Fetch from TradeLocker
│   ├── train_encoder.py             # Stage 1 training
│   ├── train_policy.py              # Stage 2 RL training
│   ├── backtest.py                  # Run validation
│   ├── optimize_hparams.py          # Optuna HPO
│   └── deploy.py                    # Start FastAPI server
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_encoder_validation.ipynb
│   ├── 04_rl_diagnostics.ipynb
│   └── 05_backtest_analysis.ipynb
└── tests/
    ├── __init__.py
    ├── test_data/
    ├── test_features/
    ├── test_models/
    ├── test_rl/
    └── test_backtest/
```

## Next Steps

1. **Review this architecture** - Confirm design decisions
2. **Set up project structure** - Create directories and skeleton files
3. **Migrate core infrastructure** - TradeLocker, Postgres, .env
4. **Implement data pipeline** - Download → DuckDB → Features
5. **Build encoder** - Causal Transformer with multi-task heads
6. **Create RL environment** - Gymnasium trading simulator
7. **Train and validate** - Walk-forward testing
8. **Deploy and monitor** - Production readiness

## Questions to Address

1. **Training approach**: 2-stage (encoder → RL) or direct RL?
   → **Recommendation**: 2-stage for interpretability

2. **Encoder architecture**: Causal Transformer vs TCN vs LSTM?
   → **Recommendation**: Start with TCN (fast), upgrade to Transformer later

3. **RL algorithm**: PPO vs SAC?
   → **Recommendation**: PPO (stable, well-tested)

4. **Action space**: Discrete vs continuous vs hybrid?
   → **Recommendation**: Discrete hybrid (easier to start)

5. **Deployment timeline**: Paper trade duration before live?
   → **Recommendation**: Minimum 30 days paper trading

---

**Author**: Claude
**Date**: 2025-10-24
**Version**: 1.0
**Status**: Design Phase
## Infrastructure Reuse

Core connectivity is lifted from AG1:
- `src/infrastructure/config_loader.py` � `.env` loader with required DB/TL keys (`MANDATORY_KEYS`).
- `src/infrastructure/database.py` � PostgreSQL connector plus table DDL and persistence helpers.
- `src/infrastructure/tradelocker_api.py` � typed TradeLocker client mirroring AG1 behaviour.
- `src/infrastructure/order_service.py` � retrying order wrapper (uses `src/utils/constants.py`).

Environment template lives in `.env.example`; install dependencies from `requirements.txt` before running the broker or DB code.
### Stage 1 Implementation Status
- `src/models/encoder.py` now instantiates the configurable StageOneEncoder (TCN backbone + multi-task heads).
- `src/data/price_action_dataset.py` loads parquet windows with column `features` (flattened window � feature vectors) plus optional labels (`regime_label`, `sr_heatmap`, `future_returns`, `pattern_label`).
- Train with `python scripts/train_encoder.py --data data/encoder_windows.parquet --config config/encoder.yaml`; checkpoints write to `models/encoders/` and optionally log to MLflow if enabled in the YAML.
