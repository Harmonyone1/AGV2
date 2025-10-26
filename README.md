# AGV2 - Advanced Trading Agent

> End-to-end reinforcement learning-based trading system that learns price-action patterns across multiple asset classes (crypto, metals, indices).

## Overview

AGV2 uses a **2-stage learning approach**:

1. **Stage 1**: Self-supervised encoder that learns market structure, support/resistance, and regime changes
2. **Stage 2**: RL policy (PPO) that makes trading decisions with explicit risk and cost modeling

**Target Markets**: ETH (crypto), XAUUSD/XAGUSD (metals), US30/NAS100/RUSSELL (indices)

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.8.0+
- GPU recommended (but not required)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd AGV2

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional, for live trading)
cp .env.example .env
# Edit .env with your TradeLocker credentials
```

### Basic Workflow

#### 1. Prepare Data

Download historical data from TradeLocker:

```bash
python scripts/download_data.py --symbol ETH --start 2024-01-01 --end 2024-12-31
```

Generate encoder training windows:

```bash
python scripts/embed_windows.py \
  --input data/processed/eth_1m.parquet \
  --output data/features/encoder_windows_eth.parquet \
  --config config/encoder.yaml
```

#### 2. Train Encoder (Stage 1)

```bash
python scripts/train_encoder.py \
  --config config/encoder.yaml \
  --data data/features/encoder_windows_eth.parquet \
  --output models/encoders/encoder_best.pt
```

This trains a Temporal Convolutional Network (TCN) with multi-task heads:
- Masked candle reconstruction
- Regime classification
- Support/Resistance heatmap prediction
- Quantile forecasting

#### 3. Generate Embeddings

```bash
python scripts/embed_windows.py \
  --input data/features/encoder_windows_eth.parquet \
  --model models/encoders/encoder_best.pt \
  --output data/features/encoder_windows_eth_emb.parquet
```

#### 4. Train RL Policy (Stage 2)

```bash
python scripts/train_policy.py \
  --config config/rl_policy.yaml \
  --data data/features/encoder_windows_eth_emb.parquet \
  --timesteps 500000
```

This trains a PPO agent using the encoder embeddings as observations.

#### 5. Backtest the Policy

```bash
python scripts/backtest_policy.py \
  --config config/backtest.yaml \
  --model models/policies/ppo_trading_env.zip \
  --episodes 100
```

Expected output:
```
Backtest results:
  Episodes      : 100
  Total return  : 0.1234
  Sharpe        : 1.56
  Max drawdown  : -0.0456
```

## Project Structure

```
AGV2/
├── config/              # YAML configuration files
│   ├── encoder.yaml     # Encoder architecture & training
│   ├── rl_policy.yaml   # RL policy hyperparameters
│   ├── backtest.yaml    # Backtesting settings
│   ├── markets.yaml     # Market specifications (tick size, sessions)
│   └── costs.yaml       # Transaction cost models
├── data/
│   ├── raw/             # Downloaded OHLCV from TradeLocker
│   ├── processed/       # Aggregated bars (1m, 5m, 15m, 1h)
│   └── features/        # Windowed features & embeddings
├── models/
│   ├── encoders/        # Trained encoder checkpoints (.pt)
│   └── policies/        # Trained RL policies (.zip)
├── src/
│   ├── data/            # Data ingestion, aggregation, datasets
│   ├── features/        # Price-action feature engineering
│   ├── models/          # Encoder architecture (TCN)
│   ├── rl/              # Gymnasium trading environment
│   ├── backtest/        # Vectorized backtesting
│   ├── infrastructure/  # Database, API clients, logging
│   └── utils/           # Common utilities
├── scripts/             # Training & evaluation scripts
├── tests/               # Pytest test suite
├── notebooks/           # Jupyter notebooks (Colab training)
└── docs/                # Additional documentation

```

## Key Features

### Price-Action First
- **NO lagging indicators** (RSI, MACD, Bollinger Bands)
- Pure price patterns: log returns, ranges, wicks, gaps
- Support/Resistance from fractal pivots + KDE clustering

### Realistic Execution
- TradeLocker-style execution semantics
- Market orders, limit orders, bracket orders (TP/SL)
- Session-aware trading (Asia/London/NY)
- Realistic cost modeling: spread + slippage + commission + financing

### Cost-Aware Learning
- Transaction costs in reward function
- Trade frequency penalty
- Risk-adjusted returns (Sharpe-based)
- Domain randomization during training

### Production-Ready
- Modular, testable, type-annotated Python
- Configuration-driven (YAML)
- Comprehensive logging with correlation IDs
- Kill-switch and circuit breakers (planned)

## Configuration

### Markets (`config/markets.yaml`)

Define market specifications for each symbol:

```yaml
markets:
  ETH:
    tick_size: 0.01
    lot_size: 0.001
    min_notional: 10.0
    session_config:
      enabled: false  # 24/7 market
    target_volatility: 0.02
```

### Costs (`config/costs.yaml`)

Transaction cost models per symbol:

```yaml
costs:
  ETH:
    maker_bps: 1.0
    taker_bps: 2.5
    spread_bps: 2.0
    slippage:
      model: proportional
      base_bps: 0.5
      volatility_multiplier: 1.0
      size_impact: 0.1
```

### RL Policy (`config/rl_policy.yaml`)

Configure the trading environment and PPO hyperparameters:

```yaml
environment:
  symbol: ETH
  episode_length: 256
  max_position: 1.0
  reward:
    trading_cost_bps: 1.0
    holding_cost_bps: 0.2
    risk_aversion: 0.001

training:
  total_timesteps: 500000
  learning_rate: 0.0003
  batch_size: 512
  n_steps: 2048
```

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
pytest tests/test_rl/test_environment.py -v
pytest tests/test_features/test_price_features.py -v
pytest tests/test_backtest/test_simulator.py -v
```

## Advanced Usage

### Custom Reward Functions

Modify `src/rl/rewards.py` to implement custom reward shaping:

```python
@dataclass
class RewardConfig:
    trading_cost_bps: float = 1.0
    holding_cost_bps: float = 0.2
    risk_aversion: float = 0.001
    sharpe_bonus: float = 0.0  # Add new parameter
```

### Multi-Asset Training

Train policies on multiple symbols:

```bash
for symbol in ETH XAUUSD US30; do
  python scripts/train_policy.py \
    --config config/rl_policy.yaml \
    --symbol $symbol \
    --timesteps 500000
done
```

### Walk-Forward Validation

Use `scripts/backtest_policy.py` with time-based splits:

```bash
python scripts/backtest_policy.py \
  --config config/backtest.yaml \
  --model models/policies/ppo_trading_env.zip \
  --start-date 2024-10-01 \
  --end-date 2024-12-31
```

## MLflow Tracking

View experiment results:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Navigate to http://localhost:5000 to explore:
- Training metrics (loss curves, accuracy)
- Hyperparameters
- Model checkpoints
- Saliency maps and attention visualizations

## Design Principles

1. **Leakage-Free Validation**: Purged cross-validation, embargo periods
2. **Cost-Aware Learning**: Transaction costs, slippage, financing in reward
3. **Explainability**: Attention rollout, Integrated Gradients (planned)
4. **Production-First**: Clean code, comprehensive logging, fail-safes

## Troubleshooting

### Common Issues

**1. CUDA out of memory during encoder training**
```bash
# Reduce batch size in config/encoder.yaml
batch_size: 32  # Instead of 64
```

**2. Policy not improving**
```bash
# Check reward scaling, increase exploration
ent_coef: 0.01  # Add entropy bonus in config/rl_policy.yaml
```

**3. Data file not found**
```bash
# Ensure data path is absolute or relative to project root
data_path: data/features/encoder_windows_eth_emb.parquet
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Architecture

For detailed system architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

For TradeLocker execution mapping, see [docs/execution_mapping.md](docs/execution_mapping.md).

## License

[Specify License]

## Acknowledgments

- Built on PyTorch, Stable-Baselines3, and Gymnasium
- Inspired by research in deep reinforcement learning for trading
- TradeLocker API integration for realistic execution simulation

## Contact

[Your Contact Information]

---

**Status**: Active Development | Production-Ready for Backtesting | Paper Trading in Progress
