# AGV2 Dashboard - Complete UI/UX Design & API Specification

## Table of Contents
1. [System Architecture](#system-architecture)
2. [UI Screens & Components](#ui-screens--components)
3. [API Endpoints](#api-endpoints)
4. [Data Models](#data-models)
5. [Real-Time Features](#real-time-features)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (React + Tailwind)                  │
├─────────────────────────────────────────────────────────────────┤
│  Dashboard  │  Training  │  Backtesting  │  Live Trading  │  Settings
│  Overview   │  Monitor   │  Results      │  Monitor       │  Config
└─────────────────────────────────────────────────────────────────┘
                              ↕ WebSocket + REST API
┌─────────────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND (Python)                       │
├─────────────────────────────────────────────────────────────────┤
│  Inference  │  Training  │  Backtest  │  Risk     │  Monitoring │
│  Engine     │  Manager   │  Engine    │  Controls │  & Logs     │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                    ML MODELS & ENVIRONMENT                       │
├─────────────────────────────────────────────────────────────────┤
│  Encoder    │  Policy    │  Trading   │  Position │  Risk       │
│  (TCN/Tf)   │  (PPO)     │  Env       │  Sizer    │  Controller │
└─────────────────────────────────────────────────────────────────┘
```

---

## UI Screens & Components

### 1. **Dashboard Overview** (Main Screen)

```
╔════════════════════════════════════════════════════════════════╗
║  AGV2 Trading System               [User Icon] [Settings] [🔔] ║
╠════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          ║
║  │ Total PnL    │ │ Today's PnL  │ │ Sharpe Ratio │          ║
║  │ +$12,345.67  │ │ +$234.56     │ │ 1.82         │          ║
║  │ ↑ +5.2%      │ │ ↑ +1.3%      │ │ ↑ +0.15      │          ║
║  └──────────────┘ └──────────────┘ └──────────────┘          ║
║                                                                 ║
║  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          ║
║  │ Win Rate     │ │ Max Drawdown │ │ Active Pos   │          ║
║  │ 56.8%        │ │ -8.2%        │ │ 2 / 3        │          ║
║  │ ↑ +2.1pp     │ │ ↓ -1.5%      │ │ ETH, XAUUSD  │          ║
║  └──────────────┘ └──────────────┘ └──────────────┘          ║
║                                                                 ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │ Equity Curve (Last 30 Days)                             │  ║
║  │                                             ╱            │  ║
║  │                                    ╱╲      ╱             │  ║
║  │                           ╱╲      ╱  ╲    ╱              │  ║
║  │                 ╱╲       ╱  ╲    ╱    ╲  ╱               │  ║
║  │        ╱╲      ╱  ╲     ╱    ╲  ╱      ╲╱                │  ║
║  │  ─────╱──╲────╱────╲───╱──────╲╱────────────────────   │  ║
║  │  Dec 1              Dec 15             Dec 30           │  ║
║  └─────────────────────────────────────────────────────────┘  ║
║                                                                 ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │ Active Positions                            [View All →] │  ║
║  ├─────────────────────────────────────────────────────────┤  ║
║  │ Symbol  │ Side │ Size  │ Entry   │ Current │ PnL      │  ║
║  │ ETH     │ LONG │ 0.75  │ 2,450.0 │ 2,512.3 │ +$46.73  │  ║
║  │ XAUUSD  │ LONG │ 0.50  │ 2,025.0 │ 2,031.5 │ +$3.25   │  ║
║  └─────────────────────────────────────────────────────────┘  ║
║                                                                 ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │ Recent Trades                               [View All →] │  ║
║  ├─────────────────────────────────────────────────────────┤  ║
║  │ Time    │ Symbol │ Side  │ PnL     │ Exit Reason       │  ║
║  │ 10:32   │ BTC    │ LONG  │ +$12.50 │ Take Profit       │  ║
║  │ 09:15   │ ETH    │ SHORT │ -$5.20  │ Stop Loss         │  ║
║  │ 08:45   │ XAUUSD │ LONG  │ +$8.75  │ Trailing Stop     │  ║
║  └─────────────────────────────────────────────────────────┘  ║
╚════════════════════════════════════════════════════════════════╝
```

**Components**:
- **Metric Cards**: Total PnL, Today's PnL, Sharpe, Win Rate, Max DD, Active Positions
- **Equity Curve Chart**: Line chart (Recharts/Victory)
- **Active Positions Table**: Real-time positions with P&L
- **Recent Trades Table**: Last 10 trades with exit reasons
- **Status Indicator**: Trading status (live, paper, stopped, kill-switch)

---

### 2. **Training Monitor** Screen

```
╔════════════════════════════════════════════════════════════════╗
║  Training Monitor                                   [Stop] [⏸]  ║
╠════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │ Current Training Job                                      │ ║
║  │ Job ID: train_eth_20251025_143022                        │ ║
║  │ Status: Running (Epoch 45/100)                           │ ║
║  │ Progress: ████████████░░░░░░░░░░░ 45%                    │ ║
║  │ ETA: 2h 15m                                              │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                                                                 ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │ Training Metrics (Live)                                   │ ║
║  │                                                           │ ║
║  │  Train Loss               Validation Loss                │ ║
║  │  ╲                        ╲                              │ ║
║  │   ╲                        ╲                             │ ║
║  │    ╲___                     ╲___                         │ ║
║  │        ───                      ───                      │ ║
║  │                                                           │ ║
║  │  Policy Reward            Entropy                        │ ║
║  │         ╱╲                       ╲                       │ ║
║  │        ╱  ╲╱╲                     ╲                      │ ║
║  │  ─────╱───────╲╱                   ╲─                   │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                                                                 ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │ Stage 1: Encoder Training                        ✓ Done  │ ║
║  │ └─ Masked Reconstruction Loss: 0.0234                    │ ║
║  │ └─ Regime Classifier Accuracy: 78.5%                     │ ║
║  │ └─ S/R Heatmap BCE Loss: 0.1234                          │ ║
║  │                                                           │ ║
║  │ Stage 2: Policy Training                         🔄 Running│ ║
║  │ └─ Episode Return: +0.0523                               │ ║
║  │ └─ Value Loss: 0.0123                                    │ ║
║  │ └─ Timesteps: 245,000 / 500,000                          │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                                                                 ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │ Console Logs                               [Clear] [↓]   │ ║
║  ├──────────────────────────────────────────────────────────┤ ║
║  │ [10:32:15] Epoch 45/100 - Train Loss: 0.0234            │ ║
║  │ [10:32:14] Validation Sharpe: 1.52                       │ ║
║  │ [10:32:10] Checkpoint saved: models/encoder_epoch_45.pt  │ ║
║  └──────────────────────────────────────────────────────────┘ ║
╚════════════════════════════════════════════════════════════════╝
```

**Components**:
- **Job Status Card**: Current training job with progress bar
- **Live Metrics Charts**: Train/val loss, policy reward, entropy (real-time updates)
- **Stage Progress**: Encoder (Stage 1) and Policy (Stage 2) status
- **Console Log Stream**: Real-time training logs (WebSocket)

---

### 3. **Backtesting** Screen

```
╔════════════════════════════════════════════════════════════════╗
║  Backtesting                                      [New Test]    ║
╠════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  ┌─────────────────────────┐  ┌───────────────────────────┐   ║
║  │ Configuration           │  │ Results Summary           │   ║
║  │                         │  │                           │   ║
║  │ Model:                  │  │ Total Return: +12.3%      │   ║
║  │ [ppo_eth_v2.zip    ▼]  │  │ Sharpe Ratio: 1.82        │   ║
║  │                         │  │ Max Drawdown: -8.2%       │   ║
║  │ Symbol:                 │  │ Win Rate: 56.8%           │   ║
║  │ [ETH               ▼]  │  │ Total Trades: 127         │   ║
║  │                         │  │ Avg Trade: +$12.50        │   ║
║  │ Date Range:             │  │                           │   ║
║  │ 2024-01-01 to 2024-12-31│  │ [Export CSV] [Share]     │   ║
║  │                         │  │                           │   ║
║  │ Episodes: [100]         │  │                           │   ║
║  │                         │  │                           │   ║
║  │ [Run Backtest]          │  │                           │   ║
║  └─────────────────────────┘  └───────────────────────────┘   ║
║                                                                 ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │ Equity Curve                                              │ ║
║  │  $12,000                                    ╱             │ ║
║  │                                    ╱╲      ╱              │ ║
║  │  $11,000              ╱╲          ╱  ╲    ╱               │ ║
║  │                 ╱╲   ╱  ╲        ╱    ╲  ╱                │ ║
║  │  $10,000  ─────╱──╲─╱────╲──────╱──────╲╱                │ ║
║  │                                                            │ ║
║  │  Jan     Mar     May     Jul     Sep     Nov              │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                                                                 ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │ Drawdown Chart                                            │ ║
║  │   0%  ────────────────────────────────────────           │ ║
║  │                                                            │ ║
║  │  -5%              ╲                    ╲                  │ ║
║  │                    ╲                    ╲                 │ ║
║  │ -10%                ╲                    ╲                │ ║
║  │                      ╲___                 ╲___            │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                                                                 ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │ Trade Analysis                          [Filters ▼]       │ ║
║  ├──────────────────────────────────────────────────────────┤ ║
║  │ Date       │ Symbol │ Side │ PnL    │ Duration │ Exit    │ ║
║  │ 2024-01-05 │ ETH    │ LONG │ +$12.5 │ 2.5h     │ TP      │ ║
║  │ 2024-01-06 │ ETH    │ SHORT│ -$5.2  │ 1.2h     │ SL      │ ║
║  │ 2024-01-08 │ XAUUSD │ LONG │ +$8.7  │ 4.1h     │ Trail   │ ║
║  │ ... (124 more trades)                                     │ ║
║  └──────────────────────────────────────────────────────────┘ ║
╚════════════════════════════════════════════════════════════════╝
```

**Components**:
- **Configuration Panel**: Model selection, symbol, date range, episodes
- **Results Summary Card**: Key metrics with export options
- **Equity Curve**: Interactive chart with zoom/pan
- **Drawdown Chart**: Underwater equity chart
- **Trade Analysis Table**: All trades with filters and sorting

---

### 4. **Live Trading Monitor** Screen

```
╔════════════════════════════════════════════════════════════════╗
║  Live Trading                    [Paper Mode ▼] [🔴 KILL SWITCH]║
╠════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  ┌───────────────────────────────────────────────────────────┐ ║
║  │ System Status                                              │ ║
║  │ ● Trading Active  │  ● Risk Controls OK  │  ● API Connected│ ║
║  │ Daily PnL: +$234.56  │  Daily Trades: 5/10  │  Heat: 12%  │ ║
║  └───────────────────────────────────────────────────────────┘ ║
║                                                                 ║
║  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          ║
║  │ Model A      │ │ Model B      │ │ Ensemble     │          ║
║  │ ETH v2.1     │ │ ETH v2.0     │ │ 50/50 Split  │          ║
║  │ Traffic: 50% │ │ Traffic: 50% │ │ ✓ Enabled    │          ║
║  │ PnL: +$120   │ │ PnL: +$114   │ │              │          ║
║  │ Sharpe: 1.85 │ │ Sharpe: 1.78 │ │ [Settings]   │          ║
║  └──────────────┘ └──────────────┘ └──────────────┘          ║
║                                                                 ║
║  ┌───────────────────────────────────────────────────────────┐ ║
║  │ Live Positions                                             │ ║
║  ├───────────────────────────────────────────────────────────┤ ║
║  │ Symbol │ Model │ Side │ Size │ Entry  │ Stop  │ Trail │ PnL│ ║
║  │ ETH    │ A     │ LONG │ 0.75 │ 2,450  │ 2,400 │ 2,490 │+$46│ ║
║  │ XAUUSD │ B     │ LONG │ 0.50 │ 2,025  │ 2,010 │ -     │+$3 │ ║
║  │                                       [Force Close All]    │ ║
║  └───────────────────────────────────────────────────────────┘ ║
║                                                                 ║
║  ┌───────────────────────────────────────────────────────────┐ ║
║  │ Risk Controls                                              │ ║
║  │ ┌────────────────────┐ ┌────────────────────┐             │ ║
║  │ │ Daily Loss Limit   │ │ Max Position       │             │ ║
║  │ │ -5.0% (-$500)      │ │ 1.0 (normalized)   │             │ ║
║  │ │ Current: -1.2%     │ │ Current: 0.75      │             │ ║
║  │ │ ████░░░░░░ 24%     │ │ ███████░░░ 75%     │             │ ║
║  │ └────────────────────┘ └────────────────────┘             │ ║
║  │                                                            │ ║
║  │ ┌────────────────────┐ ┌────────────────────┐             │ ║
║  │ │ Trade Cooldown     │ │ Portfolio Heat     │             │ ║
║  │ │ 3 bars (5 min)     │ │ Max: 20%           │             │ ║
║  │ │ Next trade: 2 min  │ │ Current: 12%       │             │ ║
║  │ │ ██████░░░░ 67%     │ │ ██████░░░░ 60%     │             │ ║
║  │ └────────────────────┘ └────────────────────┘             │ ║
║  └───────────────────────────────────────────────────────────┘ ║
║                                                                 ║
║  ┌───────────────────────────────────────────────────────────┐ ║
║  │ Order Flow (Last 10 minutes)                  [Live Feed] │ ║
║  ├───────────────────────────────────────────────────────────┤ ║
║  │ 10:32:45  ETH    LONG  0.75  LIMIT  2,450.0  ✓ FILLED    │ ║
║  │ 10:30:12  XAUUSD LONG  0.50  MARKET 2,025.0  ✓ FILLED    │ ║
║  │ 10:28:34  BTC    CLOSE 1.00  MARKET 45,120   ✓ FILLED    │ ║
║  └───────────────────────────────────────────────────────────┘ ║
╚════════════════════════════════════════════════════════════════╝
```

**Components**:
- **System Status Bar**: Trading mode, risk status, API connection
- **A/B Testing Cards**: Model A vs Model B performance comparison
- **Live Positions Table**: Real-time positions with stops/trails
- **Risk Controls Dashboard**: Visual gauges for all risk limits
- **Order Flow Stream**: Real-time order execution log
- **KILL SWITCH Button**: Emergency stop (red, prominent)

---

### 5. **Explainability** Screen

```
╔════════════════════════════════════════════════════════════════╗
║  Model Explainability                         [Select Trade ▼] ║
╠════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │ Trade Details                                             │ ║
║  │ Symbol: ETH  │  Side: LONG  │  Entry: 2,450.0  │  PnL: +$46│ ║
║  │ Timestamp: 2024-12-25 10:32:45                            │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                                                                 ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │ Attention Rollout (Which candles influenced this trade?) │ ║
║  │                                                           │ ║
║  │  Bar -50  -40   -30   -20   -10    0 (entry)            │ ║
║  │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐              │ ║
║  │  │░░░│░░░│░░░│██░│███│███│██░│░░░│░░░│███│  Attention   │ ║
║  │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘              │ ║
║  │  Low ────────────────────────────────── High              │ ║
║  │                                                           │ ║
║  │  Key Insight: Model focused on bars -10 to -5            │ ║
║  │               (strong uptrend formation)                  │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                                                                 ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │ Feature Attribution (Integrated Gradients)                │ ║
║  │                                                           │ ║
║  │  logret_12      ████████████████ +0.45                   │ ║
║  │  realized_vol   ████████████░░░░ +0.35                   │ ║
║  │  sr_heatmap     ████████░░░░░░░░ +0.22                   │ ║
║  │  range_pct      ████░░░░░░░░░░░░ +0.12                   │ ║
║  │  regime_label   ██░░░░░░░░░░░░░░ +0.08                   │ ║
║  │  upper_wick     █░░░░░░░░░░░░░░░ -0.05                   │ ║
║  │  gap_pct        ░░░░░░░░░░░░░░░░ -0.02                   │ ║
║  │                                                           │ ║
║  │  Interpretation: 12-bar momentum + low vol + S/R support │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                                                                 ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │ Support/Resistance Context                                │ ║
║  │                                                           │ ║
║  │  2,500 ─ Resistance ─────────────────────────            │ ║
║  │                                                           │ ║
║  │  2,450 ─ Entry ● ──────────────────────────              │ ║
║  │                                                           │ ║
║  │  2,400 ─ Support ────────────────────────── ✓ Held       │ ║
║  │                                                           │ ║
║  │  Model identified entry near strong support              │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                                                                 ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │ Policy Confidence & Alternatives                          │ ║
║  │                                                           │ ║
║  │  LONG 0.75   ████████████████████░░ 85% (chosen)         │ ║
║  │  LONG 0.50   ████░░░░░░░░░░░░░░░░░░ 12%                 │ ║
║  │  FLAT        █░░░░░░░░░░░░░░░░░░░░░  3%                 │ ║
║  │                                                           │ ║
║  │  High confidence decision (85%)                           │ ║
║  └──────────────────────────────────────────────────────────┘ ║
╚════════════════════════════════════════════════════════════════╝
```

**Components**:
- **Trade Selector**: Dropdown to pick any historical trade
- **Attention Heatmap**: Visual of which candles the model focused on
- **Feature Attribution**: Integrated Gradients bar chart
- **S/R Context Chart**: Price chart with support/resistance levels
- **Policy Confidence**: Probability distribution over actions

---

### 6. **Settings** Screen

```
╔════════════════════════════════════════════════════════════════╗
║  Settings                                     [Save] [Cancel]   ║
╠════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │ 📊 Trading Configuration                                 │  ║
║  │                                                          │  ║
║  │  Trading Mode:                                           │  ║
║  │  ○ Paper Trading (TradeLocker Testnet)                  │  ║
║  │  ○ Live Trading  (TradeLocker Production) ⚠️             │  ║
║  │  ● Backtest Only                                         │  ║
║  │                                                          │  ║
║  │  Default Symbol:   [ETH            ▼]                   │  ║
║  │  Max Position:     [1.0            ]                    │  ║
║  │  Episode Length:   [256            ]                    │  ║
║  └─────────────────────────────────────────────────────────┘  ║
║                                                                 ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │ 🛡️ Risk Controls                                         │  ║
║  │                                                          │  ║
║  │  Daily Loss Limit:        [-5.0]  %                     │  ║
║  │  Daily Profit Target:     [10.0]  %                     │  ║
║  │  Max Trades Per Day:      [10  ]                        │  ║
║  │  Trade Cooldown (bars):   [3   ]                        │  ║
║  │  Min Hold Time (bars):    [5   ]                        │  ║
║  │  Max Portfolio Heat:      [20.0]  %                     │  ║
║  │                                                          │  ║
║  │  ☑️ Enable Kill-Switch                                   │  ║
║  │  ☑️ Enable Trailing Stops                                │  ║
║  │  ☑️ Use Structure-Aware Stops                            │  ║
║  └─────────────────────────────────────────────────────────┘  ║
║                                                                 ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │ 📐 Position Sizing                                       │  ║
║  │                                                          │  ║
║  │  Method:                                                 │  ║
║  │  ○ Fixed                                                 │  ║
║  │  ● Volatility Targeting                                  │  ║
║  │  ○ Fractional Kelly                                      │  ║
║  │                                                          │  ║
║  │  Target Volatility:       [2.0 ]  % (daily)            │  ║
║  │  Kelly Fraction:          [0.25]                        │  ║
║  └─────────────────────────────────────────────────────────┘  ║
║                                                                 ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │ 🔧 Model Configuration                                   │  ║
║  │                                                          │  ║
║  │  Encoder Model:    [encoder_best.pt           ▼]       │  ║
║  │  Policy Model:     [ppo_trading_env.zip       ▼]       │  ║
║  │  Encoder Type:     [TCN                       ▼]       │  ║
║  │                                                          │  ║
║  │  ☑️ Use Pre-computed Volatility                          │  ║
║  │  ☑️ Include Position in Observations                     │  ║
║  │  ☑️ Enable Structured Logging                            │  ║
║  └─────────────────────────────────────────────────────────┘  ║
║                                                                 ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │ 🔌 API Configuration                                     │  ║
║  │                                                          │  ║
║  │  TradeLocker Environment: [testnet           ▼]        │  ║
║  │  TradeLocker Email:       [user@email.com    ]         │  ║
║  │  TradeLocker Server:      [demo.tradelocker  ]         │  ║
║  │  Account Number:          [12345             ]         │  ║
║  │  API Key:                 [********************]        │  ║
║  │                                          [Test Connection]│  ║
║  └─────────────────────────────────────────────────────────┘  ║
╚════════════════════════════════════════════════════════════════╝
```

**Components**:
- **Trading Config**: Mode, symbol, position limits
- **Risk Controls**: All risk parameters with validation
- **Position Sizing**: Method selection with parameters
- **Model Config**: Model selection and feature flags
- **API Config**: TradeLocker credentials with test button

---

## API Endpoints

### Base URL: `http://localhost:8000/api/v1`

### **1. Dashboard / Metrics**

#### `GET /metrics/summary`
**Response**:
```json
{
  "total_pnl": 12345.67,
  "total_pnl_pct": 5.2,
  "today_pnl": 234.56,
  "today_pnl_pct": 1.3,
  "sharpe_ratio": 1.82,
  "win_rate": 0.568,
  "max_drawdown": -0.082,
  "active_positions": 2,
  "max_positions": 3
}
```

#### `GET /metrics/equity-curve?days=30`
**Response**:
```json
{
  "timestamps": ["2024-12-01T00:00:00Z", ...],
  "equity": [10000, 10050, 10120, ...],
  "drawdown": [0, -0.01, -0.02, ...]
}
```

---

### **2. Positions**

#### `GET /positions/active`
**Response**:
```json
{
  "positions": [
    {
      "symbol": "ETH",
      "side": "LONG",
      "size": 0.75,
      "entry_price": 2450.0,
      "current_price": 2512.3,
      "pnl": 46.73,
      "pnl_pct": 1.9,
      "stop_loss": 2400.0,
      "trailing_stop": 2490.0,
      "entry_time": "2024-12-25T10:32:45Z",
      "model": "model_a"
    }
  ]
}
```

#### `POST /positions/close`
**Request**:
```json
{
  "symbol": "ETH",
  "reason": "manual"
}
```

#### `POST /positions/close-all`
**Request**:
```json
{
  "reason": "kill_switch_manual"
}
```

---

### **3. Trades**

#### `GET /trades/recent?limit=10`
**Response**:
```json
{
  "trades": [
    {
      "id": "trade_123",
      "timestamp": "2024-12-25T10:32:00Z",
      "symbol": "BTC",
      "side": "LONG",
      "entry_price": 45000.0,
      "exit_price": 45120.0,
      "size": 1.0,
      "pnl": 12.50,
      "duration_seconds": 3600,
      "exit_reason": "take_profit",
      "model": "model_a"
    }
  ]
}
```

#### `GET /trades/{trade_id}/explain`
**Response**:
```json
{
  "trade_id": "trade_123",
  "attention_weights": [0.1, 0.2, 0.8, ...],
  "feature_attribution": {
    "logret_12": 0.45,
    "realized_vol": 0.35,
    "sr_heatmap": 0.22
  },
  "policy_confidence": {
    "LONG_0.75": 0.85,
    "LONG_0.50": 0.12,
    "FLAT": 0.03
  },
  "sr_levels": [2400.0, 2500.0]
}
```

---

### **4. Training**

#### `GET /training/jobs`
**Response**:
```json
{
  "jobs": [
    {
      "job_id": "train_eth_20251025_143022",
      "status": "running",
      "stage": "policy",
      "progress": 0.45,
      "epoch": 45,
      "total_epochs": 100,
      "eta_seconds": 8100,
      "metrics": {
        "train_loss": 0.0234,
        "val_loss": 0.0267,
        "policy_reward": 0.0523
      }
    }
  ]
}
```

#### `POST /training/start`
**Request**:
```json
{
  "symbol": "ETH",
  "config": "config/rl_policy.yaml",
  "data_path": "data/features/encoder_windows_eth_emb.parquet",
  "timesteps": 500000
}
```

#### `POST /training/{job_id}/stop`

#### `WS /training/logs`
**WebSocket** for streaming logs

---

### **5. Backtesting**

#### `POST /backtest/run`
**Request**:
```json
{
  "model_path": "models/policies/ppo_trading_env.zip",
  "symbol": "ETH",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "episodes": 100
}
```

**Response**:
```json
{
  "backtest_id": "bt_123",
  "status": "running"
}
```

#### `GET /backtest/{backtest_id}/results`
**Response**:
```json
{
  "summary": {
    "total_return": 0.123,
    "sharpe_ratio": 1.82,
    "max_drawdown": -0.082,
    "win_rate": 0.568,
    "total_trades": 127
  },
  "equity_curve": [...],
  "trades": [...]
}
```

---

### **6. Inference (Live Trading)**

#### `POST /inference/predict`
**Request**:
```json
{
  "symbol": "ETH",
  "observation": [0.12, 0.45, ...],  // 259-dim
  "current_price": 2450.0,
  "current_vol": 0.03
}
```

**Response**:
```json
{
  "action": 2,  // Discrete action index
  "position_size": 0.75,
  "order_type": "market",
  "confidence": 0.85,
  "model_version": "model_a"
}
```

#### `GET /inference/status`
**Response**:
```json
{
  "trading_active": true,
  "mode": "paper",
  "models": {
    "model_a": {
      "version": "eth_v2.1",
      "traffic": 0.5,
      "pnl_today": 120.0,
      "sharpe_today": 1.85
    },
    "model_b": {
      "version": "eth_v2.0",
      "traffic": 0.5,
      "pnl_today": 114.0,
      "sharpe_today": 1.78
    }
  }
}
```

---

### **7. Risk Controls**

#### `GET /risk/status`
**Response**:
```json
{
  "daily_pnl": 234.56,
  "daily_loss_limit": -500.0,
  "daily_loss_pct": -0.012,
  "daily_loss_limit_pct": -0.05,
  "daily_trades": 5,
  "max_daily_trades": 10,
  "portfolio_heat": 0.12,
  "max_portfolio_heat": 0.20,
  "kill_switch_active": false,
  "cooldown_remaining_seconds": 120
}
```

#### `POST /risk/kill-switch`
**Request**:
```json
{
  "activate": true,
  "reason": "manual_user_triggered"
}
```

---

### **8. Configuration**

#### `GET /config`
**Response**: Returns current configuration

#### `PUT /config`
**Request**: Updates configuration (with validation)

---

### **9. Health & Status**

#### `GET /health`
**Response**:
```json
{
  "status": "healthy",
  "api_version": "1.0.0",
  "model_loaded": true,
  "tradelocker_connected": true,
  "database_connected": true
}
```

---

## Real-Time Features (WebSocket)

### `WS /ws/training`
Streams training logs and metrics

### `WS /ws/trading`
Streams live trading events (orders, fills, position updates)

### `WS /ws/metrics`
Streams real-time metrics updates (PnL, positions)

---

## Tech Stack Summary

### **Frontend**
- **Framework**: React 18+ with TypeScript
- **Styling**: Tailwind CSS 3+
- **Charts**: Recharts or Victory
- **State**: Redux Toolkit or Zustand
- **WebSocket**: Socket.io-client
- **HTTP**: Axios

### **Backend**
- **Framework**: FastAPI (Python 3.10+)
- **WebSocket**: FastAPI WebSocket support
- **ML**: PyTorch, Stable-Baselines3
- **Database**: PostgreSQL (state), Redis (cache)
- **Monitoring**: Prometheus + Grafana (optional)

---

This completes the UI/UX design and API specification. Next, I'll implement the components!
