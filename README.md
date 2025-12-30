# Polymarket RL Trading

PPO agent trading 4 concurrent 15-minute binary prediction markets (BTC, ETH, SOL, XRP) on Polymarket. Built with MLX for Apple Silicon.

**Key features:**
- Trades 4 markets simultaneously with shared policy
- Fuses Binance spot/futures + Polymarket orderbook data (18-dim state)
- Pure realized PnL reward - sparse signal at market resolution
- On-device training via MLX

## Training Journal

### Training Status: 72 updates, 4,875 trades

**Phase 1: Shaped Rewards (Updates 1-36)**
- Entropy collapsed 1.09 → 0.36 (policy became deterministic)
- Cause: Reward shaping dominated actual PnL signal
- Result: $3.90 PnL, 1,545 trades, 20.2% win rate

**Phase 2: Pure PnL Reward (Updates 37-72)**
- Switched to pure realized PnL reward, doubled entropy coefficient (0.05 → 0.10)
- Entropy recovered to 1.05, model exploring properly
- Result: $10.93 PnL, 3,330 trades, 21.2% win rate

| Phase | Updates | Entropy | PnL | Trades | Win Rate |
|-------|---------|---------|-----|--------|----------|
| 1 | 36 | 0.36 | $3.90 | 1,545 | 20.2% |
| 2 | 36 | 1.05 | $10.93 | 3,330 | 21.2% |

**Context**: $10 base capital, 50% position sizing ($5/trade). Phase 2 PnL of $10.93 = **109% ROI** on base capital.

**Key insight**: Win rate ~21% but profitable due to asymmetric payoffs - winners pay more than losers cost.

See [TRAINING_JOURNAL.md](TRAINING_JOURNAL.md) for full analysis.

## Architecture

```
├── run.py                    # Main trading engine
├── dashboard.py              # Real-time Flask-SocketIO web dashboard
├── strategies/
│   ├── base.py               # Base classes (Action, MarketState, Strategy)
│   ├── rl_mlx.py             # PPO implementation with MLX
│   ├── momentum.py           # Momentum baseline
│   ├── mean_revert.py        # Mean reversion baseline
│   ├── fade_spike.py         # Spike fading baseline
│   └── gating.py             # Ensemble gating strategy
└── helpers/
    ├── polymarket_api.py     # Polymarket REST API
    ├── binance_wss.py        # Binance spot price streaming
    ├── binance_futures.py    # Futures data (funding, OI, CVD, liquidations)
    └── orderbook_wss.py      # Polymarket CLOB orderbook streaming
```

## Features

### State Space (18 dimensions)

| Category | Features | Description |
|----------|----------|-------------|
| **Momentum** | `returns_1m`, `returns_5m`, `returns_10m` | Ultra-short price momentum |
| **Order Flow** | `ob_imbalance_l1`, `ob_imbalance_l5`, `trade_flow`, `cvd_accel` | Buy/sell pressure signals |
| **Microstructure** | `spread_pct`, `trade_intensity`, `large_trade_flag` | Market quality metrics |
| **Volatility** | `vol_5m`, `vol_expansion` | Short-term vol regime |
| **Position** | `has_position`, `position_side`, `position_pnl`, `time_remaining` | Current exposure |
| **Regime** | `vol_regime`, `trend_regime` | Market environment context |

### Action Space (3 actions)

| Action | Description |
|--------|-------------|
| `HOLD (0)` | No action |
| `BUY (1)` | Long UP token (50% position size) |
| `SELL (2)` | Long DOWN token (50% position size) |

*Note: Originally 7 actions with variable sizing (25/50/100%), simplified in Phase 2.*

### Reward Signal

Pure realized PnL on position close - sparse but aligned with actual profit:
- No shaping rewards (removed momentum bonuses, transaction penalties)
- Reward only fires when position closes
- Cleaner learning signal, harder credit assignment

## Data Sources

### Real-time Streams
- **Binance WebSocket** - Spot prices for BTC, ETH, SOL, XRP
- **Binance Futures WebSocket** - Trade flow, CVD, large trade detection
- **Polymarket CLOB WebSocket** - Orderbook depth, bid/ask spreads

### Polling Data
- **Binance Futures API** - Funding rates, open interest, mark price
- **Binance Klines** - Multi-timeframe returns (1m, 5m, 10m, 15m, 1h)
- **Polymarket API** - Active markets, token prices

## Usage

### Training Mode
```bash
# Train RL agent with $100 position sizing
python run.py --strategy rl --train --size 100

# Train with custom buffer size (more data per update)
python run.py --strategy rl --train --size 50
```

### Inference Mode
```bash
# Load trained model and run
python run.py --strategy rl --load rl_model --size 100
```

### Dashboard
```bash
# Run in separate terminal
python dashboard.py --port 5001
# Open http://localhost:5001
```

### Baseline Strategies
```bash
python run.py --strategy momentum   # Follow short-term trends
python run.py --strategy mean_revert # Fade extreme moves
python run.py --strategy fade_spike  # Fade large spikes
python run.py --strategy random      # Random baseline
```

## PPO Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lr_actor` | 1e-4 | Actor learning rate |
| `lr_critic` | 3e-4 | Critic learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda for advantage estimation |
| `clip_epsilon` | 0.2 | PPO surrogate clipping range |
| `entropy_coef` | 0.10 | Entropy bonus coefficient (was 0.05 in Phase 1) |
| `buffer_size` | 512 | Experience buffer before update |
| `batch_size` | 64 | Mini-batch size for SGD |
| `n_epochs` | 10 | PPO epochs per update |

## Network Architecture

```
Actor (Policy Network)
├── Linear(18 → 128) + Tanh
├── Linear(128 → 128) + Tanh
└── Linear(128 → 3) + Softmax

Critic (Value Network)
├── Linear(18 → 128) + Tanh
├── Linear(128 → 128) + Tanh
└── Linear(128 → 1)
```

## Dashboard Features

- **Live market cards** - Probability, time remaining, velocity
- **Position tracking** - Current exposure with unrealized P&L
- **P&L chart** - Rolling performance visualization
- **RL metrics** - Buffer progress, policy/value loss, entropy, KL divergence
- **Trade feed** - Recent trade history with outcomes

## Requirements

```
mlx>=0.5.0        # Apple Silicon ML framework
websockets>=12.0  # Async WebSocket client
flask>=3.0.0      # Web framework
flask-socketio>=5.3.0  # Real-time dashboard
numpy>=1.24.0     # Numerical computing
requests>=2.31.0  # HTTP client
```

## Installation

```bash
cd experiments/03_polymarket
python -m venv venv
source venv/bin/activate
pip install mlx websockets flask flask-socketio numpy requests
```

## Paper Trading

This system is for **paper trading only**. It does not execute real trades on Polymarket. All positions are simulated using live market data.

To connect to real trading, you would need:
1. Polymarket CLOB API credentials
2. Wallet with USDC on Polygon
3. Order execution logic (not implemented)

## Performance Notes

- Markets refresh every 15 minutes
- Decision loop runs at ~2 Hz (500ms tick)
- PPO updates take ~1-2 seconds on M1/M2
- Dashboard updates every tick for responsiveness

## Known Limitations

- No real order execution
- Single-threaded decision loop
- No persistence of training state across restarts (save manually)
- Markets may have low liquidity during off-hours

## Live Trading Considerations

Paper trading results won't directly translate to live performance due to:

**Latency**
- Paper trading assumes instant fills at mid-price
- Real orders face network latency (50-200ms to Polymarket)
- Fast-moving markets may move against you before fill
- Cross-exchange arb signals decay quickly - stale by 100ms+

**Orderbook Impact**
- Paper trades don't consume liquidity
- Real orders eat into the book, especially at size
- Thin books during off-hours = significant slippage
- Large trades ($100+) may need to walk the book

**Execution Quality**
- No maker/taker fee modeling in paper mode
- Partial fills not simulated
- Queue position matters for limit orders
- Aggressive orders pay the spread

**Market Microstructure**
- Other traders react to your orders
- Visible size on book attracts front-running
- Repeated patterns get arbitraged away
- Alpha decay as strategy becomes known

For live trading, expect 20-50% performance degradation from paper results depending on position size and market conditions.
