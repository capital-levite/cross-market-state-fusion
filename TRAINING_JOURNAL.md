# RL Trading on Polymarket: Training Journal

## Overview

Training a reinforcement learning agent to trade 15-minute binary prediction markets on Polymarket. The agent uses PPO (Proximal Policy Optimization) with an actor-critic architecture, implemented in MLX for Apple Silicon.

**Goal**: Learn to predict short-term price movements in crypto prediction markets by observing orderbook dynamics, Binance futures data, and market microstructure.

---

## What Makes This Interesting

### 1. Concurrent Multi-Asset Trading

Unlike typical RL trading systems that focus on a single asset, this agent trades **4 markets simultaneously** (BTC, ETH, SOL, XRP). Each 15-minute window spawns 4 independent binary markets, and the agent must:
- Allocate attention across all active markets
- Learn asset-specific patterns while sharing a single policy
- Handle asynchronous market expirations and refreshes

The same neural network makes decisions for all assets - learning generalizable crypto trading patterns rather than overfitting to one market.

### 2. Unique Market Structure

Polymarket's 15-minute binary markets are unusual:
- **Binary outcome**: Market resolves to $1.00 or $0.00 based on price direction
- **Known resolution time**: Exact 15-minute windows, not continuous trading
- **Orderbook-based**: Real CLOB with bid/ask spreads, not AMM
- **Cross-exchange arbitrage**: Polymarket prices should track Binance, but don't always

This creates exploitable inefficiencies - the agent can observe Binance price movements and bet on Polymarket before the orderbook fully adjusts.

### 3. Multi-Source Real-Time Data Fusion

The agent fuses data from multiple WebSocket streams in real-time:
```
Binance Spot WSS     → Price, returns, momentum
Binance Futures WSS  → Funding rates, CVD, liquidations
Polymarket CLOB WSS  → Orderbook depth, bid/ask spreads
```

This creates an 18-dimensional state space that captures both the underlying asset dynamics AND the prediction market microstructure.

### 4. Sparse Reward Challenge

Unlike continuous markets where PnL accrues gradually, binary markets only pay out at resolution:
- Entry at prob=0.55, exit at resolution
- Either +$0.45 profit (if correct) or -$0.55 loss (if wrong)
- No intermediate reward signal during the 15-minute window

This makes credit assignment harder - the agent must learn which early signals predict final outcomes.

### 5. MLX on Apple Silicon

Training runs on-device using Apple's MLX framework, enabling:
- GPU-accelerated PPO updates on M1/M2/M3
- No cloud GPU costs for experimentation
- Real-time training during live market hours

---

## The Setup

### Market Structure
- **Asset**: 15-minute binary markets on BTC, ETH, SOL, XRP
- **Question**: "Will [ASSET] price go UP or DOWN in the next 15 minutes?"
- **Tokens**: UP token (probability 0-1) and DOWN token (1 - UP probability)
- **Resolution**: Based on Binance spot price at market close

### State Space (18 features)
```
Momentum (3):
- returns_1m, returns_5m, returns_10m: Ultra-short price momentum

Order Flow (4):
- ob_imbalance_l1, ob_imbalance_l5: Orderbook imbalance at levels 1 & 5
- trade_flow_imbalance: Buy vs sell pressure
- cvd_acceleration: Cumulative volume delta acceleration

Microstructure (3):
- spread_pct: Bid-ask spread as % of prob
- trade_intensity: Recent trade frequency
- large_trade_flag: Large trade detected

Volatility (2):
- vol_5m: 5-minute realized volatility
- vol_expansion: Current vol vs recent average

Position (4):
- has_position, position_side, position_pnl, time_remaining

Regime (2):
- vol_regime, trend_regime: Market environment context
```

### Action Space (Current - Phase 2)
- **HOLD (0)**: No action
- **BUY (1)**: Buy UP token (betting price goes up)
- **SELL (2)**: Sell UP token / buy DOWN (betting price goes down)
- Fixed 50% position sizing ($5 on $10 base)

*Phase 1 used 7 actions with variable sizing (25/50/100%) - see Changes Made section.*

### Architecture
```
Actor:  Linear(18, 128) -> tanh -> Linear(128, 128) -> tanh -> Linear(128, 3) -> softmax
Critic: Linear(18, 128) -> tanh -> Linear(128, 128) -> tanh -> Linear(128, 1)
```

---

## Training Status

**Date**: December 29, 2025, 19:08 - 20:54
**Total Updates**: 72 (36 + 36)
**Total Trades**: 4,875 (1,545 + 3,330)
**Capital**: $10 base, 50% position sizing ($5/trade)
**Current PnL**: $10.93 (109% ROI)

---

### Phase 1: Shaped Rewards (Updates 1-36)

**Duration**: ~52 minutes
**Trades**: 1,545

### Configuration
```python
buffer_size = 512      # Experiences before update
batch_size = 64
n_epochs = 10
lr_actor = 1e-4
lr_critic = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
entropy_coef = 0.05    # Low - contributed to collapse
```

### Reward Function (Original)
```python
def _compute_step_reward(self, state, action, pos):
    reward = 0.0

    # PnL delta (scaled down)
    pnl_delta = current_pnl - previous_pnl
    reward += pnl_delta * 0.1

    # Transaction cost penalty
    if action != HOLD:
        reward -= 0.001

    # Momentum alignment bonus
    if action.is_buy and state.prob_velocity > 0:
        reward += 0.002

    # Position sizing bonus
    reward += 0.001 * action.size_multiplier

    return reward
```

### Results Summary

| Metric | Start | End | Notes |
|--------|-------|-----|-------|
| Entropy | 1.09 | 0.36 | Collapsed - premature convergence |
| PnL | $0.00 | $3.90 | Positive but volatile |
| Win Rate | 0% | 20.2% | Below random (33%) |
| Buffer Win Rate | 40% | 76% | Diverged from reality |

### Training Progression (Selected Updates)

| Update | Entropy | Value Loss | PnL | Trades | Win Rate |
|--------|---------|------------|-----|--------|----------|
| 1 | 1.09 | 3.29 | $3.25 | 126 | 19.8% |
| 10 | 0.79 | 6.78 | $4.40 | 881 | 17.3% |
| 20 | 0.40 | 3.60 | $1.37 | 1215 | 19.4% |
| 30 | 0.47 | 1.80 | $2.95 | 1443 | 19.2% |
| 36 | 0.36 | 1.83 | $3.90 | 1545 | 20.2% |

### Key Observations

**1. Entropy Collapse**
```
Update 1:  entropy=1.09  (good exploration)
Update 10: entropy=0.79
Update 20: entropy=0.40
Update 36: entropy=0.36  (nearly deterministic)
```
The policy converged too quickly. With `entropy_coef=0.05`, the entropy bonus wasn't strong enough to maintain exploration.

**2. Reward Signal Misalignment**

The buffer win rate (90%+) measured how often `reward > 0`, not actual profitable trades. The shaping rewards dominated:
- Transaction cost: -0.001 per trade
- Momentum bonus: +0.002 when aligned
- Size bonus: +0.001

These micro-bonuses drowned out the actual PnL signal (scaled by 0.1). The agent learned to satisfy shaping rewards, not maximize profit.

**3. Value Loss Spikes**
```
Update 5:  value_loss=38.17  (spike)
Update 14: value_loss=20.93  (spike)
Update 26: value_loss=42.64  (spike)
```
The critic struggled to predict returns, indicating the reward signal was noisy/inconsistent.

---

## Diagnosis

### Problem: Reward Shaping Dominated PnL

The original reward function had multiple components:
1. **PnL delta × 0.1** - The actual signal we care about
2. **Transaction cost** - Constant penalty
3. **Momentum bonus** - Correlation bonus
4. **Sizing bonus** - Encourages larger positions

With typical PnL deltas of $0.01-0.05, the scaled signal was 0.001-0.005. The shaping rewards were of similar magnitude, creating a noisy signal where the agent could achieve positive rewards without profitable trading.

### Buffer Win Rate vs Cumulative Win Rate

- **Buffer Win Rate**: % of experiences with reward > 0 (includes shaping bonuses)
- **Cumulative Win Rate**: % of closed trades that were profitable

The divergence (91% vs 20%) revealed the agent was optimizing for shaping rewards, not actual profits.

---

## Changes Made

### 1. Simplified Reward Function

**Before**: Step-by-step PnL delta + shaping rewards
**After**: Pure realized PnL on position close

```python
def _compute_step_reward(self, cid, state, action, pos):
    """Compute reward signal for RL training - pure realized PnL."""
    # Only reward on position close - cleaner signal
    return self.pending_rewards.pop(cid, 0.0)

# In position close logic:
pnl = (exit_price - entry_price) * size  # or inverse for DOWN
self.pending_rewards[cid] = pnl  # Pure realized PnL
```

**Rationale**:
- No more micro-bonuses that can be gamed
- Sparse but meaningful signal
- Directly aligned with what we want to optimize

### 2. Increased Entropy Coefficient

**Before**: `entropy_coef = 0.05`
**After**: `entropy_coef = 0.10`

**Rationale**: Prevent premature convergence, maintain exploration longer.

### 3. Simplified Action Space

**Before**: 7 actions with variable sizing
```
HOLD, BUY_SMALL (25%), BUY_MEDIUM (50%), BUY_LARGE (100%)
SELL_SMALL (25%), SELL_MEDIUM (50%), SELL_LARGE (100%)
```

**After**: 3 actions with fixed sizing
```
HOLD (0), BUY (1), SELL (2) - all at 50% position size
```

**Rationale**: Reduce complexity. Let the model learn when to trade before learning how much.

### 4. Reset Reward Normalization

```python
# Reset stats while keeping weights
np.savez("rl_model_stats.npz",
    reward_mean=0.0,
    reward_std=1.0,
    reward_count=0
)
```

**Rationale**: Old stats were calibrated to shaped rewards (mean=-0.002, std=0.01). New pure-PnL rewards have different distribution.

---

### Phase 2: Pure PnL Reward (Updates 37-72)

**Duration**: December 29, 2025, 20:02 - 20:54 (~52 minutes)
**Trades**: 3,330

**Capital Context**: $10 base capital, 50% position sizing ($5/trade).

#### Configuration Changes
```python
entropy_coef = 0.10  # Doubled
# Reward: pure realized PnL (no shaping)
```

#### Results (36 updates)

| Update | Entropy | Value Loss | Avg Reward | PnL | Trades | Win Rate |
|--------|---------|------------|------------|-----|--------|----------|
| 1 | 0.68 | 149.5 | +0.40 | $5.20 | 27 | 33.3% |
| 5 | 0.93 | 2.09 | +0.02 | $5.65 | 226 | 20.8% |
| 10 | 1.06 | 7.16 | +0.02 | $9.55 | 616 | 22.9% |
| 15 | 1.07 | 14.21 | -0.07 | $8.80 | 1,129 | 21.0% |
| 20 | 1.05 | 3.10 | -0.01 | $5.85 | 1,672 | 21.1% |
| 25 | 1.04 | 3.75 | +0.03 | $9.48 | 2,245 | 21.2% |
| 30 | 1.07 | 0.49 | +0.00 | $7.18 | 2,751 | 20.9% |
| 36 | 1.05 | 6.47 | +0.05 | $10.93 | 3,330 | 21.2% |

**Final ROI**: $10.93 PnL on $10 base = **109% return** on capital.

#### Observations

**1. Entropy Fully Recovered**
```
Phase 1 end:   entropy=0.36 (collapsed)
Phase 2 start: entropy=0.68 (loaded weights)
Phase 2 end:   entropy=1.05 (full exploration)
```
The higher entropy coefficient worked - entropy recovered from 0.36 to 1.05, near the theoretical maximum (~1.1 for 3 actions). The policy maintained proper stochasticity throughout Phase 2.

**2. Value Loss Spikes**

Multiple value loss spikes observed:
- Update 1: 149.5 (reward scale change)
- Update 7: 69.95 (large reward variance)
- Updates 8-9: 18-20 (gradual stabilization)
- Update 10: 7.16 (settling down)

The critic is adapting to the pure PnL reward signal which has higher variance than shaped rewards.

**3. Consistent Positive PnL**

Cumulative PnL grew steadily through Phase 2, ending at $10.93 on $10 base capital (109% ROI). Buffer rewards consistently positive, indicating the model found profitable trades on average with the pure PnL signal.

**4. Win Rate Plateau**

Win rate stable around 21-24%, which is below random (33%) but the model is profitable due to asymmetric payoffs - winners pay more than losers cost.

---

## Technical Details

### PPO Implementation

Using MLX for Apple Silicon optimization:

```python
def update(self):
    # Compute GAE advantages
    advantages = self._compute_gae(rewards, values, dones)
    returns = advantages + values

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for epoch in range(n_epochs):
        for batch in batches:
            # Policy loss with clipping
            ratio = new_probs / old_probs
            surr1 = ratio * advantages
            surr2 = clip(ratio, 1-epsilon, 1+epsilon) * advantages
            policy_loss = -min(surr1, surr2).mean()

            # Value loss
            value_loss = MSE(values, returns)

            # Entropy bonus
            entropy = -(probs * log(probs)).sum(-1).mean()

            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

### Reward Normalization

Running mean/std normalization to stabilize training:

```python
def normalize_reward(self, reward):
    self.reward_count += 1
    delta = reward - self.reward_mean
    self.reward_mean += delta / self.reward_count
    self.reward_std = sqrt(running_variance / count)
    return (reward - self.reward_mean) / (self.reward_std + 1e-8)
```

---

## Conclusions

This experiment demonstrated:

1. **Reward shaping can backfire** - The agent learned to game shaping rewards instead of maximizing PnL. Pure realized PnL worked better despite being sparse.

2. **Entropy coefficient matters** - 0.05 led to policy collapse; 0.10 maintained healthy exploration throughout training.

3. **Win rate isn't everything** - 21% win rate sounds bad, but asymmetric payoffs (binary markets) made it profitable. Winners paid more than losers cost.

4. **Action space simplification helped** - Reducing from 7 to 3 actions let the model focus on timing rather than sizing.

**Final result**: 109% ROI on base capital over 72 updates (~1.75 hours of live market training).

---

## Appendix: File Structure

```
experiments/03_polymarket/
├── run.py                 # Main trading engine
├── dashboard.py           # Real-time visualization
├── strategies/
│   ├── base.py           # Action/State definitions
│   └── rl_mlx.py         # PPO implementation
├── helpers/
│   ├── polymarket.py     # Market data fetching
│   ├── binance_wss.py    # Price streaming
│   ├── orderbook_wss.py  # Orderbook streaming
│   └── training_logger.py # CSV logging
├── logs/
│   ├── trades_*.csv      # All executed trades
│   └── updates_*.csv     # PPO update metrics
├── rl_model.safetensors  # Model weights
└── rl_model_stats.npz    # Reward normalization stats
```

---

*Last updated: December 29, 2025*
