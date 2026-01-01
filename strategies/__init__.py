"""
Trading strategies for Polymarket.

Usage:
    from strategies import create_strategy, AVAILABLE_STRATEGIES

    strategy = create_strategy("mean_revert")
    action = strategy.act(state)
"""
from .base import Strategy, MarketState, Action
from .random_strat import RandomStrategy
from .mean_revert import MeanRevertStrategy
from .momentum import MomentumStrategy
from .fade_spike import FadeSpikeStrategy
try:
    from .rl_mlx import RLStrategy  # MLX-based PPO with proper autograd
except ImportError:
    RLStrategy = None
from .rl_jax import RLJaxStrategy
from .rl_her_ddpg import RLHERDDPGStrategy
from .rl_her_ppo import RLHERPPOStrategy
from .gating import GatingStrategy


AVAILABLE_STRATEGIES = [
    "random",
    "mean_revert",
    "momentum",
    "fade_spike",
    "rl",
    "rl_jax",
    "rl_her_ddpg",
    "rl_her_ppo",
    "gating",
]


def create_strategy(name: str, **kwargs) -> Strategy:
    """Factory function to create strategies."""
    strategies = {
        "random": RandomStrategy,
        "mean_revert": MeanRevertStrategy,
        "momentum": MomentumStrategy,
        "fade_spike": FadeSpikeStrategy,
        "rl": RLStrategy,
        "rl_jax": RLJaxStrategy,
        "rl_her_ddpg": RLHERDDPGStrategy,
        "rl_her_ppo": RLHERPPOStrategy,
    }

    if name == "gating":
        # Create gating with default experts
        experts = [
            MeanRevertStrategy(),
            MomentumStrategy(),
            FadeSpikeStrategy(),
        ]
        return GatingStrategy(experts, **kwargs)

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}")

    if strategies[name] is None:
        raise ImportError(f"Strategy '{name}' requires dependencies that are not installed (e.g., mlx).")

    return strategies[name](**kwargs)


__all__ = [
    # Base
    "Strategy",
    "MarketState",
    "Action",
    # Strategies
    "RandomStrategy",
    "MeanRevertStrategy",
    "MomentumStrategy",
    "FadeSpikeStrategy",
    "RLStrategy",
    "RLJaxStrategy",
    "RLHERDDPGStrategy",
    "RLHERPPOStrategy",
    "GatingStrategy",
    # Factory
    "create_strategy",
    "AVAILABLE_STRATEGIES",
]
