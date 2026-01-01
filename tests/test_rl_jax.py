#!/usr/bin/env python3
"""
Test script for RLJaxStrategy.
"""
import numpy as np
import jax
from strategies.rl_jax import RLJaxStrategy
from strategies.base import MarketState, Action

def test_rl_jax_basic():
    print("Testing RLJaxStrategy basic functionality...")
    
    # Initialize strategy
    strategy = RLJaxStrategy(input_dim=18, buffer_size=10, batch_size=5, n_epochs=2)
    strategy.train()
    
    # Create a dummy state
    state = MarketState(
        asset="BTC",
        prob=0.5,
        time_remaining=0.5,
        prob_history=[0.5] * 10
    )
    
    # Test act
    print("Testing act()...")
    action = strategy.act(state)
    print(f"  Action: {action}")
    assert isinstance(action, Action)
    
    # Test store
    print("Testing store()...")
    next_state = MarketState(
        asset="BTC",
        prob=0.51,
        time_remaining=0.49,
        prob_history=[0.5] * 10 + [0.51]
    )
    strategy.store(state, action, 1.0, next_state, False)
    assert len(strategy.experiences) == 1
    
    # Fill buffer to trigger update
    print("Filling buffer for update...")
    for i in range(9):
        strategy.store(state, action, 0.1, next_state, i == 8)
    
    assert len(strategy.experiences) == 10
    
    # Test update
    print("Testing update()...")
    metrics = strategy.update()
    print(f"  Metrics: {metrics}")
    assert metrics is not None
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert len(strategy.experiences) == 0
    
    # Test save/load
    print("Testing save/load()...")
    save_path = "test_model.pkl"
    strategy.save(save_path)
    
    new_strategy = RLJaxStrategy(input_dim=18)
    new_strategy.load(save_path)
    
    # Verify params are loaded (indirectly by checking if act works)
    new_action = new_strategy.act(state)
    print(f"  New Action: {new_action}")
    
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        
    print("RLJaxStrategy basic tests passed!")

if __name__ == "__main__":
    test_rl_jax_basic()
