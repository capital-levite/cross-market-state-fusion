#!/usr/bin/env python3
"""
Test script for RLHERPPOStrategy.
"""
import numpy as np
import jax
from strategies.rl_her_ppo import RLHERPPOStrategy
from strategies.base import MarketState, Action

def test_rl_her_ppo_basic():
    print("Testing RLHERPPOStrategy basic functionality...")
    
    # Initialize strategy
    strategy = RLHERPPOStrategy(input_dim=18, buffer_size=100, batch_size=10)
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
    
    # Test store and HER relabeling
    print("Testing store and HER...")
    next_state = MarketState(
        asset="BTC",
        prob=0.51,
        time_remaining=0.49,
        prob_history=[0.5] * 10 + [0.51]
    )
    
    # Store a few steps
    for i in range(5):
        strategy.store(state, action, 0.1, next_state, i == 4)
    
    # Buffer should have original + HER transitions
    print(f"  Buffer size after 5 steps (1 episode): {len(strategy.buffer)}")
    assert len(strategy.buffer) > 5
    
    # Fill buffer to trigger update
    print("Filling buffer for update...")
    for _ in range(20):
        for i in range(5):
            strategy.store(state, action, 0.1, next_state, i == 4)
    
    # Test update
    print("Testing update()...")
    metrics = strategy.update()
    print(f"  Metrics: {metrics}")
    assert metrics is not None
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    
    # Test save/load
    print("Testing save/load()...")
    save_path = "test_her_ppo_model.pkl"
    strategy.save(save_path)
    
    new_strategy = RLHERPPOStrategy(input_dim=18)
    new_strategy.load(save_path)
    
    # Verify params are loaded
    new_action = new_strategy.act(state)
    print(f"  New Action: {new_action}")
    
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        
    print("RLHERPPOStrategy basic tests passed!")

if __name__ == "__main__":
    test_rl_her_ppo_basic()
