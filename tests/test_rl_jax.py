#!/usr/bin/env python3
"""
Test script for the new Distributional Temporal RLJaxStrategy.
"""
import numpy as np
import jax
import jax.numpy as jnp
from strategies.rl_jax import RLJaxStrategy
from strategies.base import MarketState, Action
import time

def test_rl_jax_distributional_temporal():
    print("\n--- Testing Distributional Temporal RLJaxStrategy ---")
    
    # Initialize strategy with small buffer for testing
    # Typical params: n_atoms=51, history_len=5
    strategy = RLJaxStrategy(
        input_dim=18, 
        buffer_size=20, 
        batch_size=10, 
        n_epochs=2,
        history_len=5,
        n_atoms=51
    )
    strategy.train()
    
    # helper to create a dummy state
    def make_dummy_state(asset="BTC", prob=0.5, time_rem=0.5):
        # Create a state that will produce 18 features via to_features()
        return MarketState(
            asset=asset,
            prob=prob,
            time_remaining=time_rem,
            prob_history=[prob] * 10
        )

    print("1. Testing temporal state tracking and act()...")
    state = make_dummy_state(prob=0.5)
    
    # Act multiple times to check temporal history accumulation
    for i in range(3):
        action = strategy.act(state)
        print(f"   Step {i} Action: {action.name}")
        assert isinstance(action, Action)
        assert len(strategy._state_history["BTC"]) == i + 1

    print("2. Testing distributional properties...")
    # Get the distribution directly for a state
    features = state.to_features().reshape(1, -1)
    temporal = strategy._last_temporal_state.reshape(1, -1)
    
    dist = strategy.critic_state.apply_fn(
        {'params': strategy.critic_state.params}, 
        jnp.array(features), 
        jnp.array(temporal)
    )
    
    print(f"   Distribution shape: {dist.shape} (Expected: (1, 51))")
    assert dist.shape == (1, 51)
    # Check if probabilities sum to 1
    prob_sum = float(jnp.sum(dist))
    print(f"   Probability sum: {prob_sum:.6f}")
    assert np.isclose(prob_sum, 1.0, atol=1e-5)

    print("3. Testing experience storage and update()...")
    # Fill buffer
    for i in range(20):
        s = make_dummy_state(prob=0.5 + i*0.01)
        # Act to set temporal state
        a = strategy.act(s)
        # Next state
        ns = make_dummy_state(prob=0.5 + (i+1)*0.01)
        # Store
        strategy.store(s, a, 1.0 if a == Action.BUY else -0.5, ns, i == 19)
    
    assert len(strategy.experiences) == 20
    
    print("   Running update (Projected Distributional Bellman)...")
    start_time = time.time()
    metrics = strategy.update()
    end_time = time.time()
    
    print(f"   Metrics: {metrics}")
    print(f"   Update time: {end_time - start_time:.4f}s")
    
    assert metrics is not None
    assert "value_loss" in metrics  # Should be cross-entropy
    assert len(strategy.experiences) == 0

    print("4. Testing save/load consistency...")
    save_path = "test_dist_temp_model.pkl"
    strategy.save(save_path)
    
    new_strategy = RLJaxStrategy(input_dim=18)
    new_strategy.load(save_path)
    
    # Act and check if it yields valid output
    new_action = new_strategy.act(state)
    print(f"   Loaded Strategy Action: {new_action.name}")
    assert isinstance(new_action, Action)
    
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        
    print("\n[SUCCESS] Distributional Temporal RLJaxStrategy passed all tests!")

if __name__ == "__main__":
    try:
        test_rl_jax_distributional_temporal()
    except Exception as e:
        print(f"\n[FAILURE] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
