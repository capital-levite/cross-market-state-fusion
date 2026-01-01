#!/usr/bin/env python3
"""
JAX-based DDPG with Hindsight Experience Replay (HER) strategy.
Uses Flax for neural networks and Optax for optimization.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass
import os
import pickle
import random

from .base import Strategy, MarketState, Action

@dataclass
class HERExperience:
    """Experience tuple for HER."""
    state: np.ndarray
    goal: np.ndarray  # Target price movement or regime
    action: float     # Continuous action [-1, 1]
    reward: float
    next_state: np.ndarray
    done: bool
    achieved_goal: np.ndarray

class DDPGActor(nn.Module):
    """Actor network: (state, goal) -> continuous action."""
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x, goal):
        # Concatenate state and goal
        inp = jnp.concatenate([x, goal], axis=-1)
        x = nn.Dense(self.hidden_size)(inp)
        x = jax.nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = jax.nn.relu(x)
        x = nn.Dense(1)(x)
        return jax.nn.tanh(x)  # Action in [-1, 1]

class DDPGCritic(nn.Module):
    """Critic network: (state, goal, action) -> Q-value."""
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x, goal, action):
        # Concatenate state, goal, and action
        inp = jnp.concatenate([x, goal, action], axis=-1)
        x = nn.Dense(self.hidden_size)(inp)
        x = jax.nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = jax.nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class RLHERDDPGStrategy(Strategy):
    """DDPG strategy with HER for sparse trading rewards."""

    def __init__(
        self,
        input_dim: int = 18,
        goal_dim: int = 1,  # e.g., target price change
        hidden_size: int = 128,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,  # Soft update parameter
        buffer_size: int = 10000,
        batch_size: int = 64,
        her_ratio: float = 0.8,  # Fraction of HER transitions
        noise_std: float = 0.1,
        seed: int = 42,
    ):
        super().__init__("rl_her_ddpg")
        self.input_dim = input_dim
        self.goal_dim = goal_dim
        self.hidden_size = hidden_size

        # Hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.her_ratio = her_ratio
        self.noise_std = noise_std

        # JAX PRNG Keys
        self.rng = jax.random.PRNGKey(seed)
        
        # Networks
        self.actor_module = DDPGActor(hidden_size=hidden_size)
        self.critic_module = DDPGCritic(hidden_size=hidden_size)

        # Initialize parameters
        init_rng, self.rng = jax.random.split(self.rng)
        dummy_state = jnp.zeros((1, input_dim))
        dummy_goal = jnp.zeros((1, goal_dim))
        dummy_action = jnp.zeros((1, 1))
        
        actor_params = self.actor_module.init(init_rng, dummy_state, dummy_goal)['params']
        critic_params = self.critic_module.init(init_rng, dummy_state, dummy_goal, dummy_action)['params']

        # Optimizers
        actor_tx = optax.adam(learning_rate=lr_actor)
        critic_tx = optax.adam(learning_rate=lr_critic)

        # Train states
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor_module.apply,
            params=actor_params,
            tx=actor_tx
        )
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic_module.apply,
            params=critic_params,
            tx=critic_tx
        )

        # Target networks
        self.target_actor_params = actor_params
        self.target_critic_params = critic_params

        # Experience buffer
        self.buffer: List[HERExperience] = []
        
        # Current episode buffer for HER relabeling
        self.episode_buffer: List[HERExperience] = []

        # Running stats for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

    def _get_current_goal(self, state: MarketState) -> np.ndarray:
        """Define the 'goal' for the current state. 
        For trading, the goal could be 'achieve a positive return'."""
        # We'll use a simple goal: target price change of 0.5% (0.005)
        return np.array([0.005], dtype=np.float32)

    def _get_achieved_goal(self, state: MarketState, next_state: MarketState) -> np.ndarray:
        """What actually happened? The price change."""
        return np.array([next_state.prob - state.prob], dtype=np.float32)

    def act(self, state: MarketState) -> Action:
        """Select action using current policy."""
        features = state.to_features()
        goal = self._get_current_goal(state)
        
        features_jax = jnp.array(features.reshape(1, -1))
        goal_jax = jnp.array(goal.reshape(1, -1))

        # Get continuous action
        action_cont = self.actor_state.apply_fn({'params': self.actor_state.params}, features_jax, goal_jax)
        action_val = float(action_cont[0, 0])

        if self.training:
            # Add exploration noise
            noise_rng, self.rng = jax.random.split(self.rng)
            action_val += float(jax.random.normal(noise_rng) * self.noise_std)
            action_val = np.clip(action_val, -1.0, 1.0)

        # Map continuous [-1, 1] to discrete Action
        if action_val > 0.33:
            return Action.BUY
        elif action_val < -0.33:
            return Action.SELL
        else:
            return Action.HOLD

    def store(self, state: MarketState, action: Action, reward: float,
              next_state: MarketState, done: bool):
        """Store experience in episode buffer for HER relabeling."""
        # Map Action enum back to continuous value for training
        action_val = 0.0
        if action == Action.BUY: action_val = 1.0
        elif action == Action.SELL: action_val = -1.0

        goal = self._get_current_goal(state)
        achieved_goal = self._get_achieved_goal(state, next_state)

        exp = HERExperience(
            state=state.to_features(),
            goal=goal,
            action=action_val,
            reward=reward,
            next_state=next_state.to_features(),
            done=done,
            achieved_goal=achieved_goal
        )
        self.episode_buffer.append(exp)

        if done:
            self._process_episode()

    def _process_episode(self):
        """Relabel episode with HER and move to main buffer."""
        for i, exp in enumerate(self.episode_buffer):
            # 1. Store original experience
            self.buffer.append(exp)

            # 2. HER: Relabel with achieved goals from future steps in the same episode
            if random.random() < self.her_ratio:
                # Pick a future step's achieved goal
                future_idx = random.randint(i, len(self.episode_buffer) - 1)
                future_goal = self.episode_buffer[future_idx].achieved_goal
                
                # Re-compute reward for this new goal
                # Reward is the PnL if the goal was the actual price movement
                # exp.action is in [-1, 1] (BUY=1, SELL=-1), future_goal[0] is the price change
                # We use the share-based PnL formula: (exit - entry) / entry
                entry_price = max(0.01, exp.state[0]) # Use first feature (prob) as entry price proxy if needed, but better to use actual prob
                # Actually, in MarketState.to_features, the first 3 are returns. 
                # Let's just use the price change directly scaled.
                new_reward = exp.action * future_goal[0] * 10.0 
                
                her_exp = HERExperience(
                    state=exp.state,
                    goal=future_goal,
                    action=exp.action,
                    reward=new_reward,
                    next_state=exp.next_state,
                    done=exp.done,
                    achieved_goal=exp.achieved_goal
                )
                self.buffer.append(her_exp)

        # Clear episode buffer
        self.episode_buffer.clear()

        # Limit main buffer size
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]

    @staticmethod
    @jax.jit
    def update_step(
        actor_state: train_state.TrainState,
        critic_state: train_state.TrainState,
        target_actor_params: Any,
        target_critic_params: Any,
        states: jnp.ndarray,
        goals: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_states: jnp.ndarray,
        dones: jnp.ndarray,
        gamma: float,
        tau: float
    ):
        # 1. Update Critic
        def critic_loss_fn(params):
            # Target Q value
            next_actions = actor_state.apply_fn({'params': target_actor_params}, next_states, goals)
            next_q = critic_state.apply_fn({'params': target_critic_params}, next_states, goals, next_actions)
            target_q = rewards + (1.0 - dones) * gamma * next_q
            
            # Current Q value
            current_q = critic_state.apply_fn({'params': params}, states, goals, actions)
            loss = jnp.mean((current_q - target_q) ** 2)
            return loss

        critic_grad_fn = jax.value_and_grad(critic_loss_fn)
        critic_loss, critic_grads = critic_grad_fn(critic_state.params)
        new_critic_state = critic_state.apply_gradients(grads=critic_grads)

        # 2. Update Actor
        def actor_loss_fn(params):
            actions_pred = actor_state.apply_fn({'params': params}, states, goals)
            q_values = critic_state.apply_fn({'params': new_critic_state.params}, states, goals, actions_pred)
            return -jnp.mean(q_values)

        actor_grad_fn = jax.value_and_grad(actor_loss_fn)
        actor_loss, actor_grads = actor_grad_fn(actor_state.params)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)

        # 3. Soft update target networks
        def soft_update(target_params, online_params, tau):
            return jax.tree_util.tree_map(
                lambda t, o: (1 - tau) * t + tau * o,
                target_params, online_params
            )

        new_target_actor_params = soft_update(target_actor_params, new_actor_state.params, tau)
        new_target_critic_params = soft_update(target_critic_params, new_critic_state.params, tau)

        return new_actor_state, new_critic_state, new_target_actor_params, new_target_critic_params, {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss
        }

    def update(self) -> Optional[Dict[str, float]]:
        """Update policy using DDPG."""
        if len(self.buffer) < self.batch_size * 2:
            return None

        # Sample batch
        batch = random.sample(self.buffer, self.batch_size)
        
        states = jnp.array([e.state for e in batch])
        goals = jnp.array([e.goal for e in batch])
        actions = jnp.array([e.action for e in batch]).reshape(-1, 1)
        rewards = jnp.array([e.reward for e in batch]).reshape(-1, 1)
        next_states = jnp.array([e.next_state for e in batch])
        dones = jnp.array([e.done for e in batch], dtype=jnp.float32).reshape(-1, 1)

        # Update step
        self.actor_state, self.critic_state, self.target_actor_params, self.target_critic_params, metrics = self.update_step(
            self.actor_state,
            self.critic_state,
            self.target_actor_params,
            self.target_critic_params,
            states,
            goals,
            actions,
            rewards,
            next_states,
            dones,
            self.gamma,
            self.tau
        )

        return {
            "actor_loss": float(metrics["actor_loss"]),
            "critic_loss": float(metrics["critic_loss"]),
            "buffer_size": len(self.buffer)
        }

    def reset(self):
        """Clear episode buffer."""
        self.episode_buffer.clear()

    def save(self, path: str):
        """Save model and training state."""
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        save_data = {
            "actor_params": self.actor_state.params,
            "critic_params": self.critic_state.params,
            "target_actor_params": self.target_actor_params,
            "target_critic_params": self.target_critic_params,
        }
        
        with open(path, "wb") as f:
            pickle.dump(save_data, f)

    def load(self, path: str):
        """Load model and training state."""
        if not os.path.exists(path):
            print(f"  [HER-DDPG] Warning: Model file {path} not found.")
            return

        with open(path, "rb") as f:
            load_data = pickle.load(f)
        
        self.actor_state = self.actor_state.replace(params=load_data["actor_params"])
        self.critic_state = self.critic_state.replace(params=load_data["critic_params"])
        self.target_actor_params = load_data["target_actor_params"]
        self.target_critic_params = load_data["target_critic_params"]
