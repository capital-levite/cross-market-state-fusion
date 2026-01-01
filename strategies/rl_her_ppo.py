#!/usr/bin/env python3
"""
JAX-based PPO with Hindsight Experience Replay (HER) strategy.
Combines the activity of PPO with the trend-learning of HER.
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
class HERPPOExperience:
    """Experience tuple for HER-PPO."""
    state: np.ndarray
    goal: np.ndarray
    action: int
    reward: float
    log_prob: float
    value: float
    done: bool
    achieved_goal: np.ndarray

class HERActor(nn.Module):
    """Actor network: (state, goal) -> action distribution."""
    hidden_size: int = 128
    action_dim: int = 3

    @nn.compact
    def __call__(self, x, goal):
        inp = jnp.concatenate([x, goal], axis=-1)
        x = nn.Dense(self.hidden_size)(inp)
        x = jax.nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = jax.nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return jax.nn.softmax(x)

class HERCritic(nn.Module):
    """Critic network: (state, goal) -> value."""
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x, goal):
        inp = jnp.concatenate([x, goal], axis=-1)
        x = nn.Dense(self.hidden_size)(inp)
        x = jax.nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = jax.nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class RLHERPPOStrategy(Strategy):
    """PPO strategy with HER for sparse trading rewards."""

    def __init__(
        self,
        input_dim: int = 18,
        goal_dim: int = 1,
        hidden_size: int = 128,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.05,  # Slightly lower than base PPO to encourage trends
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        buffer_size: int = 512,
        batch_size: int = 64,
        n_epochs: int = 10,
        her_ratio: float = 0.5,
        seed: int = 42,
    ):
        super().__init__("rl_her_ppo")
        self.input_dim = input_dim
        self.goal_dim = goal_dim
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.her_ratio = her_ratio

        # JAX PRNG Keys
        self.rng = jax.random.PRNGKey(seed)
        
        # Networks
        self.actor_module = HERActor(hidden_size=hidden_size)
        self.critic_module = HERCritic(hidden_size=hidden_size)

        # Initialize parameters
        init_rng, self.rng = jax.random.split(self.rng)
        dummy_state = jnp.zeros((1, input_dim))
        dummy_goal = jnp.zeros((1, goal_dim))
        
        actor_params = self.actor_module.init(init_rng, dummy_state, dummy_goal)['params']
        critic_params = self.critic_module.init(init_rng, dummy_state, dummy_goal)['params']

        # Optimizers
        actor_tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=lr_actor)
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=lr_critic)
        )

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

        # Buffers
        self.buffer: List[HERPPOExperience] = []
        self.episode_buffer: List[HERPPOExperience] = []

    def _get_current_goal(self, state: MarketState) -> np.ndarray:
        return np.array([0.005], dtype=np.float32)

    def _get_achieved_goal(self, state: MarketState, next_state: MarketState) -> np.ndarray:
        return np.array([next_state.prob - state.prob], dtype=np.float32)

    def act(self, state: MarketState) -> Action:
        features = state.to_features()
        goal = self._get_current_goal(state)
        
        features_jax = jnp.array(features.reshape(1, -1))
        goal_jax = jnp.array(goal.reshape(1, -1))

        probs = self.actor_state.apply_fn({'params': self.actor_state.params}, features_jax, goal_jax)
        
        if self.training:
            act_rng, self.rng = jax.random.split(self.rng)
            action_idx = jax.random.categorical(act_rng, jnp.log(probs[0]))
            action_idx = int(action_idx)
        else:
            action_idx = int(jnp.argmax(probs[0]))

        return [Action.HOLD, Action.BUY, Action.SELL][action_idx]

    def store(self, state: MarketState, action: Action, reward: float,
              next_state: MarketState, done: bool):
        features = state.to_features()
        goal = self._get_current_goal(state)
        achieved_goal = self._get_achieved_goal(state, next_state)
        
        # Get log_prob and value for PPO
        features_jax = jnp.array(features.reshape(1, -1))
        goal_jax = jnp.array(goal.reshape(1, -1))
        
        probs = self.actor_state.apply_fn({'params': self.actor_state.params}, features_jax, goal_jax)
        value = self.critic_state.apply_fn({'params': self.critic_state.params}, features_jax, goal_jax)
        
        action_idx = [Action.HOLD, Action.BUY, Action.SELL].index(action)
        log_prob = float(jnp.log(probs[0, action_idx] + 1e-8))
        
        exp = HERPPOExperience(
            state=features,
            goal=goal,
            action=action_idx,
            reward=reward,
            log_prob=log_prob,
            value=float(value[0, 0]),
            done=done,
            achieved_goal=achieved_goal
        )
        self.episode_buffer.append(exp)

        if done:
            self._process_episode()

    def _process_episode(self):
        for i, exp in enumerate(self.episode_buffer):
            # 1. Original
            self.buffer.append(exp)

            # 2. HER Relabel
            if random.random() < self.her_ratio:
                future_idx = random.randint(i, len(self.episode_buffer) - 1)
                future_goal = self.episode_buffer[future_idx].achieved_goal
                
                # PnL-based reward for HER
                # Action: 0=HOLD, 1=BUY, 2=SELL
                action_val = 0.0
                if exp.action == 1: action_val = 1.0
                elif exp.action == 2: action_val = -1.0
                
                new_reward = action_val * future_goal[0] * 10.0
                
                her_exp = HERPPOExperience(
                    state=exp.state,
                    goal=future_goal,
                    action=exp.action,
                    reward=new_reward,
                    log_prob=exp.log_prob, # Keep old log_prob (off-policy PPO)
                    value=exp.value,
                    done=exp.done,
                    achieved_goal=exp.achieved_goal
                )
                self.buffer.append(her_exp)

        self.episode_buffer.clear()
        if len(self.buffer) > self.buffer_size * 2:
            self.buffer = self.buffer[-self.buffer_size * 2:]

    @staticmethod
    @jax.jit
    def train_step(
        actor_state: train_state.TrainState,
        critic_state: train_state.TrainState,
        states: jnp.ndarray,
        goals: jnp.ndarray,
        actions: jnp.ndarray,
        old_log_probs: jnp.ndarray,
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
        old_values: jnp.ndarray,
        clip_epsilon: float,
        entropy_coef: float,
        value_coef: float
    ):
        def actor_loss_fn(params):
            probs = actor_state.apply_fn({'params': params}, states, goals)
            action_probs = jnp.take_along_axis(probs, actions, axis=1).squeeze()
            log_probs = jnp.log(action_probs + 1e-8)
            
            ratio = jnp.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
            
            entropy = -jnp.mean(jnp.sum(probs * jnp.log(probs + 1e-8), axis=1))
            total_loss = policy_loss - entropy_coef * entropy
            return total_loss, (policy_loss, entropy)

        def critic_loss_fn(params):
            values = critic_state.apply_fn({'params': params}, states, goals).squeeze()
            # Clipped value loss
            v_clipped = old_values + jnp.clip(values - old_values, -clip_epsilon, clip_epsilon)
            loss1 = (values - returns) ** 2
            loss2 = (v_clipped - returns) ** 2
            value_loss = 0.5 * jnp.mean(jnp.maximum(loss1, loss2))
            return value_loss

        actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
        (actor_loss, (policy_loss, entropy)), actor_grads = actor_grad_fn(actor_state.params)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)

        critic_grad_fn = jax.value_and_grad(critic_loss_fn)
        critic_loss, critic_grads = critic_grad_fn(critic_state.params)
        new_critic_state = critic_state.apply_gradients(grads=critic_grads)

        return new_actor_state, new_critic_state, {
            "policy_loss": policy_loss,
            "value_loss": critic_loss,
            "entropy": entropy
        }

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.buffer) < self.buffer_size:
            return None

        # Prepare data
        states = jnp.array([e.state for e in self.buffer])
        goals = jnp.array([e.goal for e in self.buffer])
        actions = jnp.array([e.action for e in self.buffer]).reshape(-1, 1)
        old_log_probs = jnp.array([e.log_prob for e in self.buffer])
        rewards = jnp.array([e.reward for e in self.buffer])
        old_values = jnp.array([e.value for e in self.buffer])
        dones = jnp.array([e.done for e in self.buffer], dtype=jnp.float32)

        # Compute GAE
        # Since HER makes it slightly off-policy, we'll just use simple returns/advantages for now
        # or treat the buffer as a single trajectory (approximate)
        returns = []
        gae = 0
        last_value = 0 # Simplified
        for i in reversed(range(len(self.buffer))):
            delta = rewards[i] + self.gamma * last_value * (1 - dones[i]) - old_values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            returns.insert(0, gae + old_values[i])
            last_value = old_values[i]
        
        returns = jnp.array(returns)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update epochs
        metrics_list = []
        for _ in range(self.n_epochs):
            # Shuffle and batch
            idx = np.random.permutation(len(self.buffer))
            for i in range(0, len(self.buffer), self.batch_size):
                batch_idx = idx[i:i+self.batch_size]
                if len(batch_idx) < self.batch_size: continue
                
                self.actor_state, self.critic_state, metrics = self.train_step(
                    self.actor_state, self.critic_state,
                    states[batch_idx], goals[batch_idx], actions[batch_idx],
                    old_log_probs[batch_idx], advantages[batch_idx],
                    returns[batch_idx], old_values[batch_idx],
                    self.clip_epsilon, self.entropy_coef, self.value_coef
                )
                metrics_list.append(metrics)

        # Clear buffer after update (PPO style, though HER makes it hybrid)
        self.buffer.clear()

        if not metrics_list: return None
        
        return {
            "policy_loss": float(np.mean([m["policy_loss"] for m in metrics_list])),
            "value_loss": float(np.mean([m["value_loss"] for m in metrics_list])),
            "entropy": float(np.mean([m["entropy"] for m in metrics_list])),
            "buffer_size": 0
        }

    def save(self, path: str):
        dir_name = os.path.dirname(path)
        if dir_name: os.makedirs(dir_name, exist_ok=True)
        save_data = {
            "actor_params": self.actor_state.params,
            "critic_params": self.critic_state.params,
        }
        with open(path, "wb") as f:
            pickle.dump(save_data, f)

    def load(self, path: str):
        if not os.path.exists(path): return
        with open(path, "rb") as f:
            load_data = pickle.load(f)
        self.actor_state = self.actor_state.replace(params=load_data["actor_params"])
        self.critic_state = self.critic_state.replace(params=load_data["critic_params"])
