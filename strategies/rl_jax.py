#!/usr/bin/env python3
"""
JAX-based PPO (Proximal Policy Optimization) strategy.
Uses Flax for neural networks and Optax for optimization.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import os
import pickle

from .base import Strategy, MarketState, Action

@dataclass
class Experience:
    """Single experience tuple."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float

class Actor(nn.Module):
    """Policy network: state -> action probabilities."""
    hidden_size: int = 128
    output_dim: int = 3

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = jax.nn.tanh(x)
        x = nn.Dense(self.hidden_size)(x)
        x = jax.nn.tanh(x)
        x = nn.Dense(self.output_dim)(x)
        return jax.nn.softmax(x, axis=-1)

class Critic(nn.Module):
    """Value network: state -> expected return."""
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = jax.nn.tanh(x)
        x = nn.Dense(self.hidden_size)(x)
        x = jax.nn.tanh(x)
        x = nn.Dense(1)(x)
        return x

class RLJaxStrategy(Strategy):
    """PPO-based strategy with actor-critic architecture using JAX/Flax."""

    def __init__(
        self,
        input_dim: int = 18,
        hidden_size: int = 128,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.10,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        buffer_size: int = 512,
        batch_size: int = 64,
        n_epochs: int = 10,
        target_kl: float = 0.02,
        seed: int = 42,
    ):
        super().__init__("rl_jax")
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = 3

        # Hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.target_kl = target_kl

        # JAX PRNG Keys
        self.rng = jax.random.PRNGKey(seed)
        
        # Networks
        self.actor_module = Actor(hidden_size=hidden_size, output_dim=self.output_dim)
        self.critic_module = Critic(hidden_size=hidden_size)

        # Initialize parameters
        init_rng, self.rng = jax.random.split(self.rng)
        dummy_input = jnp.zeros((1, input_dim))
        
        actor_params = self.actor_module.init(init_rng, dummy_input)['params']
        critic_params = self.critic_module.init(init_rng, dummy_input)['params']

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

        # Experience buffer
        self.experiences: List[Experience] = []

        # Running stats for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

        # For storing last action's log prob and value
        self._last_log_prob = 0.0
        self._last_value = 0.0

    def act(self, state: MarketState) -> Action:
        """Select action using current policy."""
        features = state.to_features()
        features_jax = jnp.array(features.reshape(1, -1))

        # Get action probabilities and value
        probs = self.actor_state.apply_fn({'params': self.actor_state.params}, features_jax)
        value = self.critic_state.apply_fn({'params': self.critic_state.params}, features_jax)

        probs_np = np.array(probs[0])
        value_np = float(value[0, 0])

        if self.training:
            # Sample from distribution
            sample_rng, self.rng = jax.random.split(self.rng)
            action_idx = int(jax.random.choice(sample_rng, self.output_dim, p=probs[0]))
        else:
            # Greedy
            action_idx = int(np.argmax(probs_np))

        # Store for experience collection
        self._last_log_prob = float(np.log(probs_np[action_idx] + 1e-8))
        self._last_value = value_np

        return Action(action_idx)

    def store(self, state: MarketState, action: Action, reward: float,
              next_state: MarketState, done: bool):
        """Store experience for training."""
        # Update running reward stats for normalization
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        self.reward_std = np.sqrt(
            ((self.reward_count - 1) * self.reward_std**2 + delta * (reward - self.reward_mean))
            / max(1, self.reward_count)
        )

        # Normalize reward
        norm_reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)

        exp = Experience(
            state=state.to_features(),
            action=action.value,
            reward=norm_reward,
            next_state=next_state.to_features(),
            done=done,
            log_prob=self._last_log_prob,
            value=self._last_value,
        )
        self.experiences.append(exp)

        # Limit buffer size
        if len(self.experiences) > self.buffer_size:
            self.experiences = self.experiences[-self.buffer_size:]

    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                     dones: np.ndarray, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            # TD error
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]

            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    @staticmethod
    @jax.jit
    def train_step(
        actor_state: train_state.TrainState,
        critic_state: train_state.TrainState,
        states: jnp.ndarray,
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
            probs = actor_state.apply_fn({'params': params}, states)
            
            # Get log probs for taken actions
            batch_size = actions.shape[0]
            action_indices = jnp.arange(batch_size)
            selected_probs = probs[action_indices, actions]
            log_probs = jnp.log(selected_probs + 1e-8)

            # PPO clipped objective
            ratio = jnp.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

            # Entropy bonus
            entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
            entropy_mean = jnp.mean(entropy)
            
            total_actor_loss = policy_loss - entropy_coef * entropy_mean
            
            # Metrics
            approx_kl = jnp.mean(old_log_probs - log_probs)
            clip_frac = jnp.mean(
                ((ratio < 1 - clip_epsilon) | (ratio > 1 + clip_epsilon)).astype(jnp.float32)
            )
            
            return total_actor_loss, (policy_loss, entropy_mean, approx_kl, clip_frac)

        def critic_loss_fn(params):
            values = critic_state.apply_fn({'params': params}, states).squeeze()
            
            # Value loss with clipping
            values_clipped = old_values + jnp.clip(
                values - old_values, -clip_epsilon, clip_epsilon
            )
            value_loss1 = (returns - values) ** 2
            value_loss2 = (returns - values_clipped) ** 2
            value_loss = 0.5 * jnp.mean(jnp.maximum(value_loss1, value_loss2))
            
            return value_loss

        # Actor update
        actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
        (actor_loss, (policy_loss, entropy, approx_kl, clip_frac)), actor_grads = actor_grad_fn(actor_state.params)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)

        # Critic update
        critic_grad_fn = jax.value_and_grad(critic_loss_fn)
        critic_loss, critic_grads = critic_grad_fn(critic_state.params)
        new_critic_state = critic_state.apply_gradients(grads=critic_grads)

        metrics = {
            "policy_loss": policy_loss,
            "value_loss": critic_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_fraction": clip_frac,
        }
        
        return new_actor_state, new_critic_state, metrics

    def update(self) -> Optional[Dict[str, float]]:
        """Update policy using PPO."""
        if len(self.experiences) < self.buffer_size:
            return None

        # Convert experiences to arrays
        states = np.array([e.state for e in self.experiences], dtype=np.float32)
        actions = np.array([e.action for e in self.experiences], dtype=np.int32)
        rewards = np.array([e.reward for e in self.experiences], dtype=np.float32)
        dones = np.array([e.done for e in self.experiences], dtype=np.float32)
        old_log_probs = np.array([e.log_prob for e in self.experiences], dtype=np.float32)
        old_values = np.array([e.value for e in self.experiences], dtype=np.float32)

        # Compute next value for GAE
        next_state_jax = jnp.array(self.experiences[-1].next_state.reshape(1, -1))
        next_value = float(self.critic_state.apply_fn({'params': self.critic_state.params}, next_state_jax)[0, 0])

        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards, old_values, dones, next_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to JAX arrays
        states_jax = jnp.array(states)
        actions_jax = jnp.array(actions)
        old_log_probs_jax = jnp.array(old_log_probs)
        advantages_jax = jnp.array(advantages.astype(np.float32))
        returns_jax = jnp.array(returns.astype(np.float32))
        old_values_jax = jnp.array(old_values)

        n_samples = len(self.experiences)
        all_metrics = []

        # Multiple epochs over the data
        for epoch in range(self.n_epochs):
            # Shuffle indices
            indices = np.random.permutation(n_samples)
            
            epoch_kl = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]

                # Get batch
                batch_states = states_jax[batch_idx]
                batch_actions = actions_jax[batch_idx]
                batch_old_log_probs = old_log_probs_jax[batch_idx]
                batch_advantages = advantages_jax[batch_idx]
                batch_returns = returns_jax[batch_idx]
                batch_old_values = old_values_jax[batch_idx]

                # Train step
                self.actor_state, self.critic_state, metrics = self.train_step(
                    self.actor_state,
                    self.critic_state,
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_advantages,
                    batch_returns,
                    batch_old_values,
                    self.clip_epsilon,
                    self.entropy_coef,
                    self.value_coef
                )
                
                all_metrics.append(metrics)
                epoch_kl += float(metrics["approx_kl"])
                n_batches += 1

            # Early stopping on KL divergence
            avg_kl = epoch_kl / max(1, n_batches)
            if avg_kl > self.target_kl:
                # print(f"  [RL-JAX] Early stop epoch {epoch}, KL={avg_kl:.4f}")
                break

        # Clear buffer after update
        self.experiences.clear()

        # Compute explained variance
        y_pred = old_values
        y_true = returns
        var_y = np.var(y_true)
        explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else 0.0

        # Aggregate metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))
        
        avg_metrics["explained_variance"] = float(explained_var)
        
        return avg_metrics

    def reset(self):
        """Clear experience buffer."""
        self.experiences.clear()

    def save(self, path: str):
        """Save model and training state."""
        # Ensure directory exists
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        # Save params using pickle (simplest for Flax)
        save_data = {
            "actor_params": self.actor_state.params,
            "critic_params": self.critic_state.params,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "reward_count": self.reward_count,
        }
        
        with open(path, "wb") as f:
            pickle.dump(save_data, f)

    def load(self, path: str):
        """Load model and training state."""
        if not os.path.exists(path):
            print(f"  [RL-JAX] Warning: Model file {path} not found.")
            return

        with open(path, "rb") as f:
            load_data = pickle.load(f)
        
        self.actor_state = self.actor_state.replace(params=load_data["actor_params"])
        self.critic_state = self.critic_state.replace(params=load_data["critic_params"])
        self.reward_mean = load_data["reward_mean"]
        self.reward_std = load_data["reward_std"]
        self.reward_count = load_data["reward_count"]
