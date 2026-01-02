#!/usr/bin/env python3
"""
JAX-based Distributional Temporal PPO (Proximal Policy Optimization) strategy.
Uses Flax for neural networks and Optax for optimization.
Implementation features:
1. Temporal Architecture: Captures momentum/trends from state history.
2. Distributional RL (C51): Models the full distribution of returns.
3. Asymmetric Actor-Critic: Larger critic for better value estimation.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
import os
import pickle

from .base import Strategy, MarketState, Action

@dataclass
class Experience:
    """Single experience tuple with temporal context."""
    state: np.ndarray
    temporal_state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_temporal_state: np.ndarray
    done: bool
    log_prob: float
    value: float  # Expected value from distribution

class TemporalEncoder(nn.Module):
    """Encodes temporal sequence of states into momentum/trend features."""
    input_dim: int = 18
    history_len: int = 5
    output_dim: int = 32

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, history_len * input_dim)
        x = nn.Dense(64)(x)
        x = nn.LayerNorm()(x)
        x = jax.nn.tanh(x)
        x = nn.Dense(self.output_dim)(x)
        x = nn.LayerNorm()(x)
        x = jax.nn.tanh(x)
        return x

class Actor(nn.Module):
    """Policy network with temporal awareness."""
    input_dim: int = 20
    hidden_size: int = 64
    output_dim: int = 3
    history_len: int = 5
    temporal_dim: int = 32

    def setup(self):
        self.encoder = TemporalEncoder(
            input_dim=self.input_dim, 
            history_len=self.history_len, 
            output_dim=self.temporal_dim
        )
        self.fc1 = nn.Dense(self.hidden_size)
        self.ln1 = nn.LayerNorm()
        self.fc2 = nn.Dense(self.hidden_size)
        self.ln2 = nn.LayerNorm()
        self.fc3 = nn.Dense(self.output_dim)

    def __call__(self, current_state, temporal_state):
        # current_state: (batch, 18), temporal_state: (batch, 5*18)
        temp_features = self.encoder(temporal_state)
        combined = jnp.concatenate([current_state, temp_features], axis=-1)
        
        x = self.fc1(combined)
        x = self.ln1(x)
        x = jax.nn.tanh(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = jax.nn.tanh(x)
        
        logits = self.fc3(x)
        return jax.nn.softmax(logits, axis=-1)

class DistributionalCritic(nn.Module):
    """Distributional value network: state -> categorical distribution over returns."""
    input_dim: int = 20
    hidden_size: int = 96
    history_len: int = 5
    temporal_dim: int = 32
    n_atoms: int = 51

    def setup(self):
        self.encoder = TemporalEncoder(
            input_dim=self.input_dim, 
            history_len=self.history_len, 
            output_dim=self.temporal_dim
        )
        self.fc1 = nn.Dense(self.hidden_size)
        self.ln1 = nn.LayerNorm()
        self.fc2 = nn.Dense(self.hidden_size)
        self.ln2 = nn.LayerNorm()
        self.fc3 = nn.Dense(self.n_atoms)

    def __call__(self, current_state, temporal_state):
        temp_features = self.encoder(temporal_state)
        combined = jnp.concatenate([current_state, temp_features], axis=-1)
        
        x = self.fc1(combined)
        x = self.ln1(x)
        x = jax.nn.tanh(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = jax.nn.tanh(x)
        
        logits = self.fc3(x)
        # Returns probabilities over atoms
        return jax.nn.softmax(logits, axis=-1)

class RLJaxStrategy(Strategy):
    """Categorical Temporal PPO strategy using JAX/Flax."""

    def __init__(
        self,
        input_dim: int = 20,
        hidden_size: int = 64,
        critic_hidden_size: int = 96,
        history_len: int = 5,
        temporal_dim: int = 32,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.03,
        v_min: float = -10.0,
        v_max: float = 10.0,
        n_atoms: int = 51,
        max_grad_norm: float = 0.5,
        buffer_size: int = 64,
        batch_size: int = 64,
        n_epochs: int = 10,
        target_kl: float = 0.02,
        seed: int = 42,
    ):
        super().__init__("rl_jax")
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.history_len = history_len
        self.temporal_dim = temporal_dim
        self.output_dim = 3

        # Hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.target_kl = target_kl

        # Distributional params
        self.v_min = v_min
        self.v_max = v_max
        self.n_atoms = n_atoms
        self.support = jnp.linspace(v_min, v_max, n_atoms)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # JAX PRNG Keys
        self.rng = jax.random.PRNGKey(seed)
        
        # Networks
        self.actor_module = Actor(
            input_dim=input_dim, 
            hidden_size=hidden_size, 
            output_dim=self.output_dim,
            history_len=history_len,
            temporal_dim=temporal_dim
        )
        self.critic_module = DistributionalCritic(
            input_dim=input_dim,
            hidden_size=critic_hidden_size,
            history_len=history_len,
            temporal_dim=temporal_dim,
            n_atoms=n_atoms
        )

        # Initialize parameters
        init_rng, self.rng = jax.random.split(self.rng)
        dummy_state = jnp.zeros((1, input_dim))
        dummy_temporal = jnp.zeros((1, history_len * input_dim))
        
        actor_params = self.actor_module.init(init_rng, dummy_state, dummy_temporal)['params']
        critic_params = self.critic_module.init(init_rng, dummy_state, dummy_temporal)['params']

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

        # Temporal state history (per-market, keyed by asset)
        self._state_history: Dict[str, deque] = {}

        # Running stats for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

        # For storing last action's log prob and value
        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._last_temporal_state: Optional[np.ndarray] = None

    def _get_temporal_state(self, asset: str, current_features: np.ndarray) -> np.ndarray:
        """Get stacked temporal state for an asset."""
        if asset not in self._state_history:
            self._state_history[asset] = deque(maxlen=self.history_len)

        history = self._state_history[asset]
        history.append(current_features.copy())

        if len(history) < self.history_len:
            padding = [np.zeros(self.input_dim, dtype=np.float32)] * (self.history_len - len(history))
            stacked = np.concatenate(padding + list(history))
        else:
            stacked = np.concatenate(list(history))

        return stacked.astype(np.float32)

    def act(self, state: MarketState) -> Action:
        """Select action using current policy with temporal context."""
        features = state.to_features()
        temporal_state = self._get_temporal_state(state.asset, features)
        
        features_jax = jnp.array(features.reshape(1, -1))
        temporal_jax = jnp.array(temporal_state.reshape(1, -1))

        # Get action probabilities and value distribution
        probs = self.actor_state.apply_fn({'params': self.actor_state.params}, features_jax, temporal_jax)
        dist = self.critic_state.apply_fn({'params': self.critic_state.params}, features_jax, temporal_jax)
        
        # Calculate expected value from distribution
        value = jnp.sum(dist * self.support, axis=-1)

        probs_np = np.array(probs[0])
        value_np = float(value[0])

        if self.training:
            sample_rng, self.rng = jax.random.split(self.rng)
            action_idx = int(jax.random.choice(sample_rng, self.output_dim, p=probs[0]))
        else:
            action_idx = int(np.argmax(probs_np))

        self._last_log_prob = float(np.log(probs_np[action_idx] + 1e-8))
        self._last_value = value_np
        self._last_temporal_state = temporal_state

        return Action(action_idx)

    def store(self, state: MarketState, action: Action, reward: float,
              next_state: MarketState, done: bool):
        """Store experience for training with temporal context."""
        # Update running reward stats
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        self.reward_std = np.sqrt(
            ((self.reward_count - 1) * self.reward_std**2 + delta * (reward - self.reward_mean))
            / max(1, self.reward_count)
        )

        norm_reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        
        next_features = next_state.to_features()
        next_temporal_state = self._get_temporal_state(next_state.asset, next_features)

        exp = Experience(
            state=state.to_features(),
            temporal_state=self._last_temporal_state if self._last_temporal_state is not None else np.zeros(self.history_len * self.input_dim, dtype=np.float32),
            action=action.value,
            reward=norm_reward,
            next_state=next_features,
            next_temporal_state=next_temporal_state,
            done=done,
            log_prob=self._last_log_prob,
            value=self._last_value,
        )
        self.experiences.append(exp)

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
            next_val = next_value if t == n - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
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
        temporal_states: jnp.ndarray,
        actions: jnp.ndarray,
        old_log_probs: jnp.ndarray,
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
        old_values: jnp.ndarray,
        support: jnp.ndarray,
        v_min: float,
        v_max: float,
        delta_z: float,
        clip_epsilon: float,
        entropy_coef: float
    ):
        def actor_loss_fn(params):
            probs = actor_state.apply_fn({'params': params}, states, temporal_states)
            
            batch_size = actions.shape[0]
            action_indices = jnp.arange(batch_size)
            selected_probs = probs[action_indices, actions]
            log_probs = jnp.log(selected_probs + 1e-8)

            ratio = jnp.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

            entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
            entropy_mean = jnp.mean(entropy)
            
            total_actor_loss = policy_loss - entropy_coef * entropy_mean
            
            approx_kl = jnp.mean(old_log_probs - log_probs)
            clip_frac = jnp.mean(((ratio < 1 - clip_epsilon) | (ratio > 1 + clip_epsilon)).astype(jnp.float32))
            
            return total_actor_loss, (policy_loss, entropy_mean, approx_kl, clip_frac)

        def critic_loss_fn(params):
            # C51 implementation: Minimize cross-entropy between projected target and current distribution
            dist = critic_state.apply_fn({'params': params}, states, temporal_states)
            
            # For PPO Distributional RL, we often regress towards the GAE-calculated returns
            # using a projected categorical distribution of the returns.
            # target_z = reward + gamma * support (if we had next_dist)
            # But here we use calculated 'returns' as the distribution target.
            
            # Projected distribution of 'returns' onto support
            target_dist = RLJaxStrategy.project_distribution(returns, support, v_min, v_max, delta_z)
            
            # Cross-entropy loss
            loss = -jnp.sum(target_dist * jnp.log(dist + 1e-8), axis=-1)
            return jnp.mean(loss)

        actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
        (actor_loss, (policy_loss, entropy, approx_kl, clip_frac)), actor_grads = actor_grad_fn(actor_state.params)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)

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

    @staticmethod
    def project_distribution(returns, support, v_min, v_max, delta_z):
        """Projects a set of scalar returns onto the categorical support."""
        batch_size = returns.shape[0]
        n_atoms = support.shape[0]
        
        # Clamp returns to [v_min, v_max]
        returns_clamped = jnp.clip(returns, v_min, v_max)
        
        # Calculate atom indices
        b = (returns_clamped - v_min) / delta_z
        l = jnp.floor(b).astype(jnp.int32)
        u = jnp.ceil(b).astype(jnp.int32)
        
        # Adjust for edge cases where l == u
        l = jnp.where((u > 0) & (l == u), l - 1, l)
        u = jnp.where((l < n_atoms - 1) & (l == u), u + 1, u)
        
        # Calculate weights
        target_dist = jnp.zeros((batch_size, n_atoms))
        
        # We need to distribute 1.0 probability for each return at its location
        # target_dist[idx, l] += (u - b)
        # target_dist[idx, u] += (b - l)
        
        # Use jax.lax.scatter or simply calculate probabilities
        prob_l = (u.astype(jnp.float32) - b)
        prob_u = (b - l.astype(jnp.float32))
        
        # Build distribution
        idx = jnp.arange(batch_size)
        
        def update_dist(dist, i):
            dist = dist.at[i, l[i]].add(prob_l[i])
            dist = dist.at[i, u[i]].add(prob_u[i])
            return dist
            
        # Optimization: use jax.ops.segment_sum or similar if possible, but for small batch/n_atoms, 
        # we can build it manually. Actually, let's use a cleaner vectorized way:
        
        target = jnp.zeros((batch_size, n_atoms))
        target = target.at[jnp.arange(batch_size), l].add(prob_l)
        target = target.at[jnp.arange(batch_size), u].add(prob_u)
        
        return target

    def update(self) -> Optional[Dict[str, float]]:
        """Update policy using PPO with Distributional RL and Temporal Context."""
        if len(self.experiences) < self.buffer_size:
            return None

        states = np.array([e.state for e in self.experiences], dtype=np.float32)
        temporal_states = np.array([e.temporal_state for e in self.experiences], dtype=np.float32)
        actions = np.array([e.action for e in self.experiences], dtype=np.int32)
        rewards = np.array([e.reward for e in self.experiences], dtype=np.float32)
        dones = np.array([e.done for e in self.experiences], dtype=np.float32)
        old_log_probs = np.array([e.log_prob for e in self.experiences], dtype=np.float32)
        old_values = np.array([e.value for e in self.experiences], dtype=np.float32)

        # Compute next value for GAE
        next_features = jnp.array(self.experiences[-1].next_state.reshape(1, -1))
        next_temporal = jnp.array(self.experiences[-1].next_temporal_state.reshape(1, -1))
        next_dist = self.critic_state.apply_fn({'params': self.critic_state.params}, next_features, next_temporal)
        next_value = float(jnp.sum(next_dist * self.support, axis=-1)[0])

        advantages, returns = self._compute_gae(rewards, old_values, dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_jax = jnp.array(states)
        temporal_jax = jnp.array(temporal_states)
        actions_jax = jnp.array(actions)
        old_log_probs_jax = jnp.array(old_log_probs)
        advantages_jax = jnp.array(advantages.astype(np.float32))
        returns_jax = jnp.array(returns.astype(np.float32))
        old_values_jax = jnp.array(old_values)

        n_samples = len(self.experiences)
        all_metrics = []

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            epoch_kl = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]

                self.actor_state, self.critic_state, metrics = self.train_step(
                    self.actor_state,
                    self.critic_state,
                    states_jax[batch_idx],
                    temporal_jax[batch_idx],
                    actions_jax[batch_idx],
                    old_log_probs_jax[batch_idx],
                    advantages_jax[batch_idx],
                    returns_jax[batch_idx],
                    old_values_jax[batch_idx],
                    self.support,
                    self.v_min,
                    self.v_max,
                    self.delta_z,
                    self.clip_epsilon,
                    self.entropy_coef
                )
                
                all_metrics.append(metrics)
                epoch_kl += float(metrics["approx_kl"])
                n_batches += 1

            if epoch_kl / max(1, n_batches) > self.target_kl:
                break

        self.experiences.clear()

        # Aggregate metrics
        avg_metrics = {key: float(np.mean([m[key] for m in all_metrics])) for key in all_metrics[0].keys()}
        
        # Explained variance
        y_true = returns
        y_pred = old_values
        var_y = np.var(y_true)
        avg_metrics["explained_variance"] = float(1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else 0.0)
        
        return avg_metrics

    def reset(self):
        """Clear experience buffer and temporal history."""
        self.experiences.clear()
        self._state_history.clear()
        self._last_temporal_state = None

    def save(self, path: str):
        """Save model and training state."""
        dir_name = os.path.dirname(path)
        if dir_name: os.makedirs(dir_name, exist_ok=True)
        
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
        if not os.path.exists(path): return
        with open(path, "rb") as f:
            load_data = pickle.load(f)
        
        self.actor_state = self.actor_state.replace(params=load_data["actor_params"])
        self.critic_state = self.critic_state.replace(params=load_data["critic_params"])
        self.reward_mean = load_data["reward_mean"]
        self.reward_std = load_data["reward_std"]
        self.reward_count = load_data["reward_count"]
