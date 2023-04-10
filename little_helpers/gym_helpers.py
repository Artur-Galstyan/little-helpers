from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import torch
from gymnasium import Env
from jaxtyping import Array, Float
from torch.utils.data import Dataset


class RLDataset(Dataset):
    def __init__(self, rewards, actions, obs):
        self.rewards = rewards
        self.actions = actions
        self.obs = obs

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, index):
        reward = torch.tensor(self.rewards[index])
        action = torch.tensor(self.actions[index])
        ob = torch.tensor(self.obs[index])
        return ob, action, reward


def get_trajectory(
    env: Env,
    key: jax.random.PRNGKeyArray,
    act_fn: Callable[..., int],
    act_fn_kwargs: dict,
    obs_preprocessing_fn: Optional[Callable] = None,
    obs_preprocessing_fn_kwargs: Optional[dict] = None,
    render=False,
) -> Tuple[Float[Array, "n_env_steps 1"], jnp.ndarray, jnp.ndarray]:
    if obs_preprocessing_fn is None:

        def obs_preprocessing_fn(x):
            return x

    if obs_preprocessing_fn_kwargs is None:
        obs_preprocessing_fn_kwargs = {}

    obs, _ = env.reset()
    rewards = []
    eps_obs = []
    eps_actions = []

    while True:
        key, subkey = jax.random.split(key)

        obs = obs_preprocessing_fn(obs, **obs_preprocessing_fn_kwargs)
        eps_obs.append(obs)
        action = act_fn(obs, **act_fn_kwargs, key=subkey)
        obs, reward, terminated, truncated, _ = env.step(int(action))

        if render:
            env.render()

        rewards.append(reward)
        eps_actions.append(action)
        if terminated or truncated:
            break
    eps_obs = jnp.stack(eps_obs)
    eps_actions = jnp.array(eps_actions)

    return rewards, eps_obs, eps_actions
