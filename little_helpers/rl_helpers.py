import logging

import jax.numpy as jnp
from jaxtyping import Array
import jax

_logger = logging.getLogger(__name__)

__author__ = "Artur A. Galstyan"
__copyright__ = "Artur A. Galstyan"
__license__ = "MIT"

@jax.jit
def get_discounted_rewards(rewards: jnp.ndarray, gamma=0.99) -> float:
    """Calculate the discounted rewards for a given set of rewards.
    Args:
        rewards: The rewards to calculate the discounted rewards for.
        gamma: The discount factor.

    Returns:
        The discounted rewards.
    """
    def body_fn(i: int, val: float):
        return val + (gamma ** i) * rewards[i]
    
    discounted_rewards = jnp.zeros(())
    num_rewards = len(rewards)
    discounted_rewards = jax.lax.fori_loop(0, num_rewards, body_fn, discounted_rewards)
    
    return discounted_rewards

@jax.jit
def get_total_discounted_rewards(rewards: jnp.array, gamma=0.99) -> jnp.array:
    """Calculate the total discounted rewards for a given set of rewards.
    Args:
        rewards: The rewards to calculate the total discounted rewards for.
        gamma: The discount factor.

    Returns:
        The total discounted rewards.
    """
    total_discounted_rewards = jnp.zeros(shape=rewards.shape)
    for i in range(len(rewards)):
        current_slice = rewards[i:]
        discounted_reward = get_discounted_rewards(current_slice, gamma)
        total_discounted_rewards = total_discounted_rewards.at[i].set(discounted_reward)
    return total_discounted_rewards.reshape(-1, 1)


def get_future_rewards(rewards: Array, gamma=0.99) -> Array:
    """Calculate the future rewards for a given set of rewards.
    Args:
        rewards: The rewards to calculate the future rewards for.
        gamma: The discount factor.

    Returns:
        The future rewards.
    """
    returns = jnp.zeros_like(rewards)
    future_returns = 0

    for t in range(len(rewards) - 1, -1, -1):
        future_returns = rewards[t] + gamma * future_returns
        returns = returns.at[t].set(future_returns)

    return returns



if __name__ == "__main__":
    pass
