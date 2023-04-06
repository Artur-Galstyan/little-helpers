import logging

import jax.numpy as jnp
from jaxtyping import Array

_logger = logging.getLogger(__name__)

__author__ = "Artur A. Galstyan"
__copyright__ = "Artur A. Galstyan"
__license__ = "MIT"



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
