import jax.numpy as jnp

from otflow.potential import PotentialOperator

def test_potential_forward():

    in_dim = 2
    hidden_dim = 2
    num_hidden = 2
    rank = 10
    seed = 0

    phi = PotentialOperator(in_dim, hidden_dim, num_hidden, rank, seed)

    x = jnp.array([0.1, 0.2, 0.3])

    y = phi(x)
