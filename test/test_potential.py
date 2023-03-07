import jax
import pytest

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from otflow.potential import PotentialOperator

@pytest.fixture(scope='function')
def phi_net_2x5x2():
    in_size = 2
    hidden_size = 5
    num_hidden = 2
    rank = 10
    seed = 0

    key = jr.PRNGKey(seed)

    phi = PotentialOperator(in_size, hidden_size, num_hidden, rank, seed)

    w0_init = jnn.initializers.constant(0.1)
    b0_init = jnn.initializers.constant(0.2)

    phi.N.initialize_layer_params(key, 0, w0_init, b0_init)
    print(phi.N.layers[0].weight)

    w_init = jnn.initializers.constant(0.3)
    b_init = jnn.initializers.constant(0.3)
    phi.N.initialize_layer_params(key, 1, w_init, b_init)

    return phi


def phi_net_8x5x5(scope='function'):

    in_size = 8
    hidden_size = 5
    num_hidden = 5
    rank = 10
    seed = 0

    key = jr.PRNGKey(seed)

    phi = PotentialOperator(in_size, hidden_size, num_hidden, rank, seed)

    w0_init = jnn.initializers.constant(0.1)
    b0_init = jnn.initializers.constant(0.2)

    phi.N.initialize_layer_params(key, 0, w0_init, b0_init)

    for i in range(1, phi.N.num_hidden_layers-1):
        w_init = jnn.initializers.constant(0.3)
        b_init = jnn.initializers.constant(0.3)
        phi.N.initialize_layer_params(key, i, w_init, b_init)

    return phi


def test_potential_forward(phi_net_2x5x2):
    """
    Test the correctness of the forward evaluation of Phi with the first hidden
    layer larger than the input.
    """

    phi = phi_net_2x5x2

    s = jnp.array([[1.0, 4.0, 0.5],
                   [2.0, 5.0, 0.6],
                   [3.0, 6.0, 0.7],
                   [0.0, 0.0, 0.0]])

    y = jax.vmap(phi)(s)
    y_ref = jnp.array([[16.2522],
                       [21.6208],
                       [29.0313],
                       [10.7258]])
    assert jnp.allclose(y, y_ref, rtol=1e-4)


def test_potential_hesstrace(phi_net_2x5x2):
    """
    Test the correctness of the manual gradient and Hessian trace computation
    with the first hidden layer larger than the input.
    """

    phi = phi_net_2x5x2

    s = jnp.array([[1.0, 4.0, 0.5],
                   [2.0, 5.0, 0.6],
                   [3.0, 6.0, 0.7],
                   [0.0, 0.0, 0.0]])

    g, h = jax.vmap(phi.hessian_trace)(s)

    g_ref = jnp.array([[2.5680485, 1.7230235, 0.3075932],
                       [ 4.155053, 2.2732747, -0.26098657],
                       [ 5.7059603, 2.7874281, -0.86566424],
                       [ 0.22874565, 0.22874565, 0.22874565]])

    h_ref = jnp.array([1.6357629, 1.599934, 1.5677052, 1.7060733])

    assert jnp.allclose(g, g_ref, rtol=1e-6)
    assert jnp.allclose(h, h_ref, rtol=1e-6)
