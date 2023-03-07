import jax
import jax.numpy as jnp

from otflow.resnet import ResNet

def test_resnet():
    """
    Test the ResNet model and it's forward evaluation
    """

    s = jnp.array([0.1, 0.2, 0.3])
    in_size = s.shape[0]-1
    hidden_size = 2
    num_hidden = 2
    seed = 0

    # Setup the ResNet and perform a forward evaluation
    net = ResNet(in_size, hidden_size, num_hidden, seed)

    y_test = net(s)

    # Now, we will do a verification against the mathematical definition of a
    # ResNet.
    activation = \
            lambda x: (jnp.abs(x) + jnp.log(1+jnp.exp(-2*jnp.abs(x))))

    step = 1.0/(hidden_size-1)

    k0 = net.layers[0].weight
    b0 = net.layers[0].bias
    k1 = net.layers[1].weight
    b1 = net.layers[1].bias

    u0 = activation(k0@s + b0)
    u1 = u0 + step*activation(k1@u0 + b1)

    assert jnp.allclose(y_test, u1, rtol=1.0e-6)

