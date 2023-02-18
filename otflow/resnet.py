import jax

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr


class ResNet(eqx.Module):
    """
    ResNet model to use as a trainable subnet within the OT-Flow model.
    """
    input_dimension: int
    hidden_dimension: int
    num_hidden_layers: int
    step_size: float
    layers: list
    activation: callable


    def __init__(self,
                 in_size: int,
                 hidden_size: int,
                 num_hidden: int = 2,
                 seed: int = 0):
        """
        Class Constructor for ResNet.

        Parameters
        ----------
        in_size:  int
            Dimension of the input vector.
        hidden_size : int
            Dimension of each hidden layer.
        num_hidden : int
            Number of hidden (residual) layers. Default is 2.
        seed : int
            Random number seed. Default is 0.
        """

        if num_hidden < 2:
            raise ValueError("Number of ResNet layers must be >= 2.")

        self.input_dimension = in_size
        self.hidden_dimension = hidden_size
        self.num_hidden_layers = num_hidden

        self.step_size = 1.0/(self.hidden_dimension-1)

        self.layers = []

        # Initialize the PRNG state with a user-provided seed
        key = jr.PRNGKey(seed)

        # Create the first layer, mapping the input size to the hidden layer
        # size
        opening_layer = eqx.nn.Linear(self.input_dimension+1,
                                      self.hidden_dimension,
                                      use_bias=True,
                                      key=key)

        # For the activation function, we use the antiderivative of the tanh
        # function. This helps with analytic gradients downstream in the
        # potential operator.
        self.activation = \
            lambda x: (jnp.abs(x) + jnp.log(1+jnp.exp(-2*jnp.abs(x))))

        # Collect the opening layer
        self.layers.append(opening_layer)

        # Create the hidden layers
        keys = jr.split(key, self.num_hidden_layers-1)
        for i in range(self.num_hidden_layers-1):

            self.layers.append(eqx.nn.Linear(self.hidden_dimension,
                                             self.hidden_dimension,
                                             use_bias=True,
                                             key=keys[i]))

    def __call__(self, x):
        """
        Convenience function for invoking the forward model.

        Parameters
        ----------
        x : jax.numpy.ndarray
            Input data.

        Returns
        -------
        jax.numpy.ndarray
        """
        return self.forward(x)

    def forward(self, x):
        """
        Invoke the model in the forward direction.

        A ResNet has the following functional form.

        Let s be the input tensor.

        u0 = activation(K0*s + b0)
        u1 = u0 + h*activation(K1*u0 + b1)
        y = u1

        Parameters
        ----------
        x : jax.numpy.ndarray
            Input data.

        Returns
        -------
        jax.numpy.ndarray
        """

        # Apply the opening layer
        y = self.activation(self.layers[0](x))

        # Apply the remaining layers
        for i in range(1, self.num_hidden_layers):
            y = y.at[:].add(self.step_size*self.activation(self.layers[i](y)))

        return y
