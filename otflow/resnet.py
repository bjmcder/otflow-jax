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

        self.step_size = 1.0/(self.num_hidden_layers-1)

        self.layers = []

        # Initialize the PRNG state with a user-provided seed
        key = jr.PRNGKey(seed)

        # Create the first layer, which maps the input size to the hidden layer
        # size
        opening_layer = eqx.nn.Linear(self.input_dimension+1,
                                      self.hidden_dimension,
                                      use_bias=True,
                                      key=key)

        # For the activation function, we use the antiderivative of the tanh
        # function. This helps with analytic gradients downstream in the
        # potential operator.
        integral_tanh = \
            lambda x: (jnp.abs(x) + jnp.log(1+jnp.exp(-2*jnp.abs(x))))

        self.activation = jax.jit(integral_tanh)

        # Collect the opening layer
        self.layers.append(opening_layer)

        # Create the hidden layers
        keys = jr.split(key, self.num_hidden_layers-1)
        for i in range(self.num_hidden_layers-1):
            newlayer = eqx.nn.Linear(self.hidden_dimension,
                                     self.hidden_dimension,
                                     use_bias=True,
                                     key=keys[i])

            self.layers.append(newlayer)

    def __call__(self, x):
        """
        Convenience function for invoking the forward pass of the model.

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

        Let x be the input tensor, consisting of data values concatenated with the
        time t.

        x = concat(data,t)

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

    def evaluate_layer(self, x, layer_idx):
        """
        Invoke the affine operation of a single layer.
        """
        return self.layers[layer_idx](x)

    def initialize_layer_params(self,
                                key: jax.random.PRNGKey,
                                layer_idx: int,
                                w_init: jax.nn.initializers.Initializer,
                                b_init: jax.nn.initializers.Initializer):
        """
        Initialize the weights and biases of a layer using custom initializers.
        This overrides the default Equinox random normal initializer.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key needed by initializers
        layer_idx : int
            Index of the layer being initialized.
        w_init : jax.nn.initializers.Initializer
            Initializer for the layer weights.
        b_init : jax.nn.initializers.Initializer
            Initializer for the layer biases.
        """
        key1, key2 = jr.split(key, 2)

        self.initialize_layer_weights(key1, layer_idx, w_init)
        self.initialize_layer_biases(key2, layer_idx, b_init)

    def initialize_layer_weights(self,
                                 key: jax.random.PRNGKey,
                                 layer_idx: int,
                                 w_init: jax.nn.initializers.Initializer):
        """
        Initialize the weights of a layer using custom initializers. This
        overrides the default Equinox random normal initializer.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key needed by initializers
        layer_idx : int
            Index of the layer being initialized.
        w_init : jax.nn.initializers.Initializer
            Initializer for the layer weights.
        """
        param_select = lambda layer: layer.weight
        init_func = w_init(key, self.layers[layer_idx].weight.shape)

        self.layers[layer_idx] = eqx.tree_at(param_select,
                                             self.layers[layer_idx],
                                             init_func)

    def initialize_layer_biases(self,
                                key: jax.random.PRNGKey,
                                layer_idx: int,
                                b_init: jax.nn.initializers.Initializer):
        """
        Initialize the biases of a layer using custom initializers. This
        overrides the default Equinox random normal initializer.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key needed by initializers
        layer_idx : int
            Index of the layer being initialized.
        b_init : jax.nn.initializers.Initializer
            Initializer for the layer biases.
        """
        param_select = lambda layer: layer.bias
        init_func = b_init(key, self.layers[layer_idx].bias.shape)

        self.layers[layer_idx] = eqx.tree_at(param_select,
                                             self.layers[layer_idx],
                                             init_func)

    def evaluate_forward(self, x):
        """
        Invoke the forward pass of the model, storing all intermediate
        activation values.

        Parameters
        ----------
        x : jax.numpy.ndarray
            Forward input data vector.
        """

        u = jnp.zeros([self.num_hidden_layers,self.hidden_dimension])

        u = u.at[0,:].set(self.activation(self.layers[0](x)))

        for i in range(1, self.num_hidden_layers):
            lay = self.layers[i](u[i-1])
            u = u.at[i,:].set(u[i-1] + self.step_size*self.activation(lay))

        return u

    def evaluate_reverse(self, x, w, u):
        """
        Invoke the reverse pass through the model, storing all intermediate layer values.

        Parameters
        ----------
        w : jax.numpy.ndarray
            Vector of values being backpropagated.

        u : jax.numpy.ndarray
            Results of the forward pass

        Returns
        -------
        jax.numpy.ndarray
        """
        h = self.step_size
        n = self.num_hidden_layers

        z = jnp.zeros([self.num_hidden_layers-1, self.hidden_dimension])

        # Evaluate the layers in revese order, storing the intermediate
        # activations in z.
        for i in range(n-1, 0, -1):
            if i == n-1:
                term = w.T
            else:
                term = z[i+1]

            K_i = self.layers[i].weight
            dlayer = jax.nn.tanh(self.layers[i](u[i-1]))

            hku = jnp.expand_dims(h*(K_i.T @ dlayer).T, axis=1)
            term = term.reshape(term.shape[0],1)

            new_z = (term + jnp.multiply(hku, term)).squeeze(1)
            z = z.at[i-1].set(new_z)

        # Finally, handle the input layer separately, which is not necessarily
        # the same size as the hidden layers
        K_0 = self.layers[0].weight
        tanh_0 = jax.nn.tanh(self.evaluate_layer(x, 0))

        z_0 = K_0.T @ (tanh_0.T * z[1])

        return z_0, z

    def evaluate(self, x, w):
        """
        Evaluate the forward and reverse passes, storing the intermediate
        activations of each.

        Parameters
        ----------
        x : jax.numpy.ndarray
            Forward input data vector.
        w : jax.numpy.ndarray
            Reverse input data vector.
        """

        fwd = jax.jit(self.evaluate_forward)
        rev = self.evaluate_reverse

        u = fwd(x)
        z_0, z = rev(x, w, u)

        return u, z_0, z
