import jax

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn

from otflow.resnet import ResNet

class PotentialOperator(eqx.Module):
    N: eqx.Module
    A: jnp.ndarray
    c: eqx.Module
    w: eqx.Module

    def __init__(self,
                 in_size,
                 hidden_size,
                 num_hidden=2,
                 rank=10,
                 seed=0,
                 test_mode=False):
        """
        Class constructor
        """

        key = jr.PRNGKey(seed)

        # Split up the random keys to initialize the model's various parameter
        # sets.
        key_n, key_a, key_c, key_w = jr.split(key, 4)

        # Create the ResNet subnet used in the potential model
        seed_n = int(jnp.sum(key_n))
        self.N = ResNet(in_size,
                        hidden_size,
                        num_hidden,
                        seed=seed_n,
                        test_mode=test_mode)

        # Create the A matrix operator
        a_init = jnn.initializers.glorot_uniform()
        self.A = a_init(key_a, [jnp.minimum(rank, in_size+1), in_size+1])

        # Create the c and w functions, which we can express as linear layer
        # modules.
        self.c = eqx.nn.Linear(in_size+1, 1, use_bias=True, key=key_c)
        self.w = eqx.nn.Linear(hidden_size, 1, use_bias=False, key=key_w)

        # To initialize the c and w parameters, we have to use PyTree
        # modification functions, since Jax disallows inplace assignment.
        cw_init = jnn.initializers.zeros(self.c.weight, self.c.weight.shape)
        cb_init = jnn.initializers.zeros(self.c.bias, self.c.bias.shape)
        ww_init = jnn.initializers.ones(self.w.weight, self.w.weight.shape)

        self.c = eqx.tree_at(lambda l: l.weight, self.c, cw_init)
        self.c = eqx.tree_at(lambda l: l.bias, self.c, cb_init)
        self.w = eqx.tree_at(lambda l: l.weight, self.w, ww_init)

    def __call__(self, x:jnp.ndarray):
        """
        Convenience function for calling the forward pass.
        """

        fwd = jax.jit(self.forward)

        return fwd(x)

    def forward(self, x:jnp.ndarray):
        """
        Evaluate the model in the forward direction. The forward model is not
        invoked typically in an OT-Flow problem, as the gradient and Hessian
        trace are the terms that actually appear in the ODE.

        Parameters
        ----------
        x : jax.numpy.ndarray
            The input tensor, with the time variable concatenated.
        """

        # Symmetrize A by doing: A'A
        A_symm = self.A.T @ self.A

        # Compute Φ(x) = w'N(x) + 0.5x'(A'A)x + b'x + c
        wn = self.w(self.N(x))
        ax = 0.5 * jnp.sum((x @ A_symm)*x, keepdims=True)
        bc = self.c(x)

        return wn + ax + bc

    def jacobian(self, x):

        jac = jax.jit(jax.jacobian(self.forward))

        return jac(x)

    def hessian_trace(self, x, grad_only=False):
        """
        """

        d = self.N.input_dimension
        m = self.N.hidden_dimension
        h = self.N.step_size

        # 1. Get the Jacobian by manual backpropagation

        # Symmetrize A by doing: A'A
        A_symm = self.A.T @ self.A

        # Evaluate forward and reverse passes of the ResNet
        u, z = self.N.evaluate(x, self.w.weight)

        # Calculate the gradients from the forward/reverse results
        z_0 = z[0,0:d+1]
        print("z")
        print(z)

        grad = (z_0 + (A_symm @ x.T) + self.c.weight.T)
        print(grad)
        if grad_only:
            return grad.T

        # 2. Compute the Hessian trace
        dtanh = lambda x: 1 - jnp.power(jax.nn.tanh(x), 2)

        tanh_0 = jax.nn.tanh(self.N.evaluate_layer(x,0))
        dtanh_0 = dtanh(self.N.evaluate_layer(x,0))

        # Opening layer trace
        K_0d = self.N.layers[0].weight[:, 0:d]

        t_0a =  dtanh_0 @ z[1]
        t_0b = jnp.power(K_0d, 2)

        t_0 = jnp.sum(t_0a * t_0b)
        t_ia = tanh_0

        pad_size = jnp.abs(m-d)
        J = jnp.pad((K_0d.T @ t_ia), (0, pad_size))

        # Remaining layers
        for i in range(1, self.N.num_hidden_layers):
            K_id = self.N.layers[i].weight

            KJ = K_id.T @ J


if __name__ == "__main__":

    import time
    s = jnp.array([[1.0, 4.0, 0.5]])
    #s = jnp.array([[1.0, 4.0, 0.5],
    #               [2.0, 5.0, 0.6],
    #               [3.0, 6.0, 0.7],
    #               [0.0, 0.0, 0.0]])

    in_size = 2
    hidden_size = 5
    num_hidden = 2
    seed = 0

    phi = PotentialOperator(in_size, hidden_size, num_hidden, test_mode=True)

    y = jax.vmap(phi)(s)

    jax.vmap(phi.hessian_trace)(s)
    j = jax.vmap(phi.jacobian)(s)
    print("autodiff jacobian")
    print(j)

    #d = 400
    #m = 32
    #nex = 1000

    #net2 = PotentialOperator(d, m, 5)

    #key = jr.PRNGKey(25)
    #x = jr.normal(key, [nex,d+1])
    #y = jax.vmap(net2)(x)

    #end = time.time()
    #h = jax.vmap(net2.jacobian)(x)
    #print('traceHess takes ', time.time()-end)