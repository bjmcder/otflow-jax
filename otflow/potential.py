import jax

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from otflow.resnet import ResNet


class PotentialOperator(eqx.Module):
    N: eqx.Module
    A: jnp.ndarray
    c: eqx.Module
    w: eqx.Module

    def __init__(self, in_size, hidden_size, num_hidden=2, rank=10, seed=0):
        """
        Class constructor.
        """

        key = jr.PRNGKey(seed)

        # Split up the random keys to initialize the model's various parameter
        # sets.
        key_n, key_a, key_c, key_w = jr.split(key, 4)

        # Create the ResNet subnet used in the potential model
        seed_n = int(jnp.sum(key_n))
        self.N = ResNet(in_size, hidden_size, num_hidden, seed=seed_n)

        # Create the A matrix operator
        a_init = jnn.initializers.glorot_uniform()
        self.A = a_init(key_a, [jnp.minimum(rank, in_size + 1), in_size + 1])

        # Create the c and w functions, which we can express as linear layer
        # modules.
        self.c = eqx.nn.Linear(in_size + 1, 1, use_bias=True, key=key_c)
        self.w = eqx.nn.Linear(hidden_size, 1, use_bias=False, key=key_w)

        # To initialize the c and w parameters, we have to use PyTree
        # modification functions, since Jax disallows inplace assignment.
        cw_init = jnn.initializers.zeros(self.c.weight, self.c.weight.shape)
        cb_init = jnn.initializers.zeros(self.c.bias, self.c.bias.shape)
        ww_init = jnn.initializers.ones(self.w.weight, self.w.weight.shape)

        self.c = eqx.tree_at(lambda l: l.weight, self.c, cw_init)
        self.c = eqx.tree_at(lambda l: l.bias, self.c, cb_init)
        self.w = eqx.tree_at(lambda l: l.weight, self.w, ww_init)

    def __call__(self, x: jnp.ndarray):
        """
        Convenience function for calling the forward pass.

        Parameters
        ----------
        x : jax.numpy.ndarray
            Input data vector.

        Returns
        -------
        jax.numpy.ndarray
        """

        fwd = jax.jit(self.forward)

        return fwd(x)

    def forward(self, x: jnp.ndarray):
        """
        Evaluate the model in the forward direction. The forward model is not
        invoked typically in an OT-Flow problem, as the gradient and Hessian
        trace are the terms that actually appear in the ODE.

        Parameters
        ----------
        x : jax.numpy.ndarray
            The input tensor, with the time variable concatenated.

        Returns
        -------
        jax.numpy.ndarray
        """

        # Symmetrize A by doing: A'A
        A_symm = self.A.T @ self.A

        # Compute Φ(x) = w'N(x) + 0.5x'(A'A)x + b'x + c
        wn = self.w(self.N(x))
        ax = 0.5 * jnp.sum((x @ A_symm) * x, keepdims=True)
        bc = self.c(x)

        return wn + ax + bc

    def jacobian(self, x):
        """
        Compute the exact Jacobian matrix of the potential operator w.r.t. the
        input vector x. Computation uses native Jax automatic differentiation.

        Parameters
        ----------
        x : jax.numpy.ndarray
            Input data vector.

        Returns
        -------
        jax.numpy.ndarray
        """

        jac = jax.jit(jax.jacobian(self.forward))

        return jac(x)

    def hessian_trace(self, x, grad_only=False):
        """
        Compute the gradient and exact Hessian trace of the potential operator
        using the analytical method described in eqs. 11-15 of the OT-flow
        paper.

        Parameters
        ----------
        x : jax.numpy.ndarray
            Input data vector
        grad_only : bool
            If True, compute the gradient only, skipping the Hessian trace
            calculation.

        Returns
        -------
        g : jax.numpy.ndarray
            The gradient of the potential model w.r.t. the input vector x.
        h : jax.numpy.ndarray
            The exact Hessian trace w.r.t. the input vector x.
        """

        d = self.N.input_dimension
        h = self.N.step_size

        # 1. Compute the forward and backward passes of the ResNet model, which
        # we will need for the Hessian calculation.

        # Symmetrize A by doing: A_symm = A'A
        A_symm = self.A.T @ self.A

        # Evaluate forward and reverse passes of the ResNet
        u, z_0, z = self.N.evaluate(x, self.w.weight)

        # Calculate the gradients from the forward/reverse results
        grad = z_0 + (A_symm @ x.T) + self.c.weight

        if grad_only:
            return grad.squeeze(0), None

        # 2. Compute the Hessian trace
        dtanh = lambda x: 1 - jnp.power(jax.nn.tanh(x), 2)

        tanh_0 = jax.nn.tanh(self.N.evaluate_layer(x, 0))
        dtanh_0 = dtanh(self.N.evaluate_layer(x, 0))

        # Compute the trace of the opening layer
        K_0d = self.N.layers[0].weight[:, 0:d]

        tr_h = jnp.sum((dtanh_0.T * z[0]) * jnp.power(K_0d, 2).T)
        jac = K_0d.T * tanh_0

        # Compute the trace of the remaining layers
        for i in range(1, self.N.num_hidden_layers):
            K_id = self.N.layers[i].weight
            KJ_i = K_id @ jac.T

            if i == self.N.num_hidden_layers - 1:
                term = self.w.weight.T
            else:
                term = z[i + 1]

            lay_i = self.N.evaluate_layer(u[i - 1], i)
            dtanh_i = dtanh(lay_i)

            t_ia = dtanh_i * term.T
            t_ib = jnp.power(KJ_i, 2)
            t_i = jnp.sum(t_ia * t_ib.T)

            tr_h += h * t_i

        tr_h += jnp.trace(A_symm[0:d, 0:d])

        return grad.squeeze(0), tr_h
