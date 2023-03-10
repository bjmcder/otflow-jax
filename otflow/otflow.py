import diffrax
import jax

import equinox as eqx
import jax.numpy as jnp

from otflow.potential import PotentialOperator


class OTOperator(eqx.Module):
    phi: eqx.Module
    alpha: list[float]

    """
    Wrapper class for the OT-Flow neural ODE operator.
    """

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        num_hidden: int = 2,
        rank: int = 10,
        seed: int = 0,
        alpha: list[float] = [1.0, 1.0, 1.0],
    ):
        self.phi = PotentialOperator(in_size, hidden_size, num_hidden, rank, seed)

        self.alpha = alpha

    def apply(self, t: float, x: jax.numpy.ndarray):
        """
        Neural ODE combining the characteristics and log-determinant
        (see Eq. (2)), the transport costs (see Eq. (5)), and the HJB
        regularizer (see Eq. (7)).

        d_t  [x ; l ; v ; r] = odefun( [x ; l ; v ; r] , t )

        x - particle position
        l - log determinant
        v - accumulated transport costs (Lagrangian)
        r - accumulates violation of HJB condition along trajectory

        Parameters
        ----------
        t : float
            The current time.
        x : jax.numpy.ndarray
            The input data vector.

        Returns
        -------
        jax.numpy.ndarray
        """

        d = self.phi.N.input_dimension

        # Evaluate the gradient and Hessian trace
        g, h = self.phi.hessian_trace(s)

        # Evaluate the Neural ODE operator and update the state
        dx = -(1.0 / self.alpha[0]) * g[0:d]
        dl = -(1.0 / self.alpha[0]) * h
        dv = 0.5 * jnp.sum(jnp.power(dx, 2))
        dr = jnp.abs(g[-1] + self.alpha[0] * dv)

        state = jnp.concatenate(
            [dx, jnp.expand_dims(dl, 0), jnp.expand_dims(dv, 0), jnp.expand_dims(dr, 0)]
        )
        return state


class OTFlow(eqx.Module):
    operator: eqx.Module
    """
    An OT-Flow neural ODE model. OT-Flow is a continuous normalizing flow (CNF)
    that uses an optimal transport (OT) regularizer to optimize the
    normalization trajectory of the model.
    """

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        num_hidden: int = 2,
        rank: int = 10,
        seed: int = 0,
        alpha: list[float] = [1.0, 1.0, 1.0],
    ):
        """
        Class constructor.
        """
        self.operator = OTOperator(in_size, hidden_size, num_hidden, rank, seed, alpha)

    def solve(self, y0, tmax, dt):
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)


if __name__ == "__main__":
    in_size = 2
    hidden_size = 5
    num_hidden = 2
    rank = 10
    seed = 0

    model = OTFlow(in_size, hidden_size, num_hidden, rank, seed)

    s = jnp.array([[1.0, 4.0, 0.5], [2.0, 5.0, 0.6], [3.0, 6.0, 0.7], [0.0, 0.0, 0.0]])

    result = jax.vmap(model.apply, in_axes=0)(s)
    print(result)
