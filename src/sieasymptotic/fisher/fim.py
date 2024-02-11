# Computes the fisher information matrix for the SIE lens moel with asymptotic expansion using jax
import jax
from jax import numpy as jnp
from jax import grad, hessian
from jax import random
from sieasymptotic.solver import solve_image_positions_polar, solve_image_positions_cartesian
from sieasymptotic.profile import fermat_potential_dimensionless_cartesian, fermat_potential_dimensionless_polar

def chi_squared(T_star, dL, f, source_r, source_phi, omegatilde=0):
    """Calculate the chi-squared value for a given set of parameters for the SIE lens model based on asymptotic expansion.

    Args:
        T_star (jnp.array): The time-delay factor
        dL (jnp.array): The luminosity distance.
        f (jnp.array): The axis ratio
        source_r (jnp.array): The radial coordinate of the source position.
        source_phi (jnp.array): The polar coordinate of the source position.

    Returns:
        jnp.array: The calculated chi-squared value.
    """
    return None