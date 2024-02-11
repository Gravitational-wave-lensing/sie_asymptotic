# Computes the fisher information matrix for the SIE lens moel with asymptotic expansion using jax
import jax
from jax import numpy as jnp
from jax import grad, hessian
from jax import random
from sieasymptotic.solver import solve_image_positions_cartesian
from sieasymptotic.profile import fermat_potential_dimensionless_cartesian
from sieasymptotic.solver.solve_image_positions import solve_effective_luminosity_distances_and_time_delays

def chi_squared(log_T_star, log_dL, f, source_r, source_phi, log_dL_effectives_median, log_time_delays_median, log_sigma_dL_effectives=jnp.ones(4)*0.1, log_sigma_time_delays=jnp.ones(3)*0.03, omegatilde=0):
    """Calculate the chi-squared value for a given set of parameters for the SIE lens model based on asymptotic expansion.

    Args:
        T_star (jnp.array): The time-delay factor
        dL (jnp.array): The luminosity distance.
        f (jnp.array): The axis ratio
        source_r (jnp.array): The radial coordinate of the source position.
        source_phi (jnp.array): The polar coordinate of the source position.
        log_dL_effectives_median (jnp.array): The median logarithm of the effective luminosity distances.
        log_time_delays_median (jnp.array): The median logarithm of the time delays.
        log_sigma_dL_effectives (jnp.array, optional): The logarithm of the effective luminosity distance uncertainties. Defaults to jnp.ones(4)*0.1.
        log_sigma_time_delays (jnp.array, optional): The logarithm of the time delay uncertainties. Defaults to jnp.ones(3)*0.03.
        omegatilde (float, optional): The angle between the source and the lens. Defaults to 0.

    Returns:
        jnp.array: The calculated chi-squared value.
    """
    log_dL_effectives, log_time_delays = solve_effective_luminosity_distances_and_time_delays(log_T_star, log_dL, f, source_r, source_phi, omegatilde)
    chi_squared = 0
    chi_squared += jnp.sum((log_dL_effectives - log_dL_effectives_median)**2/log_sigma_dL_effectives**2)
    chi_squared += jnp.sum((log_time_delays - log_time_delays_median)**2/log_sigma_time_delays**2)
    return None