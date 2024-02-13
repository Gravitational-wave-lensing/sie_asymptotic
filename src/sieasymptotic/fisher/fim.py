# Computes the fisher information matrix for the SIE lens moel with asymptotic expansion using jax
import jax
from jax import numpy as jnp
from jax import grad, hessian
from jax import random
import sieasymptotic.profile as profile
import sieasymptotic.solver as solver
import sieasymptotic.utils as utils
from jax import config
config.update("jax_enable_x64", True)

def chi_squared(log_T_star, log_dL, f, source_r, source_phi, log_dL_effectives_median, log_time_delays_median, log_sigma_dL_effectives=jnp.ones(4)*1, log_sigma_time_delays=jnp.ones(3)*1, omegatilde=0):
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
    log_dL_effectives, log_time_delays = solver.solve_effective_luminosity_distances_and_time_delays(log_T_star, log_dL, f, source_r, source_phi, omegatilde)
    print("params", log_dL_effectives, log_time_delays)
    chi_squared = jnp.sum((log_dL_effectives - log_dL_effectives_median)**2/log_sigma_dL_effectives**2) + jnp.sum((log_time_delays - log_time_delays_median)**2/log_sigma_time_delays**2)
    return chi_squared

def fisher_information_matrix( log_T_star, log_dL, f, source_r, source_phi, log_dL_effectives_median, log_time_delays_median, log_sigma_dL_effectives=jnp.ones(4)*0.1, log_sigma_time_delays=jnp.ones(3)*0.03, omegatilde=0):
    """Computes the fisher information matrix for the SIE lens moel with asymptotic expansion using jax

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
        jnp.array: The fisher information matrix.
    """
    # The fisher information matrix is the hessian of the chi_squared distribution with respect to T_star, dL, f, source_r, source_phi
    chi_squared_hessian =  hessian(lambda x: chi_squared(x[0], x[1], x[2], x[3], x[4], log_dL_effectives_median, log_time_delays_median, log_sigma_dL_effectives, log_sigma_time_delays, omegatilde )/2.)
    grad_chi_sq = grad(lambda x: chi_squared(x[0], x[1], x[2], x[3], x[4], log_dL_effectives_median, log_time_delays_median, log_sigma_dL_effectives, log_sigma_time_delays, omegatilde ))
    x = jnp.array([log_T_star, log_dL, f, source_r, source_phi])
    fisher_information = chi_squared_hessian(x)
    #print("fisher", fisher_information)
    print("grad", grad_chi_sq(x))
    return fisher_information

# If running as main, test the fisher information matrix
if __name__ == "__main__":
    from astropy.cosmology import Planck18
    from astropy import units
    source_x = jnp.array(0.01)  # example source x-coordinates
    source_y = jnp.array(0.03)  # example source y-coordinates
    f = jnp.array(0.5)  # example axis ratio
    omegatilde = jnp.array(0)  # example angle between major axis and x-axis
    source_r, source_phi = utils.transform_cartesian_to_polar(source_x, source_y, omegatilde)
    image_r, image_phi = solver.solve_image_positions_polar(source_r, source_phi, f, omegatilde)
    # Calculate the dimensionless Fermat potential in polar coordinates
    fermat_polar = profile.fermat_potential_dimensionless_polar(image_r, image_phi, source_r, source_phi, f, omegatilde)
    # print("Dimensionless Fermat potential in polar coordinates:", fermat_polar)
    # Calculate the magnification in polar coordinates
    magnification_polar = profile.magnification_sie_polar(image_r, image_phi, f, omegatilde)
    log_sqrt_abs_magnification = jnp.log(jnp.sqrt(jnp.abs(magnification_polar)))
    # print("Magnification in polar coordinates:", magnification_polar)
    # Calculate the time delay factor
    zl = 0.5
    zs = 1.0
    c_km_s = 299792.0
    sigma = 200.0/c_km_s
    T_star = profile.time_delay_factor(zl, zs, sigma)
    arrival_times = T_star*fermat_polar
    # Compute the fisher information matrix
    log_T_star = jnp.log(T_star)
    log_arrival_times = log_T_star + jnp.log(fermat_polar)
    log_time_delays = log_T_star + jnp.log(fermat_polar - fermat_polar[0])[1:]
    log_dL = jnp.log(Planck18.luminosity_distance(zl).to(units.Mpc).value)
    log_dL_effectives = log_dL - log_sqrt_abs_magnification
    # Compute the chi_squared value
    chi_sq = chi_squared(log_T_star, log_dL, f, source_r, source_phi, log_dL_effectives, log_time_delays)
    print("chisq", chi_sq)
    print(grad(jax.jit(lambda x: chi_squared(x, log_dL, f, source_r, source_phi, log_dL_effectives, log_time_delays)))(log_T_star))
    # # Set the median value to be the true value for now
    # log_dL_effectives_median = log_dL_effectives
    # log_time_delays_median = log_time_delays
    # fisher = fisher_information_matrix( log_T_star, log_dL, f, source_r, source_phi, log_dL_effectives_median, log_time_delays_median)
    # # Compute the inverse of the fisher information matrix
    # fisher_inv = jnp.linalg.inv(fisher)
    # # Print out the diagonal values
    # print("Fisher information matrix:", fisher)
    # print("Inverse Fisher information matrix:", fisher_inv)
    # print("Diagonal values:", jnp.diag(fisher_inv))
    