# Lens equation solver for the SIE lens, based on asymptotic expansion.
from jax import numpy as jnp
import jax
import numpy as np
import sieasymptotic.profile as profile
import sieasymptotic.solver as solver
import sieasymptotic.utils as utils

def solve_image_positions_polar(source_r, source_phi, f, omegatilde=0, sort_time_delay=False):
    ''' Solve the lens equation for the SIE lens, using the asymptotic expansion, in polar coordinates.
    
    Args:
        source_r (jnp.array): The radial coordinate of the source position.
        source_phi (jnp.array): The angular coordinate of the source position.
        f (jnp.array): The axis ratio of the lens.
        omegatilde (jnp.array, optional): The angle between the major axis of the lens and the x-axis.
        sort_time_delay (bool, optional): Whether to sort the image positions by arrival time. Defaults to True.
    
    Returns:
        jnp.array: The radial coordinate of the image positions.
    '''
    f, omegatilde = utils.convert_f_omegatilde(f, omegatilde, source_r.shape) # Ensure that f, omegatilde are jnp.arrays
    # There are 4 solutions. We expand to first order.
    # Asymptotic 0th order first image position:
    # (Eq. 6 of https://academic.oup.com/mnras/article/442/1/428/1244014)
    #k = jnp.arange(4)
    image_phi_0 = jnp.array([
        0. * jnp.pi/2-jnp.transpose(omegatilde),
        1. * jnp.pi/2-jnp.transpose(omegatilde),
        2. * jnp.pi/2-jnp.transpose(omegatilde),
        3. * jnp.pi/2-jnp.transpose(omegatilde)
    ])
    # (Eq. 9 and 10 of https://academic.oup.com/mnras/article/442/1/428/1244014)
    f_prime = jnp.sqrt(1-f**2)
    image_r_0 = jnp.array([
        jnp.sqrt(f)/f_prime * jnp.arcsinh(f_prime/f),
        jnp.sqrt(f)/f_prime * jnp.arcsin(f_prime),
        jnp.sqrt(f)/f_prime * jnp.arcsinh(f_prime/f),
        jnp.sqrt(f)/f_prime * jnp.arcsin(f_prime)
    ])
    
    # Asymptotic 1st order perturbations to the image positions:
    # (Eq. 14 and 15 of https://academic.oup.com/mnras/article/442/1/428/1244014)
    image_delta_phi, image_delta_r = first_order_image_perturbation(source_r, source_phi, f, omegatilde, f_prime)
    
    # The image positions, to first order, are the sum of the first-order solution plus the perturbation
    image_r = image_r_0+image_delta_r #jnp.transpose(image_r_0 + jnp.transpose(image_delta_r))
    image_phi = image_phi_0+image_delta_phi#jnp.transpose(image_phi_0 + jnp.transpose(image_delta_phi))
    image_positions = jnp.array([image_r, image_phi])
    # if sort_time_delay:
    #     # Sort the time delays (fermat potential)
    #     fermat_polar = profile.fermat_potential_dimensionless_polar(image_positions[0], image_positions[1], source_r, source_phi, f, omegatilde)
    #     image_positions, fermat_polar = utils.sort_images_by_arrival_time(image_positions, fermat_polar)
    return image_positions

def first_order_image_perturbation(source_r, source_phi, f, omegatilde, f_prime):
    """
    Calculate the first-order image perturbation for a given set of parameters.

    Args:
        source_r (jnp.array): The radial distance of the source.
        source_phi (jnp.array): The angular position of the source.
        f (jnp.array): The lensing potential.
        omegatilde (jnp.array): The angular position of the perturbation.
        f_prime (jnp.array): The derivative of the lensing potential.

    Returns:
        jnp.array: An array containing the first-order image perturbation.

    """
    image_delta_phi = jnp.array([
       source_r*jnp.sin(source_phi+omegatilde)/(jnp.sqrt(f)*(1./f_prime*jnp.arcsinh(f_prime/f)-1)),
       source_r*jnp.cos(source_phi+omegatilde)/(jnp.sqrt(f)*(1./f-1./f_prime*jnp.arcsin(f_prime)) ),
       -1*source_r*jnp.sin(source_phi+omegatilde)/(jnp.sqrt(f)*(1./f_prime*jnp.arcsinh(f_prime/f)-1)),
       -1*source_r*jnp.cos(source_phi+omegatilde)/(jnp.sqrt(f)*(1./f-1./f_prime*jnp.arcsin(f_prime)) )
    ])
    image_delta_r = jnp.array([
        source_r*jnp.cos(source_phi+omegatilde),
        source_r*jnp.cos(source_phi+omegatilde),
        -1*source_r*jnp.cos(source_phi+omegatilde),
        -1*source_r*jnp.cos(source_phi+omegatilde),
    ])
    return jnp.array([image_delta_phi,image_delta_r])

def solve_image_positions_cartesian(source_x, source_y, f, omegatilde=0):
    ''' Solve the lens equation for the SIE lens, using the asymptotic expansion, in Cartesian coordinates.
    
    Args:
        source_x (jnp.array): The x-coordinate of the source position.
        source_y (jnp.array): The y-coordinate of the source position.
        f (jnp.array): The axis ratio of the lens.
        omegatilde (jnp.array, optional): The angle between the major axis of the lens and the x-axis.
    
    Returns:
        jnp.array: The x and y coordinates of the image positions.
    '''
    # Transform the source position to polar coordinates
    source_polar = utils.transform_cartesian_to_polar(source_x, source_y, omegatilde)
    source_r = source_polar[0]
    source_phi = source_polar[1]
    # Solve the lens equation in polar coordinates
    images_polar = solve_image_positions_polar(source_r, source_phi, f, omegatilde)
    image_r = images_polar[0]
    image_phi = images_polar[1]
    # Transform the image positions back to Cartesian coordinates
    images_cartesian = utils.transform_polar_to_cartesian(image_r, image_phi, omegatilde)
    image_x = images_cartesian[0]
    image_y = images_cartesian[1]
    return jnp.array([image_x, image_y])

#from sieasymptotic.solver import solve_dL_effectives_and_time_delays

def solve_effective_luminosity_distances_and_time_delays(log_T_star, log_dL, f, source_r, source_phi, omegatilde):
    """ Solve the effective luminosity distances and time delays for the SIE lens model based on asymptotic expansion.

    Args:
        T_star (jnp.array): The time-delay factor
        dL (jnp.array): The luminosity distance.
        f (jnp.array): The axis ratio
        source_r (jnp.array): The radial coordinate of the source position.
        source_phi (jnp.array): The polar coordinate of the source position.
        omegatilde (float, optional): The angle between the source and the lens. Defaults to 0.

    Returns:
        jnp.array: The logarithm of the effective luminosity distances.
        jnp.array: The logarithm of the time delays.
    """
    # Solve the image positions
    image_r, image_phi = solve_image_positions_polar(source_r, source_phi, f, omegatilde)
    # Solve the Fermat potential
    fermat_polar = profile.fermat_potential_dimensionless_polar(image_r, image_phi, source_r, source_phi, f, omegatilde)
    # Calculate the magnification in polar coordinates
    magnification_polar = profile.magnification_sie_polar(image_r, image_phi, f, omegatilde)
    # The effective luminosity distances are the luminosity distances divided by the square root of the absolute value of the magnification
    log_effective_luminosity_distances = log_dL - 0.5*jnp.log(jnp.abs(magnification_polar)) # effective_luminosity_distances = jnp.exp(log_dL)/jnp.sqrt(jnp.abs(magnification_polar))
    # Time delays are the differences in arrival times
    log_time_delays = log_T_star + jnp.log(fermat_polar - fermat_polar[0])[1:] # time_delays = jnp.exp(log_T_star)*(fermat_polar - fermat_polar[0])
    # Solve the effective luminosity distances and time delays
    return log_effective_luminosity_distances, log_time_delays

# If executed as main, test that the solver work
if __name__ == "__main__":
    source_x = jnp.array([0.01, 0.015])  # example source x-coordinates
    source_y = jnp.array([0.03, 0.035])  # example source y-coordinates
    f = 0.5  # example axis ratio
    omegatilde = 0  # example angle between major axis and x-axis
    image_positions = solve_image_positions_cartesian(source_x, source_y, f, omegatilde)
    
    def print_images(image_positions):
        image_x = image_positions[0]
        image_y = image_positions[1]
        print("image_x_0:", image_x[0])
        print("image_x_1:", image_x[1])
        print("image_x_2:", image_x[2])
        print("image_x_3:", image_x[3])
        print("image_y_0:", image_y[0])
        print("image_y_1:", image_y[1])
        print("image_y_2:", image_y[2])
        print("image_y_3:", image_y[3])
    print_images(image_positions)
    
    # Now try to solve the image positions not as arrays
    image_positions = solve_image_positions_cartesian(source_x[0], source_y[0], 0.5, 0)
    print_images(image_positions)


