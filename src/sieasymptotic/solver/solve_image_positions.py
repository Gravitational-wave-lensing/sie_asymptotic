# Lens equation solver for the SIE lens, based on asymptotic expansion.
from jax import numpy as jnp
import jax
import numpy as np

def transform_cartesian_to_polar(x, y, omegatilde=0):
    ''' Transform Cartesian coordinates to polar coordinates.
    
    Args:
        x (jnp.array): The x-coordinate of the position.
        y (jnp.array): The y-coordinate of the position.
        omegatilde (float, optional): The angle between the major axis of the lens and the x-axis.
    
    Returns:
        jnp.array: The radial and angular coordinates of the position.
    '''
    r = jnp.sqrt(x**2 + y**2)
    phi = jnp.arctan2(y, x)-omegatilde
    return jnp.array([r, phi])

def transform_polar_to_cartesian(r, phi, omegatilde=0):
    ''' Transform polar coordinates to Cartesian coordinates.
    
    Args:
        r (jnp.array): The radial coordinate of the position.
        phi (jnp.array): The angular coordinate of the position.
        omegatilde (float, optional): The angle between the major axis of the lens and the x-axis.
    
    Returns:
        jnp.array: The x and y coordinates of the position.
    '''
    x = r*jnp.cos(phi+omegatilde)
    y = r*jnp.sin(phi+omegatilde)
    return jnp.array([x, y])

def solve_image_positions_polar(source_r, source_theta, f, omegatilde=0):
    ''' Solve the lens equation for the SIE lens, using the asymptotic expansion, in polar coordinates.
    
    Args:
        source_r (jnp.array): The radial coordinate of the source position.
        source_theta (jnp.array): The angular coordinate of the source position.
        f (float): The axis ratio of the lens.
        omegatilde (float, optional): The angle between the major axis of the lens and the x-axis.
    
    Returns:
        jnp.array: The radial coordinate of the image positions.
    '''
    # There are 4 solutions. We expand to first order.
    # Asymptotic 0th order first image position:
    # (Eq. 6 of https://academic.oup.com/mnras/article/442/1/428/1244014)
    k = jnp.arange(4)
    image_phi_0 = k * jnp.pi/2-omegatilde
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
    image_delta_phi, image_delta_r = first_order_image_perturbation(source_r, source_theta, f, omegatilde, f_prime)
    
    # The image positions, to first order, are the sum of the first-order solution plus the perturbation
    image_r = jnp.transpose(image_r_0 + jnp.transpose(image_delta_r))
    image_phi = jnp.transpose(image_phi_0 + jnp.transpose(image_delta_phi))
    return jnp.array([image_r, image_phi])

def first_order_image_perturbation(source_r, source_theta, f, omegatilde, f_prime):
    """
    Calculate the first-order image perturbation for a given set of parameters.

    Args:
        source_r (float): The radial distance of the source.
        source_theta (float): The angular position of the source.
        f (float): The lensing potential.
        omegatilde (float): The angular position of the perturbation.
        f_prime (float): The derivative of the lensing potential.

    Returns:
        numpy.ndarray: An array containing the first-order image perturbation.

    """
    image_delta_phi = jnp.array([
       source_r*jnp.sin(source_theta+omegatilde)/(jnp.sqrt(f)*(1./f_prime*jnp.arcsinh(f_prime/f)-1)),
       source_r*jnp.cos(source_theta+omegatilde)/(jnp.sqrt(f)*(1./f-1./f_prime*jnp.arcsin(f_prime)) ),
       -1*source_r*jnp.sin(source_theta+omegatilde)/(jnp.sqrt(f)*(1./f_prime*jnp.arcsinh(f_prime/f)-1)),
       -1*source_r*jnp.cos(source_theta+omegatilde)/(jnp.sqrt(f)*(1./f-1./f_prime*jnp.arcsin(f_prime)) )
    ])
    image_delta_r = jnp.array([
        source_r*jnp.cos(source_theta+omegatilde),
        source_r*jnp.cos(source_theta+omegatilde),
        -1*source_r*jnp.cos(source_theta+omegatilde),
        -1*source_r*jnp.cos(source_theta+omegatilde),
    ])
    return jnp.array([image_delta_phi,image_delta_r])

def solve_image_positions_cartesian(source_x, source_y, f, omegatilde=0):
    ''' Solve the lens equation for the SIE lens, using the asymptotic expansion, in Cartesian coordinates.
    
    Args:
        source_x (jnp.array): The x-coordinate of the source position.
        source_y (jnp.array): The y-coordinate of the source position.
        f (float): The axis ratio of the lens.
        omegatilde (float, optional): The angle between the major axis of the lens and the x-axis.
    
    Returns:
        jnp.array: The x and y coordinates of the image positions.
    '''
    # Transform the source position to polar coordinates
    source_polar = transform_cartesian_to_polar(source_x, source_y, omegatilde)
    source_r = source_polar[0]
    source_theta = source_polar[1]
    # Solve the lens equation in polar coordinates
    images_polar = solve_image_positions_polar(source_r, source_theta, f, omegatilde)
    image_r = images_polar[0]
    image_phi = images_polar[1]
    # Transform the image positions back to Cartesian coordinates
    images_cartesian = transform_polar_to_cartesian(image_r, image_phi, omegatilde)
    image_x = images_cartesian[0]
    image_y = images_cartesian[1]
    return jnp.array([image_x, image_y])

# If executed as main, test that the solver work
if __name__ == "__main__":
    source_x = jnp.array([0.01, 0.015])  # example source x-coordinates
    source_y = jnp.array([0.03, 0.035])  # example source y-coordinates
    f = 0.5  # example axis ratio
    omegatilde = 0  # example angle between major axis and x-axis
    solve_image_positions_cartesian_jax = jax.jit(solve_image_positions_cartesian)
    image_positions = solve_image_positions_cartesian_jax(source_x, source_y, f, omegatilde)
    
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
    image_positions = solve_image_positions_cartesian_jax(source_x[0], source_y[0], f, omegatilde)
    print_images(image_positions)
