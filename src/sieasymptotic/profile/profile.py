# Deflection potential, dimensionless fermat potential, magnification, and time delay for the SIE lens
from jax import numpy as jnp
from sieasymptotic.geom.geometry import transform_cartesian_to_polar, transform_polar_to_cartesian

def psi_dimensionless_polar(image_r, image_phi, f, omegatilde=0):
    ''' Calculate the dimensionless deflection potential for the SIE lens in polar coordinates.
    
    Args:
        image_r (jnp.array): The radial coordinate of the image position.
        image_phi (jnp.array): The angular coordinate of the image position.
        f (float): The axis ratio of the lens.
        omegatilde (float, optional): The angle between the major axis of the lens and the x-axis.
        
    Returns:
        jnp.array: The dimensionless deflection potential.
    '''
    r = image_r
    phi_tilde = image_phi - omegatilde
    f_prime = jnp.sqrt(1-f**2)
    # (Eq. 8 of https://academic.oup.com/mnras/article/442/1/428/1244014)
    psi = jnp.sqrt(f)/f_prime * r * (jnp.sin(phi_tilde)* jnp.arcsin(f_prime*jnp.sin(phi_tilde)) + jnp.cos(phi_tilde)*jnp.arcsinh(f_prime/f*jnp.cos(phi_tilde)))
    return psi

def psi_dimensionless_cartesian(image_x, image_y, f, omegatilde=0):
    ''' Calculate the dimensionless deflection potential for the SIE lens in Cartesian coordinates.
    
    Args:
        image_x (jnp.array): The x-coordinate of the image position.
        image_y (jnp.array): The y-coordinate of the image position.
        f (float): The axis ratio of the lens.
        omegatilde (float, optional): The angle between the major axis of the lens and the x-axis.
        
    Returns:
        jnp.array: The dimensionless deflection potential.
    '''
    image_polar = transform_cartesian_to_polar(image_x, image_y, omegatilde)
    psi = psi_dimensionless_polar(image_polar[0], image_polar[1], f, omegatilde)
    return psi
