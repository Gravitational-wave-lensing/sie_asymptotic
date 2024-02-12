# Deflection potential, dimensionless fermat potential, magnification, and time delay for the SIE lens
from jax import numpy as jnp
import sieasymptotic.utils as utils
import sieasymptotic.solver as solver

def psi_dimensionless_polar(image_r, image_phi, f, omegatilde=0):
    ''' Calculate the dimensionless deflection potential for the SIE lens in polar coordinates.
    
    Args:
        image_r (jnp.array): The radial coordinate of the image position.
        image_phi (jnp.array): The angular coordinate of the image position.
        f (jnp.array): The axis ratio of the lens.
        omegatilde (jnp.array, optional): The angle between the major axis of the lens and the x-axis.
        
    Returns:
        jnp.array: The dimensionless deflection potential.
    '''
    f, omegatilde = utils.convert_f_omegatilde(f, omegatilde, image_r.shape)
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
        f (jnp.array): The axis ratio of the lens.
        omegatilde (jnp.array, optional): The angle between the major axis of the lens and the x-axis.
        
    Returns:
        jnp.array: The dimensionless deflection potential.
    '''
    image_polar = utils.transform_cartesian_to_polar(image_x, image_y, omegatilde)
    psi = psi_dimensionless_polar(image_polar[0], image_polar[1], f, omegatilde)
    return psi

def fermat_potential_dimensionless_polar(image_r, image_phi, source_r, source_phi, f, omegatilde=0):
    ''' Calculate the dimensionless Fermat potential for the SIE lens in polar coordinates.
    
    Args:
        image_r (jnp.array): The radial coordinate of the image position.
        image_phi (jnp.array): The angular coordinate of the image position.
        source_r (jnp.array): The radial coordinate of the source position.
        source_phi (jnp.array): The angular coordinate of the source position.
        f (jnp.array): The axis ratio of the lens.
        omegatilde (jnp.array, optional): The angle between the major axis of the lens and the x-axis.
        
    Returns:
        jnp.array: The dimensionless Fermat potential.
    '''
    # Deflection potential:
    psi = psi_dimensionless_polar(image_r, image_phi, f, omegatilde)
    # 0.5*(image-source)**2
    geometrical_time_delay = (image_r**2 + source_r**2 - 2*image_r*source_r*jnp.cos(image_phi-source_phi))/2.
    # Fermat potential
    fermat = geometrical_time_delay - psi
    return fermat

def fermat_potential_dimensionless_cartesian(image_x, image_y, source_x, source_y, f, omegatilde=0):
    ''' Calculate the dimensionless Fermat potential for the SIE lens in Cartesian coordinates.
    
    Args:
        image_x (jnp.array): The x-coordinate of the image position.
        image_y (jnp.array): The y-coordinate of the image position.
        source_x (jnp.array): The x-coordinate of the source position.
        source_y (jnp.array): The y-coordinate of the source position.
        f (jnp.array): The axis ratio of the lens.
        omegatilde (jnp.array, optional): The angle between the major axis of the lens and the x-axis.
        
    Returns:
        jnp.array: The dimensionless Fermat potential.
    '''
    image_polar = utils.transform_cartesian_to_polar(image_x, image_y, omegatilde)
    source_polar = utils.transform_cartesian_to_polar(source_x, source_y, omegatilde)
    fermat = fermat_potential_dimensionless_polar(image_polar[0], image_polar[1], source_polar[0], source_polar[1], f, omegatilde)
    return fermat

def magnification_sie_polar(image_r, image_phi, f, omegatilde=0):
    '''Calculate the magnification for a point in the polar coordinate system using the Singular Isothermal Ellipsoid (SIE) profile.

    Parameters:
    image_r (jnp.array): The radial coordinate of the point.
    image_phi (jnp.array): The angular coordinate of the point.
    f (jnp.array): Axis ratio of lens
    omegatilde (jnp.array, optional): The external shear parameter. Defaults to 0.

    Returns:
    jnp.array: The magnification at the given point.
    '''
    f, omegatilde = utils.convert_f_omegatilde(f,omegatilde,image_r.shape)
    rho = image_r*jnp.sqrt(jnp.cos(image_phi)**2+f**2*jnp.sin(image_phi)**2)
    kappa = jnp.sqrt(f)/(2*rho)
    mu = 1./(1.-2*kappa)
    return mu
    
def magnification_sie_cartesian(image_x, image_y, f, omegatilde=0):
    """
    Calculate the magnification for a point in the Cartesian coordinate system using the SIE (Singular Isothermal Ellipsoid) profile.

    Parameters:
    image_x (jnp.array): The x-coordinate of the image point.
    image_y (jnp.array): The y-coordinate of the image point.
    f (jnp.array): The focal length of the lens.
    omegatilde (jnp.array, optional): The external shear parameter. Defaults to 0.

    Returns:
    jnp.array: The magnification at the given image point.
    """
    image_polar = utils.transform_cartesian_to_polar(image_x, image_y, omegatilde)
    image_r = image_polar[0]
    image_phi = image_polar[1]
    mu = magnification_sie_polar(image_r, image_phi, f, omegatilde)
    return mu

from astropy.cosmology import Planck18
from astropy import units
def time_delay_factor(zl, zs, sigma, cosmo=Planck18):
    """
    Calculate the time delay factor for gravitational lensing.

    Parameters:
    zl (float): Redshift of the lens.
    zs (float): Redshift of the source.
    sigma (float): Velocity dispersion of the lens in units of c (i.e., sigma=1 equals speed of light).
    cosmo (astropy.cosmology object, optional): Cosmology object. Default is Planck18.

    Returns:
    float: The time delay factor in seconds.
    
    Note: This function is not jax-able.
    """
    # theta_E = 4*jnp.pi*sigma**2*Dls/Ds
    Dl = cosmo.angular_diameter_distance(zl).value
    Ds = cosmo.angular_diameter_distance(zs).value
    Dls= cosmo.angular_diameter_distance_z1z2(zl, zs).value
    c  = 9.716e-15 # Mpc/seconds
    return jnp.array(((1+zl)/c)*(Dl*Dls/Ds)*(4*jnp.pi*sigma**2)**2)

if __name__ == "__main__":
    import jax
    source_x = jnp.array([0.01, 0.015])  # example source x-coordinates
    source_y = jnp.array([0.03, 0.035])  # example source y-coordinates
    f = 0.5  # example axis ratio
    omegatilde = 0  # example angle between major axis and x-axis
    source_r, source_phi = utils.transform_cartesian_to_polar(source_x, source_y, omegatilde)
    image_positions = solver.solve_image_positions_polar(source_r, source_phi, f, omegatilde)
    print("Image positions, polar coordinates:", image_positions[0])
    image_r, image_phi = image_positions

    # Calculate the dimensionless Fermat potential in polar coordinates
    fermat_polar = fermat_potential_dimensionless_polar(image_positions[0], image_positions[1], source_r, source_phi, f, omegatilde)
    print("Dimensionless Fermat potential in polar coordinates:", fermat_polar)
    
    # Sort the image positions by arrival time
    image_positions, fermat_polar = utils.sort_images_by_arrival_time(image_positions, fermat_polar)
    print("Sorted:")
    print("Image positions, polar coordinates:", image_positions[0])
    print("Dimensionless Fermat potential in polar coordinates:", fermat_polar)
    image_r, image_phi = image_positions

    
    # Calculate the magnification in polar coordinates
    magnification_polar = magnification_sie_polar(image_r, image_phi, f, omegatilde)
    print("Magnification in polar coordinates:", magnification_polar)

    # Calculate the time delay factor
    zl = 0.5
    zs = 1.0
    c_km_s = 299792.0
    sigma = 200.0/c_km_s
    time_delay = time_delay_factor(zl, zs, sigma)
    print("Time delay factor:", time_delay)
