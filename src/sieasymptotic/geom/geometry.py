from jax import numpy as jnp

def transform_cartesian_to_polar(x, y, omegatilde=0):
    """Transform Cartesian coordinates to polar coordinates.

    Args:
        x (jnp.array): The x-coordinate of the position.
        y (jnp.array): The y-coordinate of the position.
        omegatilde (int, optional): The angle between the major axis of the lens and the x-axis. Defaults to 0.

    Returns:
        jnp.array: The radial and angular coordinates of the position.
    """
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
