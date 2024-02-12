from jax import numpy as jnp

def convert_type_to_jnp_array(x, shape):
    """
    Converts x to `jnp.array` if they are not already.

    Parameters:
    x (float or jnp.array): The variable to convert (e.g. float or jnp.array).
    shape (tuple): The shape of the desired array

    Returns:
    jnp.array: The jnp array
    """
    if type(x) != jnp.array:
        x = x*jnp.ones(shape)
    return x

def convert_f_omegatilde(f,omegatilde,shape):
    f, omegatilde = convert_type_to_jnp_array(f, shape), convert_type_to_jnp_array(omegatilde, shape)
    return f, omegatilde

def sort_images_by_arrival_time(image_positions, fermat_potential):
    """
    Sort the image positions by arrival time (same as sorting by fermat potential).

    Parameters:
    image_positions (jnp.array): The image positions.
    fermat_potential (jnp.array): The fermat potential.

    Returns:
    jnp.array: The sorted image positions.
    """
    sorted_indices = jnp.argsort(fermat_potential)
    return image_positions[sorted_indices], fermat_potential[sorted_indices]