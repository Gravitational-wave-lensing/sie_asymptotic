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
    if len(fermat_potential.shape) == 2:
        sorted_indices = jnp.argsort(fermat_potential, axis=0)
        fermat_potential_sorted = fermat_potential[sorted_indices,jnp.arange(fermat_potential.shape[1])]
        image_positions_0_sorted = image_positions[0][sorted_indices,jnp.arange(fermat_potential.shape[1])]
        image_positions_1_sorted = image_positions[1][sorted_indices,jnp.arange(fermat_potential.shape[1])]
    elif len(fermat_potential.shape) <= 1:
        sorted_indices = jnp.argsort(fermat_potential)
        fermat_potential_sorted = fermat_potential[sorted_indices]
        image_positions_0_sorted = image_positions[0][sorted_indices]
        image_positions_1_sorted = image_positions[1][sorted_indices]
    else:
        raise ValueError("Something went horribly wrong")
    image_positions_sorted = jnp.array([image_positions_0_sorted, image_positions_1_sorted])
    return image_positions_sorted, fermat_potential_sorted

