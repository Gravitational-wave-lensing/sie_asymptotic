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
    if isinstance(x, jnp.array) == False:
        x = x*jnp.ones(shape)
    return x

def convert_f_omegatilde(f,omegatilde,shape):
    f, omegatilde = convert_type_to_jnp_array(f, shape), convert_type_to_jnp_array(omegatilde, shape)
    return f, omegatilde
