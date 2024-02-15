import numpy as np
import jax.numpy as jnp
from numba import njit
from scipy import LowLevelCallable # https://www.evanmcurtin.com/blog/numba-integrals/
from numba import cfunc, types, carray
from scipy.integrate import nquad, quad
from ncephes import hyp2f1

# Quick semi-analytical integration for cosmology
zmax = 100
zmin = 0
Om0max = 0.97
Om0min = 0.03
xvalsmin_ = ((Om0min*(1 + zmin)**3)/(1-Om0min))
xvalsmax_ = ((Om0max*(1 + zmax)**3)/(1-Om0max))
xvals_ = jnp.geomspace(xvalsmin_,xvalsmax_,100000)
hyp2f1_fixed_vals = jnp.array([hyp2f1(1/3,1/2,4/3,-xvals_[i]) for i in range(len(xvals_))])

def hyp2f1_fixed(x):
    ''' 
    Computes hyp2f1(1/3,1/2,4/3,-x)
    
    Parameters:
        x (jnp.array): The input value for the function.
        
    Returns:
        jnp.array: The result of the hyp2f1 function.
    '''
    return jnp.interp(x,xvals_,hyp2f1_fixed_vals)

def integrate_fast(z, Om_m):
    """
    Calculate the fast integration of a function with respect to redshift (z) and matter density (Om_m).

    Parameters:
    z (jnp.array): Redshift values.
    Om_m (jnp.array): Matter density.

    Returns:
    jnp.array: The result of the integration.
    """
    Om_L = 1 - Om_m
    integral = ( hyp2f1_fixed(((Om_m*(1 + z)**3)/Om_L)) *(1 + z)*jnp.sqrt(1 + (Om_m*(1 + z)**3)/Om_L))/jnp.sqrt(Om_L + Om_m*(1 + z)**3) \
        - (hyp2f1_fixed(((Om_m*(1 + 0)**3)/Om_L))*(1 + 0)*jnp.sqrt(1 + (Om_m*(1 + 0)**3)/Om_L))/jnp.sqrt(Om_L + Om_m*(1 + 0)**3)
    return integral

def E(z, Om_m):
    """
    Calculate the value of E(z) in the cosmological model.

    Parameters:
    z (jnp.array): The redshift value.
    Om_m (jnp.array): The matter density parameter.

    Returns:
    jnp.array: The value of E(z) in the cosmological model.
    """
    Om_L = 1 - Om_m  # Omega lambda
    return jnp.sqrt(Om_m * (1 + z) ** 3 + Om_L)

def integrate_slow(z, Om_m, n=10000):
    """
    Calculate the integral of Einv from 0 to z for each value of z.

    Parameters:
    z (jnp.array): The values of z.
    Om_m (jnp.arraylike): The values of Om_m.
    n (int, optional): The number of points to use for integration. Default is 10000.

    Returns:
    jnp.array: The integral values for each value of z.
    """
    z0 = jnp.linspace(0, 40, n)
    dz = jnp.diff(z0)[0]
    integral = jnp.zeros(len(z))
    for i in range(len(z)):
        zi = z[i]
        Om_m_i = Om_m[i]
        Einv = 1. / E(z0[z0 <= zi], Om_m_i)
        integral[i] = jnp.sum(Einv) * dz
    return integral

def dC(z, H0, Om_m):
    """
    Calculate the comoving distance at redshift z.

    Parameters:
    z (jnp.array): The redshift.
    H0 (jnp.array): The Hubble constant in km/s/Mpc.
    Om_m (jnp.array): The matter density parameter.

    Returns:
    jnp.array: The comoving distance in Mpc.
    """
    c = 299792.458 # km/s
    dH = c/H0 # Mpc
    return dH*integrate_fast(z,Om_m) # Mpc

def dC_fast(z, H0, Om_m):
    ''' 
    Calculate the comoving distance using the fast integration method.

    Parameters:
    z (jnp.array): Redshift values.
    H0 (jnp.array): Hubble constant values in km/s/Mpc.
    Om_m (jnp.array): Matter density parameter values.

    Returns:
    jnp.array: Comoving distance values in Mpc.
    '''
    c = 299792.458 # km/s
    dH = c/H0 # Mpc
    return dH*integrate_fast(z,Om_m) # Mpc

def dM(z, H0, Om_m):
    ''' Computes the comoving distance for a given redshift z, H0 and Om_m.
    
    Parameters:
    z (jnp.array): The redshift.
    H0 (jnp.array): The Hubble constant in km/s/Mpc.
    Om_m (jnp.array): The matter density parameter.
    
    Returns:
    jnp.array: The comoving distance in Mpc.
    '''
    return dC(z, H0, Om_m)

def luminosity_distance(z, H0, Om_m):
    ''' Computes the luminosity distance for a given redshift z, H0 and Om_m.
    
    Parameters:
    z (jnp.array): The redshift.
    H0 (jnp.array): The Hubble constant in km/s/Mpc.
    Om_m (jnp.array): The matter density parameter.
    
    Returns:
    jnp.array: The luminosity distance in Mpc.
    '''
    return (1+z)*dM(z, H0, Om_m)

def angular_diameter_distance(z, H0, Om_m):
    ''' Computes the angular diameter distance for a given redshift z, H0 and Om_m.
    
    Parameters:
    z (jnp.array): The redshift.
    H0 (jnp.array): The Hubble constant in km/s/Mpc.
    Om_m (jnp.array): The matter density parameter.
    
    Returns:
    jnp.array: The angular diameter distance in Mpc.
    '''
    return dM(z,H0,Om_m)/(1+z)

def angular_diameter_distance_z12(z1, z2, H0, Om_m):
    ''' Computes the angular diameter distance between two redshifts for a given H0 and Om_m.
    
    Parameters:
    z1 (jnp.array): The first redshift.
    z2 (jnp.array): The second redshift.
    H0 (jnp.array): The Hubble constant in km/s/Mpc.
    Om_m (jnp.array): The matter density parameter.
    
    Returns:
    jnp.array: The angular diameter distance in Mpc.
    '''
    # H0 is in km/s/Mpc
    c = 299792.458 # km/s
    dH = c/H0 # Mpc
    dM1 = dM(z1,H0,Om_m)
    dM2 = dM(z2,H0,Om_m)
    return ( dM2 - dM1 )/(1+z2)

# If the user wishes, they can test that the cosmology reproduces the FlatLambdaCDM model of astropy with H0 = 70 and Om0 = 0.3
if __name__ == '__main__':
    from astropy.cosmology import FlatLambdaCDM
    import matplotlib.pyplot as plt
    import time
    N = 10000
    # Make 5 subplots
    fig, ax = plt.subplots(1,4,figsize=(16,4))
    for i in range(10):
        H0 = jnp.array(np.random.uniform(5,500))
        Om0 = jnp.array(np.random.uniform(0.05,0.8))
        H0 = jnp.array([H0])*jnp.ones(N)
        Om0 = jnp.array([Om0])*jnp.ones(N)
        z = jnp.linspace(0.01,40,N)
        # Test the luminosity distance
        dL = luminosity_distance(z, H0, Om0)
        dL_astropy = FlatLambdaCDM(H0[0], Om0[0]).luminosity_distance(z)
        ax[0].plot(z,dL, label='cosmologyfast (H0={:.2f}, Om0={:.2f})'.format(H0[0],Om0[0]))
        ax[0].plot(z,dL_astropy, label='astropy (H0={:.2f}, Om0={:.2f})'.format(H0[0],Om0[0]), ls='--')
        ax[0].set_xlabel('z')
        ax[0].set_ylabel('dL')
        # Test the angular diameter distance
        dA = angular_diameter_distance(z, H0, Om0)
        dA_astropy = FlatLambdaCDM(H0[0], Om0[0]).angular_diameter_distance(z)
        ax[1].plot(z,dA, label='cosmologyfast (H0={:.2f}, Om0={:.2f})'.format(H0[0],Om0[0]))
        ax[1].plot(z,dA_astropy, label='astropy (H0={:.2f}, Om0={:.2f})'.format(H0[0],Om0[0]), ls='--')
        ax[1].set_xlabel('z')
        ax[1].set_ylabel('dA')
        # Test the angular diameter distance between two redshifts
        dA12 = angular_diameter_distance_z12(z, z+0.1, H0, Om0)
        dA12_astropy = FlatLambdaCDM(H0[0], Om0[0]).angular_diameter_distance_z1z2(z, z+0.1)
        ax[2].plot(z,dA12, label='cosmologyfast (H0={:.2f}, Om0={:.2f})'.format(H0[0],Om0[0]))
        ax[2].plot(z,dA12_astropy, label='astropy (H0={:.2f}, Om0={:.2f})'.format(H0[0],Om0[0]), ls='--')
        ax[2].set_xlabel('z')
        ax[2].set_ylabel('dA(z1,z2)')
        # Test the comoving distance
        dC_arr = dC(z, H0, Om0)
        dC_arr_astropy = FlatLambdaCDM(H0[0], Om0[0]).comoving_distance(z)
        ax[3].plot(z,dC_arr, label='cosmologyfast (H0={:.2f}, Om0={:.2f})'.format(H0[0],Om0[0]))
        ax[3].plot(z,dC_arr_astropy, label='astropy (H0={:.2f}, Om0={:.2f})'.format(H0[0],Om0[0]), ls='--')
        ax[3].set_xlabel('z')
        ax[3].set_ylabel('dC')
    plt.show()
    # Now do the same thing but plot the differences
    fig, ax = plt.subplots(1,4,figsize=(16,4))
    for i in range(10):
        H0 = jnp.array(np.random.uniform(5,500))
        Om0 = jnp.array(np.random.uniform(0.05,0.8))
        H0 = jnp.array([H0])*jnp.ones(N)
        Om0 = jnp.array([Om0])*jnp.ones(N)
        z = jnp.linspace(0.01,40,N)
        # Test the luminosity distance
        dL = luminosity_distance(z, H0, Om0)
        dL_astropy = FlatLambdaCDM(H0[0], Om0[0]).luminosity_distance(z)
        ax[0].plot(z,dL-dL_astropy, label='cosmologyfast (H0={:.2f}, Om0={:.2f})'.format(H0[0],Om0[0]))
        ax[0].set_xlabel('z')
        ax[0].set_ylabel(r"$\Delta dL \, [Mpc]$")
        # Test the angular diameter distance
        dA = angular_diameter_distance(z, H0, Om0)
        dA_astropy = FlatLambdaCDM(H0[0], Om0[0]).angular_diameter_distance(z)
        ax[1].plot(z,dA-dA_astropy, label='cosmologyfast (H0={:.2f}, Om0={:.2f})'.format(H0[0],Om0[0]))
        ax[1].set_xlabel('z')
        ax[1].set_ylabel(r"$\Delta dA \, [Mpc]$")
        # Test the angular diameter distance between two redshifts
        dA12 = angular_diameter_distance_z12(z, z+0.1, H0, Om0)
        dA12_astropy = FlatLambdaCDM(H0[0], Om0[0]).angular_diameter_distance_z1z2(z, z+0.1)
        ax[2].plot(z,dA12-dA12_astropy, label='cosmologyfast (H0={:.2f}, Om0={:.2f})'.format(H0[0],Om0[0]))
        ax[2].set_xlabel('z')
        ax[2].set_ylabel(r"$\Delta dA(z1,z2) \, [Mpc]$")
        # Test the comoving distance
        dC_arr = dC(z, H0, Om0)
        dC_arr_astropy = FlatLambdaCDM(H0[0], Om0[0]).comoving_distance(z)
        ax[3].plot(z,dC_arr-dC_arr_astropy, label='cosmologyfast (H0={:.2f}, Om0={:.2f})'.format(H0[0],Om0[0]))
        ax[3].set_xlabel('z')
        ax[3].set_ylabel(r"$\Delta dC \, [Mpc]$")
    plt.show()
