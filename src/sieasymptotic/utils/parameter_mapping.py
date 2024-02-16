import jax.numpy as jnp
from astropy.cosmology import Planck18
import sieasymptotic.utils.cosmology as cosmology

Mpc_to_seconds = 1.029e14 # Mpc to seconds
sigma_v_example = jnp.array([200/299792.]) # 200 km/s in units of c
zl_example = jnp.array([0.5])
zs_example = jnp.array([2.0])
Om_example = jnp.array([0.3])
Ol_example = jnp.array([0.7])
H0_example = jnp.array([70.]) # in km/s/Mpc
dL_example = cosmology.luminosity_distance(zs_example, H0_example, Om_example)
Dl_example = cosmology.angular_diameter_distance(zl_example, H0_example, Om_example)
Ds_example = cosmology.angular_diameter_distance(zs_example, H0_example, Om_example)
Dls_example = cosmology.angular_diameter_distance_z12(zl_example, zs_example, H0_example, Om_example)
theta_E_example = 4.*jnp.pi * sigma_v_example**2 * Dls_example / Ds_example
T_star_example = (1+zl_example) * theta_E_example**2 * Dl_example*Ds_example/Dls_example * Mpc_to_seconds

def H0_from_dL_zs(dL=dL_example, zs=zs_example, Om=Om_example, Ol=Ol_example):
    ''' Convert the dL and zs to H0.
    
    Parameters:
    dL (jnp.array): The luminosity distance.
    zs (jnp.array): The source redshift.
    Om (jnp.array): The matter density.
    Ol (jnp.array): The dark energy density.
    
    Returns:
    jnp.array: The H0.
    '''
    # Convert the dL and zs to H0:
    H0_0 = jnp.array([1.])# dL \propto 1/H0
    H0 = cosmology.luminosity_distance(zs, H0_0, Om)/dL
    return H0

def H0_from_Tstar_zs_zl_sigma_v(T_star=T_star_example, zs=zs_example, zl=zl_example, sigma_v=sigma_v_example, Om=Om_example, Ol=Ol_example):
    ''' Convert the T_star and zs to H0.
    
    Parameters:
    T_star (jnp.array): The Einstein radius crossing time.
    zs (jnp.array): The source redshift.
    zl (jnp.array): The lens redshift.
    sigma_v (jnp.array): The velocity dispersion.
    Om (jnp.array): The matter density.
    Ol (jnp.array): The dark energy density.
    
    Returns:
    jnp.array: The H0.
    '''
    H0_0 = 1.
    Dl_0 = cosmology.angular_diameter_distance(zl, H0_0, Om)
    Ds_0 = cosmology.angular_diameter_distance(zs, H0_0, Om)
    Dls_0 = cosmology.angular_diameter_distance_z12(zl, zs, H0_0, Om)
    theta_E_0 = 4.*jnp.pi * sigma_v**2 * Dls_0 / Ds_0
    T_star_0 = (1+zl) * theta_E_0**2 * Dl_0*Ds_0/Dls_0 * Mpc_to_seconds
    # T_star \propto 1/H0
    # Convert the T_star and zs to H0:
    H0 = T_star_0 / T_star
    return H0

# Compute the H0 from (dL, T_*) posteriors:
def H0_from_dL_Tstar(dL=dL_example*jnp.ones(2), T_star=T_star_example*jnp.ones(2), sigma_v=sigma_v_example*jnp.ones(2), zl=zl_example*jnp.ones(2), zs=zs_example*jnp.ones(2), Om=Om_example*jnp.ones(2), Ol=Ol_example*jnp.ones(2)):
    ''' Convert the dL and T_star posteriors to H0 posteriors.
    
    Parameters:
    dL (jnp.array): The luminosity distance.
    T_star (jnp.array): The Einstein radius crossing time.
    sigma_v (jnp.array): The velocity dispersion.
    zl (jnp.array): The lens redshift.
    zs (jnp.array): The source redshift.
    Om (jnp.array): The matter density.
    Ol (jnp.array): The dark energy density.
    
    Returns:
    jnp.array: The H0 posteriors.
    '''
    # Convert the dL and T_star posteriors to H0 posteriors:
    H0_dL = H0_from_dL_zs(dL, zs, Om, Ol) # Get H0 from dL posterior
    H0_Tstar = H0_from_Tstar_zs_zl_sigma_v(T_star, zs, zl, sigma_v, Om, Ol) # Get H0 from T_star posterior
    return H0_dL, H0_Tstar

# Compute the H0 from (dL, T_*) posteriors:
def H0_from_dL_Tstar_without_zs(dL=dL_example*jnp.ones(2), T_star=T_star_example*jnp.ones(2), sigma_v=sigma_v_example*jnp.ones(2), zl=zl_example*jnp.ones(2), Om=Om_example*jnp.ones(2), Ol=Ol_example*jnp.ones(2)):
    ''' Convert the dL and T_star posteriors to H0 posteriors.
    
    Parameters:
    dL (jnp.array): The luminosity distance.
    T_star (jnp.array): The Einstein radius crossing time.
    sigma_v (jnp.array): The velocity dispersion.
    zl (jnp.array): The lens redshift.
    Om (jnp.array): The matter density.
    Ol (jnp.array): The dark energy density.
    
    Returns:
    jnp.array: The H0 posteriors.
    '''
    # Convert the dL and T_star posteriors to H0 posteriors:
    zs_array = jnp.linspace(0.1, 10, 100)
    H0 = jnp.zeros(len(dL))
    zs = jnp.zeros(len(dL))
    for i in range(len(dL)):
        H0_dL_array = H0_from_dL_zs(dL[i], zs_array, Om[i], Ol[i]) # Get H0 from dL posterior
        H0_Tstar_array = H0_from_Tstar_zs_zl_sigma_v(T_star[i], zs_array, zl[i], sigma_v[i], Om[i], Ol[i])
        # Take the best-matching value of H0
        print("dL",H0_dL_array)
        print("Tstar", H0_Tstar_array)
        idx = jnp.argmin(( H0_dL_array - H0_Tstar_array)**2)
        H0_best, zs_best = H0_dL_array[idx], zs_array[idx]
        H0 = H0.at[i].set(float(H0_best))
        zs = zs.at[i].set(float(zs_best))
        print("H0_best",H0_best)
        print("zs_best",zs_best)
    return H0, zs

# If the user runs as main, test the parameter mapping
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    # # Test the parameter mapping
    # H0_dL, H0_Tstar = H0_from_dL_Tstar()
    # plt.plot(H0_dL, H0_Tstar, 'o')
    # plt.xlabel(r'$H_0$ from $d_L$')
    # plt.ylabel(r'$H_0$ from $T_*$')
    # plt.show()
    # print('H0_dL:', H0_dL)
    # print('H0_Tstar:', H0_Tstar)
    # print('H0_dL - H0_Tstar:', H0_dL - H0_Tstar)
    # print('H0_dL / H0_Tstar:', H0_dL / H0_Tstar)
    # Now test the parameter mapping without zs
    H0, zs = H0_from_dL_Tstar_without_zs()
    plt.plot(zs, H0, 'o')
    plt.xlabel(r'$z_s$')
    plt.ylabel(r'$H_0$')
    plt.show()
    print('H0:', H0)
    print('zs:', zs)
    
    