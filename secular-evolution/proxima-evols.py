import pathlib
import matplotlib.pyplot as plt
import pandas as pd
from vplanet import get_output
from seaborn import color_palette
import astropy.units as u
from astropy.constants import G, M_earth, R_earth
import numpy as np
from numpy import pi, sign

# Plot params
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# All usable VPLANET simulations
sims = {
        '1': ['sim1', 0.51653, 2.18208, 389.31695, 85.28869],
        '2': ['sim2', 0.39862, 1.71255, 322.54608, 466.03801],
        '3': ['sim3', 0.35092, 1.23215, 250.40924, 220.5822],
        '4': ['sim4', 0.44200, 1.13222, 277.98179, 646.02112],
        '5': ['sim5', 0.33433, 1.23139, 243.22927, 15.94921],
        '6': ['sim6', 0.32236, 1.37201, 452.43231, 124.37175]
        }
sims = pd.DataFrame.from_dict(sims, orient='index', columns=['data_dir', 'm_d', 'm_b', 'Q_d', 'Q_b'])

# Tidal heating calculation (because I didn't run these with the SurfEnFluxEqTide output omg)
# def tidal_heating(mass, tidal_Q, semi, ecc):
#     mass *= u.earthMass
#     semi *= u.AU / u.AU
#     radius = R_earth * (mass / M_earth) ** 0.274 # M-R relationship of Sotin (2007)
#     star_mass = 0.12 * u.solMass
#     k2 = 0.3
#     tide_heat = 10.5 * k2/tidal_Q * G**1.5 * star_mass**2.5 * radius**5. * semi**(-7.5) * ecc**2. * (1. - ecc**2.)**(-7.5) * (1. + 7.5 * ecc**2. + 1.875 * ecc**4. + 0.234375 * ecc**6.) # Hut (1981)
#     surf_area = 4. * pi * radius * radius
#     heat_flux = tide_heat / surf_area
#     return heat_flux.to(u.W / u.m / u.m) / (u.W / u.m / u.m)

def tidal_heating(mass, tidal_Q, semi, ecc, mean_motion, rot_freq):
    mass *= u.earthMass
    semi *= u.AU / u.AU
    mean_motion *= u.d / u.d
    rot_freq *= u.Hz * u.s/u.d
    radius = R_earth * (mass / M_earth) ** 0.274 # M-R relationship of Sotin (2007)
    star_mass = 0.12 * u.solMass
    k2 = 0.3
    big_Z = 3 * G * G * k2 * star_mass * star_mass * (star_mass + mass) * radius**5. / (semi**9.) * 1./(mean_motion * tidal_Q)
    # Phase lag signs
    lag_0 = sign(rot_freq - mean_motion)
    lag_1 = sign(2. * rot_freq - 3. * mean_motion)
    lag_2 = sign(2. * rot_freq - mean_motion)
    lag_5 = sign(mean_motion)
    # Tidal heating due to dissipation of orbital energy (obliquity is zero)
    tide_heat_orb = -big_Z/8. * (4. * lag_0 + ecc * ecc * (-20. * lag_0 + 73.5 * lag_1 + 0.5 * lag_2 - 3. * lag_5))
    # Tidal heating due to dissipation of rotational energy (obliquity is still zero)
    tide_heat_rot = big_Z/8. * rot_freq/mean_motion * (4. * lag_0 + ecc * ecc * (-20. * lag_0 + 49. * lag_1 + lag_2))
    # Combine
    tide_heat_orb = tide_heat_orb.decompose().to(u.W)
    tide_heat_rot = tide_heat_rot.decompose().to(u.W)
    tide_heat = tide_heat_orb + tide_heat_rot
    surf_area = 4. * pi * radius * radius
    heat_flux = tide_heat / surf_area
    return heat_flux.to(u.W / u.m / u.m) / (u.W / u.m / u.m)

# Retrieve data
path = pathlib.Path() / 'vplanet-sims'
times = []
b_eccs = []
d_eccs = []
b_change_semis = []
d_change_semis = []
b_heats = []
keys = sims.index
for i in keys:
    out = get_output(path/sims['data_dir'][i], units=False)
    time = out.proxima.Time
    b_ecc = out.b.Eccentricity
    d_ecc = out.d.Eccentricity
    b_semi = out.b.SemiMajorAxis
    d_semi = out.d.SemiMajorAxis
    b_mm = out.b.MeanMotion
    d_mm = out.d.MeanMotion
    b_rfrq = np.array([2.*pi/p for p in out.b.RotPer])
    b_change_semi = (b_semi/b_semi[0]) - 1.
    d_change_semi = (d_semi/d_semi[0]) - 1.
    # b_heat = tidal_heating(sims['m_b'][i], sims['Q_b'][i], b_semi, b_ecc)
    b_heat = tidal_heating(sims['m_b'][i], sims['Q_b'][i], b_semi, b_ecc, b_mm, b_rfrq)
    times.append(time)
    b_eccs.append(b_ecc)
    d_eccs.append(d_ecc)
    b_change_semis.append(b_change_semi)
    d_change_semis.append(d_change_semi)
    b_heats.append(b_heat)
io_heat = 2. # Moore et al. (2007)
earth_heat = 0.0073 # Munk & Wunsch (1998)

# Make figure
palette = color_palette('colorblind', len(times))
fig, ax = plt.subplot_mosaic([['a', 'b'], ['c', 'd'], ['e', 'e']], figsize=(9, 10))
for i, time in enumerate(times):
    ax['a'].plot(time/1.e9, d_eccs[i], c=palette[i], alpha=0.5, label=keys[i], zorder=-1)
    ax['b'].plot(time/1.e9, d_change_semis[i], c=palette[i], alpha=0.5, zorder=-1)
    ax['c'].plot(time/1.e9, b_eccs[i], c=palette[i], alpha=0.5, zorder=-1)
    ax['d'].plot(time/1.e9, b_change_semis[i], c=palette[i], alpha=0.5, zorder=-1)
    ax['e'].semilogy(time/1.e9, b_heats[i], c=palette[i], alpha=0.5, zorder=-1)
ax['e'].axhline(io_heat, c='k', ls='dashed', label='Io')
ax['e'].axhline(earth_heat, c='k', ls='dotted', label='Earth')
ax['a'].set_ylabel(r'$e_d$')
ax['b'].set_ylabel(r'$\Delta a_d / a_{d0}$')
ax['c'].set_ylabel(r'$e_b$')
ax['d'].set_ylabel(r'$\Delta a_b / a_{b0}$')
ax['e'].set_ylabel(r'Tidal heating of planet b (W m$^{-2}$)')
for a in [ax['a'], ax['b'], ax['c'], ax['d'], ax['e']]:
    a.set_xlabel('Time (Gyr)')
    a.set_xlim(0., 7.)
    a.set_rasterization_zorder(0)
ax['a'].legend(loc=0)
ax['e'].legend(loc=0)
fig.tight_layout()
fig.savefig('proxima-evols.pdf', dpi=600)
