import pathlib
import matplotlib.pyplot as plt
import pandas as pd
from vplanet import get_output
from seaborn import color_palette
import astropy.units as u
from astropy.constants import G, M_earth, R_earth
from numpy import pi

# Plot params
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# All usable VPLANET simulations
sims = {
        '1': ['vplanet-sims/sim1', 2.18208, 85.28869],
        '2': ['vplanet-sims/sim2', 1.71255, 466.03801],
        '3': ['vplanet-sims/sim3', 1.23215, 220.5822],
        '4': ['vplanet-sims/sim4', 1.13222, 646.02112],
        '5': ['vplanet-sims/sim5', 1.23139, 15.94921],
        '6': ['vplanet-sims/sim6', 1.37201, 124.37175]
        }
sims = pd.DataFrame.from_dict(sims, orient='index', columns=['data_dir', 'm_b', 'Q_b'])

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
path = pathlib.Path()
times = []
eccs = []
change_semis = []
heats = []
keys = sims.index
for i in keys:
    out = get_output(path/sims['data_dir'][i], units=False)
    time = out.proxima.Time
    b_ecc = out.b.Eccentricity
    b_semi = out.b.SemiMajorAxis
    b_mm = out.b.MeanMotion
    b_rfrq = np.array([2.*pi/p for p in out.b.RotPer])
    b_change_semi = (b_semi/b_semi[0]) - 1.
    # b_heat = tidal_heating(sims['m_b'][i], sims['Q_b'][i], b_semi, b_ecc)
    b_heat = tidal_heating(sims['m_b'][i], sims['Q_b'][i], b_semi, b_ecc, b_mm, b_rfrq)
    times.append(time)
    eccs.append(b_ecc)
    change_semis.append(b_change_semi)
    heats.append(b_heat)
io_heat = 2. # Moore et al. (2007)
earth_heat = 0.0073 # Munk & Wunsch (1998)

# Make figure
palette = color_palette('colorblind', len(times))
fig, ax = plt.subplot_mosaic([['a', 'b'], ['c', 'c']], figsize=(9, 8))
for i, time in enumerate(times):
    ax['a'].plot(time/1.e9, eccs[i], c=palette[i], alpha=0.5, label=keys[i])
    ax['b'].plot(time/1.e9, change_semis[i], c=palette[i], alpha=0.5)
    ax['c'].semilogy(time/1.e9, heats[i], c=palette[i], alpha=0.5)
ax['c'].axhline(io_heat, c='k', ls='dashed', label='Io')
ax['c'].axhline(earth_heat, c='k', ls='dotted', label='Earth')
ax['a'].set_ylabel('Eccentricity')
ax['b'].set_ylabel('Fractional change in semi-major axis')
ax['c'].set_ylabel(r'Tidal heating (W m$^{-2}$)')
for a in [ax['a'], ax['b'], ax['c']]:
    a.set_xlabel('Time (Gyr)')
    a.set_xlim(0., 7.)
ax['a'].legend(loc=0)
ax['c'].legend(loc=0)
fig.tight_layout()
fig.savefig('proxima-b-evols.pdf', dpi=600)
