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
def tidal_heating(mass, tidal_Q, semi, ecc):
    mass *= u.earthMass
    semi *= u.AU / u.AU
    radius = R_earth * (mass / M_earth) ** 0.274 # M-R relationship of Sotin (2007)
    star_mass = 0.12 * u.solMass
    k2 = 0.3
    tide_heat = 10.5 * k2/tidal_Q * G**1.5 * star_mass**2.5 * radius**5. * semi**(-7.5) * ecc**2. * (1. - ecc**2.)**(-7.5) * (1. + 7.5 * ecc**2. + 1.875 * ecc**4. + 0.234375 * ecc**6.) # Hut (1981)
    surf_area = 4. * pi * radius * radius
    heat_flux = tide_heat / surf_area
    return heat_flux.to(u.W / u.m / u.m)

# Retrieve data
path = pathlib.Path()
times = []
eccs = []
change_semis = []
heats = []
for i in sims.index:
    out = get_output(path/sims['data_dir'][i], units=False)
    time = out.proxima.Time
    b_ecc = out.b.Eccentricity
    b_semi = out.b.SemiMajorAxis
    b_change_semi = (b_semi/b_semi[0]) - 1.
    b_heat = tidal_heating(sims['m_b'][i], sims['Q_b'][i], b_semi, b_ecc)
    times.append(time)
    eccs.append(b_ecc)
    change_semis.append(b_change_semi)
    heats.append(b_heat)

# Make figure
keys = sims.index
palette = color_palette('colorblind', len(times))
fig, ax = plt.subplot_mosaic([['a', 'b'], ['c', 'c']], figsize=(9, 8))
for i, time in enumerate(times):
    ax['a'].plot(time/1.e9, eccs[i], c=palette[i], alpha=0.5, label=keys[i])
    ax['b'].plot(time/1.e9, change_semis[i], c=palette[i], alpha=0.5)
    ax['c'].semilogy(time/1.e9, heats[i], c=palette[i], alpha=0.5)
ax['a'].set_ylabel('Eccentricity')
ax['b'].set_ylabel('Fractional change in semi-major axis')
ax['c'].set_ylabel(r'Tidal heating (W m$^{-2}$)')
for a in [ax['a'], ax['b'], ax['c']]:
    a.set_xlabel('Time (Gyr)')
    a.set_xlim(0., 7.)
ax['a'].legend(loc=0)
fig.tight_layout()
fig.savefig('proxima-b-evols.pdf', dpi=600)
