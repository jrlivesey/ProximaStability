"""
This script runs five REBOUND simulations of 1 Myr of evolution of the system in
high-e stable states to ensure stability on longer timescales than presented in
Fig. 1 of the paper.
"""

import rebound
import numpy as np
import pandas as pd
import astropy.units as u
import multiprocessing as mp


def phase_delta(d, b):
    """
    Calculate the separation in phase space between the two planets. This verifies the exponential
    increase in the separation for chaotic (MEGNO != 2) evolution.
    """
    xdiff = b.x - d.x
    ydiff = b.y - d.y
    zdiff = b.z - d.z
    vxdiff = b.vx - d.vx
    vydiff = b.vy - d.vy
    vzdiff = b.vz - d.vz
    phase_delta_vec = np.array([xdiff, ydiff, zdiff, vxdiff, vydiff, vzdiff])
    return np.linalg.norm(phase_delta_vec)


def conjunction_longitude(d, b):
    return (2.*b.l - d.l) % (2.*np.pi)


def resonant_argument_1(d, b):
    return (2.*b.l - d.l - d.pomega) % (2.*np.pi)


def resonant_argument_2(d, b):
    return (2.*b.l - d.l - b.pomega) % (2.*np.pi)


def resonant_argument_3(d, b):
    return (3.*b.l - d.l - 2.*d.pomega) % (2.*np.pi)


def resonant_argument_32(d, b):
    return (3.*b.l - 2.*d.l - d.pomega) % (2.*np.pi)


def simulation_with_output(par, integrator='ias15', step_size=100, primary_mass=0.12, M=None, num=0):
    """
    Runs a REBOUND simulation with the specified orbital elements for Proxima d & b. Returns DataFrame with
    snapshots of the orbital elements taken every 100 timesteps.
    """
    m1, m2, a1, a2, e1, e2, i1, i2, Omega1, Omega2, pomega1, pomega2 = par # unpack parameters
    sim = None
    sim = rebound.Simulation()
    sim.integrator = integrator
    if integrator == 'whfast':
        sim.ri_whfast.safe_mode = False
        sim.ri_whfast.corrector = 11
    sim.add(m=primary_mass)                      # Proxima Centauri
    if M == None:
        sim.add(m=m1, a=a1, e=e1, inc=i1, Omega=Omega1, pomega=pomega1)  # Proxima d
        sim.add(m=m2, a=a2, e=e2, inc=i2, Omega=Omega2, pomega=pomega2)  # Proxima b
    else:
        M1, M2 = M
        sim.add(m=m1, a=a1, e=e1, inc=i1, Omega=Omega1, pomega=pomega1, M=M1)  # Proxima d
        sim.add(m=m2, a=a2, e=e2, inc=i2, Omega=Omega2, pomega=pomega2, M=M2)  # Proxima b
    star = sim.particles[0]
    d = sim.particles[1]
    b = sim.particles[2]
#     sim.dt = 0.05 * d.P # timestep is 5% of Proxima d's orbital period
    sim.dt = 0.01 * d.P # For System A from Barnes et al. (2015)
    sim.move_to_com()
    if integrator == 'ias15':
        sim.ri_ias15.min_dt = 1e-4 * sim.dt
    sim.init_megno()
    sim.exit_max_distance = 20.

    energy = sim.calculate_energy()
    delta = phase_delta(d, b)
    conj = conjunction_longitude(d, b)
    phi1 = resonant_argument_1(d, b)
    phi2 = resonant_argument_2(d, b)
    phi3 = resonant_argument_3(d, b)
    phi32 = resonant_argument_32(d, b)
    output = [[0., a1, a2, e1, e2, i1, i2, Omega1, Omega2, pomega1,
               pomega2, d.l, b.l, 0., 0., 0., 0., energy, 0.,
               delta, conj, phi1, phi2, phi3, phi32]]

    try:
        i = 1
        while sim.t <= 1.e6 * 2.*np.pi:
            # Stop simulation after 1 Myr
            if i % int(1.e7) == 0:
                # Print current simulation time in years, every so often
                print(str(num)+': '+str(sim.t / (2.*np.pi)))
            sim.step()
            if i % int(1.e5) == 0:
                # Write data every 10,000 timesteps
                do = d.calculate_orbit(primary=star)
                bo = b.calculate_orbit(primary=star)
                megno = sim.calculate_megno()
                energy = sim.calculate_energy()
                delta = phase_delta(d, b)
                conj = conjunction_longitude(do, bo)
                phi1 = resonant_argument_1(do, bo)
                phi2 = resonant_argument_2(do, bo)
                phi3 = resonant_argument_3(do, bo)
                phi32 = resonant_argument_32(do, bo)
                output.append([sim.t, do.a, bo.a, do.e, bo.e, do.inc, bo.inc, do.Omega, bo.Omega,
                               do.pomega, bo.pomega, do.l, bo.l, do.n, bo.n, do.d, bo.d, energy,
                               megno, delta, conj, phi1, phi2, phi3, phi32])
                if (do.d >= 20.) or (bo.d >= 20.):
                    print('ESCAPE!')
                    break
            i += 1
    except rebound.Escape as ex:
        print(ex)
        return None
    except rebound.Collision as ex:
        print(ex)
        return None

    megno = sim.calculate_megno()
    column_names = ['time', 'd_semi', 'b_semi', 'd_ecc', 'b_ecc', 'd_inc', 'b_inc', 'd_Omega', 'b_Omega',
                    'd_pomega', 'b_pomega', 'd_lambda', 'b_lambda', 'd_mm', 'b_mm', 'd_dist', 'b_dist',
                    'energy', 'megno', 'delta', 'conj', 'phi1', 'phi2', 'phi3', 'phi32']
    return pd.DataFrame(data=output, columns=column_names)


def run_simulation(e1, e2, num):
    min_mass1 = float(0.26 * u.earthMass/u.solMass)
    min_mass2 = float(1.07 * u.earthMass/u.solMass)
    obs_inc1 = 133.
    obs_inc2 = 133.
    m1 = min_mass1 / np.sin(obs_inc1)
    m2 = min_mass2 / np.sin(obs_inc2)
    a1 = 0.029
    a2 = 0.049
    i1 = 0.
    i2 = 0.
    Omega1 = 0.
    Omega2 = np.pi
    pomega1 = 0.
    pomega2 = np.pi
    par = (m1, m2, a1, a2, e1, e2, i1, i2, Omega1, Omega2, pomega1, pomega2)
    sim = simulation_with_output(par, step_size=100, num=num)
    sim.to_csv('myr-sim-'+str(e1)+'-'+str(e2)+'.csv', index_label=False)


def main():
    cores = 5
    ecc1 = [0.675, 0.855, 0.486, 0.162, 0.054]
    ecc2 = [0.3, 0.56, 0.58, 0.79, 0.36]
    workers = []
    for i in range(cores):
        workers.append(mp.Process(target=run_simulation, args=(ecc1[i], ecc2[i], i)))
    for w in workers:
        w.start()


if __name__ == '__main__':
    main()
