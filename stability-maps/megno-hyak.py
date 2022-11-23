import rebound
import numpy as np
import pathlib
import sys
import multiprocessing as mp
import astropy.units as u


def makefile(name, serial_num):
    """
    Creates the data file that will contain all initial conditions and MEGNO
    values.
    """
    global file
    cwd = pathlib.Path()
    file = cwd / name
    # if file.exists():
    #     makefile(name, serial_num+1)
    if not file.exists():
        if serial_num == 1:
            f = open(file, 'x')
        else:
            name += '_' + str(serial_num)
            file = cwd / name
            f = open(file, 'x')


def simulation_whfast(par):
    """
    Based on an example from the REBOUND docs. Runs a simulation with given
    initial conditions and calculates the MEGNO.
    """
    m1, m2, a1, a2, e1, e2, i1, i2, Omega1, Omega2, pomega1, pomega2 = par # unpack parameters
    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.ri_whfast.safe_mode = False
    sim.add(m=0.12)                      # Proxima Centauri
    sim.add(m=m1, a=a1, e=e1, inc=i1, Omega=Omega1, pomega=pomega1)  # Proxima d
    sim.add(m=m2, a=a2, e=e2, inc=i2, Omega=Omega2, pomega=pomega2)  # Proxima b
    d_orbit = sim.particles[1].P
    b_orbit = sim.particles[2].P
    sim.dt = 0.05 * d_orbit # timestep is 5% of Proxima d's orbital period
    sim.move_to_com()
    sim.init_megno()
    sim.exit_max_distance = 20.
    try:
        sim.integrate(1e6 * b_orbit, exact_finish_time=False) # Integrate ~1 million orbits of Proxima b
        megno = sim.calculate_megno()
        return megno
    except rebound.Escape:
        return 10. # At least one particle got ejected, returning large MEGNO.


def simulation_ias15(par):
    """
    Same as the function simulation_whfast, but uses the IAS15 integrator.
    WHFAST tends not to conserve energy for the unstable configurations in our
    sample. Double-checking with the more accurate IAS15 ensures that the
    instabilities we find are not due to numerical error.
    """
    m1, m2, a1, a2, e1, e2, i1, i2, Omega1, Omega2, pomega1, pomega2 = par # unpack parameters
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.add(m=0.12)                      # Proxima Centauri
    sim.add(m=m1, a=a1, e=e1, inc=i1, Omega=Omega1, pomega=pomega1)  # Proxima d
    sim.add(m=m2, a=a2, e=e2, inc=i2, Omega=Omega2, pomega=pomega2)  # Proxima b
    d_orbit = sim.particles[1].P
    b_orbit = sim.particles[2].P
    sim.dt = 0.05 * d_orbit # timestep is 5% of Proxima d's orbital period
    sim.ri_ias15.min_dt = 1e-3 * sim.dt
    sim.move_to_com()
    sim.init_megno()
    sim.exit_max_distance = 20.
    try:
        sim.integrate(1e6 * b_orbit, exact_finish_time=False) # Integrate ~1 million orbits of Proxima b
        megno = sim.calculate_megno()
        return megno
    except rebound.Escape:
        return 10. # At least one particle got ejected, returning large MEGNO.


def ecc_parameter_sweep(Nsim, core_num, total_cores):
    """
    Runs a parameter sweep of size Nsim/total_cores simulations.
    """
    Nx, Ny = Nsim
    # Create data file
    makefile('ecc-megno-bleh.txt', 1)
    # Determine the initial conditions for all simulations
    params = []
    ecc1 = None
    if core_num == total_cores - 1:
        ecc1 = np.linspace(core_num*0.9/total_cores, 0.9, Nx-core_num*int(np.floor(Nx/total_cores)))
    else:
        ecc1 = np.linspace(core_num*0.9/total_cores, (core_num+1)*0.9/total_cores, int(np.floor(Nx/total_cores)))
    ecc2 = np.linspace(0., 0.9, Ny)
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
    for e1 in ecc1:
        for e2 in ecc2:
            params.append((m1, m2, a1, a2, e1, e2, i1, i2, Omega1, Omega2, pomega1, pomega2))
    for par in params:
        # Initialize simulation, integrate
        m1, m2, a1, a2, e1, e2, i1, i2, Omega1, Omega2, pomega1, pomega2 = par
        megno = simulation_whfast(par)
        if megno == 10.:
            # Re-run with IAS15 to rule out numerical error
            megno = simulation_ias15(par)
        # Write output as line in data file (order doesn't matter; all active
        # processes can contribute data simultaneously)
        with open(file, 'a') as f:
            f.write('{} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(m1, m2, a1,
                    a2, e1, e2, i1, i2, Omega1, Omega2, pomega1, pomega2, megno))


def mass_parameter_sweep(Nsim, core_num, total_cores):
    """
    Runs a parameter sweep of size Nsim/total_cores simulations.
    """
    Nx, Ny = Nsim
    # Create data file
    makefile('mass-megno.txt', 1)
    # Determine the initial conditions for all simulations
    params = []
    mass1 = None
    if core_num == total_cores - 1:
        mass1 = np.linspace(core_num*9.79/total_cores+0.21, 10., Nx-core_num*int(np.floor(Nx/total_cores)))
    else:
        mass1 = np.linspace(core_num*9.79/total_cores+0.21, (core_num+1)*9.79/total_cores+0.21, int(np.floor(Nx/total_cores)))
    mass2 = np.linspace(1.01, 10., Ny)
    mass1 = [float(m1 * u.earthMass/u.solMass) for m1 in mass1]
    mass2 = [float(m2 * u.earthMass/u.solMass) for m2 in mass2]
    a1 = 0.029
    a2 = 0.049
    e1 = 0.
    e2 = 0.
    i1 = 0.
    i2 = 0.
    Omega1 = 0.
    Omega2 = np.pi
    pomega1 = 0.
    pomega2 = np.pi
    for m1 in mass1:
        for m2 in mass2:
            params.append((m1, m2, a1, a2, e1, e2, i1, i2, Omega1, Omega2, pomega1, pomega2))
    for par in params:
        # Initialize simulation, integrate
        m1, m2, a1, a2, e1, e2, i1, i2, Omega1, Omega2, pomega1, pomega2 = par
        megno = simulation_whfast(par)
        if megno == 10.:
            # Re-run with IAS15 to rule out numerical error
            megno = simulation_ias15(par)
        # Write output as line in data file (order doesn't matter; all active
        # processes can contribute data simultaneously)
        with open(file, 'a') as f:
            f.write('{} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(m1, m2, a1,
                    a2, e1, e2, i1, i2, Omega1, Omega2, pomega1, pomega2, megno))


def inc_parameter_sweep(Nsim, core_num, total_cores):
    """
    Runs a parameter sweep of size Nsim/total_cores simulations.
    """
    Nx, Ny = Nsim
    # Create data file
    makefile('inc-megno.txt', 1)
    # Determine the initial conditions for all simulations
    params = []
    inc1 = None
    if core_num == total_cores - 1:
        inc1 = np.linspace(core_num*np.pi/total_cores, np.pi, Nx-core_num*int(np.floor(Nx/total_cores)))
    else:
        inc1 = np.linspace(core_num*np.pi/total_cores, (core_num+1)*np.pi/total_cores, int(np.floor(Nx/total_cores)))
    inc2 = np.linspace(0., np.pi, Ny)
    min_mass1 = float(0.26 * u.earthMass/u.solMass)
    min_mass2 = float(1.07 * u.earthMass/u.solMass)
    obs_inc1 = 133.
    obs_inc2 = 133.
    m1 = min_mass1 / np.sin(obs_inc1)
    m2 = min_mass2 / np.sin(obs_inc2)
    a1 = 0.029
    a2 = 0.049
    e1 = 0.
    e2 = 0.
    Omega1 = 0.
    Omega2 = np.pi
    pomega1 = 0.
    pomega2 = np.pi
    for i1 in inc1:
        for i2 in inc2:
            params.append((m1, m2, a1, a2, e1, e2, i1, i2, Omega1, Omega2, pomega1, pomega2))
    for par in params:
        # Initialize simulation, integrate
        m1, m2, a1, a2, e1, e2, i1, i2, Omega1, Omega2, pomega1, pomega2 = par
        megno = simulation_whfast(par)
        if megno == 10.:
            # Re-run with IAS15 to rule out numerical error
            megno = simulation_ias15(par)
        # Write output as line in data file (order doesn't matter; all active
        # processes can contribute data simultaneously)
        with open(file, 'a') as f:
            f.write('{} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(m1, m2, a1,
                    a2, e1, e2, i1, i2, Omega1, Omega2, pomega1, pomega2, megno))


def main(arg):
    """
    Runs a full parameter sweep in parallel.
    """
    Nsim = (100, 100)
    workers = []
    max_cores = mp.cpu_count()
    # cores = max_cores - max_cores % 5
    cores = max_cores
    for i in range(cores):
        if arg == 'ecc':
            workers.append(mp.Process(target=ecc_parameter_sweep, args=(Nsim, i+1, cores)))
        elif arg == 'mass':
            workers.append(mp.Process(target=mass_parameter_sweep, args=(Nsim, i+1, cores)))
        elif arg == 'inc':
            workers.append(mp.Process(target=inc_parameter_sweep, args=(Nsim, i+1, cores)))
        else:
            print('Invalid keyword: use \'ecc\', \'mass\', or \'inc\'.')
            exit(1)
    for w in workers:
        w.start()


if __name__ == '__main__':
    main(sys.argv[1])
