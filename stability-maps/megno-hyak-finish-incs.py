import rebound
import numpy as np
import pathlib
import astropy.units as u
from multiprocessing import Process


def makefile(file_name, serial_num):
    """
    Creates the data file that will contain all initial conditions and MEGNO
    values.
    """
    global cwd
    global file
    cwd = pathlib.Path()
    file = cwd / file_name
    if file.exists():
        pass
    else:
        if serial_num == 1:
            f = open(file, 'x')
        else:
            name += '_' + str(serial_num)
            file = cwd / name
            f = open(file, 'x')


def checkpoint(file_name):
    """
    If for some reason the simulations do not complete (as has happened to me),
    this function can retrieve the data collected so far.
    """
    cp_data = []
    cp_file = cwd / file_name
    if cp_file.exists():
        with open(cp_file, 'r') as f:
            content = [line.strip().split() for line in f.readlines()]
            for line in content:
                if line:
                    cp_data.append(tuple([float(l) for l in line]))
        return cp_data
    else:
        return []


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
    sim.ri_ias15.min_dt = 1e-4 * sim.dt
    sim.move_to_com()
    sim.init_megno()
    sim.exit_max_distance = 20.
    try:
        sim.integrate(1e6 * b_orbit, exact_finish_time=False) # Integrate ~1 million orbits of Proxima b
        megno = sim.calculate_megno()
        return megno
    except rebound.Escape:
        return 10. # At least one particle got ejected, returning large MEGNO.


def parameter_sweep(core_num):
    """
    Runs a parameter sweep of size Nsim/total_cores simulations.
    """
    i = int(np.floor(core_num/4))
    j = (core_num % 4) * 25
    k = (core_num % 4 + 1) * 25
    # Create data file
    makefile('inc-megno-finish.txt', 1)
    # Collect checkpointed data
    cp_data = checkpoint('inc-megno-cp.txt')
    # Determine the initial conditions for all simulations
    params = []
    inc1 = np.linspace(0., np.pi/20., 5)[:-1][i]
    inc2 = np.linspace(0., np.pi, 100)[j:k]
    min_mass1 = float(0.26 * u.earthMass/u.solMass)
    min_mass2 = float(1.07 * u.earthMass/u.solMass)
    obs_inc1 = 133. * np.pi/180.
    obs_inc2 = 133. * np.pi/180.
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
    i1 = inc1
    for i2 in inc2:
        params.append((m1, m2, a1, a2, e1, e2, i1, i2, Omega1, Omega2, pomega1, pomega2))
    for par in params:
        in_cp_file = False
        # Unpack list of parameters
        m1, m2, a1, a2, e1, e2, i1, i2, Omega1, Omega2, pomega1, pomega2 = par
        # Read in checkpoint data, if available
        for line in cp_data:
            if np.abs(line[6] - i1) <= 1e-3 and np.abs(line[7] - i2) <= 1e-3:
                megno = line[12]
                with open(file, 'a') as f:
                    f.write('{} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(m1, m2, a1,
                            a2, e1, e2, i1, i2, Omega1, Omega2, pomega1, pomega2, megno))
                    in_cp_file = True
                break
        if not in_cp_file:
            # Execute REBOUND simulation
            megno = simulation_whfast(par)
            if megno == 10.:
                # Re-run with IAS15 to rule out numerical error
                megno = simulation_ias15(par)
            # Write output as line in data file (order doesn't matter; all active
            # processes can contribute data simultaneously)
            with open(file, 'a') as f:
                f.write('{} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(m1, m2, a1,
                        a2, e1, e2, i1, i2, Omega1, Omega2, pomega1, pomega2, megno))


def main():
    """
    Runs a full parameter sweep in parallel.
    """
    workers = []
    for i in range(16):
        workers.append(Process(target=parameter_sweep, args=(i,)))
    for w in workers:
        w.start()


if __name__ == '__main__':
    main()
