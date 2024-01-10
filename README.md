# ProximaStability

The code in this repository can be used to reproduce the results of Livesey, Barnes & Deitrick (2024). The following instructions take you through the steps of reproducing all figures and quantitative results in the paper.

## Hill stability

First, compile and execute the Hill stability calculator by running

```
gcc -o proxhillstab prox_hill_stab.c -lm
./proxhillstab
```

This will create two (large) data files containing the input parameters and Hill stability determinations for one million configurations of the system. Then, one may perform the analysis outlined in `Hill-Stability.ipynb`.

## Stability maps

First, the data must be collected by executing `megno-hyak.py`. Please note that these are not computationally inexpensive, especially since a large portion of the simulations will be re-run with a higher-order integrator to limit energy error. This was originally meant for use on the University of Washington's Hyak supercomputer system. Each of the following commands will begin a job will run for 2–3 days on a single CPU.

**Eccentricity dependence.** Running

```
python megno-hyak.py ecc
```

will start 10,000 REBOUND simulations varying the eccentricities of both planets (these simulations are called Set I in the paper).

**Mass dependence.** Running

```
python megno-hyak.py mass
```

will start 10,000 simulations varying the planetary masses (Set II).

**Inclination dependence.** Running

```
python megno-hyak.py inc
```

will start 10,000 simulations varying the planets' orbital inclinations with respect to the fundamental plane of the system (Set III). Once these jobs have completed, there will be three text files containing the relevant data from the 30,000 simulations. Running the cells in `proxima-stability.ipynb` will produce the figures (and more).

## Secular evolution

The six `VPLanet` simulations in `secular-evolution/vplanet-sims` must first be run. Note that these are computationally expensive, and will require multiple weeks of wall time to finish. Then, running the scripts `proxima-b-evols.py` and `proxima-evols.py` will produce in order the secular evolution figures from the paper.

## Using REBOUND

Any difficulties encountered using REBOUND should be addressed by consulting the REBOUND [repository](https://github.com/hannorein/rebound) or [documentation](https://rebound.readthedocs.io/en/latest/). Neither author of this paper is involved in the development or maintenance of this excellent $N$-body code.
