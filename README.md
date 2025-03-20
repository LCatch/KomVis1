MolecularDynamics.py
Authors: Liya Charlaganova, Bryce Benz
Date: 20/03/2025

Usage:
python MolecularDynamics.py [d/p/c/q (optional)]

This code will simulate the evolution of Argon particles in a box at
some temperature T and density rho. There are three basic options
for running the code:
    Option 'd' (demo) will run the simulation for three default values
    which reproduces the plots discussed in the report (except for
    the triple point plots). Initial state is an fcc lattice
                    rho     T
            --------------------
            gas :   0.3     3
            liquid: 0.8     1
            solid:  1.2     0.5
        at dt = 0.001, timesteps = 1500

    Option 'p' (params) will allow the user to input custom values for
    rho and T for the simulation and choose how many times to run
    the simulation

    Option 'c' (custom) will run the custom code block in MolecularDynamics.py.

    (Option 'q' (quit))

The code allows for a lot of customization. The SimulationBatch class is used
to run multiple simulations of Argon in an FCC lattice. The following
customization is available:
    - propagation method: Verlet (default), Euler
    - Custom list of rho, T values
    - Default solid/liquid/gas values
    - Size and number of steps ('dt', 'timesteps')

The Simulation() class allows for much more customization, but will only run
1 simulation. This class is useful to test certain aspects of the simulation.
The following customization is available:
    - rho, T, dt, timesteps, method
    - Number of unit cells of FCC lattice
    - Number of dimenstions of the simulation (2D/3D)
    - Number of particles
    - Size of simulation (box length)
Contradictory input parameters are resolved (for example, if the 
simulation is initialized in 2 dimensions with some number of
blocks for FCC, the simulation will default to 2D with a default
number of particles.)