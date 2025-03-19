'''
Simulation 

Created as part of the Computational Physics course at Leiden University
Authors: Bryce Benz, Liya Charlaganova

Usage:
python MolecularDynamics.py [d/p/c/q]

'''

'''
TODO:
- fix axes, better plot names
- add more colored print?

- remove unnecessary shit
- add docstrings, comments
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.constants
import time
import math
from matplotlib.lines import Line2D
import sys
import os

import scipy.signal


class Simulation:
    ''' Class for a single simulation with set parameters '''
    # TODO: are these constants necesary?

    k_B = scipy.constants.k
    A = 1e-10
    SIGMA = 3.405 * A
    # EPSILON = 119.8 * k_B
    EPSILON = 119.8
    M = 39.948 * scipy.constants.u

    def __init__(self, rho=1, T=100,
                 n_particles=None, box_length=None, dims=3, timesteps=1000, dt=0.001, blocks=3,
                 state=None, method='Verlet', output_dir='plots'):
        ''' 
        Initialize Simulation
        By default, the simulation is setup in 3 dimensional fcc-lattice mode
        with 3x3x3 unit blocks. In this mode, density 'rho' and temperature 'T'
        (both in natural units) are used to determine the size and initial
        conditions of the simulation.
        If n_particles and box_length are specified, a simulation is set up with either
        a random initial particle distribution (for 3D), or a slightly
        randomized grid (for 2D). 
        Initial velicity distribution is determined from the temperature.
        'method' can be Verlet or Euler
        '''

        # Set basic parameters
        self.dims = dims
        self.timesteps = timesteps
        self.dt = dt
        self.T = T
        self.rho = rho
        self.state = state
        self.blocks = blocks

        self.pair_corr_nbins = 100      # Used for pair correlation histogram

        # Set more dynamic parameters
        if n_particles and box_length:  # custom size setup
            self.n_particles = n_particles
            self.box_length = box_length
        elif blocks:                    # FCC setup
            self.blocks = blocks
            self.n_particles = self.blocks ** self.dims * 4
                # set box length box_length to based on the density
            self.box_length = (self.n_particles / self.rho) ** (1/3)
        else:                           # default values (blocks=0 set explicitly)
            self.n_particles = 3
            self.box_length = 6

        if output_dir:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

        self.validate_init()

        if method == 'Euler':
            self.method_step = self.Euler_step
        else:
            if method != 'Verlet':
                print('Unclear method, default to Verlet...')
            self.method_step = self.Verlet_step
        
        self.setup()

    def validate_init(self):
        ''' Check if init parameters are valid. '''
        conditions = {
            "dims must be 2 or 3": self.dims not in [2, 3],
            "box_length must be non-negative": self.box_length < 0,
            "n_particles must be non-negative": self.n_particles < 0,
            "rho must be non-negative": self.rho < 0,
            "T must be non-negative": self.T < 0,
            "blocks must be non-negative": self.blocks < 0,
            "timesteps must be >= dt": self.timesteps < self.dt,
            "dt must be non-negative": self.dt < 0
        }

        errors = [msg for msg, condition in conditions.items() if condition]
        if errors:
            print("Error(s) detected:\n" + "\n".join(errors))
            sys.exit(1)


    def setup(self):
        ''' Create arrays for storing simulation states '''
        self.positions = np.zeros([self.timesteps, self.n_particles, self.dims])
        self.velocities = np.zeros([self.timesteps, self.n_particles, self.dims])
        self.forces =  np.zeros([self.n_particles, self.dims])
        self.forces_prev =  np.zeros([self.n_particles, self.dims])
        self.energies_pot = np.zeros(self.timesteps)
        self.energies_kin = np.zeros(self.timesteps)
        self.energies_tot = np.zeros(self.timesteps)
        self.distance_hist = np.zeros([self.timesteps, self.pair_corr_nbins])
        self.pair_correlation_g = np.zeros(self.pair_corr_nbins)
        self.pressures = np.zeros(self.timesteps)
        self.final_pressure = 0
        self.ti_initial = 0
        self.current_dx = np.zeros([self.n_particles, self.n_particles])
        self.current_r = np.zeros([self.n_particles, self.n_particles])

    def cleanup(self):
        ''' Delete unnecessary temporary arrays '''
        self.forces = []
        self.forces_prev = []
        self.distance_hist =[]
        self.pressures = []
        self.current_dx = []
        self.current_r = []

    def set_initial_positions_FCC(self):
        ''' 
        Create a fcc lattice based on size box_length and the number of unit blocks
        and set it as the initial conditions. Only works in 3D
         
        '''
        unit_block = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]) + 0.001
        shift_lattice = []
        for i in range(self.blocks):
            for j in range(self.blocks):
                for k in range(self.blocks):
                    shift_lattice.append(np.tile([i, j, k], (4, 1)))
        shift_lattice = np.vstack(shift_lattice)
        lattice = (np.tile(unit_block, (self.blocks ** 3, 1)) + shift_lattice)
        self.positions[0, :, :] = lattice * self.box_length / self.blocks

    def set_initial_2Dgrid(self):
        ''' 
        Set initial conditions in 2D. Random positions are picked
        from a grid which ensures that initial positions are not too 
        close to each other. 
        '''
        # Determine the number of allowed particles based on a minimal 
        # distance of 1.5
        pos_available_1d = (self.box_length // 1.5)
        pos_available = int(pos_available_1d ** 2)

        if self.n_particles > pos_available:
            msg = f"Too many particles, default to {pos_available}"
            print("\033[91m {}\033[00m".format(msg))
            self.n_particles = pos_available
            self.setup()
        # Choose random positions in grid and translate to coordinates
        rnd_pos = np.random.choice(pos_available, size=self.n_particles, replace=False)
        self.positions[0,:,:] = np.array([rnd_pos // pos_available_1d, rnd_pos % pos_available_1d]).transpose() * 1.5 + 1
        # self.positions += np.random.normal(0, 0.2, [self.n_particles, self.dims])


    def set_initial_velocities(self):
        ''' Set initial velocity distribution from a Gaussian distribution
        with mean = 0 and a variance T**2.'''
        self.velocities[0, :, :] = np.random.normal(0, np.sqrt(self.T), [self.n_particles, self.dims])

    def set_initial_conditions(self):
        '''
        Wrapper function to set conditions based on dimensionality of
        the simulation.
        '''
        if self.blocks:
            self.set_initial_positions_FCC()
        elif self.dims == 2:
            self.set_initial_2Dgrid()
        else:
            self.positions[0,:,:] = np.random.uniform(0, self.box_length, [self.n_particles, self.dims])
            # self.velocities[0,:,:] = np.random.normal(0, 0.2, [self.n_particles, self.dims])
        self.set_initial_velocities()

    def set_current_dx(self, pos0):
        '''
        Set current_dx, an n_particles x n_particles x dims array,
        to hold the distances (in all dimensions) between particles
        based on positions pos0.
        '''
        x1 = np.repeat([pos0], repeats=self.n_particles, axis=0)
        x2 = np.transpose(x1, axes=[1,0,2])
        self.current_dx = (x2 - x1 + self.box_length/2) % self.box_length - self.box_length/2
    
    def calc_current_r(self):
        ''' TODO update
        Calculate current_r, an n_particles x n_particles array, to hold
        the absolute distances between particles based on 
        current_dx.
        (For proper function, you need to call set_current_dx first)
        '''
        dx = self.current_dx
        return np.sqrt(np.sum(dx * dx, axis=2))

    def measure_gradU(self, r):
        ''' Measure the potential gradient based on current positions. '''
        # r = self.current_r
        with np.errstate(divide='ignore', invalid='ignore'):
            grad_U = np.nan_to_num((4 * ( -12 * r ** (-13) + 6 * r ** (-7))), nan=0)
        return grad_U
    
    def measure_Epot(self, r):
        ''' Measure the potential energy based on current positions. '''
        r = self.current_r
        with np.errstate(divide='ignore', invalid='ignore'):
            U = np.nan_to_num(4 * (r ** (-12) - r ** (-6)), nan=0)
        return np.sum(U) / 2

    def measure_Ekin(self, vel):
        ''' Measure the kinetic energy based on current velocities. '''
        return 0.5 * np.sum(vel * vel)

    def measure_current_forces(self, r):
        '''
        Set forces, an n_particles x dims array, to hold the forces
        that each particle experiences in each direction based on
        the current_dx and current_r
        '''
        dx = self.current_dx
        grad_U = self.measure_gradU(r)
        force_mat = -1 * dx * np.repeat(grad_U[:, :, np.newaxis], repeats=self.dims, axis=2)
        self.forces = force_mat.sum(axis=1)

    def measure_pressure(self, r):
        ''' 
        Measure the current pressure of the system based on 
        current_dx and current_r.
        '''
        # r = self.current_r
        with np.errstate(divide='ignore', invalid='ignore'):
            a = np.nan_to_num(((-12) * r**(-12) + 6 * r**(-6)), nan=0) # betwen brackets part
        return self.rho * (self.T  - np.sum(a) / (3 * self.n_particles))

    def measure_correlation(self, ti):
        ''' Measure and set the pair correlation histogram at a time ti. '''
        r = self.current_r

        hist, _ = np.histogram(r[np.tril_indices_from(r)], bins=self.pair_corr_nbins, range=[0.01, self.box_length/2])
        self.distance_hist[ti] = hist
    
    def rescale_velocities(self, ti, Ekin_target):
        ''' Rescale velocities in order to have E_kin = Ekin_target'''
        scale = np.sqrt(Ekin_target / self.measure_Ekin(self.velocities[ti]))
        # scale = self.relaxation_scale(self.velocities[ti])
        self.velocities[ti] =  self.velocities[ti] * scale

    def relaxation_loop(self, ti, relax_steps, init_relax_steps):
        ''' TODO '''
        if ti == 0:
            relax_steps = init_relax_steps

        for i, (pos, vel) in enumerate(zip(self.positions[ti:ti+relax_steps], self.velocities[ti:ti+relax_steps])):
            self.method_step(ti+i, pos, vel)
        return ti + relax_steps

    def velocity_relaxation(self):
        ''' TODO '''
        Ekin_target = (self.n_particles - 1) * (3 / 2) * self.T
        ti = 0
        E_diff = 1e6

        relax_steps = 75
        init_relax_steps = 300
        E_err = Ekin_target * 0.05
        
        while E_diff > E_err:
            if ti >= (self.timesteps - relax_steps):
                print("No convergence")
                E_diff = 0
            else:
                ti = self.relaxation_loop(ti, relax_steps, init_relax_steps)
                self.rescale_velocities(ti, Ekin_target)
                E_diff = np.abs(self.energies_kin[ti-1] - Ekin_target)

        self.ti_initial = ti

    def Euler_step(self, ti, pos0, vel0):
        ''' 
        Calculate the new positions and velocities at step ti + 1
        based on previous positions pos0 and velocities vel0 using
        the Euler method.
        '''
        MAX = 1e3
        self.set_current_dx(pos0)
        r = self.calc_current_r()

        self.energies_kin[ti] = self.measure_Ekin(vel0)
        self.energies_pot[ti] = self.measure_Epot(r)
        self.pressures[ti] = self.measure_pressure(r)
        self.measure_correlation(ti)

        self.measure_current_forces(r)

        self.positions[ti+1, :, :] = (pos0 + vel0 * self.dt) % self.box_length
        self.velocities[ti+1,: , :] = np.clip(vel0 + self.forces * self.dt / self.m, a_min=-MAX, a_max=MAX)


    def Verlet_step(self, ti, pos0, vel0):
        ''' 
        Calculate the new positions and velocities at step ti + 1
        based on previous positions pos0 and velocities vel0 using
        the Verlet method.
        '''
        MAX = 1e3

        if ti == 0:
            self.set_current_dx(pos0)
            self.current_r = self.calc_current_r()
            self.measure_current_forces(self.current_r)
            self.forces_prev = self.forces
        r = self.current_r
        self.energies_kin[ti] = self.measure_Ekin(vel0)
        self.energies_pot[ti] = self.measure_Epot(r)
        self.pressures[ti] = self.measure_pressure(r)
        self.measure_correlation(ti)

        newpos = (pos0 + vel0 * self.dt + self.dt ** 2 / 2 * self.forces_prev) % self.box_length
        self.set_current_dx(newpos)
        self.current_r = self.calc_current_r()
        self.positions[ti+1, :, :] = newpos

        self.measure_current_forces(self.current_r)

        F_tmp = self.forces
        self.velocities[ti+1, :, :] = np.clip(vel0 + self.dt / 2 * (F_tmp + self.forces_prev), a_min=-MAX, a_max=MAX)
        self.forces_prev = F_tmp
        

    def calculate_pair_corr(self):
        '''
        Calculate the final pair correlation function g(r) based on all
        distance histograms after the system has relaxed.
        '''

        r = np.linspace(0.01, self.box_length/2, self.pair_corr_nbins)
        avg = np.average(self.distance_hist[self.ti_initial:], axis=0)
        V = self.box_length ** self.dims
        dr = self.box_length / (2 * self.pair_corr_nbins)

        g =  avg * 2 * V / (self.n_particles * (self.n_particles - 1) * 4 * np.pi * r ** 2 * dr)
        self.pair_correlation_g = g

    def determine_state(self):
        '''
        Determine the state of matter based on the number of peaks in the
        the final pair correlation function pair_correlation_g.
        '''
        peaks, _ = scipy.signal.find_peaks(self.pair_correlation_g,
                                prominence=0.2)
        
        if len(peaks) >= 4:
            self.state = 'solid'
        elif len(peaks) <= 1:
            self.state = 'gas'
        else:
            self.state = 'liquid'
        return peaks
    
    def check_output_dir(self):
        ''' Create output_dir if it does not yet exist. '''
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_positions_2d(self, ax):
        '''
        Plot the last STEPS (default 500) of the simulation in 2D 
        of this simulation instance.
        If 'ax' is provided, the data is plotted onto that axis. Otherwise
        a new figure is created and saved.
        '''
        STEPS = 500
        if STEPS > self.timesteps:
            STEPS = self.timesteps
        save = False

        if not ax:
            plt.figure()
            ax = plt.gca()
            plt.title('Simulation evolution of last {STEPS} steps\n' +
                r'$\rho$' + f' = {self.rho}, T = {self.T}, dt = {self.dt}')
            save = True
        else:
            ax.set_title(f"{self.state}: " * int(self.state) + r"$\rho$" + f" = {self.rho}, T = {self.T}")

        for ni in range(self.n_particles):
            ax.scatter(self.positions[-STEPS:, ni, 0], self.positions[-STEPS:, ni, 1], marker='.', s=2)
            ax.scatter(self.positions[-1, ni, 0], self.positions[-1, ni, 1], c='k',
                zorder=5)
        ax.set_xlim(0,self.box_length)
        ax.set_ylim(0,self.box_length)
        ax.set_xlabel('Position x [n.u.]')
        ax.set_ylabel('Position y [n.u.]')

        if save:
            plt.savefig(f'{self.output_dir}/Positions.png')

    def plot_positions_3d(self, ax=None):
        '''
        Plot the last STEPS (default 500) of the simulation in 3D 
        of this simulation instance.
        If 'ax' is provided, the data is plotted onto that axis. Otherwise
        a new figure is created and saved.
        '''
        STEPS = 500
        if STEPS > self.timesteps:
            STEPS = self.timesteps
        save = False
        if not ax:
            plt.figure()
            ax = plt.axes(projection ='3d')
            plt.title(r"$\rho$" + f" = {self.rho}, T = {self.T}" +
                  f", dt = {self.dt}, steps = {self.timesteps}")
            save = True
        else:
            ax.set_title(f"{self.state}: " + r"$\rho$" + f" = {self.rho}, T = {self.T}")

        for ni in range(self.n_particles):
            with plt.rc_context({'lines.markersize': 2}):
                ax.scatter(self.positions[-STEPS:, ni, 0], self.positions[-STEPS:, ni, 1], self.positions[-STEPS:, ni, 2],
                        marker='.', alpha=0.02)
            ax.scatter(self.positions[-1, ni, 0], self.positions[-1, ni, 1], self.positions[-1, ni, 2],
                        marker='.', alpha=0.8, c='k')
        ax.set_xlim(0, self.box_length)
        ax.set_ylim(0, self.box_length)
        ax.set_zlim(0, self.box_length)

        ax.set_xlabel('Position x [n.u.]')
        ax.set_ylabel('Position y [n.u.]')
        ax.set_zlabel('Position z [n.u.]')

        if save:
            plt.savefig(f'{self.output_dir}/Positions.png')

    def plot_positions(self, ax=None):
        '''
        Wrapper function to call proper position plotting function.
        If 'ax' is provided, the data is plotted onto that axis. Otherwise
        a new figure is created and saved.
        '''

        if self.dims == 2:
            self.plot_positions_2d(ax)
        else:
            self.plot_positions_3d(ax)
    
    def plot_energies(self, ax=None):
        '''
        Plot the evolution of the total energy, kinetic energy and potential energy
        of this simulation instance.
        If 'ax' is provided, the data is plotted onto that axis. Otherwise
        a new figure is created and saved.
        '''

        E_target = (self.n_particles - 1) * (3 / 2) * self.T
        t = np.arange(0, self.timesteps*self.dt, self.dt)[:-1]
        save=False

        if not ax:
            plt.figure()
            ax = plt.gca()
            save = True
            plt.suptitle("Energy evolutions")
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.plot(t, self.energies_pot[:-1], label=r"$E_{pot}$")
        ax.plot(t, self.energies_kin[:-1], label=r"$E_{kin}$")
        ax.plot(t, self.energies_tot[:-1], label=r"$E_{tot}$")
        ax.axhline(E_target, c='orange', ls='--', label=r"$E_{kin-target}$")
        ax.axvline(t[self.ti_initial], c='k', ls='--', label='System relaxed')
        
        ax.set_xlabel('Time [n.u.]')
        ax.set_ylabel('Energy [n.u.]')

        ax.set_title(r"$\rho$" + f' = {self.rho}, T = {self.T},' +
                     f' dt = {self.dt}, steps = {self.timesteps}' +
                     (self.state == True) * f'({self.state})')

        if save:
            plt.savefig(f'{self.output_dir}/Energies.png')
            # print('Eplot.png')

    def plot_pair_correlation(self, ax=None):
        '''
        Plot the pair-correlation function g(r) of this simulation instance.
        If 'ax' is provided, the data is plotted onto that axis. Otherwise
        a new figure is created and saved.
        '''
        r = np.linspace(0.01, self.box_length / 2, self.pair_corr_nbins)
        save = False
        if not ax:
            plt.figure()
            plt.title(r"$\rho$" + f" = {self.rho}, T = {self.T}, dt = {self.dt}, steps = {self.timesteps}")
            ax = plt.gca()
            save = True
            label = ''

        else:   # axis provided
            label = fr"$\rho$: {self.rho}, T = {self.T}"
            if self.state: 
                label += f" ({self.state})"                 
            
        ax.plot(r, self.pair_correlation_g, label=label)
        peaks = self.determine_state()
        ax.scatter(r[peaks], self.pair_correlation_g[peaks])
        ax.set_xlabel('Distance [n.u.]')
        ax.set_ylabel('Pair correlation g(r)')

        if save:
            plt.savefig(f"{self.output_dir}/Pair_corr.png")

        
    def run_simulation(self, print_progress=True):
        self.set_initial_conditions()

        start = time.perf_counter()
        self.velocity_relaxation()

        for i, (pos0, vel0) in enumerate(zip(self.positions[self.ti_initial:-1], self.velocities[self.ti_initial:-1])):
            ti = self.ti_initial + i
            self.method_step(ti, pos0, vel0)
            if print_progress:
                if (ti) % 100 == 0:
                    # print(f"{ti} steps")
                    print(f"\rProgress: {(100 * ti / (self.timesteps-100)):.0f}%", end="")
        end = time.perf_counter()
        if print_progress:
            print("", end="\r")
            print(f"Speed: {(self.timesteps / (end-start)):.0f} steps / second")
        
        self.energies_tot = self.energies_kin + self.energies_pot
        self.final_pressure = np.average(self.pressures[self.ti_initial:])
        self.calculate_pair_corr()
        self.cleanup()

class SimulationBatch:
    def __init__(self, dt=0.001, timesteps=1000, states='', 
                 params=[], repeats=1, plot=True, method='Verlet',
                 summary_file='summary', output_dir='plots'):
        self.states = states
        self.dt = dt
        self.timesteps = timesteps
        self.repeats = repeats
        self.saved_sims = []
            # if plot == 'force', all saved simulations will be plotted
            # which can lead to huge plots, so not recommended
        self.plot = plot
        self.params = params
        self.method = method
        self.summary_file = f"{summary_file}.txt"
        self.output_dir = output_dir

        self.write_summary()

    def write_summary(self):
        '''
        Write simulation summary in summary_file:
            simulation parameters (timesteps, dt)
            rho, T and resulting pressure
        '''
        with open(self.summary_file, "w") as file:
            file.write("Parameters for simulation(s):\n")
            file.write(f"Method: {self.method}, steps: {self.timesteps}, dt = {self.dt}\n")
            file.write(f"{self.repeats} simulations per configuration\n\n")

    def default_params(self, state):
        ''' Return rho, T for default states gas 'g', liquid 'l' and solid 's'. '''
        if state == 'g':
            return 0.3, 3
        if state == 'l':
            return 0.8, 1
        if state == 's':
            return 1.2, 0.5

    def run_simulation(self, rho, T, state=''):
        if state:
            print(f"Running {state}... [{self.repeats} sim(s)]")
        else:
            print(f"Running rho = {rho}, T = {T}... [{self.repeats} sim(s)]")
        pressures = np.zeros(self.repeats)
        pair_correlations = []
        for i in range(self.repeats):
            sim = Simulation(rho=rho, T=T, dt=self.dt, timesteps=self.timesteps, state=state,
                             method=self.method, output_dir=self.output_dir)
            sim.run_simulation()
            pressures[i] = sim.final_pressure
            pair_correlations.append(sim.pair_correlation_g)

        # Use final simulation to store total
        sim.pair_correlation_g = np.mean(np.array(pair_correlations), axis=0)
        
        self.P = sim.P = np.mean(pressures)
        self.P_std = sim.P_std = np.std(pressures)
        sim.determine_state()       # determine state from combined g(r)

        roundoff = 3
        if self.P_std != 0:
            roundoff = int(-math.floor(math.log10(abs(self.P_std))))
        # self.P = sim.P = round(self.P, roundoff)
        # self.P_std = sim.P_std = round(self.P_std, roundoff)

        self.saved_sims.append(sim)

        with open(self.summary_file, "a") as file:
            # file.write(f'{self.P:.{roundoff}f}')
            file.write(f'rho = {rho}, T = {T}, ' +
                       f'P = {self.P:.{roundoff}f} +/- {self.P_std:.{roundoff}f} ' +
                        f'({sim.state})\n')

    def run_states(self):
        '''
        Run simulations for default states gas 'g', liquid 'l' and solid 's'
        if these have been provided in the 'states' variable.
        '''
        if 'g' in self.states:
            rho, T = self.default_params('g')
            self.run_simulation(rho, T, state='gas')
        if 'l' in self.states: 
            rho, T = self.default_params('l')
            self.run_simulation(rho, T, state='liquid')
        if 's' in self.states:
            rho, T = self.default_params('s')
            self.run_simulation(rho, T, state='solid')

    def run_params(self):
        ''' Run multiple parameters rho, T TODO '''
        for i, (rho, T) in enumerate(self.params):
            print(f"Running configuration {i+1} / {len(self.params)}")
            self.run_simulation(rho, T)
        
    def plot_correlation(self):
        ''' 
        Plot the pair- correlation function for all saved simulations 
        in one figure. 
        '''
        plt.figure()
        ax = plt.gca()
        for i, sim in enumerate(self.saved_sims):
            sim.plot_pair_correlation(ax)
        ax.legend()
        plt.title(f'Pair correlation function with dt = {self.dt}')
        plt.savefig(f"{self.output_dir}/Pair_corr.png")

    def plot_positions(self):
        ''' Plot the positions for all saved simulations in one figure. '''
        n_sims = len(self.saved_sims)

        fig = plt.figure(figsize=[5*n_sims, 5])

        for i, sim in enumerate(self.saved_sims):
            ax = fig.add_subplot(1, n_sims, i+1,  projection='3d')
            sim.plot_positions(ax)
        plt.suptitle(f"Simulation evolution for dt = {self.dt}")
        plt.savefig(f"{self.output_dir}/Positions.png")
    
    def plot_energy(self):
        ''' Plot the energy evolutions for all saved simulations in one figure. '''
        n_sims = len(self.saved_sims)

        fig = plt.figure(figsize=[6.5, 4*n_sims])
        for i, sim in enumerate(self.saved_sims):
            ax = fig.add_subplot(n_sims, 1, i+1)
            sim.plot_energies(ax)
            if i == 0:
                ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.suptitle("Energy evolutions", fontsize=18)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/Energies.png')

    def plot_phase_space(self):
        ''' Plot the phase space of simulations provided in 'params'. '''
        mrk = {
            'gas': {'marker': 'o', 'color': 'gold'},   # Circle, Blue
            'liquid': {'marker': '^', 'color': 'deepskyblue'},  # Triangle up, Green
            'solid': {'marker': 's', 'color': 'tomato'},    # Square, Red
        }
        handleg = Line2D([0], [0], label='gas', marker=mrk['gas']['marker'], color=mrk['gas']['color'], linestyle='')
        handlel = Line2D([0], [0], label='liquid', marker=mrk['liquid']['marker'], color=mrk['liquid']['color'], linestyle='')
        handles = Line2D([0], [0], label='solid', marker=mrk['solid']['marker'], color=mrk['solid']['color'], linestyle='')

        plt.figure()
        for sim in self.saved_sims:
            # if sim.final_pressure > 0:
            plt.scatter(sim.rho, sim.T, marker=mrk[sim.state]['marker'],
                            color=mrk[sim.state]['color'])
        plt.xlabel('Density [n.u.]')
        plt.ylabel('Temperature [n.u.]')
        plt.title('Phase space of Argon')
        plt.legend(handles=[handleg, handlel, handles])
        plt.savefig(f'{self.output_dir}/Phase_space.png')

    def simulate(self):
        '''
        Wrapper function to run the default states (g,l,s) if provided and
        run custom rho, T values stored in params.
        '''
        if self.states:
            self.run_states()
        if self.params:
            self.run_params()
        if self.plot:
            print("Plotting results...")
            if len(self.saved_sims) <= 5 or (self.plot == 'force'):
                self.plot_positions()
                self.plot_correlation()
                self.plot_energy()
            if len(self.saved_sims) > 1 or (self.plot == 'force'):
                self.plot_phase_space()
        print(f"Summary written to {self.summary_file}")
        print(f"Plots stored in {self.output_dir}")

def quit():
    ''' Exit program '''
    print('Goodbye!')
    sys.exit(0)

def demo():
    '''
    Run the simulation demo which performs simulations of the three
    states of argon.
    '''
    print('Running demo...')
    batch = SimulationBatch(repeats=1, dt=0.001, timesteps=1500, states='gls', plot=True)
    batch.simulate()

def check_input(inpt):
    ''' Test whether (user input) inpt is a number '''
    try:
        inpt = float(inpt)
        return inpt
    except:
        print('Input must be an number')
        quit()

def params():
    '''
    Run simulations for parameters rho, T that will be provided by the
    user. User can also run multiple simulations for this configuration
    and obtain error margins.
    '''
    print('Please input desired density rho (in n.u.):')
    rho = check_input(input())
    print('Please input desired density T (in n.u.):')
    T = check_input(input())
    print("How many simulations do you want?")
    repeats = int(check_input(input()))

    batch = SimulationBatch(repeats=repeats, dt=0.001, timesteps=1000, params=[[rho, T]], plot=True)
    batch.simulate()
    print(f'Pressure: {batch.P}' + int(batch.P_std) * f' +/- {batch.P_std}')

def custom():
    ''' Run custom code to be specified below: '''

    print('Running custom...')
    # sim = Simulation(n_particles=3, box_length=1, dt=0.001, timesteps=1000, dims=2, blocks=2, T=1)
    # sim = Simulation(n_particles=108, dt=0.001, timesteps=1000, dims=3, blocks=3, rho=1.2, T=0.5)
    # sim = Simulation(rho=0.5, T=3, dt=0.001, timesteps=1000, blocks=3)
    
    # sim = Simulation(rho=0.8, T=1, dt=0.001, timesteps=1000)
    # sim = Simulation(T=0.35, dt=0.001, timesteps=1000, dims=2,
                    #  n_particles=30, box_length=5)

    # sim.run_simulation()
    # sim.save_sim()
    # sim.load_positions()
    # print(sim.positions[0])

    # print(sim.final_pressure)

    # sim.plot_positions()
    # sim.plot_energies()
    # sim.plot_pair_correlation()

    params = []

    for i, rho in enumerate(np.linspace(0.1, 2, 8)):
    # rho = 0.85
        for j, T in enumerate(np.linspace(0.1, 2, 8)):
            # if i + j < 10:
            params.append(np.round([rho, T], decimals=3))

    batch = SimulationBatch(repeats=1, dt=0.01, timesteps=2000, states='', plot=True,
                            method='Verlet', params=params)
    # batch = SimulationBatch(repeats=1, dt=0.01, timesteps=2000, states='', plot=True,
                            # method='Verlet', params=[[0.3, 3]])
    # batch = SimulationBatch(repeats=10, dt=0.001, timesteps=1500, states='l', plot=True,
    #                         method='Verlet')
    batch.simulate()
def select_option(opt):
    ''' Wrapper function for running a valid option 'opt' '''
    if opt == 'q':
        quit()
    elif opt == 'd':
        demo()
    elif opt == 'c':
        custom()
    else:
        params()
    quit()

def main():
    '''
    Run MolecularDynamics.py simulation. User is prompted with 4 options:
        q: quit
        d: run demo
        p: run for rho, T as specified by user
        c: run custom code
    If program is called with an additional valid argument, that option
    is run immediately without prompting the user.
    '''

    if len(sys.argv) > 1:
        if sys.argv[1] in  ['d', 'p', 'c', 'q']:
            select_option(sys.argv[1])
            quit()

    print('''Welcome to Molecular Dynamics simulation of Argon!
    Authors: Bryce, Liya''')
    opt=False
    while not opt:
        print('''Options:
    Demo: 'd'
    Parameter (T,rho) input: 'p'
    Custom code: 'c'
    Exit: 'q'
            ''')
        opt = input()
        if opt not in ['d', 'p', 'c', 'q']:
            opt=False
            print('Invalid input')
    select_option(opt)

if __name__ == "__main__":
    main()
