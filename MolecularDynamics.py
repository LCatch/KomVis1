'''
Simulation 

Created as part of the Computational Physics course at Leiden University
Authors: Bryce Benz, Liya Charlaganova
'''

'''
TODO:
- fix axes
- do the os thing that stores the latest simulation?
- add folder for plots, better plot names

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

import scipy.signal


class Simulation:
    # TODO: are these constants necesary?

    k_B = scipy.constants.k
    A = 1e-10
    SIGMA = 3.405 * A
    # EPSILON = 119.8 * k_B
    EPSILON = 119.8
    M = 39.948 * scipy.constants.u

    def __init__(self, rho=1, T=100,
                 n_particles=108, L=0, dims=3, timesteps=1000, dt=0.001, blocks=3,
                 state=None, method='Verlet'):
        # self.m = self.M
        # self.sigma = self.SIGMA
        # self.epsilon = self.EPSILON

        self.dims = dims
        self.timesteps = timesteps
        self.dt = dt
        self.T = T
        self.rho = rho
        self.state = state
        self.blocks = blocks

        self.r_range = 100      # change name
        self.m = 1
        # self.sigma = 1
        # self.epsilon = 1

        # self.kBT = T / self.EPSILON
        self.kBT = T    # fix to just T, kbT never needed ... 

        if n_particles:
            self.n_particles = n_particles
        else:
            self.n_particles = self.blocks ** self.dims * 4

        if L:
            self.L = L
        else:
            self.L = (self.n_particles * self.m / self.rho) ** (1/3) 

        if method == 'Euler':
            self.method_step = self.Euler_step
        else:
            if method != 'Verlet':
                print('Unclear method, default to Verlet...')
            self.method_step = self.Verlet_step

        self.setup()

    def setup(self):
        self.positions = np.zeros([self.timesteps, self.n_particles, self.dims])
        self.velocities = np.zeros([self.timesteps, self.n_particles, self.dims])
        self.forces =  np.zeros([self.n_particles, self.n_particles, self.dims])
        self.force_prev =  np.zeros([self.n_particles, self.dims])
        self.energies_pot = np.zeros(self.timesteps)
        self.energies_kin = np.zeros(self.timesteps)
        self.energies_tot = np.zeros(self.timesteps)
        self.distance_hist = np.zeros([self.timesteps, self.r_range])
        self.pair_correlation_g = np.zeros(self.r_range)
        self.pressures = np.zeros(self.timesteps)
        self.final_pressure = 0
        self.ti_initial = 0
        self.current_dx = np.zeros([self.n_particles, self.n_particles])
        self.current_r = np.zeros([self.n_particles, self.n_particles])

    def cleanup(self):
        self.forces = []
        self.force_prev = []
        self.distance_hist =[]
        self.pressures = []
        self.current_dx = []


    def set_initial_positions_FCC(self):
        unit_block = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        shift_lattice = []
        for i in range(self.blocks):
            for j in range(self.blocks):
                for k in range(self.blocks):
                    shift_lattice.append(np.tile([i, j, k], (4, 1)))
        shift_lattice = np.vstack(shift_lattice)
        lattice = (np.tile(unit_block, (self.blocks ** 3, 1)) + shift_lattice)
        self.positions[0, :, :] = lattice * self.L / self.blocks

    def set_initial_2Dgrid(self):
        ''' 2D solution!! doesn't work in 3D...'''
        pos_available_1d = (self.L // 1.5)
        pos_available = int(pos_available_1d ** 2)
        if self.n_particles > pos_available:
            msg = f"Too many particles, default to {pos_available}"
            print("\033[91m {}\033[00m".format(msg))
            self.n_particles = pos_available
            self.re_init()

        rnd_pos = np.random.choice(pos_available, size=self.n_particles, replace=False)
        self.positions[0,:,:] = np.array([rnd_pos // pos_available_1d, rnd_pos % pos_available_1d]).transpose() * 1.5 + 1
        self.positions += np.random.normal(0, 0.2, [self.n_particles, self.dims])


    def set_initial_velocities(self):
        self.velocities[0, :, :] = np.random.normal(0, np.sqrt(self.kBT), [self.n_particles, self.dims])

    def set_initial_conditions(self):
        if self.blocks and self.dims == 3:
            self.set_initial_positions_FCC()
            
        # elif self.n_particles == 2:
        #     self.positions[0,0,:] = [1, 1]
        #     self.positions[0,1,:] = [1, 2.5]
        #     self.velocities[0,0,:] = [1, 0]
        #     self.velocities[0,1,:] = [-1, 0]
        elif self.dims == 2:
            self.set_initial_2Dgrid()
        else:
            self.positions[0,:,:] = np.random.uniform(0, self.L, [self.n_particles, self.dims])
            # self.velocities[0,:,:] = np.random.normal(0, 0.2, [self.n_particles, self.dims])
        self.set_initial_velocities()

    def measure_gradU(self):
        r = self.current_r
        with np.errstate(divide='ignore', invalid='ignore'):
            grad_U = np.nan_to_num((4 * ( -12 * r**(-13) + 6 * r ** (-7))), nan=0)
        return grad_U
    
    def measure_Epot(self):
        r = self.current_r
        with np.errstate(divide='ignore', invalid='ignore'):
            U = np.nan_to_num(4 * (r ** (-12) - r ** (-6)), nan=0)
        return np.sum(U) / 2

    def measure_Ekin(self, vel):
        return 0.5 * self.m * np.sum(vel * vel)
    
    def relaxation_loop(self, ti, relax_steps):
        # relax_steps = 50
        if ti == 0:
            relax_steps = 500

        for i, (pos, vel) in enumerate(zip(self.positions[ti:ti+relax_steps], self.velocities[ti:ti+relax_steps])):
            self.method_step(ti+i, pos, vel)
        return ti + relax_steps
    
    def relaxation_scale(self, vel):
        return np.sqrt((self.n_particles-1) * 3 * self.kBT / (np.sum(vel * vel) * self.m) )

    def rescale_velocities(self, ti):
        scale = self.relaxation_scale(self.velocities[ti])
        self.velocities[ti] =  self.velocities[ti] * scale

    def velocity_relaxation(self):
        Ekin_target = (self.n_particles - 1) * (3 / 2) * self.kBT
        relax_steps = 500
        E_err = Ekin_target * 0.05
        ti = 0

        E_diff = 1e6
        # E_diff = 0      # TODO fix relaxation
        while E_diff > E_err:
            if ti >= (self.timesteps - relax_steps): # TODO will this break later?
                print("No convergence")
                E_diff = 0
            else:
                ti = self.relaxation_loop(ti, relax_steps)
                self.rescale_velocities(ti)
                E_diff = np.abs(self.energies_kin[ti-1] - Ekin_target)

        self.ti_initial = ti

    def set_current_dx(self, pos0):
        x1 = np.repeat([pos0], repeats=self.n_particles, axis=0)
        x2 = np.transpose(x1, axes=[1,0,2])

        self.current_dx = (x2 - x1 + self.L/2) % self.L - self.L/2
    
    def set_current_r(self):
        dx = self.current_dx
        self.current_r = np.sqrt(np.sum(dx * dx, axis=2))

    def measure_current_forces(self):
        # dx = self.current_set_current_dx(pos0)
        dx = self.current_dx
        # r = self.current_r
        # r = np.sqrt(np.sum(dx * dx, axis=2))
        # self.energies_pot[ti] = self.measure_Epot(r)
        grad_U = self.measure_gradU()
        self.forces = -1 * dx * np.repeat(grad_U[:, :, np.newaxis], repeats=self.dims, axis=2)


    def measure_pressure(self):
        # dx = self.current_set_current_dx(pos0)
        # dx = self.current_dx
        r = self.current_r
        # r = np.sqrt(np.sum(dx * dx, axis=2))
        with np.errstate(divide='ignore', invalid='ignore'):
            a = np.nan_to_num(((-12) * r**(-12) + 6 * r**(-6)), nan=0) # betwen brackets part
        # print("for test: ", np.sum(a))
        # self.pressures[ti] = self.rho * (self.kBT  - np.sum(a) / (3 * self.n_particles))
        return self.rho * (self.kBT  - np.sum(a) / (3 * self.n_particles))

    def measure_correlation(self, ti):
        # dx = self.current_set_current_dx(pos0)
        # dx = self.current_dx
        r = self.current_r
        # r = np.sqrt(np.sum(dx * dx, axis=2))

        hist, _ = np.histogram(r[np.tril_indices_from(r)], bins=self.r_range, range=[0.01, self.L/2])
        self.distance_hist[ti] = hist
        # return hist

    def Euler_step(self, ti, pos0, vel0): # TODO update new Force
        MAX = 1e3
        self.set_current_dx(pos0)
        self.set_current_r()

        self.energies_kin[ti] = self.measure_Ekin(vel0)
        self.energies_pot[ti] = self.measure_Epot()
        self.pressures[ti] = self.measure_pressure()
        self.measure_correlation(ti)

        self.measure_current_forces()

        self.positions[ti+1, :, :] = (pos0 + vel0 * self.dt) % self.L
        self.velocities[ti+1,: , :] = np.clip(vel0 + self.forces.sum(axis=1) * self.dt / self.m, a_min=-MAX, a_max=MAX)


    def Verlet_step(self, ti, pos0, vel0):
        MAX = 1e3

        if ti == 0:
            self.set_current_dx(pos0)
            self.set_current_r()
            self.measure_current_forces()
            self.force_prev = self.forces.sum(axis=1)

        self.energies_kin[ti] = self.measure_Ekin(vel0)
        self.energies_pot[ti] = self.measure_Epot()
        self.pressures[ti] = self.measure_pressure()
        self.measure_correlation(ti)

        newpos = (pos0 + vel0 * self.dt + self.dt ** 2 / (2 * self.m) * self.force_prev) % self.L
        self.set_current_dx(newpos)
        self.set_current_r()
        self.positions[ti+1, :, :] = newpos

        self.measure_current_forces()

        F_tmp = self.forces.sum(axis=1)
        self.velocities[ti+1, :, :] = np.clip(vel0 + self.dt / (2 * self.m) * (F_tmp + self.force_prev), a_min=-MAX, a_max=MAX)
        self.force_prev = F_tmp
        

    def calculate_pair_corr(self):
        r = np.linspace(0.01, self.L/2, self.r_range)
        avg = np.average(self.distance_hist[self.ti_initial:], axis=0)
        V = self.L ** self.dims
        dr = self.L / (2 * self.r_range)

        g =  avg * 2 * V / (self.n_particles * (self.n_particles - 1) * 4 * np.pi * r ** 2 * dr)
        self.pair_correlation_g = g
        # print('here g', g)

    def determine_state(self):
        # distance=self.r_range/10
        peaks, _ = scipy.signal.find_peaks(self.pair_correlation_g,
                                prominence=0.25)
        # print(f"n_peaks = {len(peaks)}, {self.state}")
        
        if len(peaks) >= 3:
            self.state = 'solid'
        elif len(peaks) <= 1:
            self.state = 'gas'
        else:
            self.state = 'liquid'
        return peaks

    def plot_positions_2d(self):
        plt.figure()
        for ni in range(self.n_particles):
            plt.scatter(self.positions[:, ni, 0], self.positions[:, ni, 1], marker='.', s=2)
            # plt.scatter(self.positions[1, 0, :], self.positions[1, 1, :], c='r', marker='.')
            plt.scatter(self.positions[0, ni, 0], self.positions[0, ni, 1], c='k',
                zorder=5)
            # plt.scatter(self.positions[1, 0, 0], self.positions[1, 1, 0], c='r', marker='x')
        plt.xlim(0,self.L)
        plt.ylim(0,self.L)
        plt.savefig("posplot.png")
        print("posplot.png")

    def plot_positions_3d(self, ax=None):
        STEPS = 1000
        # alphas = np.linspace(0.1, 1, self.timesteps)

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
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_zlim(0, self.L)

        if save:
            plt.savefig('3dplot.png')
            print('3dplot.png')

    def plot_positions(self):
        if self.dims == 2:
            self.plot_positions_2d()
        elif self.dims == 3:
            self.plot_positions_3d()
        else:
            print("error what are you doing? (dims != 2 or 3)")
    
    def plot_energies(self, ax=None):
        E_target = (self.n_particles - 1) * (3 / 2) * self.kBT
        # print("E_target: ", E_target)
        t = np.arange(0, self.timesteps*self.dt, self.dt)[:-1]
        save=False

        if not ax:
            plt.figure()
            ax = plt.gca()
            save = True
        ax.plot(t, self.energies_pot[:-1], label="Epot")
        ax.plot(t, self.energies_kin[:-1], label="Ekin")
        ax.plot(t, self.energies_tot[:-1], label="Etot")
        ax.axhline(E_target, c='k', ls='--')
        ax.axvline(t[self.ti_initial], c='k', ls='--')
        ax.legend()

        ax.set_title(r"$\rho$" + f" = {self.rho}, T = {self.T}, dt = {self.dt}, steps = {self.timesteps}")

        if save:
            plt.savefig('Eplot.png')
            print('Eplot.png')

    def plot_pair_correlation(self, ax=None):
        r = np.linspace(0.01, self.L / 2, self.r_range)
        # r = np.linspace(0, self.L, self.r_range)

        if ax:
            label = fr"$\rho$: {self.rho}, T = {self.T}"
            if self.state: 
                label += f" ({self.state})"                 
            ax.plot(r, self.pair_correlation_g, label=label)
            peaks = self.determine_state()
            ax.scatter(r[peaks], self.pair_correlation_g[peaks])
            # ax.set_xlim(0.01, self.L/2)
        else:
            plt.figure()
            plt.plot(r, self.pair_correlation_g)
            plt.title(r"$\rho$" + f" = {self.rho}, T = {self.T}, dt = {self.dt}, steps = {self.timesteps}")
            # plt.xlim(0.01, self.L/2)

            plt.savefig("pair_correlation.png")
            print("pair_correlation.png")

    def save_sim(self):
        np.savez("pos.npz", array=self.positions)
    
    def load_positions(self):
        data = np.load("pos.npz")
        for item in data.files:
            self.positions = data[item]

    def run_simulation(self, print_progress=True):
        # if self.state:
            # print(f"Running {self.state}...")

        self.set_initial_conditions()
        # ti_initial=0

        start = time.perf_counter()
        self.velocity_relaxation()

        for i, (pos0, vel0) in enumerate(zip(self.positions[self.ti_initial:-1], self.velocities[self.ti_initial:-1])):
            ti = self.ti_initial + i
            # print(vel0)
            # print(pos0.shape)
            # self.Euler_step(ti, pos0, vel0)
            # self.Verlet_step(ti, pos0, vel0)
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
        # print("measure_pressure: ", self.pressure)

        # print(self.positions[0])
        # print(self.energies_kin)

class SimulationBatch:
    def __init__(self, dt=0.001, timesteps=1000, opt='gls', 
                 params=[], repeats=1, plot=True, method='Verlet',
                 summary_file='summary'):
        self.opt = opt
        self.dt = dt
        self.timesteps = timesteps
        self.repeats = repeats
        self.saved_sims = []
        self.plot = plot
        self.params = params
        self.method = method
        self.summary_file = f"{summary_file}.txt"

        self.write_summary()

    def write_summary(self):
        with open(self.summary_file, "w") as file:
            file.write("Parameters for simulation(s):\n")
            file.write(f"Method: {self.method}, steps: {self.timesteps}, dt = {self.dt}\n")
            file.write(f"{self.repeats} simulations per configuration\n\n")

    def sim_params(self, state):
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
                             method=self.method)
            sim.run_simulation()
            pressures[i] = sim.final_pressure
            pair_correlations.append(sim.pair_correlation_g)

        # store last simulation
        sim.g = np.mean(np.array(pair_correlations), axis=0)
        sim.determine_state()
        # print(sim.g.shape)
        P = np.mean(pressures)
        P_std = np.std(pressures)
        # if P_std == 0:
        roundoff = 2
        if P_std != 0:
            roundoff = int(math.floor(math.log10(abs(P_std))) - 1)
        # print(f'{state}: {np.mean(pressures)} +/- {np.std(pressures)}')
        self.saved_sims.append(sim)

        with open(self.summary_file, "a") as file:
            file.write(f"rho = {rho}, T = {T}, P = {P:.{roundoff}f} +/- {P_std:.{roundoff}f}\n")

    def run_options(self):
        if 'g' in self.opt:
            rho, T = self.sim_params('g')
            self.run_simulation(rho, T, state='gas')
        if 'l' in self.opt: 
            rho, T = self.sim_params('l')
            self.run_simulation(rho, T, state='liquid')
        if 's' in self.opt:
            rho, T = self.sim_params('s')
            self.run_simulation(rho, T, state='solid')

    def run_params(self):
        for i, (rho, T) in enumerate(self.params):
            print(f"Running sim {i+1} / {len(self.params)}")
            self.run_simulation(rho, T)
        
    def plot_correlation(self):
        plt.figure()
        ax = plt.gca()
        for i, sim in enumerate(self.saved_sims):
            sim.plot_pair_correlation(ax)
        ax.legend()
        plt.savefig("all_corrs.png")
        print("all_corrs.png")

        # plt.figure()
        # ax = plt.gca()
        # for i, sim in enumerate(self.saved_sims):
            # sim.determine_state()
        # ax.legend()
        # plt.savefig("fft.png")
        # print("fft.png")

    def plot_phase_space(self):
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
            if sim.final_pressure > 0:
                plt.scatter(sim.final_pressure, sim.T, marker=mrk[sim.state]['marker'],
                            color=mrk[sim.state]['color'])
        plt.xlabel('P')
        plt.ylabel('T')
        plt.title('phase spaceeee')
        plt.legend(handles=[handleg, handlel, handles])
        plt.savefig('states.png')

    def plot_positions(self):
        n_sims = len(self.saved_sims)

        fig = plt.figure(figsize=[5*n_sims, 5])

        for i, sim in enumerate(self.saved_sims):
            ax = fig.add_subplot(1, n_sims, i+1,  projection='3d')
            sim.plot_positions_3d(ax)
        plt.suptitle(f"dt = {self.dt}, steps = {self.timesteps}")
        plt.savefig("allpos.png")
        print("allpos.png")
    
    def plot_energy(self):
        n_sims = len(self.saved_sims)

        fig = plt.figure(figsize=[5, 4*n_sims])
        # fig, axs = plt.subplots(1, 3, projection ='3d')
        for i, sim in enumerate(self.saved_sims):
            ax = fig.add_subplot(n_sims, 1, i+1)
            sim.plot_energies(ax)
        plt.savefig('allenergies.png')
        print('allenergies.png')

    def run_simulations(self):
        if self.opt:
            self.run_options()
        if self.params:
            self.run_params()
        if self.plot:
            # self.plot_positions()
            # self.plot_correlation()
            # self.plot_energy()
            self.plot_phase_space()
        print(f"Summary written to {self.summary_file}")


# sim = Simulation(n_particles=3, L=1, dt=0.001, timesteps=1000, dims=2, blocks=2, T=1)
# sim = Simulation(n_particles=108, dt=0.001, timesteps=1000, dims=3, blocks=3, rho=1.2, T=0.5)
# sim = Simulation(rho=0.5, T=3, dt=0.001, timesteps=1000)
# sim = Simulation(rho=0.8, T=1, dt=0.001, timesteps=1000)
sim = Simulation(T=0.35, dt=0.001, timesteps=1000, dims=2,
                 n_particles=30, L=10)


sim.run_simulation()
# sim.save_sim()
# sim.load_positions()

print(sim.final_pressure)

sim.plot_positions()
# sim.plot_energies()
# sim.plot_pair_correlation()


params = []

# for i, rho in enumerate(np.linspace(0.2, 0.7, 10)):
# # rho = 0.85
#     for j, T in enumerate(np.linspace(0.1, 0.8, 10)):
#         # if i + j < 10:
#         params.append(np.round([rho, T], decimals=3))

# batch = SimulationBatch(repeats=10, dt=0.001, timesteps=2000, opt='', plot=True,
                        # method='Verlet', params=[[0.45, 0.35]])
# batch.run_simulations()
