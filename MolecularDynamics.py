import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.constants
from scipy.stats import qmc


# print("yea")
class Simulation:
    k_B = scipy.constants.k
    A = 1e-10
    SIGMA = 3.405 * A
    EPSILON = 119.8 * k_B
    M = 39.948 * scipy.constants.u

    def __init__(self, n_particles=3, L=5, dims=2, timesteps=1000, dt=0.001, blocks=3, T=1):
        self.n_particles = n_particles
        self.L = L
        self.dims = dims
        self.timesteps = timesteps
        self.dt = dt
        self.T = T

        self.m = 1
        self.sigma = 1
        self.epsilon = 1

        self.blocks = blocks

        # self.m = self.M
        # self.sigma = self.SIGMA
        # self.epsilon = self.EPSILON

        self.positions = np.zeros([timesteps, n_particles, dims])
        self.velocities = np.zeros([timesteps, n_particles, dims])
        self.F_matrix =  np.zeros([n_particles, n_particles, dims])
        self.F_vect_prev =  np.zeros([n_particles, dims])
        self.E_potential = np.zeros(timesteps)
        self.E_kinetic = np.zeros(timesteps)
        self.E_total = np.zeros(timesteps)

    def re_init(self):
        self.positions = np.zeros([self.timesteps, self.n_particles, self.dims])
        self.velocities = np.zeros([self.timesteps, self.n_particles, self.dims])
        self.F_matrix =  np.zeros([self.n_particles, self.n_particles, self.dims])
        self.F_vect_prev =  np.zeros([self.n_particles, self.dims])

    def set_pseudorandom(self):
        ''' 2D solution!! doesn't work in 3D...'''
        pos_available_1d = (self.L // 1.5)
        pos_available = int(pos_available_1d ** 2)
        if self.n_particles > pos_available:
            msg = f"Too many particles, default to {pos_available}"
            print("\033[91m {}\033[00m".format(msg))
            self.n_particles = pos_available
            self.re_init()

        rnd_pos = np.random.choice(pos_available, size=self.n_particles, replace=False)
        # x = rnd_pos // pos_available_1d
        # y = rnd_pos // pos_available_1d
        # print(rnd_pos)

        self.positions[0,:,:] = np.array([rnd_pos // pos_available_1d, rnd_pos % pos_available_1d]).transpose() * 1.5 + 1
        # print(a)
        self.positions += np.random.normal(0, 0.2, [self.n_particles, self.dims])
        # np.random.normal(0, 0.2, [self.n_particles, self.dims])

    def set_FCC(self):
        unit_block = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        shift_lattice = []
        for i in range(self.blocks):
            for j in range(self.blocks):
                for k in range(self.blocks):
                    shift_lattice.append(np.tile([i, j, k], (4, 1)))
        shift_lattice = np.vstack(shift_lattice)

        lattice = (np.tile(unit_block, (self.blocks ** 3, 1)) + shift_lattice)

        # print(lattice)
        # print(lattice * self.L)

        self.positions[0, :, :] = lattice * self.L / self.blocks

    def set_initial_conditions(self, method='random'):
        if self.n_particles > 3:
            random=True
        if method == 'fcc':
            self.set_FCC()
            self.velocities[0,:,:] = np.random.normal(0, np.sqrt(self.T), [self.n_particles, self.dims])
        elif method == 'fixed':
            self.positions[0,0,:] = [1, 1]
            self.positions[0,1,:] = [1, 2.5]
            if self.n_particles > 2:
                self.positions[0, 2, :] = [3,4]
            self.velocities[0,0,:] = [1, 0]
            self.velocities[0,1,:] = [-1, 0]
        elif method == 'pseudo':
            self.set_pseudorandom()
            # self.positions[0,:,:] = np.random.uniform(0, self.L, [self.n_particles, self.dims])
            self.velocities[0,:,:] = np.random.normal(0, 0.2, [self.n_particles, self.dims])
            # self.velocities[0,:,:] = np.random.normal(0, 0.2, [self.n_particles, self.dims])
        else:
            self.positions[0,:,:] = np.random.uniform(0, self.L, [self.n_particles, self.dims])
            self.velocities[0,:,:] = np.random.normal(0, 0.2, [self.n_particles, self.dims])


    def U(self, r):
        U = 4*self.epsilon * ((self.sigma/r)**12 - (self.sigma/r)**6)
        return U

    def U_(self, r):
        # sigma = 
        U_ = 4 * ((1/r)**12 - (1/r)**6)
        return U_

    def get_U(self, xi, xj):
        dx = (xi - xj + self.L/2) % self.L - self.L/2
        r = np.sqrt(np.sum(dx * dx))
        return self.U_(r)

    def grad_U(self, r):
        return 4 * ((-12) * r**(-13) + 6 * r**(-7))
        # return grad_U

    def Force_(self, xi, xj):
        # print('pos: ', xi, xj)
        # dx = xi - xj
        dx = (xi - xj + self.L/2) % self.L - self.L/2
        if self.dims == 1:
            r = dx
        else:
            r = np.sqrt(np.sum(dx * dx))
        # print("r: ", r)

        F = -1 * dx * self.grad_U(r)/r
        # arr_clipped = np.clip(arr, a_min=None, a_max=1)
        # F = np.clip(F, a_min=None, a_max=1)

        # if np.abs(np.sum(F)) > 2:
        #    return 2 * (F / np.abs(F))
        return F

    def run_test(self, xmin=0.9, xmax=5, steps=50):
        xtest = np.linspace(0.9,2,50)
        plt.plot(xtest, self.U_(xtest))
        plt.show()
        plt.plot(xtest, self.grad_U(xtest))
        plt.show()
        # plt.plot(xtest, U)
        plt.plot(self, xtest, self.Force_(xtest, np.zeros(50)))
        plt.show()
        print(self.U_(xtest))

    def inner_loop(self, ti, pos0, calc_U=True):
        U_tot = 0
        for ni in range(self.n_particles):      # TODO change to iterate over pos-n ?
            for nj in range(self.n_particles):
                if ni == nj:
                    continue
                elif ni > nj:
                    self.F_matrix[ni, nj] = -self.F_matrix[nj,  ni]
                else:
                    # print("n: ", ni, nj)
                    F = self.Force_(pos0[ni], pos0[nj])
                    U_tot += self.get_U(pos0[ni], pos0[nj])
                    # print("f: ", F)
                    self.F_matrix[ni, nj] = F
                # F_tot += F
        self.E_potential[ti] = U_tot

    def Euler_step(self, ti, pos0, vel0):
        self.positions[ti+1, :, :] = (pos0 + vel0 * self.dt) % self.L

        # print("step ", ti, pos0, vel0, self.positions[ti+1, :, :])
        # U_tot = 0
        # for ni in range(self.n_particles):      # TODO change to iterate over pos-n ?
        #     for nj in range(self.n_particles):
        #         if ni == nj:
        #             continue
        #         elif ni > nj:
        #             self.F_matrix[ni, nj] = -self.F_matrix[nj,  ni]
        #         else:
        #             # print("n: ", ni, nj)
        #             F = self.Force_(pos0[ni], pos0[nj])
        #             # print("f: ", F)
        #             self.F_matrix[ni, nj] = F
        #         # F_tot += F
        #         U_tot += self.get_U(pos0[ni], pos0[nj])
        #             # F_matrix[ni, nj] = F_tot
            
            # print(self.F_matrix.shape)
        # self.E_potential[ti] = U_tot
        self.inner_loop(ti, pos0)
        self.E_kinetic[ti] =  0.5 * self.m * np.sum(vel0 * vel0)
        MAX = 1e3
        self.velocities[ti+1,: , :] = np.clip(vel0 + self.F_matrix.sum(axis=1) * self.dt / self.m, a_min=-MAX, a_max=MAX)


    def Verlet_step(self, ti, pos0, vel0):
        MAX = 1e3
        if ti == 0:
            self.inner_loop(ti, pos0)
            self.F_vect_prev = self.F_matrix.sum(axis=1)

        newpos = (pos0 + vel0 * self.dt + self.dt ** 2 / (2 * self.m) * self.F_vect_prev) % self.L
        # print("New position: ", newpos)
        self.positions[ti+1, :, :] = newpos
        self.inner_loop(ti, self.positions[ti+1, :, :])

        F_tmp = self.F_matrix.sum(axis=1)
        self.velocities[ti+1, :, :] = np.clip(vel0 + self.dt / (2 * self.m) * (F_tmp + self.F_vect_prev), a_min=-MAX, a_max=MAX)
        self.F_vect_prev = F_tmp
        self.E_kinetic[ti] =  0.5 * self.m * np.sum(vel0 * vel0)

    def plot_positions_2d(self):
        alphas = np.linspace(0.1, 1, self.timesteps)
        # print(alpha)
        plt.figure()
        for ni in range(self.n_particles):
            # for i, pos in enumerate(self.positions):
            #     plt.scatter(pos[ni, 0], pos[ni, 1], marker='.',
            #         alpha=alphas[i])
            plt.scatter(self.positions[:, ni, 0], self.positions[:, ni, 1], marker='.',
                        alpha=1)
            # plt.scatter(self.positions[1, 0, :], self.positions[1, 1, :], c='r', marker='.')
            plt.scatter(self.positions[0, ni, 0], self.positions[0, ni, 1], c='k', marker='x',
                zorder=5)
            # plt.scatter(self.positions[1, 0, 0], self.positions[1, 1, 0], c='r', marker='x')
        plt.xlim(0,self.L)
        plt.ylim(0,self.L)
        plt.savefig("posplot.png")
        print("posplot.png")

    def plot_positions_3d(self):
        alphas = np.linspace(0.1, 1, self.timesteps)
        # print(alphas)

        fig = plt.figure()
        ax = plt.axes(projection ='3d')

        STEPS = 100

        for ni in range(self.n_particles):
            # ax.scatter(pos[ni, 0], pos[ni, 1], pos[ni, 2], alpha=0.8)
            ax.scatter(self.positions[-STEPS:, ni, 0], self.positions[-STEPS:, ni, 1], self.positions[-STEPS:, ni, 2],
                        marker='.', alpha=0.2)
            # ax.scatter(self.positions[0, ni, 0], self.positions[0, ni, 1], self.positions[0, ni, 2],
            #             marker='x', alpha=0.8, c='k')
        # ax.scatter(self.p)

        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_zlim(0, self.L)

        plt.savefig('3dplot.png')
        print('3dplot.png')

    def plot_positions(self):
        if self.dims == 2:
            self.plot_positions_2d()
        elif self.dims == 3:
            self.plot_positions_3d()
        else:
            print("error what are you doing? (dims != 2 or 3)")
    
    def plot_energies(self):
        t = np.arange(0, self.timesteps*self.dt, self.dt)
        plt.figure()
        plt.plot(t, self.E_potential, label="Epot")
        plt.plot(t, self.E_kinetic, label="Ekin")
        plt.plot(t, self.E_total, label="Etot")
        plt.legend()
        plt.savefig('Eplot.png')
        print('Eplot.png')
        
    def save_sim(self):
        np.savez("pos.npz", array=self.positions)
    
    def load_positions(self):
        data = np.load("pos.npz")
        for item in data.files:
            self.positions = data[item]

    def run_simulation(self):
        self.set_initial_conditions(method='fcc')
        # print("init vel: ", self.velocities)
        # print("init pos: ", self.positions[0, :, :])
        for ti, (pos0, vel0) in enumerate(zip(self.positions[:-1], self.velocities[:-1])):
            # print(vel0)
            # print(pos0.shape)
            # self.Euler_step(ti, pos0, vel0)
            self.Verlet_step(ti, pos0, vel0)
            if (ti) % 100 == 0:
                print(f"{ti} steps")
        self.E_total = self.E_kinetic + self.E_potential
        # print(self.positions[0])



sim = Simulation(n_particles=32, L=3, dt=0.001, timesteps=1000, dims=3, blocks=2, T=100)
# sim.set_FCC()
# sim.plot_positions_3d()

sim.run_simulation()
sim.save_sim()
# sim.load_positions()

sim.plot_positions()
sim.plot_energies()