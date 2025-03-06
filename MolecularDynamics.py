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

    def __init__(self, n_particles=3, L=5, dims=2, timesteps=1000, dt=0.001):
        self.n_particles = n_particles
        self.L = L
        self.dims = dims
        self.timesteps = timesteps
        self.dt = dt

        self.m = 1
        self.sigma = 1
        self.epsilon = 1

        # self.m = self.M
        # self.sigma = self.SIGMA
        # self.epsilon = self.EPSILON

        self.positions = np.zeros([timesteps, n_particles, dims])
        self.velocities = np.zeros([timesteps, n_particles, dims])
        self.F_matrix =  np.zeros([n_particles, n_particles, dims])
        self.E_potential = np.zeros(timesteps)
        self.E_kinetic = np.zeros(timesteps)
        self.E_total = np.zeros(timesteps)

    def set_pseudorandom(self):
        ''' 2D solution!! doesn't work in 3D...'''
        pos_available_1d = (self.L // 1.5)
        pos_available = int(pos_available_1d ** 2)
        if pos_available < self.n_particles:
            msg = f"Too many particles, default to {pos_available}"
            print("\033[91m {}\033[00m".format(msg))
            self.n_particles = pos_available
        rnd_pos = np.random.choice(pos_available, size=self.n_particles, replace=False)
        # x = rnd_pos // pos_available_1d
        # y = rnd_pos // pos_available_1d
        # print(rnd_pos)

        self.positions[0,:,:] = np.array([rnd_pos // pos_available_1d, rnd_pos % pos_available_1d]).transpose() * 1.5 + 1
        # print(a)
        self.positions += np.random.normal(0, 0.2, [self.n_particles, self.dims])
        # np.random.normal(0, 0.2, [self.n_particles, self.dims])

    def set_initial_conditions(self, random=False):
        if self.n_particles > 3:
            random=True
        if random:
            self.set_pseudorandom()
            # self.positions[0,:,:] = np.random.uniform(0, self.L, [self.n_particles, self.dims])
            self.velocities[0,:,:] = np.random.normal(0, 0.2, [self.n_particles, self.dims])
        else:
            self.positions[0,0,:] = [1, 1]
            self.positions[0,1,:] = [2, 2.5]
            if self.n_particles > 2:
                self.positions[0, 2, :] = [3,4]
            # self.velocities[0,0,:] = [0.2, 0]
            # self.velocities[0,1,:] = [-0.2, 0]
            # self.velocities[0,:,:] = np.random.normal(0, 0.2, [self.n_particles, self.dims])

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

    def Euler_step(self, ti, pos0, vel0):
        self.positions[ti+1, :, :] = (pos0 + vel0 * self.dt) % self.L

        # print("step ", ti, pos0, vel0, self.positions[ti+1, :, :])
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
                    # print("f: ", F)
                    self.F_matrix[ni, nj] = F
                # F_tot += F
                # U_tot += get_U(pos0[ni], pos0[nj])
                    # F_matrix[ni, nj] = F_tot
            
            # print(self.F_matrix.shape)
        self.E_potential[ti] = U_tot
        MAX = 1e3
        self.velocities[ti+1,: , :] = np.clip(vel0 + self.F_matrix.sum(axis=1) * self.dt / self.m, a_min=-MAX, a_max=MAX)


    def Verlet_step(self):
        pass

    def plot_positions(self):
        alphas = np.linspace(0.1, 1, self.timesteps)
        # print(alpha)
        for ni in range(self.n_particles):
            # for i, pos in enumerate(self.positions):
            #     plt.scatter(pos[ni, 0], pos[ni, 1], marker='.',
            #         alpha=alphas[i])
            plt.scatter(self.positions[:, ni, 0], self.positions[:, ni, 1], marker='.',
                alpha=alphas)
            # plt.scatter(self.positions[1, 0, :], self.positions[1, 1, :], c='r', marker='.')
            plt.scatter(self.positions[0, ni, 0], self.positions[0, ni, 1], c='k', marker='x',
                zorder=5)
            # plt.scatter(self.positions[1, 0, 0], self.positions[1, 1, 0], c='r', marker='x')
        plt.xlim(0,self.L)
        plt.ylim(0,self.L)
        plt.savefig("posplot.png")
        print("posplot.png")
    
    def plot_energies(self):
        t = np.arange(0, self.timesteps, self.dt)
        plt.plot(t, self.E_potential)
        plt.plot(t, self.E_kinetic)
        plt.plot(t, self.E_total)
        plt.savefig('Eplot.png')
        print('Eplot.png')
        

    def run_simulation(self):
        self.set_initial_conditions(random=True)
        # print("init vel: ", self.velocities)
        for ti, (pos0, vel0) in enumerate(zip(self.positions[:-1], self.velocities[:-1])):
            # print(vel0)
            # print(pos0.shape)
            self.Euler_step(ti, pos0, vel0)
        # print(self.positions[0])



sim = Simulation(n_particles=6, L=5, dt=0.0001, timesteps=30000)
sim.run_simulation()
sim.plot_positions()