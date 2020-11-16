import numpy as np
from celluloid import Camera
import matplotlib.pyplot as plt
from Functions import *
import matplotlib.animation as ani
import copy


class Individual:
    """Class for individuals.
        Used to randomly create individuals which are used next in biologically inspired algorithms
        """

    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension
        self.lB = lower_bound
        self.uB = upper_bound
        self.parameters = np.zeros(self.dimension)
        self.velocity = np.zeros(self.dimension)

    def randomize(self):
        """Randomly assign vector of length == dimensions
            Randomly generate velocity vector
        """
        self.parameters = np.random.uniform(self.lB, self.uB, self.dimension)  # .tolist()
        self.velocity = np.random.uniform(self.lB / 20, self.uB / 20, self.dimension)  # .tolist()
        self.best_par = copy.copy(self.parameters)


class SwarmOptimization:
    """Particle Swarm Optimization algorithm.
        It is used to optimize given function in particular dimension space.
         For 2d problem it's possible to make an animation.
         make swarm -> run algorithm -> make animation"""

    def __init__(self, func_name, dimensions=2):
        self.lB = func_dic[func_name][1][0] # unpacking dictionary with functions from Functions.py
        self.uB = func_dic[func_name][1][1]
        self.function = func_dic[func_name][0]
        self.d = dimensions
        self.swarm = []
        self.m = 0

    def make_swarm(self, size):
        """Make swarm of given size."""
        self.size = size
        for i in range(self.size):
            individual = Individual(self.d, self.lB, self.uB)
            individual.randomize()
            self.swarm.append(individual)
        self.generations = [[x.parameters for x in self.swarm]]  # save all generations for animation

    def weight(self):
        """Auxiliary function for weight calculation."""

        ws = 0.9
        we = 0.4
        return ws - (ws - we) * self.m / self.Mmax

    def new_velocity(self, c1, c2, x):
        """Auxiliary function for velocity calculation."""

        r1 = np.random.uniform()
        r2 = np.random.uniform()
        v = (x.velocity * self.weight()
             + r1 * c1 * (x.best_par - x.parameters)
             + r2 * c2 * (self.gBest - x.parameters))
        # check boundaries for velocity: func_low_bound/20 < v < func_high_bound/20
        v = [self.lB/20 if v_i < self.lB/20 else (self.uB/20 if v_i > self.uB/20 else v_i) for v_i in v]
        return np.array(v)

    def run(self, c1, c2, Mmax):
        """Executes algorithm of optimization"""

        self.gBest = sorted(self.swarm, key=lambda x: self.function(x.parameters))[0].parameters
        self.Mmax = Mmax
        while self.m < Mmax:
            for i, individual in enumerate(self.swarm):
                individual.velocity = self.new_velocity(c1, c2, individual)
                new_par = individual.parameters + individual.velocity
                # check boundaries for velocity: func_low_bound < x < func_high_bound
                individual.parameters = np.array([self.lB if x_i < self.lB else (self.uB if x_i > self.uB else x_i) for x_i in new_par])
                if self.function(individual.parameters) < self.function(individual.best_par):
                    individual.best_par = individual.parameters
                    if self.function(individual.best_par) < self.function(self.gBest):
                        self.gBest = individual.best_par
                new_swarm = [x.parameters for x in self.swarm]
            self.generations.append(new_swarm)  # collect all generations
            self.m += 1

    def make_animation(self):
        """Make animation of evolutionary algorithm and save it as .gif"""

        fig, ax = plt.subplots()
        camera = Camera(fig)  # use simple library from github

        for i, generation in enumerate(self.generations):  # plotting individuals for every generation
            x = [x[0] for x in generation]
            y = [x[1] for x in generation]

            ax.scatter(x, y, s=10, color='black')
            plt.title('Swarm Optimization algorithm')
            camera.snap()  # make a screenshot for animation

        x1 = np.linspace(self.lB, self.uB, 100)  # generate coordinates for plotting func
        x2 = np.linspace(self.lB, self.uB, 100)
        # grid of coordinates, result - shape1 = (1, 100), shape2 = (100, 1)
        xx1, xx2 = np.meshgrid(x1, x2, sparse=True)
        z = np.empty([xx1.shape[1], xx2.shape[0]])  # empty matrix (100, 100)

        for i in range(xx2.shape[0]):  # filling z matrix using prepared functions
            for j in range(xx1.shape[1]):
                z[i, j] = self.function([xx1[:, j][0], xx2[i][0]])

        # plotting 2d heatmap of given function
        plt.pcolormesh(xx1, xx2, z, alpha=0.6, shading='auto', )
        plt.xlim(self.lB, self.uB)
        plt.ylim(self.lB, self.uB)
        plt.colorbar()

        Writer = ani.writers['pillow']
        writer = Writer(fps=10)

        animation = camera.animate()
        animation.save('SwOpt_algorithm.gif', writer=writer)


dimensions = 2
pop_size = 15
c1 = 2
c2 = 2
M_max = 50

algorithm = SwarmOptimization('rastrigin')
algorithm.make_swarm(pop_size)
algorithm.run(c1, c2, M_max)
algorithm.make_animation()
answer = sorted(algorithm.swarm, key= lambda x: algorithm.function(x.parameters))[0]
print('Optimal value: {}'.format(algorithm.function(answer.parameters)))
print('Optimal solution: {}'.format(answer.parameters))



