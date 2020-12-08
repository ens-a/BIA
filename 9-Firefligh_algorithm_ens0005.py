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

    def __init__(self, func_name, dimensions):
        self.lB = func_dic[func_name][1][0]  # unpacking dictionary with functions from Functions.py
        self.uB = func_dic[func_name][1][1]
        self.function = func_dic[func_name][0]
        self.dimensions = dimensions
        self.parameters = np.zeros(self.dimensions)

    def randomize(self):

        self.parameters = np.random.uniform(self.lB, self.uB, self.dimensions)

    def evaluate(self):
        return self.function(self.parameters)


class FireflySwarmOptimization:


    def __init__(self, func_name, dimensions=2):

        self.func_name = func_name
        self.lB = func_dic[func_name][1][0]  # unpacking dictionary with functions from Functions.py
        self.uB = func_dic[func_name][1][1]
        self.function = func_dic[func_name][0]
        self.d = dimensions
        self.Leader = None
        self.swarm = []
        self.generations = []

    def make_swarm(self, size):
        """Make swarm of given size."""
        self.size = size
        for i in range(self.size):
            individual = Individual(self.func_name, self.d)
            individual.randomize()

            if i == 0:
                self.Leader = individual
            elif individual.evaluate() < self.Leader.evaluate():
                self.Leader = individual

            self.swarm.append(individual)
        self.generations = [[x.parameters for x in self.swarm]]  # save all generations for animation

    def distance(self, x_i, x_j):

        temp_sum = sum((i - j) ** 2 for i, j in list(zip(x_i, x_j)))
        return temp_sum ** 0.5

    def new_param(self, x_i, x_j):

        attractiveness = self.beta / (self.distance(x_i, x_j) + 1)
        random_vector = np.random.normal(0, 1, size=(1, self.d))[0]
        new_vector = x_i + attractiveness * (x_j - x_i) + self.alpha * random_vector
        new_vector = [self.lB if i < self.lB else (self.uB if i > self.uB else i) for i in new_vector]

        return np.array(new_vector)

    def run(self, alpha, beta, Mmax):
        """Executes algorithm of optimization"""

        self.alpha = alpha
        self.beta = beta
        self.Mmax = Mmax
        self.m = 0
        self.gBest = sorted(self.swarm, key=lambda x: self.function(x.parameters))[0].parameters

        while self.m < Mmax:
            for i in range(self.d):
                for j in range(self.d):
                    x_i = self.swarm[i]
                    x_j = self.swarm[j]
                    if x_i.evaluate() < x_j.evaluate():
                        new_x_i = self.new_param(x_i.parameters, x_j.parameters)
                        self.swarm[i].parameters = new_x_i
                        if self.swarm[i].evaluate() < self.Leader.evaluate():
                            self.Leader.parameters = new_x_i

            print('Current best: {}'.format(self.Leader.evaluate()))
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
            plt.title('FireFly algorithm')
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
        animation.save('Firefly_algorithm.gif', writer=writer)


dimensions = 2
pop_size = 10
alpha = 0.4
beta = 1
M_max = 100

algorithm = FireflySwarmOptimization('rastrigin')
algorithm.make_swarm(pop_size)
algorithm.run(alpha, beta, M_max)
algorithm.make_animation()

