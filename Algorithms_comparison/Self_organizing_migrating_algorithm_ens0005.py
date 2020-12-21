import numpy as np
from numpy import random as rnd
from celluloid import Camera
import matplotlib.pyplot as plt
from Functions import *
import matplotlib.animation as ani
import copy
import time

class Individual:
    """Class for individuals.
        Used to randomly create individuals which are used next in biologically inspired algorithms
        """

    def __init__(self, dimensions, func_name, PRT):
        self.d = dimensions
        self.lB = func_dic[func_name][1][0]  # unpacking dictionary with functions from Functions.py
        self.uB = func_dic[func_name][1][1]
        self.func = func_dic[func_name][0]  # evaluation function
        self.parameters = np.zeros(self.d)
        self.PRT = PRT
        self.PRTvec = []

    def randomize(self):
        """Randomly assign vector of length == dimensions
        """
        self.parameters = np.random.uniform(self.lB, self.uB, self.d)

    def evaluate(self):
        """Evaluate solution of an individual using optimizing function
        """
        return self.func(self.parameters)

    def get_PRTvec(self):
        """Randomly assign PRT vector which is used for mutation
        """
        self.PRTvec = [1 if rnd.uniform() < self.PRT else 0 for _ in range(self.d)]


class SOMA:
    """Self-organizing algorithm which was designed by Ivan Zelinka."""

    def __init__(self, func_name, dimensions=2):

        self.func_name = func_name
        self.lB = func_dic[func_name][1][0]  # unpacking dictionary with functions from Functions.py
        self.uB = func_dic[func_name][1][1]
        self.function = func_dic[func_name][0]
        self.d = dimensions
        self.Leader = None
        self.population = []
        self.generations = [] # generations are used in animation

    def make_population(self, size):
        """Make population of given size with individuals and choose initial Leader (optimal value)
        """
        for i in range(size):
            individual = Individual(self.d, self.func_name, self.PRT)
            individual.randomize()

            if i == 0:
                self.Leader = individual
            elif individual.evaluate() < self.Leader.evaluate():
                self.Leader = individual

            self.population.append(individual)
        self.generations.append([x.parameters for x in self.population])

    def calc_next(self, individ, t):
        """Calculate mutations
        """
        x_new = copy.deepcopy(individ)
        for i in range(self.d):
            x_new.parameters[i] = (individ.parameters[i] +
                                   (self.Leader.parameters[i] - individ.parameters[i]) * individ.PRTvec[i] * t)

        # check boundaries for new parameter vector
        x_new.parameters = [self.lB if x_i < self.lB else (self.uB if x_i > self.uB else x_i) for x_i in x_new.parameters]
        return x_new

    def run(self, pop_size, path_length, step, migrations, prt):
        """Execution of  the algorithm
        """

        self.PRT = prt
        self.Mmax = migrations
        self.m = 0

        self.make_population(pop_size)

        while self.m < self.Mmax:
            for i, x in enumerate(self.population):
                t = 0
                while t < path_length:
                    x.get_PRTvec()
                    x_new = self.calc_next(x, t)
                    if t == 0:
                        local_optimal = x_new
                    elif x_new.evaluate() < local_optimal.evaluate():
                        local_optimal = x_new
                    t += step
                if local_optimal.evaluate() < x.evaluate():
                    self.population[i] = local_optimal

                if local_optimal.evaluate() < self.Leader.evaluate():
                    self.Leader = local_optimal
            new_pop = [ind.parameters for ind in self.population]
            self.generations.append(new_pop)
            self.m += 1

    def make_animation(self):
        """Make animation of evolutionary algorithm and save it as .gif"""


        fig, ax = plt.subplots()
        camera = Camera(fig)  # use simple library from github

        for i, generation in enumerate(self.generations):  # plotting individuals for every generation
            x = [x[0] for x in generation]
            y = [x[1] for x in generation]

            ax.scatter(x, y, s=10, color='black')
            plt.title('SOMA algorithm')
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
        writer = Writer(fps=5)

        animation = camera.animate()
        animation.save('SOMA_algorithm.gif', writer=writer)
"""
pop_size = 30
prt = 0.4
path = 3
step = 0.3
migrations = 300
start_time = time.time()
algorithm = SOMA('sphere', 30)
algorithm.run(pop_size, path, step, migrations, prt)
#algorithm.make_animation()
#answer = sorted(algorithm.population, key= lambda x: algorithm.function(x.parameters))[0]
print('Optimal value: {}'.format(algorithm.Leader.evaluate()))
#print('Optimal solution: {}'.format(answer.parameters))
print("--- %s seconds ---" % (time.time() - start_time))"""
















