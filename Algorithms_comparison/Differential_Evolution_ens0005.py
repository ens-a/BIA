import numpy as np
from celluloid import Camera
import matplotlib.pyplot as plt
from Functions import *
import matplotlib.animation as ani
import time


class Individual:
    """Class for individuals.
        Used to randomly create individuals which are used next in biologically inspired algorithms
        """

    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension
        self.lB = lower_bound
        self.uB = upper_bound
        self.parameters = np.zeros(self.dimension)
        self.f = np.inf

    def randomize(self):
        """Randomly assign vector of length == dimensions
        """
        self.parameters = np.random.uniform(self.lB, self.uB, self.dimension).tolist()


class DifferentialEvaluation:
    """Differential Evaluation algorithm.
        It is used to optimize given function in particular dimension space.
         For 2d problem it's possible to make an animation.
         make population -> run algorithm -> make animation"""

    def __init__(self, func_name, dimensions=2):
        self.lB = func_dic[func_name][1][0]
        self.uB = func_dic[func_name][1][1]
        self.function = func_dic[func_name][0]
        self.d = dimensions
        self.population = []
        self.g = 0

    def make_population(self, np):
        """Make population of given size NP."""
        self.NP = np
        for i in range(self.NP):
            x = Individual(self.d, self.lB, self.uB)
            x.randomize()
            self.population.append(x.parameters)
        self.generations = [self.population]  # save all generations for animation

    def run(self, F, CR, Gmax):
        """Executes algorithm of optimization"""
        while self.g < Gmax:
            new_pop = self.population.copy()
            for i, x in enumerate(new_pop):
                # for every individual in population we take three another
                # which are r1 != r2 != r3 != x
                list_ind = list(range(self.NP))
                list_ind.remove(i)
                r1 = np.random.choice(list_ind)
                list_ind.remove(r1)
                r2 = np.random.choice(list_ind)
                list_ind.remove(r2)
                r3 = np.random.choice(list_ind)
                # make a new individual which is product of r1, r2, r3
                v = [(v1 - v2) * F + v3 for v1, v2, v3 in list(zip(new_pop[r1],
                                                                   new_pop[r2],

                                                                  new_pop[r3]))]
                v = [self.lB if v_i < self.lB else (self.uB if v_i > self.uB else v_i) for v_i in v ]
                u = [0] * self.d
                # randomly swap values from x and v
                j_rnd = np.random.randint(0, self.d)
                for j in range(self.d):
                    if np.random.uniform() < CR or j == j_rnd:
                        u[j] = v[j]
                    else:
                        u[j] = x[j]
                # evaluate optimizing function from vector u
                # if it's less than from initial vector x, we change it in population
                f_u = self.function(u)
                if f_u <= self.function(x):
                    new_pop[i] = u
                self.population = new_pop  # rewrite population

            self.generations.append(new_pop)  # collect all populations
            self.g += 1

    def make_animation(self):
        """Make animation of evolutionary algorithm and save it as .gif"""
        fig, ax = plt.subplots()
        camera = Camera(fig)  # use simple library from github

        for i, generation in enumerate(self.generations):  # plotting individuals for every generation
            x = [x[0] for x in generation]
            y = [x[1] for x in generation]

            ax.scatter(x, y, s=10, color='black')
            plt.title('Differential Evolution algorithm')
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
        animation.save('DE_algorithm.gif', writer=writer)

"""
dimensions = 2
NP = 20
F = 0.5
CR = 0.5
Gmax = 50

start_time = time.time()
algorithm = DifferentialEvaluation('sphere', 30)
algorithm.make_population(30)
algorithm.run(F, CR, 3000)
algorithm.make_animation()
answer = sorted(algorithm.population, key=algorithm.function)[0]
print('Optimal value: {}'.format(algorithm.function(answer)))
print('Optimal solution: {}'.format(answer))
print("--- %s seconds ---" % (time.time() - start_time))
"""