import numpy as np
from celluloid import Camera
import matplotlib.pyplot as plt
from Functions import *
import matplotlib.animation as ani
import copy


class LearningBasedOptimisation:


    def __init__(self, func_name, dimensions=2):

        self.func_name = func_name
        self.lB = func_dic[func_name][1][0]  # unpacking dictionary with functions from Functions.py
        self.uB = func_dic[func_name][1][1]
        self.function = func_dic[func_name][0]
        self.d = dimensions
        self.population = []
        self.generations = []
        self.teacher = None

    def make_population(self, size):
        """Make population of given size."""
        self.size = size
        for i in range(self.size):
            learner = np.random.uniform(self.lB, self.uB, self.d)
            self.population.append(learner)
            if i == 0:
                self.teacher = learner
            elif self.function(learner) < self.function(self.teacher):
                self.teacher = learner
        self.population = np.asarray(self.population)
        self.generations.append(self.population)  # save all generations for animation


    def run(self, iterations):
        """Executes algorithm of optimization"""
        self.Mmax = iterations
        m = 0
        # Teacher phase
        while m < self.Mmax:
            #print(m, self.function(self.teacher))
            r = np.random.uniform(0, 1)
            Tf = np.random.randint(1, 2)
            pop_mean = [self.population[:, d].mean() for d in range(self.d)]
            difference = r*(self.teacher - Tf*pop_mean)
            for i, x_i in enumerate(self.population):
                x_new = x_i + difference
                x_new =[self.lB if v < self.lB else (self.uB if v > self.uB else v) for v in x_new]
                if self.function(x_new) < self.function(x_i):
                    self.population[i] = x_new
            # Learner phase
                possible_index = set([n for n in range(self.size)]) - {i}
                j = np.random.choice(list(possible_index), 1)[0]
                x_j = self.population[j]
                if self.function(x_i) < self.function(x_j):
                    x_new = x_i + r*(x_i - x_j)
                else:
                    x_new = x_i + r*(x_j - x_i)
                if self.function(x_new) < self.function(x_i):
                    self.population[i] = x_new
            self.teacher = sorted(self.population, key=self.function)[0]
            new_pop = copy.deepcopy(self.population)
            self.generations.append(new_pop)
            m += 1
    def make_animation(self):
        """Make animation of evolutionary algorithm and save it as .gif"""

        fig, ax = plt.subplots()
        camera = Camera(fig)  # use simple library from github

        for i, generation in enumerate(self.generations):  # plotting individuals for every generation
            x = [x[0] for x in generation]
            y = [x[1] for x in generation]

            ax.scatter(x, y, s=10, color='black')
            plt.title('Teach Based Optimisation')
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
        writer = Writer(fps=20)

        animation = camera.animate()
        animation.save('Learning_Based_opt.gif', writer=writer)

"""
dimensions = 2
pop_size = 30
M_max = 3000

algorithm = LearningBasedOptimisation('sphere', 2)
algorithm.make_population(pop_size)
algorithm.run(M_max)
#algorithm.make_animation()
print(algorithm.function(algorithm.teacher))


"""