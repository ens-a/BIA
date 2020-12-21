import numpy as np
from celluloid import Camera
import matplotlib.pyplot as plt
import matplotlib.animation as ani


class Individual:
    """Class for individuals.
        Used to randomly create individuals which are used next in biologically inspired algorithms
        """

    def __init__(self, dimension, boundaries):
        self.dimension = dimension
        self.B = boundaries  # [(lB1, uB1), (lB2, uB2]
        self.parameters = []

    def randomize(self):
        """Randomly assign vector of length == dimensions
        """
        for bound in self.B:
            self.parameters.append(np.random.uniform(bound[0], bound[1]))


class Algorithm:

    def __init__(self, boundaries, dimensions, obj_functions):

        self.boundaries = boundaries
        self.d = dimensions
        self.obj_func = obj_functions  # obj_func[n] = Nth function
        self.population = []

    def make_pop(self, NP):

        self.NP = NP
        for i in range(self.NP):
            x = Individual(self.d, self.boundaries)
            x.randomize()
            self.population.append(x.parameters)

    def non_dominated_sorting(self):

        S = [[] for _ in range(self.NP * 2)]
        n = [0] * self.NP * 2
        Q = [[]]
        for i in range(self.NP * 2):
            for k in range(self.NP * 2):
                if i != k:
                    if (self.obj_func[1](self.population[i]) >= self.obj_func[1](self.population[k]) and
                            self.obj_func[2](self.population[i]) >= self.obj_func[2](self.population[k])):
                        n[i] += 1
                    elif (self.obj_func[1](self.population[i]) <= self.obj_func[1](self.population[k]) and
                          self.obj_func[2](self.population[i]) <= self.obj_func[2](self.population[k])):
                        S[i].append(k)
            if n[i] == 0:
                Q[0].append(i)
        print('n=', n)
        print('S=', S)
        m = 0
        while len(Q[-1]) != 0:

            temp = []
            for p in Q[m]:
                for q in S[p]:
                    n[q] = n[q] - 1
                    if (n[q] == 0):
                        temp.append(q)
            Q.append(temp)
            m += 1

        return Q[:-1]

    def genetic_algorithm(self, gen_number):

        self.G = gen_number

        g = 0
        while g < self.G:
            print('Iteration number:', g)
            for i in range(self.NP):
                parent_A = self.population[i]  # select the first parent
                list_i = list(range(self.NP))
                list_i.remove(i)  # exclude current index from a list
                # randomly select the second parent different from the first parent
                parent_B = self.population[np.random.choice(list_i)]

                new_parent = []
                # crossover two parents
                for j in range(self.d):
                    if np.random.uniform() < 0.5:
                        new_parent.append((parent_A[j] + parent_B[j]) / 2)
                    else:
                        new_parent.append((parent_A[j] - parent_B[j]) / 2)
                    # make a single mutation randomly
                    if np.random.uniform() < 0.5:
                        new_parent[j] = new_parent[j] + np.random.uniform(0, 1)
                for k in range(self.d):
                    lB = self.boundaries[k][0]
                    uB = self.boundaries[k][1]
                    x = new_parent[k]
                    new_parent[k] = lB if x < lB else (uB if x > uB else x)

                self.population.append(new_parent)
            ranks = self.non_dominated_sorting()
            print('Output of non-dominated sorting =', ranks)
            sorted_pop = [item for r in ranks for item in r]
            new_pop = [self.population[index] for index in sorted_pop[:self.NP]]
            self.population = new_pop
            g += 1


boundaries = [(0, 10), (0, 20)]
dim = 2


def func1(x):
    r = x[0]
    h = x[1]
    return np.pi * r * np.sqrt(r ** 2 + h ** 2)


def func2(x):
    r = x[0]
    h = x[1]
    return np.pi * r * (np.sqrt(r ** 2 + h ** 2) + r)


func_dic = {1: func1, 2: func2}

a = Algorithm(boundaries, dim, func_dic)
a.make_pop(5)
a.genetic_algorithm(100)
# print('Best solution:', a.population[0])
