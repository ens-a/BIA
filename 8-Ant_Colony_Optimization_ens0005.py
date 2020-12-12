import numpy as np
from numpy import random
from celluloid import Camera
import matplotlib.pyplot as plt
from Functions import *
import matplotlib.animation as ani
import copy
import math

class Ant:

    def __init__(self, cities, D, alpha, beta):

        self.cities = cities
        self.D = D  # dimensions of the TS problem (number of cities)
        self.a = alpha
        self.b = beta
        self.init_dist_matrix()
        self.visited_cities = []
        self.visited_cities.append(random.choice(list(self.cities.keys())))


    def init_dist_matrix(self):

        self.dist_matrix = np.zeros((self.D, self.D))
        for i in list(self.cities.keys()):
            for j in list(self.cities.keys()):
                if i != j:
                    self.dist_matrix[i][j] = self.dist(i, j)


    def dist(self, city1, city2):
        """Calculates distance between two cities"""

        x1, y1 = self.cities[city1]
        x2, y2 = self.cities[city2]
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def visit_city(self, pheromone_matrix):

        if np.random.rand() < RANDOM_FACTOR:
            self.visit_random_city()
        else:
            self.visit_probable_city(pheromone_matrix)

    def visit_random_city(self):

        all_cities = set(self.cities.keys())
        open_cities = all_cities - set(self.visited_cities)
        self.visited_cities.append(random.choice(list(open_cities)))

    def visit_probable_city(self, pheromone_matrix):

        curr_city = self.visited_cities[-1]
        all_cities = set(self.cities.keys())
        open_cities = all_cities - set(self.visited_cities)
        all_prob = []
        slices = {}
        total_prob = 0
        for city in open_cities:
            inverse_dist = math.pow(1 / self.dist_matrix[curr_city][city], self.b)
            pheromone = math.pow(pheromone_matrix[curr_city][city], self.a)
            probability = pheromone * inverse_dist
            all_prob.append(probability)
            slices[city] = [total_prob, sum(all_prob)]
            total_prob += probability

        for city in open_cities:
            slices[city] = [a / total_prob for a in slices[city]]

        choice = random.rand()
        result = [city for city in open_cities if slices[city][0] < choice < slices[city][1]][0]

        self.visited_cities.append(result)

    def total_dist(self):
        sequence = self.visited_cities
        return sum([self.dist(sequence[i], sequence[i+1-self.D]) for i in range(self.D)])


class ACO:

    def __init__(self, cities_number, ants_number, alpha, beta, evaporation):
        self.D = cities_number
        self.ants_number = ants_number
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.pheromone_matrix = np.ones((self.D, self.D))
        self.best_distance = math.inf
        self.best_ant = None
        self.all_best_ants = []
        self.init_cities()

    def init_cities(self):
        """Initialize hashable structure to store coordinates of cities"""

        # names of cities
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        self.cities = {}
        for d in range(self.D):  # randomly choose coordinates
            x = np.random.randint(800)
            y = np.random.randint(800)
            # fill in dictionary with cities and coordinates
            self.cities[d] = (x, y)

    def make_colony(self):

        self.ant_colony = []
        for _ in range(self.ants_number):
            self.ant_colony.append(Ant(self.cities, self.D, self.alpha, self.beta))

    def move_ants(self):

        for ant in self.ant_colony:
            ant.visit_city(self.pheromone_matrix)

    def update_pheromones(self):

        for x in range(self.D):
            for y in range(self.D):
                self.pheromone_matrix[x][y] = self.pheromone_matrix[x][y] * self.evaporation
                for ant in self.ant_colony:
                    if ant.visited_cities.index(x) == ant.visited_cities.index(y) - 1:
                        self.pheromone_matrix[x][y] += 1 / ant.total_dist()

    def get_best(self):
        for ant in self.ant_colony:
            distance = ant.total_dist()
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_ant = ant
                self.all_best_ants.append(self.best_ant)

    def run(self, iterations):
        for i in range(iterations):
            self.make_colony()
            for _ in range(self.D-1):
                self.move_ants()
            self.update_pheromones()
            self.get_best()
            print(i, 'SHORTEST PATH: ', self.best_distance)

    def make_animation(self):
        """Returns plot with map of journey"""

        fig, ax = plt.subplots(figsize=(5, 5))
        camera = Camera(fig)  # use simple library from github
        for ant in self.all_best_ants:

            # unpacking coordinates using dictionary with cities
            x_points = [self.cities[city][0] for city in ant.visited_cities]
            y_points = [self.cities[city][1] for city in ant.visited_cities]

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Map of the cities. Total path: %1.1f' % ant.total_dist())
            ax.set_xlim([0, 820])
            ax.set_ylim([0, 820])

            ax.plot(x_points, y_points, 'ko')  # plot points
            ax.plot(x_points, y_points, 'r-')  # plot lines between points
            ax.plot(x_points[-1], y_points[-1], 'go')
            ax.plot(x_points[0], y_points[0], 'bo')

            # place city's name on the plot
            for city in ant.visited_cities:
                x, y = self.cities[city]
                ax.text(x, y + 7, city, fontsize=11)

            camera.snap()  # make a screenshot for animation

        Writer = ani.writers['pillow']
        writer = Writer(fps=0.5)

        animation = camera.animate()
        animation.save('Ant_colony_algorithm.gif', writer=writer)


RANDOM_FACTOR = 0  # can be adjusted to add extra randomness
CITIES_NUMBER = 20
ANT_NUMBER = 20
EVAPORATION = 0.4
ALPHA = 1
BETA = 1

aco = ACO(CITIES_NUMBER, ANT_NUMBER, ALPHA, BETA, EVAPORATION)
aco.run(100)
aco.make_animation()


