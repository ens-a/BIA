

import numpy as np
import pandas as pd
import time
from Functions import *
from Differential_Evolution_ens0005 import DifferentialEvaluation
from Particle_Swarm_Optimization_ens0005 import SwarmOptimization
from Self_organizing_migrating_algorithm_ens0005 import SOMA
from Firefligh_algorithm_ens0005 import FireflySwarmOptimization
from Learning_Based_optimization_ens0005 import LearningBasedOptimisation
start_time_init = time.time()

functions_list = list(func_dic.keys())

EXPERIMENTS = 1
D = 30
NP = 30
M_max = 3000


for name in functions_list:

    data = {'DE': [], 'PSO': [], 'SOMA': [], 'FA': [], 'TLBO': []}

    # Differential Evaluation
    CR = 0.5
    F = 0.5
    for i in range(EXPERIMENTS):
        start_time = time.time()
        algorithm = DifferentialEvaluation(name, D)
        algorithm.make_population(NP)
        algorithm.run(F, CR, M_max)
        answer = sorted(algorithm.population, key=algorithm.function)[0]
        data['DE'].append(algorithm.function(answer))
        print(name, 'DE', i, 'done')
        print("--- %s seconds ---" % (time.time() - start_time))

    # Particle Swarm Optimization
    C1 = 2
    C2 = 2
    for i in range(EXPERIMENTS):
        start_time = time.time()
        algorithm = SwarmOptimization(name, D)
        algorithm.make_swarm(NP)
        algorithm.run(C1, C2, M_max)
        answer = sorted(algorithm.swarm, key=lambda x: algorithm.function(x.parameters))[0]
        data['PSO'].append(algorithm.function(answer.parameters))
        print(name, 'PSO', i, 'done')
        print("--- %s seconds ---" % (time.time() - start_time))

    # Self Organizing Migration Algorithm
    path = 3
    step = 0.3
    PRT = 0.4
    for i in range(EXPERIMENTS):
        start_time = time.time()
        algorithm = SOMA(name, D)
        algorithm.run(NP, path, step, 200, PRT)
        answer = sorted(algorithm.population, key=lambda x: algorithm.function(x.parameters))[0]
        data['SOMA'].append(algorithm.function(answer.parameters))
        print(name, 'SOMA', i, 'done')
        print("--- %s seconds ---" % (time.time() - start_time))

    # Firefly algorithm
    alpha = 0.1
    beta = 1
    for i in range(EXPERIMENTS):
        start_time = time.time()
        algorithm = FireflySwarmOptimization(name, D)
        algorithm.make_swarm(NP)
        algorithm.run(alpha, beta, 200)
        answer = sorted(algorithm.swarm, key=lambda x: x.evaluate())[0]
        data['FA'].append(algorithm.function(answer.parameters))
        print(name, 'FA', i, 'done')
        print("--- %s seconds ---" % (time.time() - start_time))

    # Teaching-Learning based algorithm
    for i in range(EXPERIMENTS):
        start_time = time.time()
        algorithm = LearningBasedOptimisation(name, D)
        algorithm.make_population(NP)
        algorithm.run(M_max)
        answer = algorithm.teacher
        data['TLBO'].append(algorithm.function(answer))
        print(name, 'TLBO', i, 'done')
        print("--- %s seconds ---" % (time.time() - start_time))


    df = pd.DataFrame(data)

    if name == 'sphere':
        df.to_excel('outcome.xlsx', sheet_name=name + ' func')
    else:
        with pd.ExcelWriter('outcome.xlsx',
                            mode='a') as writer:
            df.to_excel(writer, sheet_name=name + ' func')


print("--- %s seconds ---" % (time.time() - start_time_init), 'Total time')