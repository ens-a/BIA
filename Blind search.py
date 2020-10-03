import numpy as np
from  Functions import *
#There are many possibe ways to generate random float. In this solution numpy.random.uniform was used 
#Functions names:
#['sphere', 'ackley', 'rastrigin', 'rosenbrock', 'griewank', 'schwefel', 'levy', 'micha','zakhar']
def blind_search(func_name, dimension, n_iter, seed = 123):
    np.random.seed(seed)     #for reproducibility
    func = func_dic[func_name][0]    #choose function
    low_lim = func_dic[func_name][1][0]  #find limits
    high_lim = func_dic[func_name][1][1]
    x_rnd = np.random.uniform(low_lim, high_lim, dimension) #generate initial random vector of x
    best_value = func(x_rnd) #iniciate best value
    for i in range(n_iter):  #make a loop for a number of iterations
        x_rnd = np.random.uniform(low_lim, high_lim, dimension) #generate random vector x
        temp_value = func(x_rnd) #calculate function value at x
        if temp_value < best_value: #compere calculated value with best one
            best_value = temp_value #save new best value and best vector x
            best_x = x_rnd
    return [best_value, best_x]