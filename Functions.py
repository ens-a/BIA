#xi ∈ [-5.12, 5.12]
def sphere(x):
    f = np.sum([x**2 for x in x])
    return f

import numpy as np
pi = np.pi

#xi ∈ [-32.768, 32.768]
def ackley(x):
    a = 20
    b = 0.2
    c = 2*pi
    d = len(x)
    sum1 = np.sum([x**2 for x in x])
    sum2 = np.sum([np.cos(c*x) for x in x])
    
    f = a + np.exp(1) -a * np.exp(-b * np.sqrt(sum1/d)) - np.exp(sum2/d)
    return f

#xi ∈ [-5.12, 5.12]
def rastrigin(x):
    d = len(x)
    sum1 = np.sum([x**2 for x in x])
    sum2 = np.sum([10*np.cos(2*pi*x) for x in x])
    f = 10*d + sum1 - sum2
    return f

#xi ∈ [-5, 10]
def rosenbrock(x):
    d = len(x)
    sum1 = 0
    for i in range(d-1):
        sum1 += (100*(x[i+1] - x[i]**2))**2
    sum2 = np.sum([(x-1)**2 for x in x])
    f = sum1 + sum2
    return f

#xi ∈ [-600, 600]
def griewank(x):
    d = len(x)
    sum1 = np.sum([x**2/4000 for x in x])
    prod1 = 0
    for i in range(d):
        prod1 *= np.cos(x[i]/np.sqrt(i+1))
    f = sum1 - prod1 +1 
    return f

#xi ∈ [-500, 500]
def schwefel(x):
    d = len(x)
    c = 418.9829
    sum1 = np.sum([x*np.sin(np.sqrt(np.absolute(x))) for x in x])
    f = c*d - sum1
    return f

#xi ∈ [-10, 10]
def levy(x):
    d = len(x)
    w = [1 + (x -1)/4 for x in x]
    c1 = np.sin(pi*w[0])**2
    c2 = (w[-1] - 1)**2*(1 + np.sin(2*pi*w[-1])**2)
    sum1 = np.sum([(w - 1)**2*(1 + 10*np.sin(pi*w + 1)**2) for w in w])
    f = c1 + sum1 + c2
    return f

#xi ∈ [0, π]
def micha(x):
    m = 10
    d = len(x)
    sum1 = 0
    for i in range(d):
        sum1 += np.sin(x[i])*np.sin((i+1)/pi*x[i]**2)**(2*m)
    f = -sum1
    return f

#xi ∈ [-5, 10]
def zakhar(x):
    d = len(x)
    sum1 = np.sum([x**2 for x in x])
    sum2 = 0
    for i in range(d):
        sum2 += 0.5*(i+1)*x[i]
    f = sum1 + sum2**2 + sum2**4
    return f

func_dic = {'sphere': [sphere, [-5.12, 5.12]],
            'ackley': [ackley, [-32.768, 32.768]],
            'rastrigin': [rastrigin, [-5.12, 5.12]],
            'rosenbrock': [rosenbrock, [-5, 10]],
            'griewank': [griewank, [-600, 600]],
            'schwefel': [schwefel, [-500, 500]],
            'levy': [levy, [-10, 10]],
            'micha': [micha, [0, pi]],
            'zakhar': [zakhar, [-5, 10]]}
#small comment