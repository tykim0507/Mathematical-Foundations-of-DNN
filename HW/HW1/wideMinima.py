import numpy as np
import matplotlib.pyplot as plt


np.seterr(invalid='ignore', over='ignore')  # suppress warning caused by division by inf

def f(x):
    return 1/(1 + np.exp(3*(x-3))) * 10 * x**2  + 1 / (1 + np.exp(-3*(x-3))) * (0.5*(x-10)**2 + 50)

def fprime(x):
    return 1 / (1 + np.exp((-3)*(x-3))) * (x-10) + 1/(1 + np.exp(3*(x-3))) * 20 * x + (3* np.exp(9))/(np.exp(9-1.5*x) + np.exp(1.5*x))**2 * ((0.5*(x-10)**2 + 50) - 10 * x**2) 

x = np.linspace(-5,20,100)
N = 10000 #number of iterations

# Gradient descent
x0 = np.random.uniform(-5,20)
alpha = 0.01
for i in range(N):
    x0 -= alpha * fprime(x0)
    if i % 5 == 0:
        plt.plot(x0, f(x0), 'ro')
plt.plot(x,f(x), 'k')
plt.show()
