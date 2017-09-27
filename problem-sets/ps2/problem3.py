import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import math
import random

tau = 50

def s(t):
    return odeint(f, 0, t)

def b(t):
    res = np.zeros_like(t)
    index = 0
    # print(t)
    for i in np.nditer(t):
        if math.floor(i) % tau == 0:
            res[index] = random.uniform(1.5, 2.5)
        else:
            res[index] = 2.0
        index += 1
    
    # print(res)
    return res

def b_(t):
    if math.floor(t) % tau == 0:
        return random.uniform(2.0, 2.5)
    return 2.0

def f(s, t):
    W = 1
    # b_ = b(np.linspace(0, 200, 2000))
    return (-s + (W * s) + b_(t)) / tau

def g(t):
    res = np.zeros_like(t)
    index = 0
    for i in np.nditer(t):
        res[index] = quad(b_, 0, index)
        index += 1
    return res

def main():
    t = np.linspace(0, 200, 1000)
    plt.figure(1)
    plt.subplot(211)
    plt.ylabel("Input (b(t))")
    plt.xlabel("Time (ms)")
    plt.plot(t, b(t))
    plt.plot(t, b(t))
    # plt.plot(t, g(t))
    plt.legend()


    plt.tight_layout()
    plt.savefig("problem3.png")
    plt.show()
    



if (__name__ == "__main__"):
    main()