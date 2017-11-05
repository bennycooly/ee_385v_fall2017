import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# returns the roots of unity
def roots(j_range):
    N = j_range.size
    res = np.copy(j_range)
    for j in np.nditer(res, op_flags=["readwrite"]):
        # print(j)
        j[...] = pow(math.exp((2 * np.pi * np.sqrt(-1) * j) / N), j)

    return res

def eigenvalues(j_range, c):
    print(j_range.size)
    res = np.copy(j_range)
    for j in np.nditer(res, op_flags=["readwrite"]):
        j[...] = np.dot(roots(j_range), c)
    print(res)
    return res



def main():
    N = 5
    c = np.zeros(N)
    c_val = 2
    c[1] = c_val
    c[N-1] = c_val
    j = np.arange(0, N)
    # j = np.linspace(0, N, N)
    print(c)
    print(j)
    print(roots(j))
    print(eigenvalues(j, c))

    x = np.linspace(0, 1, 1000)
    plt.figure(1, figsize=(10, 8))
    plt.subplot(421)
    plt.title("Problem 4a: Eigenvalue Spectrum")
    plt.ylabel("Eigenvalue")
    plt.xlabel("j")
    plt.plot(j, eigenvalues(j, c))
    # plt.plot(x, L(x, beta, gamma), label="β = 4 (infinite local minima)")
    # beta = 3.9
    # plt.plot(x, L(x, beta, gamma), label="β = 3.9 (1 local minima)")
    # beta = 4.1
    # plt.plot(x, L(x, beta, gamma), label="β = 4.1 (2 local minima)")
    plt.legend()

    


    plt.tight_layout()
    plt.savefig("problem3.png")
    plt.show()
    



if (__name__ == "__main__"):
    main()
