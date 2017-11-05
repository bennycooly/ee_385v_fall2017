import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import math
import random
from scipy.linalg import circulant

# returns the roots of unity
def roots(j, N):
    res = np.arange(0, N)
    for i in np.nditer(res, op_flags=["readwrite"]):
        # print(i)
        i[...] = pow(math.exp((2 * np.pi * j) / N), i)

    return res

def eigenvalues(j_range, c):
    res = np.copy(j_range)
    for j in np.nditer(res, op_flags=["readwrite"]):
        j[...] = np.inner(roots(j, j_range.size), c)
    print(res)
    return res



def main():
    N = 100
    c = np.zeros(N)
    c_val = 1
    c[1] = c_val
    c[N-1] = c_val
    j = np.arange(0, N)

    print(circulant(c))
    print(la.eigvals(circulant(c)).argsort()[::-1])

    x = np.linspace(0, 1, 1000)
    plt.figure(1, figsize=(10, 8))
    plt.subplot(421)
    plt.title("Problem 4a: Eigenvalue Spectrum, N=" + str(N))
    plt.ylabel("Eigenvalue")
    plt.xlabel("Index (j)")
    plt.plot(j, la.eigvals(circulant(c)).argsort()[::-1], label="c=" + str(c_val))
    c_val = 2
    c[1] = c_val
    c[N-1] = c_val
    plt.plot(j, la.eigvals(circulant(c)).argsort()[::-1], label="c=" + str(c_val))
    c_val = -2
    c[1] = c_val
    c[N-1] = c_val
    plt.plot(j, la.eigvals(circulant(c)).argsort()[::-1], label="c=" + str(c_val))
    c_val = -200
    c[1] = c_val
    c[N-1] = c_val
    plt.plot(j, la.eigvals(circulant(c)).argsort()[::-1], label="c=" + str(c_val))
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
