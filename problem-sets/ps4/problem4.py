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
    N = 20
    j = np.arange(0, N)

    c = np.zeros(N)
    
    # print(la.eigvals(circulant(c)).argsort()[::-1])

    plt.figure(1, figsize=(10, 8))
    plt.subplot(421)
    plt.title("Problem 4a: Eigenvalue Spectrum, N=" + str(N))
    plt.ylabel("Eigenvalues λ^2")
    plt.xlabel("Index (j)")

    c_val = 0.99
    c[1] = c_val
    c[N-1] = c_val
    print(circulant(c))
    eigenvalues, eigenvectors = la.eig(circulant(c))
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    plt.plot(j, eigenvalues * eigenvalues, label="c=" + str(c_val))

    c_val = -1
    c[1] = c_val
    c[N-1] = c_val
    eigenvalues, eigenvectors = la.eig(circulant(c))
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    plt.plot(j, eigenvalues * eigenvalues, label="c=" + str(c_val))

    c_val = 0.75
    c[1] = c_val
    c[N-1] = c_val
    eigenvalues, eigenvectors = la.eig(circulant(c))
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    plt.plot(j, eigenvalues * eigenvalues, label="c=" + str(c_val))

    c_val = -2
    c[1] = c_val
    c[N-1] = c_val
    eigenvalues, eigenvectors = la.eig(circulant(c))
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    plt.plot(j, eigenvalues * eigenvalues, label="c=" + str(c_val))

    c_val = 0.5
    c[1] = c_val
    c[N-1] = c_val
    eigenvalues, eigenvectors = la.eig(circulant(c))
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    plt.plot(j, eigenvalues * eigenvalues, label="c=" + str(c_val))

    # c_val = -5
    # c[1] = c_val
    # c[N-1] = c_val
    # eigenvalues, eigenvectors = la.eig(circulant(c))
    # idx = eigenvalues.argsort()[::-1]
    # eigenvalues = eigenvalues[idx]
    # eigenvectors = eigenvectors[:,idx]
    # plt.plot(j, eigenvalues * eigenvalues, label="c=" + str(c_val))

    c_val = -0.5
    eigenvalues, eigenvectors = la.eig(circulant(c))
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    print(eigenvalues * eigenvalues)
    # plt.plot(x, L(x, beta, gamma), label="β = 4 (infinite local minima)")
    # beta = 3.9
    # plt.plot(x, L(x, beta, gamma), label="β = 3.9 (1 local minima)")
    # beta = 4.1
    # plt.plot(x, L(x, beta, gamma), label="β = 4.1 (2 local minima)")
    plt.legend()

    


    plt.tight_layout()
    plt.savefig("problem4.png")
    plt.show()
    



if (__name__ == "__main__"):
    main()
