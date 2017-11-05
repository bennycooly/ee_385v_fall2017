import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt
import math
import random

tau = 50

def F(x):
    return (x * np.log(x)) + ((1 - x) * np.log(1 - x)) - 1

def L(x, beta, gamma):
    W = np.full((x.size, x.size), beta)
    # b = np.full((x.size), - (beta / 2) * (1 + gamma))
    # return F(x) - (0.5 * np.transpose(x) @ W @ x) - (np.transpose(b) @ x)
    b = -(beta/2) * (1 + gamma)
    return F(x) - (0.5 * beta * x * x) - (b * x)

def L_mat(x, beta, gamma):
    W = np.full((x.size, x.size), beta)
    b = np.full((x.size), - (beta / 2) * (1 + gamma))
    return F(x) - (0.5 * np.transpose(x) @ W @ x) - (np.transpose(b) @ x)
    # b = -(beta/2) * (1 + gamma)
    # return F(x) - (0.5 * beta * x * x) - (b * x)

def get_fixed_points(x, beta, gamma):
    return x[argrelextrema(L(x, beta, gamma), np.less)]

def fixed_points(gamma, x, beta):
    res = np.copy(gamma)
    for g in np.nditer(res, op_flags=["readwrite"]):
        fixed_points = argrelextrema(L(x, beta, g), np.less)
        g[...] = x[fixed_points[0][0]]
    
    return res

def fixed_points_second(gamma, x, beta):
    res = np.copy(gamma)
    for g in np.nditer(res, op_flags=["readwrite"]):
        fixed_points = argrelextrema(L(x, beta, g), np.less)
        if fixed_points[0].size == 2:
            g[...] = x[fixed_points[0][1]]
        else:
            g[...] = None
    
    return res


def main():
    beta = 4
    gamma = 0

    x = np.linspace(0, 1, 1000)
    plt.figure(1, figsize=(10, 8))
    plt.subplot(421)
    plt.title("Problem 2d: Lyapunov Function")
    plt.ylabel("L(x)")
    plt.xlabel("x")
    plt.plot(x, L(x, beta, gamma), label="β = 4 (infinite local minima)")
    beta = 3.9
    plt.plot(x, L(x, beta, gamma), label="β = 3.9 (1 local minima)")
    beta = 4.1
    plt.plot(x, L(x, beta, gamma), label="β = 4.1 (2 local minima)")
    plt.legend()

    plt.subplot(423)
    plt.title("Problem 2e: Lyapunov Function (β=5)")
    plt.ylabel("L(x)")
    plt.xlabel("x")
    gamma = -0.25
    plt.plot(x, L(x, beta, gamma), label="γ = -0.25")
    gamma = 0
    plt.plot(x, L(x, beta, gamma), label="γ = 0")
    gamma = 0.25
    plt.plot(x, L(x, beta, gamma), label="γ = 0.25")
    plt.legend()

    
    plt.subplot(424)
    plt.title("Problem 2e: Stable Fixed Points (β=5)")
    beta = 5
    gamma = -0.25
    gamma_range = np.linspace(-0.25, 0.25)
    plt.ylabel("x")
    plt.xlabel("γ")
    plt.plot(gamma_range, fixed_points(gamma_range, x, beta), label="First fixed point")
    plt.plot(gamma_range, fixed_points_second(gamma_range, x, beta), label="Second fixed point")
    plt.legend()


    # 2-D Space

    plt.subplot(425)
    plt.title("Problem 2f: Lyapunov Function in Bistable Switch")
    plt.ylabel("L(x)")
    x1 = np.linspace(0, 1)
    x2 = np.linspace(0, 1)
    x = np.meshgrid(x1, x2)
    print(x)
    beta = -4
    plt.xlabel("x")
    plt.plot(x, L_mat(x, beta, gamma), label="beta = 4 (infinite local minima)")
    beta = -3.9
    plt.plot(x, L_mat(x, beta, gamma), label="beta = 3.9 (1 local minima)")
    beta = -4.1
    plt.plot(x, L_mat(x, beta, gamma), label="beta = 4.1 (2 local minima)")
    plt.legend()


    plt.tight_layout()
    plt.savefig("problem2.png")
    plt.show()
    



if (__name__ == "__main__"):
    main()
