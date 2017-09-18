
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

# define constants


def r(I):
    V_m = -60 * (10 ** -3)
    g_m = 0.1 * (10 ** 3)
    C_m = 1
    V_t = -50 * (10 ** -3)
    V_reset = -55 * (10 ** -3)
    return 1/((C_m/g_m) * np.log(((I/g_m) + V_m - V_reset)/((I/g_m) + V_m - V_t)))


def main():
    
    I = np.linspace(0.5, 3)
    plt.ylabel("r (Hz)")
    plt.xlabel("I (A)")
    plt.plot(I, r(I))
    # plt.show()
    plt.savefig("problem1.png")
    
    plt.show()


if (__name__ == "__main__"):
    main()
