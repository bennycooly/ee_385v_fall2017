
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

# define constants
V_m = -60 * (10 ** -3)
g_m = 0.1 * (10 ** 3)
C_m = 1
V_t = -50 * (10 ** -3)
V_reset = -55 * (10 ** -3)

spikes = np.zeros(3000)

def get_I(t):
    if t > 1 and t < 2:
        return 1
    return 0

def v_eul(t):
    v = np.zeros_like(t)
    # delta time of 0.1 ms
    delta_t = 0.1 * (10 ** -3)

    # V(0) = V_reset
    v[0] = V_reset
    for i in range(1, len(v)):
        I = 0
        if i >= 1000 and i < 2000:
            I = 2.5
        # forward euler
        v[i] = v[i - 1] + (delta_t * (((-g_m/C_m) * (v[i - 1] - V_m)) + (I/C_m)))
        if v[i] >= V_t:
            spikes[i] = 1
            v[i] = V_reset
    return v

def s_eul(t, tau_s):
    s = np.zeros_like(t)
    # delta time of 0.1 ms
    delta_t = 0.1 * (10 ** -3)

    # S(0) = 0
    s[0] = 0
    for i in range(1, len(s)):
        I = 0
        if i >= 1000 and i < 2000:
            I = 2.5
        # forward euler
        s[i] = s[i - 1] + (delta_t * ((-s[i - 1]/tau_s) + r(I)))
    return s

def r(I):
    return 1/((C_m/g_m) * np.log(((I/g_m) + V_m - V_reset)/((I/g_m) + V_m - V_t)))


def main():
    
    t = np.linspace(0, 3, 3000)
    plt.figure(1)
    plt.subplot(211)
    plt.ylabel("Voltage (v))")
    plt.xlabel("Time (s)")
    plt.plot(t, v_eul(t))

    plt.subplot(212)
    plt.plot(t, s_eul(t, 10 * (10 ** -3)), label = "10 ms")
    plt.plot(t, s_eul(t, 50 * (10 ** -3)), label = "50 ms")
    plt.plot(t, s_eul(t, 100 * (10 ** -3)), label = "100 ms")
    plt.legend()
    plt.ylabel("S")
    plt.xlabel("Time (s)")
    plt.savefig("problem2.png")
    plt.show()


if (__name__ == "__main__"):
    main()
