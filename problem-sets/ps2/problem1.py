import scipy as sp
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

W_th = 4

class Problem1:
    def __init__(self):
        self.W = W_th
        self.b = - self.W / 2
        self.db = 0

    def g(self, g):
        return np.exp(g)/(1 + np.exp(g))

    def s(self, t):
        return odeint(self.f, -0.5, t)

    def f(self, s, t):
        return -s + self.g(self.W * s + self.b + self.db)
    
    def state(self, db, initial):
        res = np.zeros_like(db)
        # initial lower state
        index = 0
        if initial == "low":
            for i in np.nditer(db, op_flags=['readwrite']):
                if (i > 3):
                    res[index] = 1
                else:
                    res[index] = 0
                index += 1
        
        # initial upper state
        elif initial == "high":
            for i in np.nditer(db, op_flags=['readwrite']):
                if (i < -1):
                    res[index] = 0
                else:
                    res[index] = 1
                index += 1
        print(res)
        return res

def main():
    functions = Problem1()
    t = np.linspace(0, 10, 1000)
    plt.figure(1, figsize=(10, 8))
    plt.subplot(321)
    plt.title("S(t) for various W")
    plt.ylabel("S")
    plt.xlabel("Time (s)")
    plt.plot(t, functions.s(t), label = "W = 4 (W*)")
    functions.W = 6
    plt.plot(t, functions.s(t), label = "W = 6")
    functions.W = 2
    plt.plot(t, functions.s(t), label = "W = 2")
    plt.legend()

    plt.subplot(322)
    t = np.linspace(0, 2, 100)
    plt.title("f(t) for various W")
    plt.ylabel("f(t)")
    plt.xlabel("Time (s)")
    functions.W = W_th
    plt.plot(t, functions.g(functions.W * (t - 0.5)), label = "W = 4 (W*)")
    functions.W = 6
    plt.plot(t, functions.g(functions.W * (t - 0.5)), label = "W = 6")
    functions.W = 2
    plt.plot(t, functions.g(functions.W * (t - 0.5)), label = "W = 2")
    plt.plot(t, t, label = "f(t) = t")
    plt.legend()

    plt.subplot(323)
    t = np.linspace(0, 10, 1000)
    functions.W = 6
    plt.plot(t, functions.s(t), label = "db = 0")
    functions.db = 3
    plt.plot(t, functions.s(t), label = "db = 3 (upper state unstable)")
    functions.db = -1
    plt.plot(t, functions.s(t), label = "db = -1 (lower state unstable)")
    plt.title("S(t) for various db (W = 6)")
    plt.ylabel("S")
    plt.xlabel("t")
    plt.legend()    

    plt.subplot(324)
    db = np.linspace(-5, 5, 1000)
    plt.title("State for various db")
    plt.ylabel("State")
    plt.xlabel("db")
    plt.plot(db, functions.state(db, "low"), label = "initial low")
    plt.plot(db, functions.state(db, "high"), label = "initial high")
    plt.legend()

    plt.tight_layout()
    plt.savefig("problem1.png")
    plt.show()
    



if (__name__ == "__main__"):
    main()
