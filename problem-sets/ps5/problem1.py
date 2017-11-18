
import matplotlib.pyplot as plt
import numpy as np
from mlp import MLP


def plot():
    plt.figure(1, figsize=(10, 8))
    plt.subplot(421)
    plt.title("MLP Error")
    plt.ylabel("Squared Error")
    plt.xlabel("Epoch (5000 iterations)")

    x = np.arange(0, 20, 1)
    y = np.array([
        8.004993983,
        8.00486582828,
        7.00831649819,
        5.01539610946,
        4.03815700898,
        2.04648382919,
        2.04648180676,
        2.04647895837,
        2.04647461668,
        2.0464670669,
        2.04644986189,
        2.04634447829,
        1.05918287277,
        1.94476624932,
        0.950615489642,
        0.950615489642,
        0.950615489642,
        0.950615489642,
        0.950615489642,
        0.950615489642
    ])
    y_1 = np.array([

    ])
    plt.plot(x, y, label="Training error")
    plt.legend()

    plt.savefig("problem1.png")
    plt.show()

def main():
    mlp = MLP("mnistabridged.mat", 25)
    # mlp.train()
    plot()


if __name__ == "__main__":
    main()
