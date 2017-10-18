
import scipy as sp
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

class Hopfield():
    def __init__(self, P, N):
        # Size of each pattern
        self.N = N

        # Number of patterns
        self.P = P

        # Weight matrix
        self.W = np.zeros((N, N))

        # Crosstalk terms
        self.C = np.zeros((P, N))

        # Training patterns
        self.patterns = np.zeros((P, N))
    
    def create_patterns(self):
        self.patterns = np.random.choice([-1, 1], size=(self.P, self.N))

    def calculate_weight_matrix(self):
        for i in range(self.N):
            for j in range(i, self.N):
                # print(str(i) + "," + str(j))
                # w_ii = 0
                if i == j:
                    self.W[i][j] = 0
                    continue

                pattern_sum = 0
                for mu in range(self.P):
                    pattern_sum += self.patterns[mu][i] * self.patterns[mu][j]
                self.W[i][j] = (1 / self.N) * pattern_sum
                # symmetric
                self.W[j][i] = self.W[i][j]

    def collect_crosstalk(self):
        for v in range(self.P):
            for i in range(self.N):
                # print(str(i) + "," + str(j))
                pattern_sum = 0
                for mu in range(self.P):
                    if mu == v:
                        continue
                    for j in range(self.N):
                        if j == i:
                            continue
                        pattern_sum += self.patterns[mu][i] * self.patterns[mu][j] * self.patterns[v][j]
                self.C[v][i] = (1 / self.N) * pattern_sum
    

class SparseHopfield(Hopfield):
    def __init__(self, P, N, f):
        super().__init__(P, N)
        self.f = f
    
    def create_patterns(self):
        self.patterns = np.random.choice([0, 1], size=(self.P, self.N), p=[1 - self.f, self.f])
    
    def collect_crosstalk(self):
        th = 0
        for v in range(self.P):
            for i in range(self.N):
                # print(str(i) + "," + str(j))
                pattern_sum = 0
                for mu in range(self.P):
                    if mu == v:
                        continue
                    for j in range(self.N):
                        if j == i:
                            continue
                        pattern_sum += (self.patterns[mu][i] - self.f) * (self.patterns[mu][j] - self.f) * self.patterns[v][j]
                self.C[v][i] = (1 / (self.N * self.f * (1 - self.f))) * pattern_sum - th

def plot_crosstalk_hist():
    N = 500
    plt.figure(1, figsize=(10, 8))

    # P = 100
    P = 50
    hopfield = Hopfield(P, N)
    hopfield.create_patterns()
    # print(hopfield.patterns)
    # hopfield.calculate_weight_matrix()
    # print(hopfield.W)
    hopfield.collect_crosstalk()
    # hopfield.plot_crosstalk_hist()

    plt.subplot(311)
    plt.title("Crosstalk with " + "(N, P) = " + "(" + str(N) + ", " + str(P) + ")")
    plt.hist(np.ndarray.flatten(hopfield.C), bins = 50)

    # P = 140
    P = 70
    hopfield = Hopfield(P, N)
    hopfield.create_patterns()
    hopfield.collect_crosstalk()
    # hopfield.plot_crosstalk_hist()

    plt.subplot(312)
    plt.title("Crosstalk with " + "(N, P) = " + "(" + str(N) + ", " + str(P) + ")")
    plt.hist(np.ndarray.flatten(hopfield.C), bins = 50)

    # P = 200
    P = 100
    hopfield = Hopfield(P, N)
    hopfield.create_patterns()
    hopfield.collect_crosstalk()
    # hopfield.plot_crosstalk_hist()

    plt.subplot(313)
    plt.title("Crosstalk with " + "(N, P) = " + "(" + str(N) + ", " + str(P) + ")")
    plt.hist(np.ndarray.flatten(hopfield.C), bins = 50)

    plt.tight_layout()
    plt.savefig("problem3a.png")
    plt.show(block=False)


def plot_crosstalk_hist_sparse():
    N = 20
    f = 0.05
    
    plt.figure(2, figsize=(10, 8))

    # P = 100
    P = 80
    hopfield = SparseHopfield(P, N, f)
    hopfield.create_patterns()
    # print(hopfield.patterns)
    # hopfield.calculate_weight_matrix()
    # print(hopfield.W)
    hopfield.collect_crosstalk()
    # hopfield.plot_crosstalk_hist()

    plt.subplot(311)
    plt.title("Crosstalk with " + "(N, P) = " + "(" + str(N) + ", " + str(P) + ")")
    plt.hist(np.ndarray.flatten(hopfield.C), bins = 50)
    print((hopfield.C > 1).sum())
    # P = 140
    P = 100
    hopfield = SparseHopfield(P, N, f)
    hopfield.create_patterns()
    hopfield.collect_crosstalk()
    # hopfield.plot_crosstalk_hist()

    plt.subplot(312)
    plt.title("Crosstalk with " + "(N, P) = " + "(" + str(N) + ", " + str(P) + ")")
    plt.hist(np.ndarray.flatten(hopfield.C), bins = 50)
    print((hopfield.C > 1).sum())

    # P = 200
    P = 120
    hopfield = SparseHopfield(P, N, f)
    hopfield.create_patterns()
    hopfield.collect_crosstalk()
    # hopfield.plot_crosstalk_hist()

    plt.subplot(313)
    plt.title("Crosstalk with " + "(N, P) = " + "(" + str(N) + ", " + str(P) + ")")
    plt.hist(np.ndarray.flatten(hopfield.C), bins = 50)
    print((hopfield.C > 1).sum())

    plt.tight_layout()
    plt.savefig("problem3b.png")
    plt.show(block=False)


def main():
    plot_crosstalk_hist()
    plot_crosstalk_hist_sparse()


if __name__ == "__main__":
    main()
