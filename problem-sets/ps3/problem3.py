
import scipy as sp
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

class Hopfield():
    def __init__(self, N, P):
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
    
    def plot_crosstalk_hist(self):
        plt.hist(self.C, bins = 50)
        plt.show()

def main():
    hopfield = Hopfield(10, 50)
    hopfield.create_patterns()
    # print(hopfield.patterns)
    hopfield.calculate_weight_matrix()
    # print(hopfield.W)
    hopfield.collect_crosstalk()
    hopfield.plot_crosstalk_hist()


if __name__ == "__main__":
    main()
