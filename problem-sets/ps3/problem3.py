
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

        # Create patterns
        self.patterns = []
    
    def create_patterns(self):
        for i in range(self.P):
            self.patterns.append(np.random.choice([-1, 1], size=(self.N)))

    def calculate_weight_matrix(self):
        for i in range(self.N):
            for j in range(self.N):
                # print(str(i) + "," + str(j))
                pattern_sum = 0
                for mu in range(self.P):
                    pattern_sum += self.patterns[mu][i] * self.patterns[mu][j]
                self.W[i][j] = (1 / self.N) * pattern_sum


def main():
    hopfield = Hopfield(1000, 100)
    hopfield.create_patterns()
    # print(hopfield.patterns)
    hopfield.calculate_weight_matrix()
    print(hopfield.W)


if __name__ == "__main__":
    main()
