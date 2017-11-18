
import numpy as np
import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


class MLP:
    def __init__(self, mat_filename, num_hidden):
        self.B = 0
        self.data = loadmat(mat_filename)
        print(self.data.keys())

        self.training_set = Dataset(self.data.get("train").transpose(), self.data.get("trainlabels"))
        # data_point = 321
        # pixels = np.reshape(self.training_set.data[data_point], (28, 28))
        # print(self.data.get("trainlabels")[data_point])
        # plt.imshow(pixels)
        # plt.show()
        self.testing_set = Dataset(self.data.get("test").transpose(), self.data.get("testlabels"))

        self.input_layer = np.zeros(self.training_set.data.shape[1])
        # print(len(self.input_layer))

        self.hidden_layer = np.zeros(num_hidden)
        # print(self.hidden_layer)

        self.output_layer = np.zeros(10)

        self.layers = [
            self.input_layer,
            self.hidden_layer,
            self.output_layer
        ]

        weights_input_hidden = np.random.rand(len(self.input_layer), len(self.hidden_layer))
        weights_hidden_output = np.random.rand(len(self.hidden_layer), len(self.output_layer))

        # weights[i] represents the weight matrix from layer i to layer i + 1
        self.weights = [
            weights_input_hidden,
            weights_hidden_output
        ]

        # print(self.weights)

        
        # store the deltas generated from backprop
        self.deltas = [
            np.zeros_like(self.hidden_layer),
            np.zeros_like(self.output_layer)
        ]

        # num_neurons = len(self.input_layer) + len(self.hidden_layer) + len(self.output_layer)
        # self.weights = np.random.rand(num_neurons, num_neurons)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)
    
    # Calculate the output for two layers based on the activation
    def activate(self, input_layer, output_layer, weights):
        j = 0
        for output in np.nditer(output_layer, op_flags=["readwrite"]):
            weighted_sum = 0
            i = 0
            for input in np.nditer(input_layer, op_flags=["readwrite"]):
                weighted_sum += input * weights[i][j]
                i += 1
            # print("sum is " + str(weighted_sum))
            output[...] = self.sigmoid(weighted_sum)
            j += 1

    # Propagate the activations forward to the output layer
    def forward_propagate(self):
        for i, layer in enumerate(self.layers):
            # don't run on the output layer
            if i == len(self.layers) - 1:
                break

            self.activate(layer, self.layers[i + 1], self.weights[i])
    
    
    # Backprop calculation
    def backward_propagate(self, solution):
        for i, layer in enumerate(reversed(self.layers)):
            # get the true layer index since we are iterating in reverse
            i = len(self.layers) - 1 - i
            
            # the index of the error layer is i - 1
            deltas_index = i - 1

            # ignore the input layer
            if i == 0:
                continue
            
            errors = np.zeros_like(layer)
            # output layer
            if i == len(self.layers) - 1:
                errors = solution - layer
                # self.deltas[errors_index] = errors
                # print("Errors for output layer: ")
                # print(errors)
            
            # a hidden layer
            else:
                for j in range(len(layer)):
                    error = 0
                    # calculate weighted sum of the errors from the current layer to the next layer
                    for k in range(len(self.layers[i + 1])):
                        error += self.weights[i][j][k] * self.deltas[i][k]
                    errors[j] = error
                # print("Errors for hidden layer: ")
                # print(errors)

            for j in range(len(layer)):
                self.deltas[deltas_index][j] = errors[j] * self.sigmoid_derivative(layer[j])

            # print("Deltas: ")
            # print(self.deltas)
            
    # Update the weights
    def update_weights(self, learning_rate):
        # print("Updating weights...")
        # for layer_index, weights in enumerate(self.weights):
        #     for i in range(weights.shape[0]):
        #         for j in range(weights.shape[1]):
        #             weights[i][j] = weights[i][j] + learning_rate * self.deltas[layer_index][j] * self.layers[layer_index + 1][j]

        for layer_index, layer in enumerate(self.layers):
            if layer_index == len(self.layers) - 1:
                break
            for i in range(len(layer)):
                for j in range(len(self.layers[layer_index + 1])):
                    self.weights[layer_index][i][j] += learning_rate * self.deltas[layer_index][j] * self.layers[layer_index][i]
        
        # print("Updated weights:")
        # print(self.weights)
        
    def train(self):
        print("Training network...")
        num_epochs = 1000
        sum_error = 0
        for epoch in range(num_epochs):
            print("Epoch: " + str(epoch))
            for i in range(len(self.training_set.data)):
                # print(i)
                # if i == 100:
                #     break
                for j in range(len(self.training_set.data[i])):
                    self.layers[0][j] = self.training_set.data[i][j]
                # print(self.layers[0])
                
                self.forward_propagate()
                # print(self.layers[1])
                # print(self.layers[2])

                solution = np.zeros(len(self.layers[len(self.layers) - 1]))
                solution[self.training_set.labels[i]] = 1
                # print("Solution:")
                # print(solution)

                # print("Output:")
                # print(self.layers[2])

                error = solution - self.layers[2]
                sum_error = np.sum(error * error)
                if sum_error < 1:
                    break
                
                self.backward_propagate(solution)

                # print("Weights before:")
                # print(self.weights)
                self.update_weights(0.4)
                # print(sum_error)
                # print("Weights after:")
                # print(self.weights)
                # print("Error:")
                # print(error)
            
            print("Error: " + str(sum_error) + '\n')
        
        # print(self.layers)
        # print("error: " + str(sum_error))
        print("Weights:")
        print(self.weights)
        for i, weights in enumerate(self.weights):
            np.savetxt("weight_" + str(i) + ".txt", weights)
               
        
        for i in range(len(self.testing_set.data)):
            for j in range(len(self.training_set.data[i])):
                self.layers[0][j] = self.training_set.data[i][j]
            self.forward_propagate()
            solution = np.zeros(10)
            solution[self.training_set.labels[0]] = 1
            print("Solution:")
            print(solution)

            print("Output:")
            print(self.layers[2])
        
