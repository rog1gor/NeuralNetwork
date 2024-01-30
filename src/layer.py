import numpy as np
import matplotlib.image as mpimg

import data_convertion as dc

class Layer:
    def __init__(self, num_neurons: int, num_next_neurons: int):
        #? Storing neurons in a N dimentional vector transposition
        self.neurons = np.zeros((1, num_neurons))

        #? Storing weigths in a N x M matrix where next layer 
        #? stores neurons in a M dimentional vector transposition
        self.weigths = np.random.uniform(-1., 1., size=(num_neurons, num_next_neurons))

    def calculate_next_neurons(self) -> np.ndarray:
        return np.dot(self.neurons, self.weigths)

    
class NeuralNetwork:
    def __init__(self, layer_dimensions: np.array, final_dimension: int):
        np.append(final_dimension, arr=layer_dimensions)

        #? Initialize all hidden layers
        #? (remember that final dimension was appended)
        self.layers = np.array(
            [
                Layer(layer_dimensions[i], layer_dimensions[i+1])
                for i in range(len(layer_dimensions)-1)
            ]
        )

        #? Create our final layer
        self.final_layer = np.zeros((1, final_dimension))

    def calculate_final_layer(self):
        for index, layer in enumerate(self.layers):
            new_neurons = layer.calculate_next_neurons()
            
            #? If we're at the final hidden layer, then update final layer
            if index == len(self.layers) - 1:
                self.final_layer = new_neurons
            #? Otherwise update next hidden layer
            else:
                self.layers[index+1] = new_neurons