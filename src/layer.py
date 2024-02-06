import numpy as np
import math

#? neuron is a float between 0. and 1.
neuron = float

class Layer:
    #& Mapping floats to neurons
    @staticmethod
    def __sigmoid(x: float) -> neuron:
        try:
            result = 1 / (1 + math.exp(-x))
        except OverflowError:
            result = float('inf')
        return result

    def __init__(self, num_neurons: int, num_following_neurons: int):
        #? num_neurons(N): number of neurons in this layer
        #? num_following_neurons(M): number of neurons in the following layer
        
        #? N dimentional vectors
        self.neurons            = np.zeros((num_neurons,))
        self.neuron_derivatives = np.zeros((num_neurons,))

        #? M dimentional vectors
        self.biases             = np.ones((num_following_neurons,))
        self.bias_derivatives   = np.zeros((num_following_neurons,))

        #? N x M matrix
        self.weigths            = np.ones((num_neurons, num_following_neurons))
        self.weigth_derivatives = np.zeros((num_neurons, num_following_neurons))

        #? sigmoid function gradient
        self.sigmoid_derivatives = np.zeros((num_following_neurons,))

    #& Calculates neurons in a following layer
    def calculate_following_neurons(self) -> np.ndarray[neuron]:
        next_layer = np.dot(self.neurons, self.weigths)
        next_layer += self.biases
        mapper = np.vectorize(Layer.__sigmoid)  #? Mapping Real numbers to [0., 1.] interval
        return mapper(next_layer)
    
    #& Sets all gradients to zero
    def __reset_derivatives(self):
        N = self.neurons.shape[0]
        M = self.biases.shape[0]
        self.neuron_derivatives     = np.zeros((N,))
        self.bias_derivatives       = np.zeros((M,))
        self.weigth_derivatives     = np.zeros((N, M))
        self.sigmoid_derivatives    = np.zeros((M,))

    def __update_sigmoid_derivatives(self, following_layer_derivatives: np.ndarray[float]) -> None:
        mapper = np.vectorize(lambda x: Layer.__sigmoid(x) * Layer.__sigmoid(1 - x) * x)
        self.sigmoid_derivatives = mapper(following_layer_derivatives)
    
    def __update_bias_derivatives(self) -> None:
        self.bias_derivatives = self.sigmoid_derivatives

    def __update_weight_derivatives(self) -> None:
        self.weigth_derivatives = np.dot(self.neurons[np.newaxis].T, self.sigmoid_derivatives[np.newaxis])

    def __update_neuron_derivatives(self) -> None:
        self.neuron_derivatives = np.apply_along_axis(lambda x: np.sum(x * self.sigmoid_derivatives), axis=1, arr=self.weigths)

    #& Calculates new gradients based on layers in the following layer
    #& Returns new neuron gradients
    def calculate_derivatives(self, prev_layer_derivatives: np.ndarray[float]) -> np.ndarray[float]:
        self.__reset_derivatives()
        self.__update_sigmoid_derivatives(prev_layer_derivatives)
        self.__update_bias_derivatives()
        self.__update_weight_derivatives()
        self.__update_neuron_derivatives()
        return self.neuron_derivatives
    
    def update_parameters(self):
        self.biases     -= self.bias_derivatives
        self.weigths    -= self.weigth_derivatives