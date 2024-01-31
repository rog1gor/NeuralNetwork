import numpy as np
import math

import data_convertion as dc

#? neuron is a float in [0., 1.] interval
neuron = float

class Layer:
    #& Mapping floats to neurons
    @staticmethod
    def __sigmoid(x: float) -> neuron:
        return 1 / (1 + math.exp(-x))
    
    def __init__(self, num_neurons: int, num_following_neurons: int):
        #? num_neurons(N): number of neurons in this layer
        #? num_following_neurons(M): number of neurons in the following layer
        
        #? N dimentional vectors
        self.neurons            = np.zeros((num_neurons,))
        self.neuron_gradients   = np.zeros((num_neurons,))

        #? M dimentional vectors
        self.biases         = np.ones((num_following_neurons,))
        self.bias_gradients = np.zeros((num_following_neurons,))

        #? N x M matrix
        self.weigths            = np.ones((num_neurons, num_following_neurons))
        self.weigth_gradients   = np.zeros((num_neurons, num_following_neurons))

        #? sigmoid function gradient
        self.sigmoid_gradient   = np.zeros((num_following_neurons,))

    #& Calculates neurons in a following layer
    def calculate_following_neurons(self) -> np.ndarray[neuron]:
        next_layer = np.dot(self.neurons, self.weigths)
        next_layer += self.biases
        mapper = np.vectorize(Layer.__sigmoid)  #? Mapping Real numbers to [0., 1.] interval
        return mapper(next_layer)
    
    #& Sets all gradients to zero
    def __reset_gradients(self):
        N = self.neurons.shape[0]
        M = self.biases.shape[0]
        self.neuron_gradients   = np.zeros((N,))
        self.bias_gradients     = np.zeros((M,))
        self.weigth_gradients   = np.zeros((N, M))
        self.sigmoid_gradient   = np.zeros((M,))

    def __update_sigmoid_gradients(self, prev_layer_gradients: np.array[float]) -> None:
        #todo
        pass
    
    def __update_bias_gradients(self) -> None:
        #todo
        pass

    def __update_weight_gradients(self) -> None:
        #todo
        pass

    def __update_neuron_gradients(self) -> None:
        #todo
        pass

    #& Calculates new gradients based on layers in the following layer
    #& Returns new neuron gradients
    def calculate_gradients(self, prev_layer_gradients: np.array[float]) -> np.array[float]:
        self.__reset_gradients()
        self.__update_sigmoid_gradients(prev_layer_gradients)
        self.__update_bias_gradients()
        self.__update_weight_gradients()
        self.__update_neuron_gradients()
        return self.neuron_gradients


class NeuralNetwork:
    #& Feeds the input layer with the image
    def __create_input_layer(self, input_path: str) -> None:
        neurons = dc.img_to_neurons(input_path)
        self.input_layer = Layer(neurons.shape[0], self.hidden_layers[0].neurons.shape[0])
        self.input_layer.neurons = neurons

    #& Creates a vector that the output layer should be simmilar to
    def __create_expected_output(self, label_position: int) -> None:
        #? We're expecting that the neural network should choose only one output.
        #? Hence the expected vector is all zeros expect at the position that relates to the label.
        self.expected_output = np.zeros((self.output_layer.shape[0]))
        self.expected_output[label_position] = 1.

    #& Updates the output layer based on the input layer
    def __update_output_layer(self) -> None:
        #? Calculate first hidden layer
        self.hidden_layers[0].neurons = self.input_layer.calculate_following_neurons()
        
        #? Dinamically calculate all other layers
        for index, layer in enumerate(self.hidden_layers):
            new_neurons = layer.calculate_following_neurons()
            
            #? If this is the last hidden layer, then update the output layer
            if index == len(self.hidden_layers) - 1:
                self.output_layer = new_neurons
            #? Otherwise update next hidden layer
            else:
                self.hidden_layers[index+1].neurons = new_neurons

    #& Evaluates how good did the neural network perform
    def __scoring_function(self) -> float:
        #? The score is a sum of the scores of each neuron
        #? Score of each neuron is a square of the difference
        #? between the neuron and the expected neuron
        mapper = np.vectorize(lambda x: x*x)
        return np.sum(
            mapper(self.output_layer - self.expected_output))
    
    def __calculate_output_layer_gradient() -> np.array[float]:
        #todo
        pass

    def __update_gradients(self) -> None:
        prev_layer_gradients = self.__calculate_output_layer_gradient()
        for layer in np.flip(self.hidden_layers):
            prev_layer_gradients = layer.calculate_gradients(prev_layer_gradients)
            

    def __update_parameters(self):
        #todo
        pass

    def __init__(self, input_path: str, layer_dimensions: np.array[int], final_dimension: int):
        assert len(layer_dimensions) > 0,   "There should be at least one layer!"
        assert min(layer_dimensions) > 0,   "All layers must have dimensions at least 1"
        assert final_dimension > 0,         "Final dimension must be at least 1"

        layer_dimensions = np.append(
            arr=layer_dimensions, values=[final_dimension])

        #? Initialize all hidden layers
        #? (remember that final dimension was appended)
        self.hidden_layers = np.array(
            [
                Layer(layer_dimensions[i], layer_dimensions[i+1])
                for i in range(len(layer_dimensions)-1)
            ]
        )

        #? Create output layer
        self.output_layer = np.zeros((final_dimension,))
        #? Create input layer
        self.__create_input_layer(input_path)

    #& Calculates the score based on current parameters
    def score_input(self, expected_output: int) -> float:
        self.__create_expected_output(expected_output)
        self.__update_output_layer()
        return self.__scoring_function()

    #& Updates parameters based on the score difference
    def backpropagation(self) -> None:
        self.__update_gradients()
        self.__update_parameters()