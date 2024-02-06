import numpy as np

import data_convertion as dc
from layer import Layer

class NeuralNetwork:
    #& Feeds the input layer with the image
    def _create_input_layer(self, input_path: str) -> None:
        neurons = dc.img_to_neurons(input_path)
        self.input_layer = Layer(
            neurons.shape[0], self.hidden_layers[0].neurons.shape[0])
        self.input_layer.neurons = neurons

    #& Creates a vector that the output layer should be simmilar to
    def _create_expected_output(self, label_position: int) -> None:
        #? We're expecting that the neural network should choose only one output.
        #? Hence the expected vector is all zeros expect at the position that relates to the label.
        self.expected_output = np.zeros((self.output_layer.shape[0]))
        self.expected_output[label_position] = 1.

    #& Updates the output layer based on the input layer
    def _update_output_layer(self) -> None:
        #? Calculate first hidden layer
        self.hidden_layers[0].neurons = \
            self.input_layer.calculate_following_neurons()
        
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
    def _evaluate_output_layer(self) -> float:
        #? The score is a sum of the scores of each neuron
        #? Score of each neuron is a square of the difference
        #? between the neuron and the expected neuron
        mapper = np.vectorize(lambda x: x*x)
        return np.sum(
            mapper(self.output_layer - self.expected_output))
    
    #& Calculated derivatives for neurons in output layer
    def _calculate_output_layer_derivatives(self) -> np.ndarray[float]:
        mapper = np.vectorize(lambda x: self.score / 2*x)
        return mapper(self.output_layer - self.expected_output)

    #& Updates derivatives for each layer
    def _update_gradient(self) -> None:
        following_layer_derivatives = self._calculate_output_layer_derivatives()
        for layer in np.flip(self.hidden_layers):
            following_layer_derivatives = \
                layer.calculate_derivatives(following_layer_derivatives)
    
        self.input_layer.calculate_derivatives(following_layer_derivatives)
        
    #& Updates every layer parameters based on calculated gradients
    def _update_parameters(self) -> None:
        self.input_layer.update_parameters()
        for layer in self.hidden_layers:
            layer.update_parameters()


    def __init__(self, input_path: str,
                 layer_dimensions: np.ndarray[int],
                 final_dimension: int):
        assert len(layer_dimensions) > 0, \
            "There should be at least one layer!"
        assert min(layer_dimensions) > 0, \
            "All layers must have dimensions at least 1"
        assert final_dimension > 0, \
            "Final dimension must be at least 1"

        layer_dimensions = np.append(
            arr=layer_dimensions, values=[final_dimension])

        #? Initialize all hidden layers
        #? (remember that final dimension was appended)
        self.hidden_layers = np.array([
            Layer(layer_dimensions[i], layer_dimensions[i+1])
            for i in range(len(layer_dimensions)-1)
        ])

        #? Create output layer
        self.output_layer = np.zeros((final_dimension,))
        #? Create input layer
        self._create_input_layer(input_path)


    #& Sets neurons of the input layer accordingly to a given input
    def feed_input_layer(self, input_path: str) -> None:
        self.input_layer.neurons = dc.img_to_neurons(input_path)


    #& Calculates the score based on current parameters
    def score_input(self, expected_output: int) -> float:
        self._create_expected_output(expected_output)
        self._update_output_layer()
        self.score = self._evaluate_output_layer()
        return self.score


    #& Updates parameters based on the score difference
    def backpropagation(self) -> None:
        self._update_gradient()
        self._update_parameters()