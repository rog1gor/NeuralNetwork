import sys
sys.path.append("../")
import neural_network as nn

def __caluculate_output_test(
        input_file: str, expected_output: int,
        layer_dimensions: list[int], final_dimension: int, failure=False):    
    neural_network = nn.NeuralNetwork(input_file, layer_dimensions, final_dimension)
    score = neural_network.score_input(expected_output)
    if failure:
        assert score > 0.01, f"Expected score is to low - {score}"
    else:
        assert score < 0.01, f"Expected score is to high - {score}"

def __backpropagation_test(
        input_file: str, expected_output: int,
        layer_dimensions: list[int], final_dimension: int):
    neural_network = nn.NeuralNetwork(input_file, layer_dimensions, final_dimension)
    for _ in range(1000):
        _ = neural_network.score_input(expected_output)
        neural_network.backpropagation()
    score = neural_network.score_input(expected_output)
    assert score < 0.1, f"Score is too high: {score}"

def neural_network_unit_tests():
    print("Running Neural Network unit test...")
    __caluculate_output_test("test.png", 0, [32, 16], 1)
    __caluculate_output_test("test.png", 0, [10, 10], 1)
    
    __caluculate_output_test("test.png", 0, [32, 16], 2, failure=True)
    __caluculate_output_test("test.png", 0, [1], 1, failure=True)

    for i in range(10):
        __backpropagation_test("test.png", i, [16, 16], 10)

    print("Neural Network unit tests: PASSED")

if __name__ == "__main__":
    neural_network_unit_tests()