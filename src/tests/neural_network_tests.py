import sys
sys.path.append("../")

import neural_network as nn


def _caluculate_output_test(
        input_file: str, expected_output: int,
        layer_dimensions: list[int], final_dimension: int, failure=False):    
    neural_network = nn.NeuralNetwork(input_file, layer_dimensions, final_dimension)
    score = neural_network.score_input(expected_output)
    if failure:
        try:
            assert score > 0.01, f"Expected score is to low - {score}"
        except AssertionError as err:
            print(err)
            return False
    else:
        try:
            assert score < 0.01, f"Expected score is to high - {score}"
        except AssertionError as err:
            print(err)
            return False
    return True


def _backpropagation_test(
        input_file: str, expected_output: int,
        layer_dimensions: list[int], final_dimension: int):
    neural_network = nn.NeuralNetwork(input_file, layer_dimensions, final_dimension)
    for _ in range(1000):
        _ = neural_network.score_input(expected_output)
        neural_network.backpropagation()
    score = neural_network.score_input(expected_output)
    try:
        assert score < 0.1, f"Score is too high: {score}"
    except AssertionError as err:
        print(err)
        return False
    return True


def neural_network_unit_tests():
    print("Running Neural Network unit test...")
    
    passed = True
    passed &= _caluculate_output_test("test.png", 0, [32, 16], 1)
    passed &= _caluculate_output_test("test.png", 0, [10, 10], 1)
    
    passed &= _caluculate_output_test("test.png", 0, [32, 16], 2, failure=True)
    passed &= _caluculate_output_test("test.png", 0, [1], 1, failure=True)

    for i in range(10):
        passed &= _backpropagation_test("test.png", i, [16, 16], 10)

    if passed:
        print("Neural Network unit tests: PASSED")
    else:
        print("Neural Network unit tests: FAILED")


if __name__ == "__main__":
    neural_network_unit_tests()