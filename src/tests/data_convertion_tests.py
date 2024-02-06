import numpy as np
import math
import sys
sys.path.append("../")

import data_convertion as dc

neuron  = float
eps     = 0.001 #? Value for comparing floats


def _classify_pixel_test(pixel: list, expected_neuron: neuron):
    pixel = np.array(pixel)
    neuron = dc._classify_pixel(pixel)

    try:
        assert isinstance(neuron, float), "Neurons must be of a float type"
        assert neuron >= 0., "Neurons must have positive values"
        assert neuron <= 1., "Neurons must have value less then 1."
        assert math.fabs(neuron - expected_neuron) < eps, f"""
            Neuron value is different than expected.
            Expected value: {expected_neuron}
            Acutall value: {neuron}
        """
    except AssertionError as err:
        print(err)
        return False
    return True


def classify_pixel_unit_tests():
    print("Running classify pixel tests...")

    passed = True
    #? These pixels should get classified as filled
    passed &= _classify_pixel_test([0.,      0.,     0.],    1.)
    passed &= _classify_pixel_test([0.25,    0.25,   0.25],  1.)
    passed &= _classify_pixel_test([0.5,     0.5,    0.5],   1.)
    passed &= _classify_pixel_test([1.,      0.5,    0.5],   1.)
    passed &= _classify_pixel_test([0.8,     0.8,    0.2],   1.)

    #? These pixels shoudl get classified as empty
    passed &= _classify_pixel_test([1.,      1.,     1.],    0.)
    passed &= _classify_pixel_test([0.9,     0.9,    0.9],   0.)
    passed &= _classify_pixel_test([0.8,     0.8,    0.8],   0.)
    passed &= _classify_pixel_test([1.,      0.8,    0.5],   0.)
    passed &= _classify_pixel_test([1.,      0.7,    0.7],   0.)

    if passed:
        print("Classify pixel tests: PASSED\n")
    else:
        print("Classify pixel tests: FAILED\n")


def img_to_neurons_unit_test(img_path: str):
    print("Running image to neurons test...")
    
    neurons = dc.img_to_neurons(img_path)
    try:
        assert len(neurons.shape) == 1, "Neurons should be a one dimentional vector"
        for neuron in neurons:
            assert neuron == 0. or neuron == 1., f"""
                Neurons must have value either 1. or 0.
                Acutall value: {neuron}
            """
    except AssertionError as err:
        print(err)
        print("Image to neurons test: FAILED\n")
    else:
        print("Image to neurons test: PASSED\n")


if __name__ == "__main__":
    classify_pixel_unit_tests()
    img_to_neurons_unit_test("test.png")
