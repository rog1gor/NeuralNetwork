import numpy as np
import math
import random
import sys
sys.path.append("../")

import layer

eps = 0.001 #? Float comparison

def _calculate_following_neurons_test(
        this_neurons: list[int], expected_neurons: list[int]):
    L = layer.Layer(len(this_neurons), len(expected_neurons))
    L.neurons = np.array(this_neurons)

    for i in range(len(expected_neurons)):
        for j in range(len(this_neurons)):
            if expected_neurons[i] < eps:
                L.weigths[j][i] = -1.
            else:
                L.weigths[j][i] = 1.

    result = L.calculate_following_neurons()
    try:
        max_diff = max([
            math.fabs(r - e) for (r, e) in zip(result, expected_neurons)
        ])
        assert max_diff <= eps, f"Difference is too high: {max_diff}"
    except AssertionError as err:
        print(err)
        return False
    return True


def layer_unit_tests():
    print("Running Layer unit tests...")
    passed = True

    passed &= _calculate_following_neurons_test(
        [1. for _ in range(10)], [0. for _ in range(10)]
    )
    passed &= _calculate_following_neurons_test(
        [1. for _ in range(10)], [1. for _ in range(10)]
    )

    for i in range(10):
        def val_at_pos(x, pos):
            if x == pos:
                return 1.
            return 0.
        
        passed &= _calculate_following_neurons_test(
            [random.random() for _ in range(100)], [val_at_pos(i, j) for j in range(10)]
        )

    if passed:
        print("Layer unit tests: PASSED\n")
    else:
        print("Layer unit tests: FAILED\n")


if __name__ == "__main__":
    layer_unit_tests()