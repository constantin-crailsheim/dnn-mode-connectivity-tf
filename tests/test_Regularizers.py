import itertools
import random
from typing import List, Tuple

import numpy as np
import pytest
import tensorflow as tf
from mode_connectivity.curves.layers import DenseCurve


@pytest.fixture(params=[2, 4])
def units(request):
    return request.param


def bool_combinations(
    lengths: List[int], pick: float = 1.0, at_least_one: bool = True
) -> List[Tuple[bool]]:
    """Returns all possible combinations of booleans in a list.

    Allows to pick only a certain percentage of elements.
    This allows to test lots of possible combinations, but not all,
    since this can be computationally expensive.

    Args:
        lengths (List[int]): List lengths to generate.
        pick (float, optional): Allows to only pick some elements. Defaults to 1.0.
        at_least_one (bool): Pick at least one combination for each length.

    Returns:
        List[Tuple[bool]]: List of Tuples with bool combinations.

    Example:

    >>> bool_combinations([2])
     [(False, False), (True, False), (False, True), (True, True)]
    >>> bool_combinations([2, 3])
     [(False, False), (True, False), ...,
      (False, False, False), (True, False, False), ..., (True, True, True)]

    >>> bool_combinations([2], pick=0.5)
     [(True, False), (True, True)]
    """
    combinations = []
    for l in lengths:
        combs = list(itertools.product([True, False], repeat=l))
        samples = combs
        if pick != 1.0:
            n_samples = int(pick * len(combs))
            samples = random.sample(combs, n_samples)
        if at_least_one and not samples:
            samples = random.sample(combs, 1)
        combinations += samples
    return combinations


@pytest.fixture(params=bool_combinations([2, 3, 4, 5, 6], pick=0.05))
def fix_points(request):
    return request.param


@pytest.fixture(params=[None, "ones"])
def kernel_initializer(request):
    return request.param


@pytest.fixture(params=[None, tf.keras.regularizers.L2(0.5)])
def kernel_regularizer(request):
    return request.param


@pytest.fixture
def initialized_curve_layer(units, fix_points, kernel_initializer, kernel_regularizer):
    return DenseCurve(
        units=units,
        fix_points=fix_points,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
    )


@pytest.fixture
def model_dense(initialized_curve_layer) -> tf.keras.Model:
    n_fix_points = len(initialized_curve_layer.fix_points)
    curve_weights = tf.ones((n_fix_points,)) / n_fix_points

    class ModelDense(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense = initialized_curve_layer

        def call(self, inputs, **kwargs):
            x = self.dense((inputs, curve_weights))
            return x

    return ModelDense()


class TestRegularizer:
    def test_fit_loss_correct(self, model_dense):
        shape = (model_dense.dense.units, model_dense.dense.units)
        loss = tf.keras.losses.MeanAbsoluteError()
        model_dense.compile(
            optimizer=tf.keras.optimizers.SGD(),
            loss=loss,
        )
        inputs = tf.ones(shape=shape)
        targets = tf.ones(shape=shape) * 2

        # Calculate loss manually
        # 1) loss from loss function
        # 2) If we regularize, add regularization loss
        # 2.1) for kernel
        # 2.2) for bias
        outputs = model_dense(inputs)
        expected_loss = loss(outputs, targets)
        kernel_reg = model_dense.dense.kernel_regularizer
        if kernel_reg:
            expected_loss += tf.math.add_n(
                kernel_reg(w) for w in model_dense.dense.curve_kernels
            )
        bias_reg = model_dense.dense.kernel_regularizer
        if bias_reg:
            expected_loss += tf.math.add_n(
                bias_reg(b) for b in model_dense.dense.curve_biases
            )

        print("Expected Loss", expected_loss)
        results = model_dense.fit(inputs, targets)
        assert np.allclose(results.history["loss"], expected_loss)
