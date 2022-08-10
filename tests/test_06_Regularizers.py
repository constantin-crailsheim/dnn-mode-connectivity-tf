import itertools
import random
from typing import List, Tuple

import numpy as np
import pytest
import tensorflow as tf
from mode_connectivity.layers import DenseCurve


@pytest.fixture(params=[4])
def units(request):
    return request.param


@pytest.fixture(params=[True, False])
def fix_start(request):
    return request.param


@pytest.fixture(params=[True, False])
def fix_end(request):
    return request.param


@pytest.fixture(params=[0, 1, 2])
def fix_points(request, fix_start, fix_end):
    """Returns all possible combinations of fixed points of different lengths."""
    return [fix_start] + [False] * request.param + [fix_end]


@pytest.fixture(params=[None, tf.keras.regularizers.L2(0.5)])
def kernel_regularizer(request):
    return request.param


@pytest.fixture(params=[None, tf.keras.regularizers.L2(0.5)])
def bias_regularizer(request):
    return request.param


@pytest.fixture
def initialized_curve_layer(units, fix_points, kernel_regularizer, bias_regularizer):
    return DenseCurve(
        units=units,
        fix_points=fix_points,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
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
    def test_fit_loss_correct(self, model_dense: tf.keras.Model):
        """Test if the losses of .fit() and a custom loop are equivalent."""
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
